import os
import ast
import json
from pathlib import Path
from typing import Dict, List

class GhostBugScanner(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.vulnerabilities = []
        self.in_async_func = False
        self.in_while_loop = False

    def report(self, issue_type, line_no, message):
        self.vulnerabilities.append({
            "file": self.filename,
            "line": line_no,
            "type": issue_type,
            "message": message
        })

    def visit_AsyncFunctionDef(self, node):
        old_async = self.in_async_func
        self.in_async_func = True
        self.generic_visit(node)
        self.in_async_func = old_async

    def visit_ExceptHandler(self, node):
        if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
            # Check if body only has 'pass'
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                self.report("SILENT_FAILURE", node.lineno, "Bare except or 'except Exception: pass' detected. This hides critical failures.")
            # Check if there is no logging or raise
            elif not any(isinstance(stmt, ast.Raise) or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr in ('error', 'exception', 'critical')) for stmt in node.body):
                self.report("SILENT_FAILURE", node.lineno, "Exception caught but neither raised nor properly logged.")
        self.generic_visit(node)

    def visit_While(self, node):
        old_while = self.in_while_loop
        self.in_while_loop = True
        
        # In async context, a while loop without await is a blocking time bomb
        if self.in_async_func:
            has_await = any(isinstance(n, ast.Await) for n in ast.walk(node))
            if not has_await:
                self.report("EVENT_LOOP_BLOCK", node.lineno, "while loop inside async function without 'await'. Blocks event loop.")
        
        self.generic_visit(node)
        self.in_while_loop = old_while

    def visit_Global(self, node):
        self.report("GLOBAL_STATE_MUTATION", node.lineno, f"Use of 'global' keyword for variables {node.names}. Highly dangerous in multi-threaded/async HFT.")
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        # Look for mutable default arguments
        for arg in node.args.defaults:
            if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                self.report("MUTABLE_DEFAULT_ARG", node.lineno, "Mutable default argument in function. State leaks across calls.")
        self.generic_visit(node)


def scan_directory(root_dir: str, exclude_dirs: List[str]) -> List[Dict]:
    all_vulns = []
    base_path = Path(root_dir)
    
    for py_file in base_path.rglob('*.py'):
        # Skip excluded dirs
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=str(py_file))
            scanner = GhostBugScanner(str(py_file))
            scanner.visit(tree)
            all_vulns.extend(scanner.vulnerabilities)
        except SyntaxError as e:
            all_vulns.append({
                "file": str(py_file),
                "line": e.lineno,
                "type": "SYNTAX_ERROR",
                "message": f"Syntax error: {e.msg}"
            })
        except Exception as e:
            # Skip unparseable files
            pass
            
    return all_vulns

if __name__ == "__main__":
    project_root = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini"
    excludes = ["venv", ".venv", "__pycache__", ".git", "graveyard", "archive", "scratch", "tests"]
    
    print(f"Scanning project root: {project_root}")
    vulnerabilities = scan_directory(project_root, excludes)
    
    output_file = os.path.join(project_root, "ghost_bugs_report.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vulnerabilities, f, indent=4)
        
    print(f"Found {len(vulnerabilities)} ghost bugs across active project files.")
    print(f"Report saved to {output_file}")
