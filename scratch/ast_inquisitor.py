import ast
import os
import json
from pathlib import Path

class BugInquisitor(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.findings = {
            "silent_get": [],
            "return_zero": [],
            "except_pass": [],
            "dataframe_instantiation": [],
            "time_sleep": [],
            "sync_predict": []
        }

    def add_finding(self, category, line, details):
        self.findings[category].append({
            "file": str(self.filepath),
            "line": line,
            "details": details
        })

    # Hunt for dict.get('key', default)
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
            if len(node.args) == 2:
                # We have a dict.get(x, y) call
                self.add_finding("silent_get", node.lineno, "dict.get('key', default) found")
        
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'DataFrame':
            # Could be pd.DataFrame or pl.DataFrame
            if isinstance(node.func.value, ast.Name) and node.func.value.id in ['pd', 'pl', 'pandas', 'polars']:
                self.add_finding("dataframe_instantiation", node.lineno, "DataFrame instantiated")

        if isinstance(node.func, ast.Attribute) and node.func.attr == 'sleep':
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'time':
                self.add_finding("time_sleep", node.lineno, "time.sleep() found")

        if isinstance(node.func, ast.Attribute) and node.func.attr == 'predict':
            # model.predict()
            self.add_finding("sync_predict", node.lineno, "model.predict() found (potential blocking)")

        self.generic_visit(node)

    # Hunt for return 0.0
    def visit_Return(self, node):
        if node.value is not None:
            if isinstance(node.value, ast.Constant):
                if node.value.value == 0.0 or node.value.value == 0:
                    self.add_finding("return_zero", node.lineno, f"return {node.value.value} found")
        self.generic_visit(node)

    # Hunt for except: pass
    def visit_ExceptHandler(self, node):
        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                self.add_finding("except_pass", node.lineno, "except: pass found")
                break
        self.generic_visit(node)


def scan_directory(root_dir):
    all_findings = {
        "silent_get": [],
        "return_zero": [],
        "except_pass": [],
        "dataframe_instantiation": [],
        "time_sleep": [],
        "sync_predict": []
    }
    
    # Exclude venv and other unnecessary directories
    exclude_dirs = {'.venv', 'venv', 'node_modules', '.git', '__pycache__', 'build', 'dist', 'tests'}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    inquisitor = BugInquisitor(filepath.relative_to(root_dir))
                    inquisitor.visit(tree)
                    
                    for k in all_findings.keys():
                        all_findings[k].extend(inquisitor.findings[k])
                except Exception as e:
                    print(f"Failed to parse {filepath}: {e}")

    return all_findings

if __name__ == "__main__":
    project_root = Path("C:/Users/jhona/Documents/Proyectos/Trader Gemini")
    print(f"Scanning project: {project_root}")
    results = scan_directory(project_root)
    
    output_file = project_root / "scratch" / "ast_findings.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

    # Summary
    print("\n--- AST INQUISITOR SUMMARY ---")
    for k, v in results.items():
        print(f"{k.upper()}: {len(v)} occurrences")
