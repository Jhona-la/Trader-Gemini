import os
import ast
from pathlib import Path

def scan_deep_flaws(directory):
    flaws = {
        'blocking_sleep_in_async': [],
        'bare_excepts': [],
        'unbounded_memory': [],
        'missing_timeouts': [],
        'print_in_critical_path': []
    }
    
    for root, _, files in os.walk(directory):
        if 'venv' in root or '__pycache__' in root or '.git' in root: continue
        for file in files:
            if not file.endswith('.py'): continue
            path = Path(root) / file
            
            try:
                content = path.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                # Check line by line heuristics
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    # 1. Memory leaks (unbounded caches)
                    if ' = {}' in line and 'cache' in line.lower():
                        flaws['unbounded_memory'].append(f"{path.name}:{i+1}")
                    if ' = []' in line and ('.append' in content and 'max' not in content and 'del ' not in content):
                        # Too complex for regex, just mark if we see generic history lists
                        if 'history =' in line or 'records =' in line:
                            flaws['unbounded_memory'].append(f"{path.name}:{i+1}")
                            
                    # 2. Print in critical path
                    if 'print(' in line and 'core' in str(path) and 'engine' in str(path):
                        flaws['print_in_critical_path'].append(f"{path.name}:{i+1}")
                        
                # AST Checks
                for node in ast.walk(tree):
                    # Blocking sleep in async function?
                    if isinstance(node, ast.AsyncFunctionDef):
                        for sub_node in ast.walk(node):
                            if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Attribute):
                                if sub_node.func.attr == 'sleep' and (isinstance(sub_node.func.value, ast.Name) and sub_node.func.value.id == 'time'):
                                    flaws['blocking_sleep_in_async'].append(f"{path.name}:{node.lineno}")
                    
                    # Bare excepts
                    if isinstance(node, ast.ExceptHandler):
                        if node.type is None:
                            flaws['bare_excepts'].append(f"{path.name}:{node.lineno}")
                            
                    # Requests missing timeouts
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if node.func.attr in ('get', 'post') and (isinstance(node.func.value, ast.Name) and node.func.value.id == 'requests'):
                            has_timeout = False
                            for kw in node.keywords:
                                if kw.arg == 'timeout': has_timeout = True
                            if not has_timeout:
                                flaws['missing_timeouts'].append(f"{path.name}:{node.lineno}")

            except Exception as e:
                pass
                
    return flaws

if __name__ == '__main__':
    project_dir = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini"
    res = scan_deep_flaws(project_dir)
    print("=== PROFUNDO AUDIT RESULTS ===")
    for k, v in res.items():
        if v:
            print(f"-- {k.upper()} --")
            for item in set(v): print(f"  {item}")
