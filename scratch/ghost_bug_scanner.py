import ast
import os
import glob

def check_file_for_time_bombs(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception as e:
        return [f"PARSE ERROR: {e}"]
        
    bombs = []
    has_asyncio = 'asyncio' in content
    
    for node in ast.walk(tree):
        # 1. Mutable default arguments
        if isinstance(node, ast.arguments):
            for default in node.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    bombs.append(f"Line {node.lineno}: Mutable default argument (list/dict/set) can cause state leakage.")
                    
        # 2. Bare Excepts or swallowed exceptions
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                bombs.append(f"Line {node.lineno}: Bare 'except:' found. Can catch KeyboardInterrupt and mask fatal errors.")
            elif getattr(node.type, 'id', '') == 'Exception':
                # Check if it lacks logger or raise
                body_code = ast.unparse(node.body) if hasattr(ast, 'unparse') else ""
                if 'log' not in body_code and 'raise' not in body_code and 'return' not in body_code:
                    bombs.append(f"Line {node.lineno}: 'except Exception:' silently swallows errors (Ghost Bug).")
                    
        # 3. Floating point direct equality
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    # Hard to know types statically, but we can flag literal float comparisons
                    for comp in [node.left] + node.comparators:
                        if isinstance(comp, ast.Constant) and isinstance(comp.value, float):
                            bombs.append(f"Line {node.lineno}: Direct float equality comparison (use math.isclose).")
                            
        # 4. time.sleep in asyncio contexts
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if getattr(node.func.value, 'id', '') == 'time' and node.func.attr == 'sleep':
                    if has_asyncio:
                        bombs.append(f"Line {node.lineno}: time.sleep() found in an asyncio module. Will block the entire event loop.")
                        
        # 5. Unbounded appending (heuristics: while True loops with appends)
        if isinstance(node, ast.While):
            body_code = ast.unparse(node.body) if hasattr(ast, 'unparse') else ""
            if '.append(' in body_code and 'len(' not in body_code and 'pop(0)' not in body_code and 'clear()' not in body_code:
                # Potential memory leak if loop runs forever
                if ast.unparse(node.test) == 'True':
                    bombs.append(f"Line {node.lineno}: 'while True:' contains '.append()' without visible boundary checks (Memory leak bomb).")

    return bombs

def main():
    root_dir = r"C:\\Users\\jhona\\Documents\\Proyectos\\Trader Gemini"
    all_py_files = glob.glob(os.path.join(root_dir, '**', '*.py'), recursive=True)
    
    total_bombs = 0
    for file in all_py_files:
        if 'venv' in file or '__pycache__' in file or 'scratch' in file:
            continue
            
        bombs = check_file_for_time_bombs(file)
        if bombs:
            print(f"\\n🚨 BUGS FANTASMAS EN: {os.path.relpath(file, root_dir)}")
            for b in bombs:
                print(f"   - {b}")
                total_bombs += 1
                
    if total_bombs == 0:
        print("\\n✅ No se detectaron Bombas de Tiempo algorítmicas mediante AST.")
    else:
        print(f"\\n⚠️ Se encontraron {total_bombs} posibles Bombas de Tiempo.")

if __name__ == '__main__':
    main()
