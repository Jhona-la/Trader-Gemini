import os
import ast
import re

base_dir = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini"

def get_files():
    files_to_check = []
    for root, _, files in os.walk(base_dir):
        if 'venv' in root or '.git' in root or '__pycache__' in root: continue
        for file in files:
            if file.endswith('.py'):
                files_to_check.append(os.path.join(root, file))
    return files_to_check

def plan_alfa_shapes_and_types(files):
    results = []
    # Look for numpy/pandas conversions, shape assertions, float32/64
    for f in files:
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            if 'float32' in content and 'float64' in content:
                results.append(f"Precision mismatch risk in {os.path.basename(f)}")
            if '.reshape(' in content or '.astype(' in content:
                results.append(f"Reshaping/Casting found in {os.path.basename(f)}")
            if 'c_portfolio_math' in content or '@njit' in content:
                results.append(f"C/Numba boundary in {os.path.basename(f)}")
    return results

def plan_bravo_async_blocks(files):
    results = []
    for fpath in files:
        if 'engine.py' not in fpath and 'ml_strategy.py' not in fpath: continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            if isinstance(sub.func, ast.Attribute) and sub.func.attr in ['sleep', 'read_csv', 'fit', 'predict']:
                                results.append(f"Blocking call '{sub.func.attr}' inside async '{node.name}' in {os.path.basename(fpath)}")
        except Exception: pass
    return results

def plan_charlie_time_calls(files):
    results = set()
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'time.time()' in content: results.add(f"time.time() in {os.path.basename(fpath)}")
                if 'time.perf_counter()' in content: results.add(f"time.perf_counter() in {os.path.basename(fpath)}")
        except: pass
    return list(results)

def plan_delta_ml(files):
    results = []
    for fpath in files:
        if 'engine.py' in fpath or 'ml_strategy.py' in fpath:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f.readlines()):
                    if 'predict(' in line or 'predict_proba(' in line:
                        results.append(f"Prediction call at {os.path.basename(fpath)}:{i+1}")
                    if 'fillna' in line or 'dropna' in line or 'NaN' in line:
                        results.append(f"NaN handling at {os.path.basename(fpath)}:{i+1}")
    return results

if __name__ == "__main__":
    files = get_files()
    print("=== PLAN ALFA ===")
    for r in plan_alfa_shapes_and_types(files)[:10]: print(r)
    print("\n=== PLAN BRAVO ===")
    for r in plan_bravo_async_blocks(files)[:10]: print(r)
    print("\n=== PLAN CHARLIE ===")
    for r in plan_charlie_time_calls(files)[:10]: print(r)
    print("\n=== PLAN DELTA ===")
    for r in plan_delta_ml(files)[:10]: print(r)
