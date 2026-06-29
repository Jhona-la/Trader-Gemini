import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
import importlib

def extract_defaults(filepath):
    import ast
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except FileNotFoundError:
        return {}
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg, default in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
                if isinstance(default, ast.Constant):
                    defaults[f"{node.name}.{arg.arg}"] = default.value
    return defaults

def compare():
    engine_defaults = extract_defaults("core/engine.py")
    backtest_defaults = extract_defaults("core/backtest_infra.py")
    god_mode_defaults = extract_defaults("scripts/run_god_mode_backtest.py")

    print("--- DIVERGENCES ---")
    for key, val in backtest_defaults.items():
        engine_key = key
        if engine_key in engine_defaults and engine_defaults[engine_key] != val:
            print(f"Divergence in {key}: Backtest={val}, Production={engine_defaults[engine_key]}")

    print("\n--- GOD MODE OVERRIDES ---")
    with open("scripts/run_god_mode_backtest.py", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Config." in line and "=" in line:
                print(f"L{i+1}: {line.strip()}")

if __name__ == "__main__":
    compare()
