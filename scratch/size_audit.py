import os, sys

files = [
    'strategies/ml_strategy.py',
    'strategies/technical.py',
    'core/portfolio.py',
    'risk/risk_manager.py',
    'sophia/intelligence.py',
    'sophia/nemesis.py',
    'core/engine.py',
    'data/binance_loader.py',
    'execution/binance_executor.py',
    'core/online_learning.py',
    'core/evolution.py',
    'core/shadow_darwin.py',
    'strategies/components/feature_engineering.py',
    'strategies/statistical.py',
    'strategies/sniper_strategy.py',
]

print("=" * 70)
print(f"{'FILE':<50} {'LINES':>8} {'KB':>8}")
print("=" * 70)

total_lines = 0
total_kb = 0
for f in files:
    if os.path.exists(f):
        lines = len(open(f, encoding='utf-8', errors='ignore').readlines())
        kb = os.path.getsize(f) / 1024
        total_lines += lines
        total_kb += kb
        flag = " ⚠️ HUGE" if lines > 2000 else (" 🔶 LARGE" if lines > 1000 else "")
        print(f"{f:<50} {lines:>8} {kb:>7.0f}{flag}")
    else:
        print(f"{f:<50} NOT FOUND")

print("=" * 70)
print(f"{'TOTAL':<50} {total_lines:>8} {total_kb:>7.0f}")

# Count total project files
total_py = 0
for root, dirs, fls in os.walk('.'):
    if '.venv' in root or '__pycache__' in root or 'node_modules' in root:
        continue
    for ff in fls:
        if ff.endswith('.py'):
            total_py += 1
print(f"\nTotal .py files in project: {total_py}")

# Memory analysis: check imports
print("\n--- HEAVY IMPORT ANALYSIS ---")
import importlib
heavy = ['sklearn', 'xgboost', 'talib', 'numpy', 'pandas', 'scipy']
for mod in heavy:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: available")
    except:
        print(f"  {mod}: NOT INSTALLED")
