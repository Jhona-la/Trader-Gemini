import os
import re

root = r'C:\Users\jhona\Documents\Proyectos\Trader Gemini'

print('=== HARDCODED MAGIC NUMBERS IN CRITICAL FILES ===')

critical_files = [
    'strategies/technical.py',
    'strategies/ml_strategy.py',
    'strategies/sniper_strategy.py',
    'strategies/statistical.py',
    'risk/risk_manager.py',
    'core/engine.py',
    'core/portfolio.py',
    'core/feedback_processor.py'
]

for cf in critical_files:
    fp = os.path.join(root, cf)
    if not os.path.exists(fp): continue
    with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = fh.readlines()
    
    magic_count = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('\"\"\"') or not stripped:
            continue
        # Find comparisons with raw numbers, like < 0.05, > 0.003
        if re.search(r'(?:>|<|>=|<=|==)\s*[-+]?0\.\d{2,}', stripped) or re.search(r'(?:>|<|>=|<=|==)\s*\d{2,}\.\d*', stripped):
            magic_count += 1
    
    print(f'{cf}: {magic_count} hardcoded thresholds')

print('\n=== CONFIG.PY ANALYSIS ===')
fp = os.path.join(root, 'config.py')
with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
    cfg = fh.read()
classes = re.findall(r'class\s+(\w+)', cfg)
print(f'Config classes: {classes}')
print(f'Config file size: {os.path.getsize(fp)/1024:.0f}KB')
attrs = len(re.findall(r'\w+\s*=\s*', cfg))
print(f'Approximate config attributes: {attrs}')

print('\n=== EVOLUTION/ADAPTABILITY INTEGRATION CHECK ===')
engine_fp = os.path.join(root, 'core/engine.py')
with open(engine_fp, 'r', encoding='utf-8', errors='ignore') as fh:
    engine_content = fh.read()

evo_modules = ['evolution', 'meta_optimizer', 'self_tuner', 'genotype']
for em in evo_modules:
    matches = re.findall(rf'from\s+core\.{em}\s+import', engine_content) + re.findall(rf'import\s+core\.{em}', engine_content)
    print(f'Is {em} integrated into Engine? {"YES" if matches else "NO"}')

print('\n=== SOPHIA/IA INTEGRATION CHECK ===')
matches = re.findall(r'from\s+sophia\..*import', engine_content)
print(f'Is Sophia directly imported in Engine? {"YES" if matches else "NO"}')
