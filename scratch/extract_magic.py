import os
import re

root = r'C:\Users\jhona\Documents\Proyectos\Trader Gemini'
fp = os.path.join(root, 'strategies', 'ml_strategy.py')

with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
    lines = fh.readlines()

print('=== MAGIC NUMBERS IN ML_STRATEGY ===')
count = 0
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if stripped.startswith('#') or stripped.startswith('\"\"\"') or not stripped:
        continue
    # Find lines with float comparisons
    if re.search(r'(?:>|<|>=|<=|==)\s*[-+]?0\.\d{2,}', stripped) or re.search(r'(?:>|<|>=|<=|==)\s*\d{2,}\.\d*', stripped):
        print(f'Line {i}: {stripped}')
        count += 1
