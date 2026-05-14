import ast
import os

SCAN_DIRS = ['core', 'execution', 'risk', 'strategies', 'utils', 'dashboard']
total_errors = 0

for d in SCAN_DIRS:
    if not os.path.isdir(d): continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if not f.endswith('.py'): continue
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                content = fh.read()
            try:
                ast.parse(content)
            except SyntaxError as e:
                print(f"🔴 SYNTAX ERROR in {fp}:{e.lineno} — {e.msg}")
                total_errors += 1

print(f"\nTotal syntax errors: {total_errors}")
