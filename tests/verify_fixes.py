"""Quick verification of the 6 critical bug fixes."""
import sys
sys.path.insert(0, '.')

print('=' * 60)
print('VERIFICACIÓN RÁPIDA DE FIXES')
print('=' * 60)

passed = 0
total = 8

# FIX 1-2: Engine duplicate imports
with open('core/engine.py', 'r', encoding='utf-8-sig') as f:
    engine_src = f.read()
    lines = engine_src.split('\n')

config_imports = [i for i, l in enumerate(lines, 1) if 'from config import Config' in l]
logger_imports = [i for i, l in enumerate(lines, 1) if 'from utils.logger import logger' in l]

ok1 = len(config_imports) == 1
ok2 = len(logger_imports) == 1
print(f"  [1] Config imports: {len(config_imports)} (expected 1) -> {'PASS' if ok1 else 'FAIL'}")
print(f"  [2] Logger imports: {len(logger_imports)} (expected 1) -> {'PASS' if ok2 else 'FAIL'}")
passed += ok1 + ok2

# FIX 3-4: Engine burst mode
has_deque_bug = 'self.events._deque.popleft()' in engine_src
has_deques_fix = 'self.events._deques[p]' in engine_src
ok3 = not has_deque_bug
ok4 = has_deques_fix
print(f"  [3] Burst _deque bug gone: {'PASS' if ok3 else 'FAIL'}")
print(f"  [4] Burst _deques fix present: {'PASS' if ok4 else 'FAIL'}")
passed += ok3 + ok4

# FIX 5-8: Portfolio imports
with open('core/portfolio.py', 'r', encoding='utf-8-sig') as f:
    port_src = f.read()

typing_line = ''
for line in port_src.split('\n'):
    if 'from typing import' in line:
        typing_line = line
        break

ok5 = 'Tuple' in typing_line
ok6 = 'Optional' in typing_line
ok7 = 'import numpy as np' in port_src
ok8 = 'from core.data_handler import get_data_handler' in port_src

print(f"  [5] Tuple imported: {'PASS' if ok5 else 'FAIL'}")
print(f"  [6] Optional imported: {'PASS' if ok6 else 'FAIL'}")
print(f"  [7] numpy imported: {'PASS' if ok7 else 'FAIL'}")
print(f"  [8] get_data_handler lazy import: {'PASS' if ok8 else 'FAIL'}")
passed += ok5 + ok6 + ok7 + ok8

all_pass = passed == total
icon = "✅" if all_pass else "❌"
print(f"\n  RESULTADO: {icon} {passed}/{total} FIXES VERIFICADOS")
