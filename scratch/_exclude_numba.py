import os
import glob
import re

# 1. Find all files with 'from numba import' or 'import numba'
numba_files = []
for d in ['core', 'risk', 'strategies', 'utils', 'analysis']:
    for filepath in glob.glob(os.path.join(d, '**', '*.py'), recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'import numba' in content or 'from numba' in content:
                    numba_files.append(os.path.basename(filepath))
        except Exception as e:
            pass

# 2. Update setup_compiler.py
with open('setup_compiler.py', 'r', encoding='utf-8') as f:
    setup_code = f.read()

exclude_match = re.search(r'EXCLUDE_FILES = \[.*?\]', setup_code, flags=re.DOTALL)
if exclude_match:
    old_exclude = exclude_match.group(0)
    current_files = re.findall(r'"([^"]+)"', old_exclude)
    current_files += re.findall(r"'([^']+)'", old_exclude)
    
    all_files = list(set(current_files + numba_files + ['setup_compiler.py', 'setup.py', 'main.py', '__init__.py']))
    
    new_exclude = 'EXCLUDE_FILES = [\n' + ',\n'.join(f'    "{f}"' for f in sorted(all_files)) + '\n]'
    setup_code = setup_code.replace(old_exclude, new_exclude)
    
    with open('setup_compiler.py', 'w', encoding='utf-8') as f:
        f.write(setup_code)
    print(f'Added {len(numba_files)} Numba files to EXCLUDE_FILES.')
