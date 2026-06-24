import os
import glob
import re

with open('setup_compiler.py', 'r', encoding='utf-8') as f:
    code = f.read()

match = re.search(r'EXCLUDE_FILES = \[(.*?)\]', code, flags=re.DOTALL)
if match:
    files = re.findall(r'"([^"]+)"', match.group(1))
    basenames = [f.replace('.py', '') for f in files]

    deleted = 0
    for d in ['core', 'risk', 'strategies', 'utils', 'analysis']:
        for filepath in glob.glob(os.path.join(d, '**', '*.pyd'), recursive=True):
            name = os.path.basename(filepath)
            module_name = name.split('.')[0]
            if module_name in basenames:
                os.remove(filepath)
                deleted += 1
                print(f'Deleted {filepath}')
    print(f'Total deleted: {deleted}')
