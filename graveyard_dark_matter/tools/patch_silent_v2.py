import re
import os

files_to_patch = [
    'execution/binance_executor.py',
    'strategies/ml_strategy.py',
    'strategies/technical.py',
    'core/engine.py',
    'risk/risk_manager.py',
    'data/binance_loader.py'
]

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    patched_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        patched_lines.append(line)
        
        # Match 'except Exception:' or 'except Exception as e:'
        match = re.match(r'^(\s*)except\s+Exception(?:.*):\s*$', line)
        if match:
            indent = match.group(1)
            # Check next lines to see if there is logging
            j = i + 1
            has_log = False
            has_raise = False
            end_of_block = j
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip() == '':
                    j += 1
                    continue
                next_indent_match = re.match(r'^(\s+)', next_line)
                if next_indent_match:
                    next_indent = next_indent_match.group(1)
                    if len(next_indent) <= len(indent):
                        end_of_block = j
                        break
                else:
                    end_of_block = j
                    break
                    
                if 'logger.error' in next_line or 'logger.exception' in next_line or 'logger.critical' in next_line:
                    has_log = True
                if 'raise' in next_line:
                    has_raise = True
                j += 1
                
            if not has_log and not has_raise:
                # Inject a logging statement right after the except block
                # Determine inner indentation
                inner_indent = indent + "    "
                for k in range(i+1, min(len(lines), j)):
                    if lines[k].strip():
                        inner_indent_match = re.match(r'^(\s+)', lines[k])
                        if inner_indent_match:
                            inner_indent = inner_indent_match.group(1)
                            break
                            
                injection = f"{inner_indent}import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)\n"
                patched_lines.append(injection)
        i += 1
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(patched_lines)

for fp in files_to_patch:
    print(f"Patching {fp}")
    patch_file(fp)
