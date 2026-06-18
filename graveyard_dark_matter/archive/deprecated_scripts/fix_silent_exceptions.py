"""Fix silent pass exceptions by replacing pass with logger.debug"""
import os
import re

def fix_silent_exceptions():
    SCAN_DIRS = ['core', 'execution', 'risk', 'strategies']
    
    for d in SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            for f in files:
                if not f.endswith('.py'):
                    continue
                fp = os.path.join(root, f)
                with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                    lines = fh.readlines()
                
                changed = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped == 'pass' and i > 0:
                        prev = lines[i-1].strip()
                        if prev.startswith('except') and 'Exception' in prev:
                            has_log = False
                            for j in range(max(0, i-3), min(len(lines), i+3)):
                                if 'logger' in lines[j] or 'print' in lines[j]:
                                    has_log = True
                            if not has_log:
                                # Extract exception variable if present
                                match = re.match(r'except\s+Exception\s+as\s+(\w+):', prev)
                                var_name = match.group(1) if match else "e"
                                
                                if not match:
                                    # If it was just 'except Exception:', change it to 'except Exception as e:'
                                    prev_indent = lines[i-1][:len(lines[i-1]) - len(lines[i-1].lstrip())]
                                    lines[i-1] = prev_indent + "except Exception as e:\n"
                                
                                indent = line[:len(line) - len(line.lstrip())]
                                lines[i] = indent + f'logger.debug(f"Silent exception caught: {{e}}")\n'
                                changed += 1
                
                if changed > 0:
                    with open(fp, 'w', encoding='utf-8') as fh:
                        fh.writelines(lines)
                    print(f"Fixed {changed} silent exceptions in {fp}")

fix_silent_exceptions()
