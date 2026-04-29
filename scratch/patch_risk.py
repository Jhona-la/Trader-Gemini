import re
import os

path = r'c:\Users\jhona\Documents\Proyectos\Trader Gemini\risk\risk_manager.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

start_idx = text.find('def check_stops')
end_idx = text.find('def _check_momentum_exit')

if start_idx == -1 or end_idx == -1:
    print('Functions not found')
    exit(1)

part1 = text[:start_idx]
part2 = text[start_idx:end_idx]
part3 = text[end_idx:]

def repl(m):
    indent = m.group(1)
    s1 = m.group(0)
    s2 = indent + "setup_type=pos.get('setup_type', 'UNKNOWN_SETUP'),"
    s3 = indent + "trade_id=pos.get('trade_id'),"
    return s1 + '\n' + s2 + '\n' + s3

new_part2 = re.sub(r'([ \t]+)signal_type=SignalType\.EXIT,', repl, part2)

with open(path, 'w', encoding='utf-8') as f:
    f.write(part1 + new_part2 + part3)

print('Done!')
