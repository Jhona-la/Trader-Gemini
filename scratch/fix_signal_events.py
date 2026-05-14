"""
FIX-FORENSIC-V82: Remove invalid 'direction=' parameter from all SignalEvent() calls in risk_manager.py.
SignalEvent does NOT have a 'direction' field (only OrderEvent does).
This causes TypeError which silently kills ALL exit signals from RiskManager.
"""
import re

filepath = r'risk\risk_manager.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# Pattern: remove lines like "                            direction=pos.get("pos_side", "LONG"),"
# These appear inside SignalEvent() constructor calls
# We need to be careful to only remove inside SignalEvent blocks, not OrderEvent blocks

lines = content.split('\n')
new_lines = []
in_signal_event = False
signal_event_depth = 0

i = 0
removed_count = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Track if we're inside a SignalEvent( block
    if 'SignalEvent(' in stripped and 'OrderEvent' not in stripped:
        in_signal_event = True
        signal_event_depth = line.count('(') - line.count(')')
        new_lines.append(line)
        i += 1
        continue
    
    if in_signal_event:
        signal_event_depth += line.count('(') - line.count(')')
        
        # Remove invalid params: direction=, quantity=, is_exit=, is_close=
        if stripped.startswith('direction=') and 'pos_side' in stripped:
            removed_count += 1
            i += 1
            continue
        if stripped.startswith('quantity=abs(qty)'):
            removed_count += 1
            i += 1
            continue
        if stripped.startswith('is_exit='):
            removed_count += 1
            i += 1
            continue
        if stripped.startswith('is_close='):
            removed_count += 1
            i += 1
            continue
            
        if signal_event_depth <= 0:
            in_signal_event = False
    
    new_lines.append(line)
    i += 1

new_content = '\n'.join(new_lines)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Removed {removed_count} invalid parameters from SignalEvent() calls")

# Verify
verify_content = open(filepath, 'r', encoding='utf-8').read()
remaining = sum(1 for i, line in enumerate(verify_content.split('\n')) 
                if 'direction=pos.get' in line)
print(f"Remaining 'direction=pos.get' lines: {remaining}")
