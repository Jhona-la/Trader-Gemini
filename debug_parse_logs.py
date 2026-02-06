
import json
import re
from collections import defaultdict

log_file = 'logs/bot_20260205.json'
symbols_found = defaultdict(int)

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            entry = json.loads(line)
            msg = entry.get('message', '')
            if '[ML ORACLE]' in msg or '[UNIFIED ORACLE]' in msg:
                match = re.search(r'(?:ML|UNIFIED) ORACLE\]\s+([A-Z0-9/]+)', msg)
                if match:
                    symbol = match.group(1)
                    symbols_found[symbol] += 1
        except:
            continue

print(f"Total symbols found with Oracle logs: {len(symbols_found)}")
for sym, count in sorted(symbols_found.items()):
    print(f"  {sym}: {count} logs")
