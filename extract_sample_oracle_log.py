
import json
target = '[UNIFIED ORACLE]'
with open('logs/bot_20260205.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in reversed(lines):
        if target in line:
            entry = json.loads(line)
            print("--- FULL MESSAGE ---")
            print(entry.get('message', ''))
            print("--- END MESSAGE ---")
            break
