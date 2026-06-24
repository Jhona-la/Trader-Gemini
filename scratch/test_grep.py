import re
with open("execution/binance_executor.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if re.match(r'^\s*async def\s+|^\s*def\s+', line):
        print(f"{i+1}: {line.strip()}")
