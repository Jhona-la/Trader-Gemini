import re

file_path = r'C:\Users\jhona\Documents\Proyectos\Trader Gemini\core\market_regime.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(
    r'df\["close"\]\.values',
    'df["close"].to_numpy() if hasattr(df["close"], "to_numpy") else df["close"].values',
    content
)
content = re.sub(
    r'df\["volume"\]\.values',
    'df["volume"].to_numpy() if hasattr(df["volume"], "to_numpy") else df["volume"].values',
    content
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch values applied to market_regime.py")
