import re

file_path = r'C:\Users\jhona\Documents\Proyectos\Trader Gemini\core\market_regime.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(
    r'df\["adx"\]\.iloc\[-1\]',
    'df["adx"].item(-1) if hasattr(df["adx"], "item") else df["adx"].iloc[-1]',
    content
)
content = re.sub(
    r'df\["atr_pct"\]\.iloc\[-1\]',
    'df["atr_pct"].item(-1) if hasattr(df["atr_pct"], "item") else df["atr_pct"].iloc[-1]',
    content
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied to market_regime.py")
