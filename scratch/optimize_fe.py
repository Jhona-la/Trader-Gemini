import re

path = r'C:/Users/jhona/Documents/Proyectos/Trader Gemini/strategies/components/feature_engineering.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(
    r"pl\.Series\('([^']+)', np\.zeros\(n_len\)\)",
    r"pl.lit(0.0).alias('\1')",
    content
)

content = re.sub(
    r"pl\.Series\('([^']+)', np\.full\(n_len, ([^\)]+)\)\)",
    r"pl.lit(\2).alias('\1')",
    content
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Optimized legacy series bindings in feature_engineering.py")
