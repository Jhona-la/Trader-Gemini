import re

file_path = "strategies/ml_strategy.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

content = re.sub(r'self\.par_engine\["([^"]+)"\]', r'self.par_engine.get("\1")', content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Replaced successfully!")
