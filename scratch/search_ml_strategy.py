filepath = "c:/Users/jhona/Documents/Proyectos/Trader Gemini/strategies/ml_strategy.py"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for idx in range(330, 365):
    if idx < len(lines):
        print(f"Line {idx+1}: {lines[idx].strip()}")
