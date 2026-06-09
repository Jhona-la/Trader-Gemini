filepath = "c:/Users/jhona/Documents/Proyectos/Trader Gemini/core/engine.py"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "sophia" in line.lower() or "prediction_tracker" in line.lower():
        print(f"Line {i+1}: {line.strip()}")
