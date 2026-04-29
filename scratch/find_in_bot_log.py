path = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\logs\bot_20260424.json"
target = "a996a033"

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if target in line:
            print(f"Line {i+1}: {line.strip()}")
