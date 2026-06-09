import json

json_path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\results\backtests\god_mode_4de14a8b_1d.json"

with open(json_path, "r") as f:
    data = json.load(f)

print("--- METRICS ---")
for k, v in data["metrics"].items():
    print(f"  {k}: {v}")

print("\n--- REJECTION REASONS ---")
for k, v in data["rejection_reasons"].items():
    if v > 0:
        print(f"  {k}: {v}")
