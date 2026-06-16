import json

with open("ghost_bugs_report.json", "r", encoding="utf-8") as f:
    d = json.load(f)

targets = [
    "execution\\binance_executor.py",
    "strategies\\ml_strategy.py",
    "strategies\\technical.py",
    "core\\engine.py",
    "risk\\risk_manager.py",
    "data\\binance_loader.py"
]

for t in targets:
    print(f"\n=== {t} ===")
    vulns = [v for v in d if t in v['file']]
    for v in vulns:
        print(f"L{v['line']}: [{v['type']}] {v['message']}")
