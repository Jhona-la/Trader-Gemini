import json
import os

root = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\results\backtests"
for file in os.listdir(root):
    if file.endswith(".json"):
        path = os.path.join(root, file)
        try:
            with open(path, "r") as f:
                data = json.load(f)
                trades = data.get("metrics", {}).get("total_trades", 0)
                if trades == 17:
                    print(f"MATCH (Total Trades): {file}")
                # Also check strategy attribution
                strat_attr = data.get("strategy_attribution", {})
                for s_id, stats in strat_attr.items():
                    if stats.get("trades") == 17:
                        print(f"MATCH (Strategy Trades): {file} -> {s_id}")
                    if stats.get("wins") == 15 and stats.get("losses") == 2:
                        print(f"MATCH (15W/2L): {file} -> {s_id}")
        except:
            pass
