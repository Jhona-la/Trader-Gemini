import os
import json

results_dir = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\results\backtests"
entry_price = 75788.0703
exit_price = 76146.5781

found = []

for filename in os.listdir(results_dir):
    if filename.endswith(".json"):
        path = os.path.join(results_dir, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
                trade_history = data.get("trade_history", {})
                
                # Check all horizons
                for horizon, trades in trade_history.items():
                    for trade in trades:
                        t_entry = trade.get("entry_price", 0)
                        t_exit = trade.get("exit_price", 0)
                        
                        # Match within small epsilon
                        if abs(t_entry - entry_price) < 0.01 and abs(t_exit - exit_price) < 0.01:
                            print(f"FOUND MATCH in {filename} ({horizon}): {t_entry} -> {t_exit}")
                            found.append(filename)
        except Exception as e:
            pass

if not found:
    print("No price match found in results/backtests/")
