import os
import json

results_dir = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\results\backtests"
target_wr = 88.2
target_wins = 15
target_losses = 2

found = []

for filename in os.listdir(results_dir):
    if filename.endswith(".json"):
        path = os.path.join(results_dir, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
                metrics = data.get("metrics", {})
                
                # Check metrics for 15W/2L or 88.2%
                wr = metrics.get("win_rate", 0)
                total = metrics.get("total_trades", 0)
                
                # Some files might have different structures, let's check strategy_attribution too
                strat_attr = data.get("strategy_attribution", {})
                for strat, stats in strat_attr.items():
                    s_wins = stats.get("wins", 0)
                    s_losses = stats.get("losses", 0)
                    s_total = stats.get("trades", 0)
                    s_wr = stats.get("win_rate", 0) * 100
                    
                    if s_wins == 15 and s_losses == 2:
                        print(f"FOUND MATCH in {filename} (Strategy: {strat}): 15W/2L")
                        found.append(filename)
                    elif abs(s_wr - target_wr) < 0.1:
                         print(f"FOUND WR MATCH in {filename} (Strategy: {strat}): {s_wr}%")
                         found.append(filename)

                if metrics.get("total_trades") == 17 and abs(wr - target_wr) < 0.1:
                    print(f"FOUND TOTAL MATCH in {filename}: 17 trades, {wr}% WR")
                    found.append(filename)
                    
        except Exception as e:
            pass

if not found:
    print("No exact match found in results/backtests/")
