import json

path = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\multi_horizon_results.json"
target_vals = [13.249748614397078, 12.951866789397753]

try:
    with open(path, "r") as f:
        data = json.load(f)
        
    for horizon, assets in data.get("horizons", {}).items():
        for asset, strategies in assets.items():
            for strat_name, metrics in strategies.items():
                curve = metrics.get("equity_curve", [])
                if any(v in curve for v in target_vals):
                    print(f"MATCH FOUND in {horizon} -> {asset} -> {strat_name}")
                    print(f"Metrics: {metrics.get('trades')} trades, WR: {metrics.get('win_rate')}%")
                    
except Exception as e:
    print(f"Error: {e}")
