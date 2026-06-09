import os
import json
import glob

def find_latest_backtest():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "backtests")
    files = glob.glob(os.path.join(results_dir, "god_mode_*.json"))
    if not files:
        print("No backtest results found in results/backtests/")
        return None
    latest_file = max(files, key=os.path.getmtime)
    print(f"Latest backtest file: {latest_file}")
    return latest_file

def analyze_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    print("\n--- METRICS ---")
    print(json.dumps(data.get("metrics"), indent=2))
    
    trade_hist = data.get("trade_history")
    print(f"\ntype(trade_history): {type(trade_hist)}")
    
    if isinstance(trade_hist, dict):
        print(f"Total keys in trade_history: {len(trade_hist)}")
        keys = list(trade_hist.keys())
        print(f"First 5 keys: {keys[:5]}")
        
        # Print value of first key
        if keys:
            print("\n--- FIRST TRADE VALUE ---")
            print(json.dumps(trade_hist[keys[0]], indent=2))
    elif isinstance(trade_hist, list):
        print(f"Total trades in trade_history: {len(trade_hist)}")
        if trade_hist:
            print("\n--- FIRST TRADE ---")
            print(json.dumps(trade_hist[0], indent=2))
    else:
        print(f"trade_history is of type: {type(trade_hist)}")

if __name__ == "__main__":
    latest = find_latest_backtest()
    if latest:
        analyze_file(latest)
