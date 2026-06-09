import os
import json

def inspect_run():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "backtests")
    file_path = os.path.join(results_dir, "god_mode_af4a2fbd_7d.json")
    if not os.path.exists(file_path):
        print("File not found.")
        return
        
    with open(file_path, "r") as f:
        data = json.load(f)
        
    print("📋 FILE NAME: god_mode_af4a2fbd_7d.json")
    print(f"Timestamp of execution: {data.get('timestamp')}")
    print(f"Version: {data.get('version')}")
    print(f"Run ID: {data.get('run_id')}")
    
    # Check config
    config = data.get("config", {})
    print("\n⚙️ CONFIG IN FILE:")
    for k, v in config.items():
        print(f"  {k}: {v}")
        
    # Check trade times
    trade_history = data.get("trade_history", {})
    scalping_trades = trade_history.get("scalping", [])
    if scalping_trades:
        print(f"\n📈 Scalping Trades Count: {len(scalping_trades)}")
        print(f"  First Trade Close: {scalping_trades[0].get('closed_at')}")
        print(f"  Last Trade Close: {scalping_trades[-1].get('closed_at')}")
        # Print entry and exit times of first trade
        first_t = scalping_trades[0]
        print(f"  First Trade Details:")
        print(f"    ID: {first_t.get('trade_id')}")
        print(f"    Symbol: {first_t.get('symbol')}")
        print(f"    Direction: {first_t.get('direction')}")
        print(f"    Entry Price: {first_t.get('entry_price')}")
        print(f"    Closed At: {first_t.get('closed_at')}")
        print(f"    Duration Seconds: {first_t.get('duration_seconds')}")
        
if __name__ == "__main__":
    inspect_run()
