import os
import json
import glob

def find_latest_backtest():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "backtests")
    files = glob.glob(os.path.join(results_dir, "god_mode_*.json"))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def run_audit():
    file_path = find_latest_backtest()
    if not file_path:
        print("No backtest file found.")
        return
    
    print(f"Auditing trade details for: {os.path.basename(file_path)}")
    with open(file_path, "r") as f:
        data = json.load(f)
        
    trade_history = data.get("trade_history", {})
    scalping_trades = trade_history.get("scalping", [])
    
    print("\n| # | Trade ID | Direction | Entry Time | Entry Price | Exit Price | Net PNL ($) | Net PNL (%) | Exit Reason |")
    print("|---|----------|-----------|------------|-------------|------------|-------------|-------------|-------------|")
    
    for i, t in enumerate(scalping_trades, 1):
        trade_id = t.get("trade_id")
        direction = t.get("direction")
        entry_time = t.get("closed_at") # using closed_at or entry_time if available
        # Let's check if there's an entry_time, else format closed_at
        entry_price = t.get("entry_price", 0)
        exit_price = t.get("exit_price", 0)
        net_pnl = t.get("net_pnl", 0)
        net_pnl_pct = t.get("net_pnl_percent", 0) * 100
        exit_reason = t.get("exit_reason")
        
        print(f"| {i} | {trade_id} | {direction} | {entry_time} | {entry_price:.2f} | {exit_price:.2f} | {net_pnl:+.4f} | {net_pnl_pct:+.4f}% | {exit_reason} |")

if __name__ == "__main__":
    run_audit()
