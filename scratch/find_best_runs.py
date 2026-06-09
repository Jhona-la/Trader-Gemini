import os
import json
import glob

def scan_runs():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "backtests")
    files = glob.glob(os.path.join(results_dir, "god_mode_*.json"))
    
    runs = []
    for f_path in files:
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
            
            metrics = data.get("metrics", {})
            run_id = data.get("run_id", "unknown")
            timestamp = data.get("timestamp", "")
            
            # Get some config info if available
            config = data.get("config", {})
            
            pnl = metrics.get("pnl_usd", metrics.get("final_capital", 13.0) - 13.0)
            wr = metrics.get("win_rate", 0)
            total_trades = metrics.get("total_trades", 0)
            
            runs.append({
                "file": os.path.basename(f_path),
                "run_id": run_id,
                "timestamp": timestamp,
                "pnl": pnl,
                "wr": wr,
                "trades": total_trades
            })
        except Exception:
            pass
            
    # Sort by win rate descending
    runs.sort(key=lambda x: x["wr"], reverse=True)
    
    print("| File | Run ID | PNL ($) | Win Rate (%) | Total Trades | Timestamp |")
    print("|------|--------|---------|--------------|--------------|-----------|")
    for r in runs[:30]:
        print(f"| {r['file']} | {r['run_id']} | {r['pnl']:+.4f} | {r['wr']:.2f}% | {r['trades']} | {r['timestamp']} |")

if __name__ == "__main__":
    scan_runs()
