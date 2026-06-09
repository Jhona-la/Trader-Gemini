import os
import json
import glob
import numpy as np

def find_latest_backtest():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "backtests")
    files = glob.glob(os.path.join(results_dir, "god_mode_*.json"))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def run_analysis():
    file_path = find_latest_backtest()
    if not file_path:
        print("No backtest file found.")
        return
    
    print(f"Analyzing: {os.path.basename(file_path)}")
    with open(file_path, "r") as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    trade_history = data.get("trade_history", {})
    
    print("\n========================================================")
    print("📋 ROOT METRICS")
    print("========================================================")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    if not trade_history:
        print("\nNo trade history found in the file.")
        return
        
    print("\n========================================================")
    print("📊 DETAILED STATISTICAL ANALYSIS BY HORIZON")
    print("========================================================")
    
    for horizon, trades in trade_history.items():
        if not trades:
            print(f"\nHorizon {horizon}: No trades recorded.")
            continue
            
        n_trades = len(trades)
        net_pnls_usd = [t.get("net_pnl", 0) for t in trades]
        net_pnls_pct = [t.get("net_pnl_percent", 0) * 100 for t in trades] # Convert to %
        sizes_usd = [t.get("size_usd", 0) for t in trades]
        fees_paid = [t.get("fees_paid", 0) for t in trades]
        durations = [t.get("duration_seconds", 0) for t in trades]
        
        wins_usd = [p for p in net_pnls_usd if p > 0]
        losses_usd = [p for p in net_pnls_usd if p <= 0]
        
        wins_pct = [p for p in net_pnls_pct if p > 0]
        losses_pct = [p for p in net_pnls_pct if p <= 0]
        
        n_wins = len(wins_usd)
        n_losses = len(losses_usd)
        win_rate = (n_wins / n_trades) * 100 if n_trades > 0 else 0
        
        avg_win_usd = np.mean(wins_usd) if wins_usd else 0
        avg_loss_usd = np.mean(losses_usd) if losses_usd else 0
        avg_trade_usd = np.mean(net_pnls_usd) if net_pnls_usd else 0
        
        avg_win_pct = np.mean(wins_pct) if wins_pct else 0
        avg_loss_pct = np.mean(losses_pct) if losses_pct else 0
        avg_trade_pct = np.mean(net_pnls_pct) if net_pnls_pct else 0
        
        gross_profit = sum(wins_usd)
        gross_loss = sum(losses_usd)
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float('inf') if gross_profit > 0 else 1.0
        
        total_fees = sum(fees_paid)
        avg_duration_mins = (np.mean(durations) / 60) if durations else 0
        
        # Mathematical Expectancy (E)
        # E = (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
        # In terms of probabilities: p_win = n_wins / n_trades, p_loss = n_losses / n_trades
        p_win = n_wins / n_trades if n_trades > 0 else 0
        p_loss = n_losses / n_trades if n_trades > 0 else 0
        expectancy_usd = (p_win * avg_win_usd) + (p_loss * avg_loss_usd)
        expectancy_pct = (p_win * avg_win_pct) + (p_loss * avg_loss_pct)
        
        print(f"\n🌅 Horizon: {horizon}")
        print(f"--------------------------------------------------------")
        print(f"  • Total Trades:          {n_trades}")
        print(f"  • Wins:                  {n_wins}")
        print(f"  • Losses:                {n_losses}")
        print(f"  • Win Rate:              {win_rate:.2f}%")
        print(f"  • Profit Factor:         {profit_factor:.2f}")
        print(f"  • Gross Profit ($):      {gross_profit:+.4f}")
        print(f"  • Gross Loss ($):        {gross_loss:+.4f}")
        print(f"  • Total Fees Paid ($):   {total_fees:.4f}")
        print(f"  • Average Size ($):      {np.mean(sizes_usd):.4f}")
        print(f"  • Average Duration:      {avg_duration_mins:.2f} mins")
        print(f"  • Average Win ($):       {avg_win_usd:+.4f} ({avg_win_pct:+.4f}%)")
        print(f"  • Average Loss ($):      {avg_loss_usd:+.4f} ({avg_loss_pct:+.4f}%)")
        print(f"  • Expectancy E ($):      {expectancy_usd:+.4f} per trade")
        print(f"  • Expectancy E (%):      {expectancy_pct:+.4f}% per trade")
        
        # Distribution profile
        print(f"  • Max Win ($):           {max(net_pnls_usd) if net_pnls_usd else 0:+.4f}")
        print(f"  • Max Loss ($):          {min(net_pnls_usd) if net_pnls_usd else 0:+.4f}")

if __name__ == "__main__":
    run_analysis()
