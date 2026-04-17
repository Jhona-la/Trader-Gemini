import json
import os

def summarize_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    print("# Multi-Horizon Backtest Summary (God Mode)")
    print("| Days | Horizon | Symbol | Strategy | PNL ($) | Win Rate (%) | Sharpe | Trades |")
    print("|------|---------|--------|----------|---------|--------------|--------|---------|")

    if isinstance(data, list):
        # New format from run_god_mode_backtest.py
        for r in data:
            horizon = r.get('horizon', 'UNKNOWN')
            days = r.get('days', 0)
            symbol = r.get('symbol', 'UNKNOWN')
            pnl = r.get('pnl_usd', 0)
            wr = r.get('win_rate', 0)
            sharpe = r.get('sharpe', 0)
            trades = r.get('trades', 0)
            
            # Sub-strategies could be extracted if we group them, but the top-level is the aggregate per horizon. 
            # We assume strategy is "Orchestrator/Aggregate" for the top level row:
            pnl_str = f"**{pnl:+.4f}**" if pnl > 0 else f"{pnl:.4f}"
            print(f"| {days}D | {horizon} | {symbol} | Aggregate | {pnl_str} | {wr:.2f}% | {sharpe:.2f} | {trades} |")
    else:
        # Legacy format
        for test_dur, symbols in data.items():
            if test_dur == "combined": continue
            for symbol, strategies in symbols.items():
                for strategy, metrics in strategies.items():
                    pnl = metrics.get('pnl_usd', 0)
                    wr = metrics.get('win_rate', 0)
                    sharpe = metrics.get('sharpe', 0)
                    trades = metrics.get('trades', 0)
                    pnl_str = f"**{pnl:+.4f}**" if pnl > 0 else f"{pnl:.4f}"
                    print(f"| {test_dur} | UNKNOWN | {symbol} | {strategy} | {pnl_str} | {wr:.2f}% | {sharpe:.2f} | {trades} |")

if __name__ == "__main__":
    summarize_results('god_mode_backtest_results.json')
