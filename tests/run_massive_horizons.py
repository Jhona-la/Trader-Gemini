import sys
import os
import multiprocessing
import time
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from tests.run_backtest import fetch_binance_data, run_backtest, calculate_metrics

def process_symbol_horizon(args):
    """
    Worker function to process a single symbol for a specific horizon.
    """
    symbol, days = args
    try:
        start_time = time.time()
        
        # 1. Fetch Data
        df = fetch_binance_data(symbol, days=days)
        if df is None or len(df) == 0:
            return (symbol, days, None, "No Data")
            
        # 2. Run Backtest
        import contextlib, io
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            results = run_backtest(df, symbol)
            metrics = calculate_metrics(results['portfolio'])
        
        elapsed = time.time() - start_time
        
        # Calculate some extra fields
        p = results['portfolio']
        pnl = p.current_capital - p.initial_capital
        total_trades = metrics['total_trades']
        win_rate = metrics['win_rate']
        sharpe = metrics['sharpe_ratio']
        max_dd = metrics['max_drawdown_pct']
        
        wins = [t for t in p.trades if t['pnl_usd'] > 0]
        losses = [t for t in p.trades if t['pnl_usd'] <= 0]
        avg_win = sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        res = {
            'symbol': symbol,
            'days': days,
            'pnl': round(pnl, 4),
            'total_return_pct': round(metrics['total_return'], 2),
            'sharpe': round(sharpe, 2),
            'win_rate': round(win_rate, 1),
            'max_dd_pct': round(max_dd, 2),
            'total_trades': total_trades,
            'payoff': round(payoff, 2),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
        }
        
        return (symbol, days, res, None)
        
    except Exception as e:
        return (symbol, days, None, str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1, help="Number of days to backtest")
    args = parser.parse_args()
    days = args.days
    
    print("="*60)
    print(f"🚀 MASSIVE PARALLEL HORIZON: {days} DAYS")
    print("="*60)
    
    symbols = Config.TRADING_PAIRS
    print(f"📋 Targets: {len(symbols)} Symbols")
    
    start_global = time.time()
    tasks = [(sym, days) for sym in symbols]
    
    with multiprocessing.Pool(processes=min(len(symbols), multiprocessing.cpu_count())) as pool:
        results = pool.map(process_symbol_horizon, tasks)
        
    print("\n" + "="*60)
    print(f"📊 AGGREGATED REPORT ({days} DAYS)")
    print("="*60)
    print(f"{'Symbol':<12} {'PnL':>10} {'Return%':>9} {'WR%':>6} {'Trades':>7} {'Sharpe':>7} {'DD%':>7} {'Payoff':>8} {'AvgWin':>9} {'AvgLoss':>9}")
    print("-" * 100)
    
    successful = 0
    all_res = {}
    
    for symbol, d, metrics, error in results:
        if error:
            print(f"❌ {symbol:<10} ERROR: {error}")
            continue
            
        icon = "🟢" if metrics['pnl'] > 0 else "🔴"
        print(f"{icon} {symbol:<10} {metrics['pnl']:>+9.4f} {metrics['total_return_pct']:>+8.2f}% {metrics['win_rate']:>5.1f} {metrics['total_trades']:>7} {metrics['sharpe']:>7.2f} {metrics['max_dd_pct']:>6.2f}% {metrics['payoff']:>7.2f} {metrics['avg_win']:>+8.4f} {metrics['avg_loss']:>+8.4f}")
        successful += 1
        all_res[symbol] = metrics
            
    print("-" * 100)
    duration = time.time() - start_global
    print(f"\n⏱️ Execution Time: {duration:.1f}s ({duration/60:.1f}m)")
    
    with open(f'{days}d_results_v6.json', 'w') as fp:
        json.dump(all_res, fp, indent=2)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
