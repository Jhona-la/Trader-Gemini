
import sys
import os
import multiprocessing
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
try:
    # Try importing from local directory if running from tests/
    from run_backtest import fetch_binance_data, run_backtest, calculate_metrics
except ImportError:
    # Try importing from package if running from root
    from tests.run_backtest import fetch_binance_data, run_backtest, calculate_metrics

def process_symbol(symbol):
    """
    Worker function to process a single symbol.
    """
    try:
        print(f"🔹 [Worker] Processing {symbol}...")
        start_time = time.time()
        
        # 1. Fetch Data
        df = fetch_binance_data(symbol, days=15)
        if df is None or len(df) == 0:
            return (symbol, None, "No Data")
            
        # 2. Run Backtest
        results = run_backtest(df, symbol)
        
        # 3. Calculate Metrics
        metrics = calculate_metrics(results['portfolio'])
        
        elapsed = time.time() - start_time
        print(f"✅ [Worker] Finished {symbol} in {elapsed:.1f}s | PnL: ${metrics['total_return']:.2f}%")
        
        return (symbol, metrics, None)
        
    except Exception as e:
        print(f"❌ [Worker] Error on {symbol}: {e}")
        return (symbol, None, str(e))

def main():
    print("="*60)
    print("🚀 TRADER GEMINI - PARALLEL BACKTEST ENGINE (PHASE 1)")
    print("="*60)
    
    # Symbols
    symbols = Config.CRYPTO_FUTURES_PAIRS[:20]
    print(f"📋 Targets: {len(symbols)} Symbols")
    print(f"⚡ CPU Cores: {multiprocessing.cpu_count()}")
    
    start_global = time.time()
    
    # Run Parallel
    with multiprocessing.Pool(processes=min(len(symbols), multiprocessing.cpu_count())) as pool:
        results = pool.map(process_symbol, symbols)
        
    print("\n" + "="*60)
    print("📊 AGGREGATED REPORT")
    print("="*60)
    print(f"{'Symbol':<10} | {'Trades':<8} | {'Win Rate':<8} | {'PnL ($)':<10} | {'Return %':<10} | {'Status'}")
    print("-" * 70)
    
    total_pnl_usd = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    
    successful = 0
    failed = 0
    
    trade_metrics = []
    
    for symbol, metrics, error in results:
        if error:
            print(f"{symbol:<10} | {'-':<8} | {'-':<8} | {'-':<10} | {'-':<10} | ❌ {error}")
            failed += 1
            continue
            
        # Extract metrics correctly (Phase 71 Fix)
        # INITIAL_CAPITAL can vary if real balance is fetched, so we trust metrics['pnl_usd']
        # if available, or calculate it properly.
        # run_backtest.py should be improved to return a cleaner metrics dict.
        
        # FIXED: Use the actual return from the portfolio simulation
        pnl = metrics.get('total_pnl_usd', metrics['final_capital'] - metrics.get('initial_capital', 15.0))
        total_pnl_usd += pnl
        total_trades += metrics['total_trades']
        wins += metrics['winning_trades']
        losses += metrics['losing_trades']
        
        status = "✅" if pnl > 0 else "🛑"
        
        print(f"{symbol:<10} | {metrics['total_trades']:<8} | {metrics['win_rate']:<8.1f} | {pnl:>+9.2f} | {metrics['total_return']:>9.2f}% | {status}")
        successful += 1
        
        if metrics['total_trades'] > 0:
            trade_metrics.append(metrics)
            
    print("-" * 70)
    print(f"🏆 GRAND TOTAL PnL: ${total_pnl_usd:.2f}")
    print(f"📈 TOTAL TRADES:    {total_trades}")
    if total_trades > 0:
        global_wr = (wins / total_trades) * 100
        print(f"🎯 GLOBAL WIN RATE: {global_wr:.1f}%")
    
    duration = time.time() - start_global
    print(f"\n⏱️ Execution Time: {duration:.1f}s ({duration/60:.1f}m)")

if __name__ == '__main__':
    # Fix for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
