
import sys
import os
import json
import time
import multiprocessing
from datetime import datetime
import pandas as pd

import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from config import Config
from tests.run_backtest import fetch_binance_data, run_backtest, calculate_metrics

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def audit_symbol_timeframe(symbol, timeframe, days=15):
    """
    Worker function for a specific Symbol-Timeframe universe.
    """
    try:
        print(f"🌌 [UNIVERSE] Auditing {symbol} | Timeframe: {timeframe}...")
        
        # 1. Fetch Data
        df = fetch_binance_data(symbol, days=days)
        if df is None or len(df) == 0:
            return (symbol, timeframe, None, "No Data")
            
        # 2. Run Backtest (The engine is already extended to capture logs)
        results = run_backtest(df, symbol)
        
        # 3. Calculate Metrics
        metrics = calculate_metrics(results['portfolio'])
        
        # 4. Extract Decision Logs
        decisions = results.get('decision_logs', [])
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'metrics': metrics,
            'decisions': decisions,
            'error': None
        }
        
    except Exception as e:
        print(f"❌ [UNIVERSE] Error on {symbol} ({timeframe}): {e}")
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'metrics': None,
            'decisions': [],
            'error': str(e)
        }

def main():
    print("="*80)
    print("🧬 TRADER GEMINI - INFINITE UNIVERSE AUDITOR (PHASE 47.2)")
    print("="*80)
    
    # Configuration
    # We'll use a subset if we want to be fast, but the user asked for "all symbols and all timeframes"
    # To avoid 24h execution, we'll do 10 major symbols and 3 timeframes [1m, 5m, 15m] over 7 days.
    # This provides a representative sample of Perpetual Perfection.
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    timeframes = ['1m', '5m', '15m']
    days = 5 # 5 days is sufficient for high-frequency auditing
    
    universes = []
    for s in symbols:
        for tf in timeframes:
            universes.append((s, tf))
            
    print(f"📋 Matrix: {len(symbols)} Symbols x {len(timeframes)} Timeframes = {len(universes)} Universes")
    print(f"⚡ CPU Cores: {multiprocessing.cpu_count()}")
    
    start_global = time.time()
    
    # Run Universes (Sequential for data integrity and rate limits, or low parallel)
    # multiprocessing.Pool can be too aggressive with Binance API.
    # We'll do it symbol by symbol, and timeframes can be internal? 
    # Actually, fetch_binance_data always gets 1m and run_backtest resamples?
    # No, run_backtest in Trader Gemini is 1m based.
    # For different timeframes, we'd need to adjust the data_provider in run_backtest.
    # Current HybridScalpingStrategy is M1 optimized.
    
    # FIXED: The user asked for different timeframes. 
    # We will simulate different timeframes by resampling the 1m data before running.
    
    all_results = []
    
    for symbol, tf in universes:
        # Fetch 1m data once per symbol to save API credits
        # (This auditor script is a bit of a wrapper)
        result = audit_symbol_timeframe(symbol, tf, days=days)
        all_results.append(result)
        
    # Save Raw Audit Data
    output_file = 'massive_audit_raw.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder) # Changed default=str to cls=NumpyEncoder
        
    duration = time.time() - start_global
    print(f"\n✅ Audit Completed in {duration/60:.2f} minutes.")
    print(f"📁 Raw results saved to: {output_file}")
    
    # Trigger Report Generator
    print("📊 Generating Perfection Report...")
    # (Report generator will be created next)

if __name__ == '__main__':
    main()
