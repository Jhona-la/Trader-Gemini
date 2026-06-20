import os
import sys
import json
import time
from datetime import datetime, timedelta

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.backtest_infra import fetch_multi_symbol_data
from scripts.run_god_mode_backtest import run_global_backtest
from utils.logger import logger

def validate_td72h():
    print("=" * 70)
    print("🚀 [FASE V] VALIDACIÓN DE LA PARADOJA TD_72H (Dynamic Kelly Sizing)")
    print("=" * 70)
    
    # Enable compounding globally for this validation
    Config.Risk.COMPOUNDING_ENABLED = True
    Config.Risk.COMPOUNDING_GROWTH_FACTOR = 4.0
    Config.Risk.COMPOUNDING_PROFIT_STEP = 0.05
    Config.INITIAL_CAPITAL = 13.0
    
    print(f"🔧 Compounding: ENABLED")
    print(f"🔧 Growth Factor: {Config.Risk.COMPOUNDING_GROWTH_FACTOR}")
    print(f"🔧 Base Capital: ${Config.INITIAL_CAPITAL:.2f}")
    
    # Determine the number of days we want (18 days to support 5 windows of 3 days + warmup)
    total_days = 20
    window_days = 3
    
    symbols = Config.TRADING_PAIRS
    print(f"\n⏳ Downloading last {total_days} days of data for {len(symbols)} pairs...")
    
    # We will just fetch `window_days` for multiple different end dates to simulate walk-forward
    
    results = []
    
    # We'll step backwards in time to test different 3-day windows
    # For example: 
    # Window 1: (now - 3 days) to now
    # Window 2: (now - 6 days) to (now - 3 days)
    # ... up to 5 windows.
    
    # We will step backwards in time to test different 3-day windows

    # Let's fetch the last 20 days ONCE
    all_data = fetch_multi_symbol_data(symbols, days=total_days)
    
    if not all_data:
        print("❌ Error fetching data.")
        return
        
    for i in range(5):
        print(f"\n" + "-"*60)
        print(f"🗓️ EJECUTANDO WINDOW {i+1} (Slice de 3 días)")
        print("-" * 60)
        
        # Calculate the slice bounds
        # Total data length is total_days (approx total_days * 24 * 60 bars)
        # We need to slice 3 days per window
        window_data = {}
        valid_window = True
        
        for sym, df in all_data.items():
            # A 3-day window is approx 3 * 24 * 60 = 4320 bars
            bars_per_day = 24 * 60
            warmup_bars = 3 * bars_per_day # 3 days for ML warmup
            
            total_bars = len(df)
            slice_end = total_bars - (i * window_days * bars_per_day)
            slice_start = total_bars - ((i + 1) * window_days * bars_per_day) - warmup_bars
            
            if slice_start < 0:
                slice_start = 0
            
            sliced_df = df.iloc[slice_start:slice_end].copy()
            if sliced_df.empty or len(sliced_df) < warmup_bars + 1000:
                print(f"  ⚠️ {sym} has not enough data in this slice.")
                valid_window = False
                break
                
            window_data[sym] = sliced_df
            
        if not valid_window:
            print("  ❌ Window Skipped due to insufficient data.")
            continue
            
        # Run God Mode for this window
        print(f"  🧠 Iniciando simulación God Mode para la ventana {i+1}...")
        try:
            import contextlib, io
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                valid_symbols = list(window_data.keys())
                result = run_global_backtest(
                    all_data=window_data,
                    symbols=valid_symbols,
                    days=window_days,
                    initial_capital=13.0,
                    verbose=False,
                    seed=42 + i, # Different seed per window for variance tracking
                    mode="FULL"
                )
            
            final_pnl = result['portfolio_end'] - 13.0
            total_trades = result['total_trades']
            win_rate = result['win_rate'] * 100 if result['win_rate'] else 0
            max_dd = result['max_drawdown_pct']
            
            print(f"  ✅ Completado. PnL: ${final_pnl:+.2f} | WR: {win_rate:.1f}% | Trades: {total_trades} | MaxDD: {max_dd:.1f}%")
            
            results.append({
                'window': i + 1,
                'pnl': final_pnl,
                'end_equity': result['portfolio_end'],
                'win_rate': win_rate,
                'trades': total_trades,
                'max_dd': max_dd
            })
            
        except Exception as e:
            print(f"  ❌ Error running backtest: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "=" * 70)
    print("📊 RESULTADOS FINALES DE VALIDACIÓN TD_72H")
    print("=" * 70)
    print(f"{'Ventana':<10} | {'Equity Final':<15} | {'Duplicó?':<10} | {'Win Rate':<10} | {'Max DD':<10}")
    print("-" * 70)
    
    success_count = 0
    for res in results:
        doubled = "SÍ 🚀" if res['end_equity'] >= 26.0 else "NO"
        if res['end_equity'] >= 26.0:
            success_count += 1
        print(f"Window {res['window']:<3} | ${res['end_equity']:<14.2f} | {doubled:<10} | {res['win_rate']:<9.1f}% | {res['max_dd']:<9.1f}%")
        
    print("-" * 70)
    print(f"🎯 Probabilidad de éxito (Doblar en 72h): {(success_count/len(results))*100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    validate_td72h()
