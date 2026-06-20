#!/usr/bin/env python3
"""
Phase 8 AITS: LOCAL BACKTEST — Uses cached historical CSV data
when Binance API is unavailable (geo-restriction).
"""
import os, sys, json

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Point to the actual project
_project_root = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini"
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import numpy as np

# --- Load local CSVs ---
data_dir = os.path.join(_project_root, "data", "historical")
symbols_available = []
all_data = {}

for fname in os.listdir(data_dir):
    if fname.endswith("_1m.csv"):
        sym_raw = fname.replace("_1m.csv", "").replace("_", "/")
        if sym_raw not in ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]: continue
        df = pd.read_csv(os.path.join(data_dir, fname))
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'timestamp' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            except ValueError:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            df.set_index('datetime', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # 1440 mins/day * 1 days = 1440
        df = df.tail(1440)
        
        all_data[sym_raw] = df
        symbols_available.append(sym_raw)
        print(f"  ✅ Loaded {sym_raw}: {len(df):,} bars ({len(df)/1440:.2f} days)")

if not all_data:
    print("❌ No local CSV data found.")
    sys.exit(1)

print(f"\n📊 Loaded {len(all_data)} symbols from local cache.")
print(f"   Symbols: {symbols_available}")

# --- Import and run God Mode Backtest ---
from scripts.run_god_mode_backtest import run_global_backtest

import logging
logging.getLogger("trader_gemini").setLevel(logging.WARNING)

import time
start_time = time.perf_counter()

results = run_global_backtest(
    all_data=all_data,
    symbols=symbols_available,
    days=1,
    initial_capital=13.0,
    verbose=False,
)

end_time = time.perf_counter()
total_time = (end_time - start_time) * 1000
print(f"\n⏱️ TOTAL BACKTEST EXECUTION TIME: {total_time:.2f} ms")

# --- Print Results ---
if results:
    print("\n" + "=" * 70)
    print("📊 PHASE 8 AITS — BACKTEST RESULTS (LOCAL DATA)")
    print("=" * 70)
    
    metrics = results["metrics"]
    trades = results["trades"]
    
    initial_cap = results['config'].get('initial_capital', 13.0)
    final_cap = metrics['final_capital']
    net_pnl = final_cap - initial_cap
    total_trades = metrics['total_trades']
    avg_trade_pnl = net_pnl / total_trades if total_trades > 0 else 0.0
    
    # Calculate Profit Factor from trades
    all_trades = results['trade_history'].get('scalping', []) + results['trade_history'].get('swing', [])
    wins_pnl = sum([t['net_pnl'] for t in all_trades if t['net_pnl'] > 0])
    losses_pnl = abs(sum([t['net_pnl'] for t in all_trades if t['net_pnl'] < 0]))
    profit_factor = wins_pnl / losses_pnl if losses_pnl > 0 else (99.9 if wins_pnl > 0 else 0.0)
    
    print(f"  💰 Final Capital:    ${final_cap:.2f}")
    print(f"  📈 Total Return:     {metrics['total_return_pct']:.2f}%")
    print(f"  🏆 Win Rate:         {metrics['win_rate']:.1f}%")
    print(f"  📊 Sharpe Ratio:     {metrics['sharpe_ratio']:.3f}")
    print(f"  📉 Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"  🔄 Total Trades:     {total_trades}")
    print(f"  💵 Avg Trade PnL:    ${avg_trade_pnl:.4f}")
    print(f"  🏭 Profit Factor:    {profit_factor:.2f}")
    
    # CompoundingEngine validation
    try:
        from core.compounding_engine import get_compounding_engine
        ce = get_compounding_engine()
        ce_metrics = ce.get_metrics()
        print(f"\n  🏦 CompoundingEngine (HORIZON-AWARE):")
        print(f"     Phase:       {ce_metrics['growth_phase']}")
        print(f"     Regime:      {ce_metrics['regime']}")
        print(f"     MICRO Alloc: {ce_metrics['micro_allocation']*100:.1f}%")
        print(f"     SCL Alloc:   {ce_metrics['scalping_allocation']*100:.1f}%")
        print(f"     SWG Alloc:   {ce_metrics['swing_allocation']*100:.1f}%")
        print(f"     Peak Equity: ${ce_metrics['peak_equity']:.2f}")
    except Exception as e:
        print(f"  ⚠️ CompoundingEngine metrics unavailable: {e}")
    
    # Save results
    out_path = os.path.join(_project_root, "scripts", "phase8_aits_results.json")
    try:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  💾 Results saved to: {out_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save results: {e}")
else:
    print("❌ Backtest returned no results.")
