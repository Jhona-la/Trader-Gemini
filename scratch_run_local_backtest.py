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
        if sym_raw not in ["BTC/USDT"]: continue  # Quick validation: BTC only
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
        
        # Quick validation: 1 day
        df = df.tail(120)
        
        all_data[sym_raw] = df
        symbols_available.append(sym_raw)
        print(f"  ✅ Loaded {sym_raw}: {len(df):,} bars ({len(df)/1440:.1f} days)")

if not all_data:
    print("❌ No local CSV data found.")
    sys.exit(1)

print(f"\n📊 Loaded {len(all_data)} symbols from local cache.")
print(f"   Symbols: {symbols_available}")

# --- Import and run God Mode Backtest ---
from scripts.run_god_mode_backtest import run_global_backtest

import logging
logging.getLogger("trader_gemini").setLevel(logging.WARNING)

results = run_global_backtest(
    all_data=all_data,
    symbols=symbols_available,
    days=1,
    initial_capital=13.0,
    verbose=False,
)

# --- Print Results ---
if results:
    print("\n" + "=" * 70)
    print("📊 PHASE 8 AITS — BACKTEST RESULTS (LOCAL DATA)")
    print("=" * 70)
    
    metrics = results.get("metrics", {})
    trades = results.get("trades", [])
    
    print(f"  💰 Final Capital:    ${results.get('final_capital', 0):.2f}")
    print(f"  📈 Total Return:     {metrics.get('total_return', 0):.2f}%")
    print(f"  🏆 Win Rate:         {metrics.get('win_rate', 0):.1f}%")
    print(f"  📊 Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  📉 Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  🔄 Total Trades:     {metrics.get('total_trades', len(trades))}")
    print(f"  💵 Avg Trade PnL:    ${metrics.get('avg_trade_pnl_usd', 0):.4f}")
    print(f"  🏭 Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
    
    # CompoundingEngine validation
    try:
        from core.compounding_engine import get_compounding_engine
        ce = get_compounding_engine()
        ce_metrics = ce.get_metrics()
        print(f"\n  🏦 CompoundingEngine:")
        print(f"     Phase:       {ce_metrics['growth_phase']}")
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
