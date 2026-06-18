#!/usr/bin/env python3
"""
MASSIVE QUANTUM AUDIT SCRIPT
This script runs a super massive backtest simulating all 3 horizons (Microscalping, Scalping, Swing)
simultaneously, and tracks the compound growth (Kelly scaling) mathematically to verify the goal
of reaching 100% ROI every 3 days.

It reads local data, executes the God Mode Backtest, and then applies the Kelly fractional rules
to check the compounded equity curve.
"""

import os
import sys
import json
import logging
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import numpy as np
from config import Config
from scripts.run_god_mode_backtest import run_global_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("QuantumAudit")

def run_quantum_audit():
    logger.info("🚀 Starting Super Massive Quantum Audit...")
    
    # 1. Load Local Data (1m bars)
    data_dir = os.path.join(_project_root, "data", "historical")
    symbols_available = []
    all_data = {}
    
    if not os.path.exists(data_dir):
        logger.error(f"❌ Data directory not found: {data_dir}")
        return

    # Load heavily traded altcoins for des-correlation + majors
    target_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT"]
    
    for fname in os.listdir(data_dir):
        if fname.endswith("_1m.csv"):
            sym_raw = fname.replace("_1m.csv", "").replace("_", "/")
            if sym_raw not in target_symbols:
                continue
            
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
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            
            # Load up to 14 days of 1m data (1440 * 14 = 20160)
            df = df.tail(20160)
            all_data[sym_raw] = df
            symbols_available.append(sym_raw)
            logger.info(f"✅ Loaded {sym_raw}: {len(df):,} bars (~{len(df)/1440:.1f} days)")

    if not all_data:
        logger.error("❌ No valid historical data loaded. Cannot run audit.")
        return

    # 2. Execute Backtest
    initial_cap = 13.0
    days_to_test = 7
    logger.info(f"⚖️ Initial Capital: ${initial_cap:.2f} | Test Window: {days_to_test} days")
    
    results = run_global_backtest(
        all_data=all_data,
        symbols=symbols_available,
        days=days_to_test,
        initial_capital=initial_cap,
        verbose=False
    )
    
    if not results:
        logger.error("❌ Backtest returned no results.")
        return

    metrics = results.get("metrics", {})
    all_trades = []
    for h in ["microscalping", "scalping", "swing"]:
        trades_for_h = results.get("trade_history", {}).get(h, [])
        for t in trades_for_h:
            t['horizon'] = h.upper()
        all_trades.extend(trades_for_h)
    
    # Sort trades chronologically
    all_trades.sort(key=lambda x: x.get('exit_time', x.get('entry_time')))
    
    logger.info("="*60)
    logger.info("📊 RAW BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Total Trades: {len(all_trades)}")
    
    wins = [t for t in all_trades if t.get('net_pnl', 0) > 0]
    losses = [t for t in all_trades if t.get('net_pnl', 0) <= 0]
    wr = len(wins) / len(all_trades) if all_trades else 0
    logger.info(f"Raw Win Rate: {wr*100:.2f}%")
    
    # 3. Quantum Compounding Simulation
    logger.info("\n" + "="*60)
    logger.info("🧪 QUANTUM COMPOUNDING AUDIT (Kelly Geometric Growth)")
    logger.info("="*60)
    
    virtual_equity = initial_cap
    peak_equity = initial_cap
    base_risk = 0.05  # 5% base risk
    
    daily_roi = {}
    
    for t in all_trades:
        exit_time = t.get('exit_time')
        if not exit_time: continue
        day_str = str(exit_time)[:10]
        
        # Calculate dynamic risk based on compounding factor
        profit_above_watermark = virtual_equity - initial_cap
        
        if profit_above_watermark > 0:
            # For every 5% profit retained, increase risk by 30%
            profit_pct = profit_above_watermark / initial_cap
            steps = int(profit_pct / Config.Risk.COMPOUNDING_PROFIT_STEP)
            current_risk = base_risk * (1 + (steps * Config.Risk.COMPOUNDING_GROWTH_FACTOR))
        else:
            current_risk = base_risk
            
        current_risk = min(current_risk, 0.60)  # Max 60% fractional Kelly
        
        # Calculate trade PnL based on this simulated risk
        pnl_pct = t.get('pnl_pct', 0)
        # In reality, risk dictates quantity. PnL % is on notional. 
        # Assume pnl_pct was based on base capital. We scale it linearly with risk exposure.
        # Original exposure was Config.MAX_RISK_PER_TRADE (0.05).
        scale_factor = current_risk / 0.05
        simulated_pnl = t.get('net_pnl', 0) * scale_factor
        
        virtual_equity += simulated_pnl
        if virtual_equity > peak_equity:
            peak_equity = virtual_equity
            
        if day_str not in daily_roi:
            daily_roi[day_str] = virtual_equity
        else:
            daily_roi[day_str] = virtual_equity
            
    logger.info(f"Initial Capital: ${initial_cap:.2f}")
    logger.info(f"Final Quantum Equity: ${virtual_equity:.2f}")
    logger.info(f"Total ROI: {((virtual_equity/initial_cap)-1)*100:.2f}% in {days_to_test} days")
    
    # Verify exponential goal
    logger.info("\n📈 DAILY PROGRESSION:")
    prev_eq = initial_cap
    for day, eq in sorted(daily_roi.items()):
        daily_g = ((eq/prev_eq)-1)*100
        logger.info(f"  {day}: ${eq:.2f} | Daily Growth: {daily_g:+.2f}%")
        prev_eq = eq

    # Dump deep audit
    audit_out = {
        "initial_capital": initial_cap,
        "final_quantum_equity": virtual_equity,
        "raw_win_rate": wr,
        "total_trades": len(all_trades),
        "daily_progression": daily_roi
    }
    with open(os.path.join(_project_root, "forensic_quantum_audit.json"), "w") as f:
        json.dump(audit_out, f, indent=2)

if __name__ == "__main__":
    run_quantum_audit()
