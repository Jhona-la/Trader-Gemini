"""
🔬 DIAGNÓSTICO DE VIABILIDAD - Micro Cuenta $13 USD
================================================
Analiza en un Backtest Mass Loop la viabilidad frente a los
costos de fees y slippage usando run_global_backtest.
"""
import sys, os
import json
import logging
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timezone
import pandas as pd
from scripts.run_god_mode_backtest import run_global_backtest
from config import Config
from data.database import DatabaseHandler

def patch_config(tp_val, sl_val, fee_val, initial_balance=13.0):
    """
    Inyecta parámetros masivos de trading al Config Singleton 
    antes de despertar al backtester.
    """
    # Force Fees
    Config.BINANCE_TAKER_FEE_BNB = fee_val
    Config.INITIAL_CAPITAL = initial_balance

    # Force params directly on Scalping params to bypass strategy initialization logic
    if not hasattr(Config, 'Strategies'):
        class DummyStrats: pass
        Config.Strategies = DummyStrats()
    
    if not hasattr(Config.Strategies, 'SCALPING_PARAMS'):
        Config.Horizons.Scalping = {}
    
    Config.Horizons.Scalping['tp_pct'] = tp_val
    Config.Horizons.Scalping['sl_pct'] = sl_val

    # Ensure technical uses this
    Config.Strategies.TECH_TP_PCT = tp_val
    Config.Strategies.TECH_SL_PCT = sl_val
    
    return True

def analyze_viability(results, fee_rate, slippage, expected_tp):
    """Analiza métricas forenses del ledger de scalping."""
    scl_ledger = results['portfolio'].scalping_ledger if hasattr(results, 'portfolio') and hasattr(results['portfolio'], 'scalping_ledger') else []
    
    if not scl_ledger:
        return {'total': 0, 'wr': 0, 'gross': 0, 'fees': 0, 'slp': 0, 'net': 0, 'viability': 0}
        
    gross = sum(t.get('gross_pnl', 0) for t in scl_ledger)
    fees_paid = sum(t.get('fees_paid', 0) for t in scl_ledger)
    
    # Simulate additional rigid slippage on exit & entry price size margin
    # slippage applies on sizing volume
    slp_cost = sum( (t.get('size_usd', 0) * slippage * 2) for t in scl_ledger) 
    
    net = gross - fees_paid - slp_cost
    wins = sum(1 for t in scl_ledger if (t.get('gross_pnl', 0) - t.get('fees_paid', 0) - (t.get('size_usd', 0)*slippage*2)) > 0)
    total = len(scl_ledger)
    
    wr = (wins / total) * 100 if total > 0 else 0
    viability = net / gross if gross > 0 else 0
    
    return {
        'total': total,
        'wr': round(wr, 1),
        'gross': gross,
        'fees': fees_paid,
        'slp': slp_cost,
        'net': net,
        'viability': viability
    }

def main():
    parser = argparse.ArgumentParser(description="Micro Account Scalping Viability Mass Sweep")
    parser.add_argument('--days', type=int, default=3)
    parser.add_argument('--symbols', type=str, default='BTC/USDT')
    parser.add_argument('--sweep-tp', action='store_true')
    args = parser.parse_args()

    symbols = args.symbols.split(',')
    
    # ── Param Sweep Lists ──
    tp_values = [0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020] if args.sweep_tp else [0.015]
    sl_val = 0.005
    fee_val = 0.00075 # 0.075% taker
    slippage = 0.0005 # 0.05% slippage

    print(f"\n{'='*70}")
    print(f"🚀 MASSIVE MICRO-ACCOUNT VIABILITY AUDIT")
    print(f"   Symbols: {symbols} | Days: {args.days}")
    print(f"   Config: Fee={fee_val*100:.3f}% | SL={sl_val*100:.2f}% | Slippage={slippage*100:.2f}%")
    print(f"{'='*70}\n")
    
    # First fetch data once (expensive op)
    from scripts.run_god_mode_backtest import fetch_multi_symbol_data
    
    print(f"\n📡 Descargando datos masivos... esto puede tomar tiempo.")
    all_data = fetch_multi_symbol_data(symbols, days=args.days, max_workers=2)
    
    for sym in symbols:
        if sym in all_data and len(all_data[sym]) > 100:
            print(f"✅ Loaded {sym}: {len(all_data[sym])} candles")
        else:
            print(f"❌ Failed to load sufficient data for {sym}")

    print("\n  ⏱️ STARTING PARAMETER SWEEP...\n")
    
    results_matrix = []
    
    for tp in tp_values:
        print(f"  ➡️ Testing TP: {tp*100:.1f}%")
        patch_config(tp, sl_val, fee_val, 13.0)
        
        # Suppress prints for the backend backtest to keep our sweep clean
        sys.stdout = open(os.devnull, 'w')
        try:
            bt_res = run_global_backtest(all_data, symbols, args.days, 13.0, verbose=False)
        finally:
            sys.stdout = sys.__stdout__
            
        metrics = analyze_viability(bt_res, fee_val, slippage, tp)
        
        row = {
            'TP_Pct': f"{tp*100:.1f}%",
            'Trades': metrics['total'],
            'WinRate': f"{metrics['wr']}%",
            'Gross PnL': f"${metrics['gross']:.3f}",
            'Fees': f"${metrics['fees']:.3f}",
            'Slippage': f"${metrics['slp']:.3f}",
            'Net PnL': f"${metrics['net']:.3f}",
            'Viability': f"{metrics['viability']*100:.1f}%"
        }
        results_matrix.append(row)
        
        # Check specific status right away
        status_color = "🔴" if metrics['net'] <= 0 else "🟢"
        print(f"     {status_color} Trades: {row['Trades']:^4} | Net: {row['Net PnL']:^8} | Viability: {row['Viability']:^6}")

    # Display final table
    print(f"\n{'='*90}")
    print(f"📊 SUMMARY OF SWEEP: SCALPING MICRO ACCOUNT OPTIMIZATION")
    print(f"{'='*90}")
    
    df_res = pd.DataFrame(results_matrix)
    print(df_res.to_string(index=False))
    print(f"\n{'-'*90}")
    
    # Identificación automática de Capital de Viabilidad
    best_row = max(results_matrix, key=lambda x: float(x['Net PnL'].replace('$','')))
    print(f"🎯 OPTIMAL TP IDENTIFIED: {best_row['TP_Pct']} with Net PnL of {best_row['Net PnL']}")

if __name__ == "__main__":
    main()
