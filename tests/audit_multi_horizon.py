"""
🔬 FORENSIC AUDIT: Multi-Horizon Virtual Ledger Isolation Test
Proves that SCALPING and SWING positions on the SAME symbol
maintain INDEPENDENT Average Entry Prices via Composite Keys.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime, timezone
from core.events import FillEvent
from core.portfolio import Portfolio
from core.enums import SignalType, OrderSide

def run_forensic_audit():
    print("="*60)
    print("🔬 TRADER GEMINI: FORENSIC AUDIT (VIRTUAL LEDGER ISOLATION)")
    print("="*60)
    print("\n[🎯 CONTEXT] Capital: $13 USD | Objective: 100% WR Scalping alongside Swing\n")
    
    portfolio = Portfolio(initial_capital=13.0)
    
    # 1. SCALP ENGINE buys BTC at 60000
    print("👉 [T=0] SCALPING ENGINE detects 1-minute Bullish Imbalance.")
    
    fill_scalp = FillEvent(
        symbol="BTC/USDT",
        timeindex=datetime.now(timezone.utc),
        quantity=0.0001, # e.g. $6 Notional
        exchange="BINANCE",
        direction=OrderSide.BUY,
        fill_price=60000.0,
        fill_cost=6.0,
        commission=0.005,
        horizon="SCALPING",
    )
    portfolio.update_fill(fill_scalp)
    
    # 2. SWING ENGINE buys BTC at 61000 (a different, higher price)
    print("\n👉 [T=1] SWING ENGINE detects 1-Day Bullish Breakout.")
    fill_swing = FillEvent(
        symbol="BTC/USDT",
        timeindex=datetime.now(timezone.utc),
        quantity=0.0002,
        exchange="BINANCE",
        direction=OrderSide.BUY,
        fill_price=61000.0,
        fill_cost=12.2,
        commission=0.010,
        horizon="SWING",
    )
    portfolio.update_fill(fill_swing)
    
    # ======= VERIFICATION =======
    print("\n" + "="*60)
    print("📊 VIRTUAL LEDGER STATE:")
    print("="*60)
    
    scalp_key = "BTC/USDT_SCALPING"
    swing_key = "BTC/USDT_SWING"
    
    has_scalp = scalp_key in portfolio.virtual_ledger
    has_swing = swing_key in portfolio.virtual_ledger
    
    print(f"  Keys in Virtual Ledger: {list(portfolio.virtual_ledger.keys())}")
    
    passed = 0
    failed = 0
    
    # TEST 1: Both composite keys exist
    if has_scalp and has_swing:
        print(f"  ✅ TEST 1 PASSED: Both {scalp_key} and {swing_key} exist independently.")
        passed += 1
    else:
        print(f"  ❌ TEST 1 FAILED: Missing keys. Scalp={has_scalp}, Swing={has_swing}")
        failed += 1
    
    # TEST 2: SCALP avg_price is 60000 (not diluted by Swing's 61000)
    if has_scalp:
        scalp_avg = portfolio.virtual_ledger[scalp_key]['avg_price']
        if abs(scalp_avg - 60000.0) < 0.01:
            print(f"  ✅ TEST 2 PASSED: SCALP Avg Entry = ${scalp_avg:.2f} (Isolated from Swing)")
            passed += 1
        else:
            print(f"  ❌ TEST 2 FAILED: SCALP Avg Entry = ${scalp_avg:.2f} (Expected $60000.00)")
            failed += 1
    
    # TEST 3: SWING avg_price is 61000 (its own independent average)
    if has_swing:
        swing_avg = portfolio.virtual_ledger[swing_key]['avg_price']
        if abs(swing_avg - 61000.0) < 0.01:
            print(f"  ✅ TEST 3 PASSED: SWING Avg Entry = ${swing_avg:.2f} (Isolated from Scalp)")
            passed += 1
        else:
            print(f"  ❌ TEST 3 FAILED: SWING Avg Entry = ${swing_avg:.2f} (Expected $61000.00)")
            failed += 1
    
    # TEST 4: Quantities are independent
    if has_scalp and has_swing:
        scalp_qty = portfolio.virtual_ledger[scalp_key]['quantity']
        swing_qty = portfolio.virtual_ledger[swing_key]['quantity']
        if abs(scalp_qty - 0.0001) < 1e-8 and abs(swing_qty - 0.0002) < 1e-8:
            print(f"  ✅ TEST 4 PASSED: SCALP Qty={scalp_qty}, SWING Qty={swing_qty} (Independent)")
            passed += 1
        else:
            print(f"  ❌ TEST 4 FAILED: SCALP Qty={scalp_qty}, SWING Qty={swing_qty}")
            failed += 1
    
    # TEST 5: Physical position is aggregated (Binance-compatible)
    phys_pos = portfolio.positions.get('BTC/USDT', {})
    phys_qty = phys_pos.get('quantity', 0)
    expected_total = 0.0003  # 0.0001 + 0.0002
    if abs(phys_qty - expected_total) < 1e-8:
        print(f"  ✅ TEST 5 PASSED: Physical Position = {phys_qty} (Aggregated for Binance)")
        passed += 1
    else:
        print(f"  ❌ TEST 5 FAILED: Physical Position = {phys_qty} (Expected {expected_total})")
        failed += 1
    
    print(f"\n{'='*60}")
    print(f"📋 RESULTS: {passed} PASSED, {failed} FAILED")
    if failed == 0:
        print("🏆 VIRTUAL LEDGER ISOLATION: CERTIFIED OPERATIONAL")
    else:
        print("🚨 VIRTUAL LEDGER: DEFECTS REMAIN")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_forensic_audit()
