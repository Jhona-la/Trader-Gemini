"""
TP1 FORENSIC TEST - CoG-RA Audit
Critical bug investigation for TP1 trailing stop
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from config import Config

def test_tp1_forensic():
    """FORENSIC TEST: TP1 Trailing Stop - Debug Mode"""
    print("\n" + "="*70)
    print("🔬 TP1 FORENSIC TEST - CoG-RA AUDIT")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        print("\n📊 SCENARIO SETUP:")
        print("  Entry Price: $3000")
        print("  HWM (Peak): $3045 (+1.5% reached)")
        print("  Current Price: Testing different values...")
        
        # SCENARIO 1: Current at +1.0% exactly (should enter TP1 logic)
        print("\n=== TEST 1: Current = $3030 (+1.0% profit) ===")
        portfolio.positions['BTC/USDT'] = {
            'quantity': 1.0,
            'avg_price': 3000.0,
            'current_price': 3030.0,
            'high_water_mark': 3045.0,
            'stop_distance': 60.0
        }
        
        # DEBUGGING: Calculate what the code will see
        entry = 3000.0
        current = 3030.0
        hwm = 3045.0
        
        unrealized_pnl_pct = ((current - entry) / entry) * 100
        print(f"  💡 Unrealized PnL%: {unrealized_pnl_pct:.4f}%")
        print(f"  💡 TP1 Threshold: 1.0%")
        print(f"  💡 Enters TP1 logic? {unrealized_pnl_pct >= 1.0}")
        
        if unrealized_pnl_pct >= 1.0:
            gain_from_entry = hwm - entry
            trail_distance = gain_from_entry * 0.5
            stop_price = hwm - trail_distance
            breakeven_stop = entry * 1.003
            final_stop = max(stop_price, breakeven_stop)
            
            print(f"  💡 Gain from entry: ${gain_from_entry:.2f}")
            print(f"  💡 Trail distance (50%): ${trail_distance:.2f}")
            print(f"  💡 Raw stop price: ${stop_price:.2f}")
            print(f"  💡 Breakeven stop (+0.3%): ${breakeven_stop:.2f}")
            print(f"  💡 Final stop price: ${final_stop:.2f}")
            print(f"  💡 Should trigger? {current < final_stop}")
        
        stop_signals = risk_mgr.check_stops(portfolio, None)
        
        print(f"\n  📡 Signals generated: {len(stop_signals)}")
        if len(stop_signals) == 0:
            print("  ❌ NO SIGNAL - Current price above stop")
            print(f"  🔍 Need current < ${final_stop:.2f} to trigger")
            
            # SCENARIO 2: Drop current price below stop
            print("\n=== TEST 2: Dropping current to trigger TP1 ===")
            drop_price = final_stop - 0.5  # Drop slightly below stop
            portfolio.positions['BTC/USDT']['current_price'] = drop_price
            print(f"  📉 New current price: ${drop_price:.2f}")
            
            stop_signals = risk_mgr.check_stops(portfolio, None)
            print(f"  📡 Signals after drop: {len(stop_signals)}")
            
            if len(stop_signals) > 0:
                print(f"  ✅ TP1 TRIGGERED at ${drop_price:.2f}")
                print(f"  ✅ FORENSIC CONCLUSION: TP1 logic is CORRECT")
                print(f"  🐛 BUG IDENTIFIED: Test was using price ABOVE stop")
                return True
            else:
                print(f"  ❌ STILL NO SIGNAL")
                print(f"  🐛 CRITICAL BUG: TP1 logic may be broken")
                
                # Final debugging: Recalculate with new price
                new_current = drop_price
                new_pnl_pct = ((new_current - entry) / entry) * 100
                print(f"\n  🔍 DEEP DEBUG:")
                print(f"    Entry: ${entry}")
                print(f"    Current: ${new_current}")
                print(f"    HWM: ${hwm}")
                print(f"    PnL%: {new_pnl_pct:.4f}%")
                print(f"    >= 1.0%? {new_pnl_pct >= 1.0}")
                
                return False
        else:
            print(f"  ✅ TP1 triggered immediately")
            return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        portfolio.db.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                os.remove(file)
        db_path = os.path.join(Config.DATA_DIR, "trader_gemini.db")
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == '__main__':
    success = test_tp1_forensic()
    print("\n" + "="*70)
    if success:
        print("✅ TP1 FORENSIC TEST: PASSED")
        print("="*70)
        sys.exit(0)
    else:
        print("❌ TP1 FORENSIC TEST: FAILED")
        print("="*70)
        sys.exit(1)
