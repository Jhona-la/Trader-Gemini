"""
Trader Gemini - Exit Management Test Suite (Simplified)

Tests for:
1. TP1 trailing stop (+1% profit, 50% retracement)
2. TP2 trailing stop (+2% profit, 25% retracement)  
3. TP3 trailing stop (+3%+ profit, 10% retracement)
4. Stop-loss triggers
5. HWM tracking
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from config import Config

def test_tp1_trailing_stop():
    """Test TP1 trailing stop (+1% profit, 50% retracement)"""
    print("\n" + "="*70)
    print("TEST 1: TP1 Trailing Stop (+1%)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Rallied $3000 -> $3045 (+1.5%), dropped to $3022
        # peak_pnl_pct = 1.5% (>= 1%, activates TP1)
        # TP1 stop = $3022.50, current = $3022 < stop
        portfolio.positions['BTC/USDT'] = {
            'quantity': 1.0,
            'avg_price': 3000.0,
            'current_price': 3022.0,
            'high_water_mark': 3045.0,
            'stop_distance': 60.0
        }
        
        stop_signals = risk_mgr.check_stops(portfolio, None)
        
        assert len(stop_signals) > 0, "TP1 should trigger"
        print(f"  ✅ TP1 triggered at $3022")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
    finally:
        portfolio.db.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                os.remove(file)
        db_path = os.path.join(Config.DATA_DIR, "trader_gemini.db")
        if os.path.exists(db_path):
            os.remove(db_path)

def test_tp2_trailing_stop():
    """Test TP2 trailing stop calculation (+2% profit, 25% retracement)"""
    print("\n" + "="*70)
    print("TEST 1: TP2 Trailing Stop (+2%)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        portfolio.positions['ETH/USDT'] = {
            'quantity': 10.0,
            'avg_price': 100.0,
            'current_price': 102.2,
            'high_water_mark': 103.0,
            'stop_distance': 2.0
        }
        
        stop_signals = risk_mgr.check_stops(portfolio, None)
        
        assert len(stop_signals) > 0, "TP2 should trigger"
        print(f"  ✅ TP2 triggered at $102.20")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
    finally:
        portfolio.db.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                os.remove(file)
        db_path = os.path.join(Config.DATA_DIR, "trader_gemini.db")
        if os.path.exists(db_path):
            os.remove(db_path)

def test_tp3_trailing_stop():
    """Test TP3 trailing stop"""
    print("\n" + "="*70)
    print("TEST 2: TP3 Trailing Stop (+3%+)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        portfolio.positions['SOL/USDT'] = {
            'quantity': 100.0,
            'avg_price': 100.0,
            'current_price': 104.4,
            'high_water_mark': 105.0,
            'stop_distance': 2.0
        }
        
        stop_signals = risk_mgr.check_stops(portfolio, None)
        
        assert len(stop_signals) > 0, "TP3 should trigger"
        print(f"  ✅ TP3 triggered at $104.40")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
    finally:
        portfolio.db.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                os.remove(file)
        db_path = os.path.join(Config.DATA_DIR, "trader_gemini.db")
        if os.path.exists(db_path):
            os.remove(db_path)

def test_stop_loss():
    """Test stop-loss trigger"""
    print("\n" + "="*70)
    print("TEST 3: Stop-Loss (-2%)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        portfolio.positions['AVAX/USDT'] = {
            'quantity': 50.0,
            'avg_price': 100.0,
            'current_price': 97.0,
            'high_water_mark': 100.0,
            'stop_distance': 2.0
        }
        
        stop_signals = risk_mgr.check_stops(portfolio, None)
        
        assert len(stop_signals) > 0, "Stop-loss should trigger"
        print(f"  ✅ Stop-loss triggered at $97")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
    finally:
        portfolio.db.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                os.remove(file)
        db_path = os.path.join(Config.DATA_DIR, "trader_gemini.db")
        if os.path.exists(db_path):
            os.remove(db_path)

def test_hwm_tracking():
    """Test High Water Mark tracking"""
    print("\n" + "="*70)
    print("TEST 4: HWM Tracking")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    
    try:
        portfolio.positions['BTC/USDT'] = {
            'quantity': 1.0,
            'avg_price': 50000.0,
            'current_price': 50000.0,
            'high_water_mark': 50000.0,
            'low_water_mark': 50000.0,
            'stop_distance': 1000.0
        }
        
        assert portfolio.positions['BTC/USDT']['high_water_mark'] == 50000.0
        print(f"  ✅ Initial HWM: $50000")
        
        portfolio.update_market_price('BTC/USDT', 51000.0)
        assert portfolio.positions['BTC/USDT']['high_water_mark'] == 51000.0
        print(f"  ✅ HWM updated: $51000")
        
        portfolio.update_market_price('BTC/USDT', 50500.0)
        assert portfolio.positions['BTC/USDT']['high_water_mark'] == 51000.0
        print(f"  ✅ HWM maintained after price drop")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
    finally:
        portfolio.db.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                os.remove(file)
        db_path = os.path.join(Config.DATA_DIR, "trader_gemini.db")
        if os.path.exists(db_path):
            os.remove(db_path)

def run_all_tests():
    """Run all exit management tests"""
    print("\n" + "="*70)
    print("🚪 TRADER GEMINI - EXIT MANAGEMENT TEST SUITE")
    print("="*70)
    
    results = {
        'TP1 Trailing Stop': test_tp1_trailing_stop(),
        'TP2 Trailing Stop': test_tp2_trailing_stop(),
        'TP3 Trailing Stop': test_tp3_trailing_stop(),
        'Stop-Loss': test_stop_loss(),
        'HWM Tracking': test_hwm_tracking()
    }
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n  Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n  🎉 ALL EXIT MANAGEMENT TESTS PASSED!")
        print("  Phase 5 (Exit Management) validated")
    
    return 0 if total_passed == total_tests else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
