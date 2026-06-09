"""
Trader Gemini - Exit Management Test Suite (Multi-Horizon Compete)

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
from datetime import datetime, timezone

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from config import Config
from core.events import SignalEvent, SignalType

def test_tp1_trailing_stop():
    """Test TP1 trailing stop (+1% profit, 50% retracement)"""
    print("\n" + "="*70)
    print("TEST 1: TP1 Trailing Stop (+1%)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades_tp1.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status_tp1.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Rallied $3000 -> $3045 (+1.5%), dropped to $3022
        # peak_pnl_pct = 1.5% (>= 1%, activates TP1)
        # Under new 3-stage adaptive trailing stop:
        # default_tp is 0.8% for SCALPING. 1.5% peak is 1.875x TP target (progress >= 90%).
        # Stage 3 tight trail protects 85% of gains from peak:
        # trail_price = 3045 - (45 * 0.15) = 3038.25
        # current = 3022 < 3038.25 (triggers TP1 trail)
        pos_data = {
            'quantity': 1.0,
            'avg_price': 3000.0,
            'current_price': 3022.0,
            'high_water_mark': 3045.0,
            'low_water_mark': 3000.0,
            'stop_distance': 60.0,
            'horizon': 'SCALPING',
            'exit_pending_time': 0
        }
        portfolio.positions['BTC/USDT'] = pos_data
        portfolio.virtual_ledger['BTC/USDT_SCALPING_LONG'] = pos_data
        
        stop_signals = risk_mgr.check_stops(portfolio, None, symbol_filter='BTC/USDT')
        
        assert len(stop_signals) > 0, "TP1 should trigger"
        assert stop_signals[0].signal_type == SignalType.EXIT
        # Corrected: removed return True to resolve PytestReturnNotNoneWarning
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        raise
    finally:
        portfolio.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass

def test_tp2_trailing_stop():
    """Test TP2 trailing stop calculation (+2% profit, 25% retracement)"""
    print("\n" + "="*70)
    print("TEST 2: TP2 Trailing Stop (+2%)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades_tp2.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status_tp2.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Entry 100, HWM 103 (+3% peak), current 102.2
        # progress = 3.0 / 0.8 = 3.75 (>= 90%)
        # Stage 3 tight trail protects 85% of gains from peak:
        # trail_price = 103 - (3.0 * 0.15) = 102.55
        # current = 102.2 < 102.55 (triggers TP2 trail)
        pos_data = {
            'quantity': 10.0,
            'avg_price': 100.0,
            'current_price': 102.2,
            'high_water_mark': 103.0,
            'low_water_mark': 100.0,
            'stop_distance': 2.0,
            'horizon': 'SCALPING',
            'exit_pending_time': 0
        }
        portfolio.positions['ETH/USDT'] = pos_data
        portfolio.virtual_ledger['ETH/USDT_SCALPING_LONG'] = pos_data
        
        stop_signals = risk_mgr.check_stops(portfolio, None, symbol_filter='ETH/USDT')
        
        assert len(stop_signals) > 0, "TP2 should trigger"
        assert stop_signals[0].signal_type == SignalType.EXIT
        print(f"  ✅ TP2 triggered at $102.20")
        # Corrected: removed return True to resolve PytestReturnNotNoneWarning
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        raise
    finally:
        portfolio.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass

def test_tp3_trailing_stop():
    """Test TP3 trailing stop"""
    print("\n" + "="*70)
    print("TEST 3: TP3 Trailing Stop (+3%+)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades_tp3.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status_tp3.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Entry 100, HWM 105 (+5% peak), current 104.0
        # progress = 5.0 / 0.8 = 6.25 (>= 90%)
        # Stage 3 tight trail protects 85% of gains:
        # trail_price = 105 - (5.0 * 0.15) = 104.25
        # current = 104.0 < 104.25 (triggers trail)
        pos_data = {
            'quantity': 100.0,
            'avg_price': 100.0,
            'current_price': 104.0,
            'high_water_mark': 105.0,
            'low_water_mark': 100.0,
            'stop_distance': 2.0,
            'horizon': 'SCALPING',
            'exit_pending_time': 0
        }
        portfolio.positions['SOL/USDT'] = pos_data
        portfolio.virtual_ledger['SOL/USDT_SCALPING_LONG'] = pos_data
        
        stop_signals = risk_mgr.check_stops(portfolio, None, symbol_filter='SOL/USDT')
        
        assert len(stop_signals) > 0, "TP3 should trigger"
        assert stop_signals[0].signal_type == SignalType.EXIT
        print(f"  ✅ TP3 triggered at $104.00")
        # Corrected: removed return True to resolve PytestReturnNotNoneWarning
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        raise
    finally:
        portfolio.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass

def test_stop_loss():
    """Test stop-loss trigger"""
    print("\n" + "="*70)
    print("TEST 4: Stop-Loss (-2%)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades_sl.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status_sl.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Entry 100, current 97 (-3% drop). Scalping SL defaults to 0.40%.
        # -3% < -0.40% triggers stop-loss
        pos_data = {
            'quantity': 50.0,
            'avg_price': 100.0,
            'current_price': 97.0,
            'high_water_mark': 100.0,
            'low_water_mark': 97.0,
            'stop_distance': 2.0,
            'horizon': 'SCALPING',
            'exit_pending_time': 0
        }
        portfolio.positions['AVAX/USDT'] = pos_data
        portfolio.virtual_ledger['AVAX/USDT_SCALPING_LONG'] = pos_data
        
        stop_signals = risk_mgr.check_stops(portfolio, None, symbol_filter='AVAX/USDT')
        
        assert len(stop_signals) > 0, "Stop-loss should trigger"
        assert stop_signals[0].signal_type == SignalType.EXIT
        print(f"  ✅ Stop-loss triggered at $97")
        # Corrected: removed return True to resolve PytestReturnNotNoneWarning
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        raise
    finally:
        portfolio.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass

def test_hwm_tracking():
    """Test High Water Mark tracking"""
    print("\n" + "="*70)
    print("TEST 5: HWM Tracking")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades_hwm.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status_hwm.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    
    try:
        pos_data = {
            'quantity': 1.0,
            'avg_price': 50000.0,
            'current_price': 50000.0,
            'high_water_mark': 50000.0,
            'low_water_mark': 50000.0,
            'stop_distance': 1000.0,
            'horizon': 'SCALPING'
        }
        portfolio.positions['BTC/USDT'] = pos_data
        portfolio.virtual_ledger['BTC/USDT_SCALPING_LONG'] = pos_data
        portfolio._last_prices['BTC/USDT'] = 50000.0
        
        assert portfolio.positions['BTC/USDT']['high_water_mark'] == 50000.0
        assert portfolio.virtual_ledger['BTC/USDT_SCALPING_LONG']['high_water_mark'] == 50000.0
        print(f"  ✅ Initial HWM: $50000")
        
        # Update with a 1% jump to avoid ghost tick protection (> 2% jump)
        portfolio.update_market_price('BTC/USDT', 50500.0)
        assert portfolio.positions['BTC/USDT']['high_water_mark'] == 50500.0
        assert portfolio.virtual_ledger['BTC/USDT_SCALPING_LONG']['high_water_mark'] == 50500.0
        print(f"  ✅ HWM updated: $50500")
        
        # Drop price, should maintain HWM
        portfolio.update_market_price('BTC/USDT', 50200.0)
        assert portfolio.positions['BTC/USDT']['high_water_mark'] == 50500.0
        assert portfolio.virtual_ledger['BTC/USDT_SCALPING_LONG']['high_water_mark'] == 50500.0
        print(f"  ✅ HWM maintained after price drop")
        # Corrected: removed return True to resolve PytestReturnNotNoneWarning
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        raise
    finally:
        portfolio.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass
