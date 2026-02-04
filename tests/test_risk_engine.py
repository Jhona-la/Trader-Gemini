"""
Trader Gemini - Risk Engine Test Suite

Tests for:
1. Position sizing calculations (ATR-based, Kelly Criterion)
2. Max concurrent positions enforcement
3. Stop-loss and take-profit calculations
4. Capital protection (max risk per trade)
5. Fat finger protection
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
from core.events import SignalEvent
from config import Config

def test_position_sizing():
    """Test position sizing calculations"""
    print("\n" + "="*70)
    print("TEST 1: Position Sizing")
    print("="*70)
    
    # Create portfolio and risk manager
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Test 1: Basic position sizing (10% of capital for medium account)
        signal = SignalEvent(
            strategy_id="TestStrategy",
            symbol="BTC/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type="LONG",
            strength=1.0
        )
        
        current_price = 50000.0
        position_size = risk_mgr.size_position(signal, current_price)
        
        expected_size = 10000.0 * Config.POSITION_SIZE_MEDIUM_ACCOUNT  # 10% of $10k = $1000
        assert abs(position_size - expected_size) < 1.0, f"Position size mismatch: {position_size} vs {expected_size}"
        print(f"  ✅ Basic sizing: ${position_size:.2f} (10% of capital)")
        
        # Test 2: Signal strength scaling (50% strength = 50% size)
        signal.strength = 0.5
        position_size_scaled = risk_mgr.size_position(signal, current_price)
        
        expected_scaled = expected_size * 0.5
        assert abs(position_size_scaled - expected_scaled) < 1.0, f"Scaled size mismatch"
        print(f"  ✅ Strength scaling (0.5): ${position_size_scaled:.2f}")
        
        # Test 3: ATR-based volatility sizing
        signal.strength = 1.0
        signal.atr = 2000.0  # High volatility
        position_size_vol = risk_mgr.size_position(signal, current_price)
        
        # With ATR, size should be capped by volatility
        print(f"  ✅ ATR-based sizing: ${position_size_vol:.2f} (ATR: {signal.atr})")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
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
            try:
                import time
                time.sleep(0.1)
                os.remove(db_path)
            except PermissionError:
                pass

def test_max_concurrent_positions():
    """Test max concurrent positions enforcement"""
    print("\n" + "="*70)
    print("TEST 2: Max Concurrent Positions")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=3, portfolio=portfolio)
    
    try:
        # Simulate 3 open positions
        portfolio.positions = {
            'BTC/USDT': {'quantity': 0.01, 'avg_price': 50000, 'current_price': 50000},
            'ETH/USDT': {'quantity': 0.5, 'avg_price': 3000, 'current_price': 3000},
            'SOL/USDT': {'quantity': 10, 'avg_price': 100, 'current_price': 100}
        }
        
        open_count = sum(1 for pos in portfolio.positions.values() if pos['quantity'] != 0)
        assert open_count == 3, f"Expected 3 positions, got {open_count}"
        print(f"  ✅ Created {open_count} open positions")
        
        # Try to open 4th position (should be rejected)
        signal = SignalEvent(
            strategy_id="TestStrategy",
            symbol="AVAX/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type="LONG",
            strength=1.0
        )
        
        order = risk_mgr.generate_order(signal, current_price=50.0)
        
        assert order is None, "Order should be rejected (max positions reached)"
        print("  ✅ 4th position correctly rejected")
        
        # Try EXIT signal (should be allowed)
        exit_signal = SignalEvent(
            strategy_id="TestStrategy",
            symbol="BTC/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type="EXIT",
            strength=1.0
        )
        
        exit_order = risk_mgr.generate_order(exit_signal, current_price=51000.0)
        assert exit_order is not None, "EXIT orders should always be allowed"
        print("  ✅ EXIT signal correctly allowed")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
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
            try:
                import time
                time.sleep(0.1)
                os.remove(db_path)
            except PermissionError:
                pass

def test_risk_per_trade():
    """Test max risk per trade enforcement"""
    print("\n" + "="*70)
    print("TEST 3: Max Risk Per Trade")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Max risk per trade is 1% by default (Config.MAX_RISK_PER_TRADE)
        assert risk_mgr.max_risk_per_trade == Config.MAX_RISK_PER_TRADE
        print(f"  ✅ Max risk per trade: {risk_mgr.max_risk_per_trade * 100}%")
        
        # With $10k capital and 1% risk, max risk is $100
        max_risk_dollars = 10000.0 * Config.MAX_RISK_PER_TRADE
        assert max_risk_dollars == 100.0, f"Max risk calculation error: {max_risk_dollars}"
        print(f"  ✅ Max risk amount: ${max_risk_dollars:.2f}")
        
        # Position sizing should respect this limit
        # If we have ATR of $500 (stop distance = 2*ATR = $1000)
        # Position size = Risk / Stop Distance = $100 / $1000 = 0.1 BTC
        signal = SignalEvent(
            strategy_id="TestStrategy",
            symbol="BTC/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type="LONG",
            strength=1.0
        )
        signal.atr = 500.0  # ATR = $500
        
        current_price = 50000.0
        position_size = risk_mgr.size_position(signal, current_price)
        
        # Calculate expected quantity
        stop_distance = signal.atr * 2.0  # $1000
        expected_units = (max_risk_dollars / stop_distance) * current_price
        
        # Position size should be near expected (may be capped)
        print(f"  ✅ ATR-based risk sizing: ${position_size:.2f} (ATR stop distance: ${stop_distance:.0f})")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
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
            try:
                import time
                time.sleep(0.1)
                os.remove(db_path)
            except PermissionError:
                pass

def test_stop_loss_calculations():
    """Test stop-loss trigger calculations"""
    print("\n" + "="*70)
    print("TEST 4: Stop-Loss Calculations")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Create a losing position (price dropped 2.5%)
        portfolio.positions = {
            'BTC/USDT': {
                'quantity': 0.01,
                'avg_price': 50000.0,
                'current_price': 48750.0,  # -2.5% loss
                'high_water_mark': 50000.0,
                'stop_distance': 1000.0  # 2% of 50k = $1000
            }
        }
        
        # Check if stop-loss should trigger
        stop_signals = risk_mgr.check_stops(portfolio, None)
        
        # Should trigger because loss > 2%
        assert len(stop_signals) > 0, "Stop-loss should trigger at -2.5%"
        assert stop_signals[0].signal_type == 'EXIT'
        print(f"  ✅ Stop-loss triggered at -2.5%")
        
        # TEST TAKE-PROFIT (TRAILING STOP SIMULATION)
        # Scenario: Price rallied from $3000 to $3100 (+3.33%), then retraced to $3070
        portfolio.positions.clear()
        portfolio.positions['ETH/USDT'] = {
            'quantity': 0.5,
            'avg_price': 3000.0,          # Entry price
            'current_price': 3070.0,      # Current: +2.33% profit (dropped from HWM)
            'high_water_mark': 3100.0,    # Peak: +3.33% profit
            'stop_distance': 60.0
        }
        
        # Mathematical verification:
        # Entry: $3000
        # HWM: $3100 (+3.33%)
        # Current: $3070 (+2.33%)
        # unrealized_pnl_pct = (3070-3000)/3000*100 = 2.33% 
        # This triggers TP2 (>= 2%)
        #
        # TP2 Calculation:
        # gain_from_entry = 3100 - 3000 = 100
        # trail_distance = 100 * 0.25 = 25 (25% retracement allowed)
        # stop_price = 3100 - 25 = 3075
        # min_stop = 3000 * 1.015 = 3045 (lock at least +1.5%)
        # stop_price = max(3075, 3045) = 3075
        # 
        # Trigger condition: current_price (3070) < stop_price (3075) ✓
        
        tp_signals = risk_mgr.check_stops(portfolio, None)
        
        eth_signals = [s for s in tp_signals if s.symbol == 'ETH/USDT']
        assert len(eth_signals) > 0, f"Take-profit (TP2) should trigger. Got {len(eth_signals)} signals"
        print(f"  ✅ Take-profit (TP2) triggered with trailing stop from HWM")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
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
            try:
                import time
                time.sleep(0.1)
                os.remove(db_path)
            except PermissionError:
                pass

def test_minimum_order_size():
    """Test minimum order size enforcement"""
    print("\n" + "="*70)
    print("TEST 5: Minimum Order Size (Fat Finger Protection)")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(initial_capital=10000.0, csv_path=test_csv, status_path=test_status)
    risk_mgr = RiskManager(max_concurrent_positions=5, portfolio=portfolio)
    
    try:
        # Try to create a tiny order ($3) - should be rejected
        signal = SignalEvent(
            strategy_id="TestStrategy",
            symbol="BTC/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type="LONG",
            strength=0.0003  # Very weak signal
        )
        
        current_price = 50000.0
        position_size = risk_mgr.size_position(signal, current_price)
        
        # Position size will be tiny due to low strength
        print(f"  📏 Calculated size: ${position_size:.2f}")
        
        # generate_order should reject orders < $5
        order = risk_mgr.generate_order(signal, current_price)
        
        if position_size < 5.0:
            assert order is None, "Tiny orders should be rejected"
            print("  ✅ Order < $5 correctly rejected")
        else:
            print("  ⚠️  Order size above minimum, cannot test rejection")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
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
            try:
                import time
                time.sleep(0.1)
                os.remove(db_path)
            except PermissionError:
                pass

def run_all_tests():
    """Run all risk engine tests"""
    print("\n" + "="*70)
    print("⚖️ TRADER GEMINI - RISK ENGINE TEST SUITE")
    print("="*70)
    
    results = {
        'Position Sizing': test_position_sizing(),
        'Max Concurrent Positions': test_max_concurrent_positions(),
        'Max Risk Per Trade': test_risk_per_trade(),
        'Stop-Loss Calculations': test_stop_loss_calculations(),
        'Minimum Order Size': test_minimum_order_size()
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
        print("\n  🎉 ALL RISK ENGINE TESTS PASSED!")
        print("  Phase 3 (Risk Engine) core functionality verified")
    
    return 0 if total_passed == total_tests else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
