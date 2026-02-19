import unittest
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from risk.risk_manager import RiskManager
from core.events import SignalEvent, OrderEvent
from core.enums import SignalType
from config import Config


import threading

class MockPortfolio:
    def __init__(self, cash=10000.0, equity=10000.0):
        self._cash = cash
        self._equity_cache = equity
        self._cash_lock = threading.Lock()
        self.positions = {}
        self.strategy_performance = {}

    def get_total_equity(self):
        return self._equity_cache
    
    def get_available_cash(self):
        with self._cash_lock:
            return self._cash
    
    def reserve_cash(self, amount):
        with self._cash_lock:
            if self._cash >= amount:
                self._cash -= amount
                return True
            return False

    def get_smart_kelly_sizing(self, symbol, strategy_id):
        # Simulation of the new portfolio method
        perf = self.strategy_performance.get(strategy_id, {'wins': 6, 'losses': 4, 'trades': 10})
        wr = perf['wins'] / perf['trades']
        b = 1.5
        kelly = (wr * b - (1-wr)) / b
        return max(0.005, min(0.05, kelly * 0.5))

class MockDataProvider:
    def get_latest_bars(self, symbol, n=5):
        import numpy as np
        # Return structured array mock
        dtype = [('timestamp', 'datetime64[ms]'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')]
        data = np.zeros(n, dtype=dtype)
        data['close'] = 50000.0
        return data

class TestRiskManager(unittest.TestCase):
    
    def setUp(self):
        # Reset cooldowns and Config overrides
        self.portfolio = MockPortfolio()
        # Mock Testnet/Demo to FALSE to avoid account scaling during tests
        # Use full path for patching to ensure it affects the risk_manager module
        self.p_testnet = patch('risk.risk_manager.Config.BINANCE_USE_TESTNET', False)
        self.p_demo = patch('risk.risk_manager.Config.BINANCE_USE_DEMO', False)
        self.p_futures = patch('risk.risk_manager.Config.BINANCE_USE_FUTURES', True) # Default to True
        
        self.p_testnet.start()
        self.p_demo.start()
        self.p_futures.start()
        
        # Mock sys.exit to prevent tests from closing
        self.p_exit = patch('sys.exit')
        self.p_exit.start()
        
        self.rm = RiskManager(portfolio=self.portfolio)
        self.data_provider = MockDataProvider()
        
    def tearDown(self):
        self.p_testnet.stop()
        self.p_demo.stop()
        self.p_futures.stop()
        self.p_exit.stop()
        # Clean up any lock files created during tests
        if os.path.exists("STOP_TRADING.LOCK"):
             try:
                 os.remove("STOP_TRADING.LOCK")
             except:
                 pass
        
    def test_position_sizing_tiers(self):
        """Rule 4.1: Verify Dynamic Position Sizing works with micro-account logic"""
        # The current code has micro-account optimization:
        # - For capital < $50: 50% of capital
        # - Virtual capital cap of $100 for testnet
        # - CVaR reduction when in drawdown
        
        # Test micro account sizing (< $50)
        self.portfolio._equity_cache = 50.0  # Phase 5: Use cache attribute
        self.portfolio._cash = 50.0
        self.rm.peak_capital = 50.0  # Set peak to avoid CVaR penalty
        sig = SignalEvent(
            strategy_id="TEST", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG, 
            atr=500.0
        )
        size = self.rm.size_position(sig, 50000)
        # Should be around 50% of $50 = $25, but Binance min is $5
        self.assertGreaterEqual(size, 5.0)  # At least Binance minimum
        self.assertLessEqual(size, 50.0)    # Not more than capital

    def test_kill_switch_activation(self):
        """Rule 4.5: Verify Kill Switch Triggers"""
        # Verify initial state
        self.assertTrue(self.rm.kill_switch.check_status())
        
        # 1. Trigger via Max Daily Losses (default is 5 for standard, but growth phase allows 7)
        # Set peak equity first
        self.rm.kill_switch.update_equity(10000.0)
        for _ in range(4):
            self.rm.record_loss()
        self.assertTrue(self.rm.kill_switch.check_status())  # Still okay at 4
        self.rm.record_loss()  # 5th loss
        self.assertFalse(self.rm.kill_switch.check_status())  # 5 losses -> KILL
        self.assertIn("DAILY_LOSSES", self.rm.kill_switch.activation_reason)
        
        # Reset for next test
        self.rm.kill_switch.active = False
        self.rm.kill_switch.daily_losses = 0
        
        # 2. Trigger via Drawdown (>15% for standard accounts)
        # Peak equity set to 10000
        self.rm.kill_switch.peak_equity = 10000.0
        # Drop to 8400 (16% loss)
        self.rm.update_equity(8400.0)
        self.assertFalse(self.rm.kill_switch.check_status())
        self.assertIn("DRAWDOWN", self.rm.kill_switch.activation_reason)

    def test_order_rejection_on_kill_switch(self):
        """Verify orders are rejected when Kill Switch is Active"""
        self.rm.kill_switch.active = True
        self.rm.kill_switch.activation_reason = "TEST"
        
        sig = SignalEvent(
            strategy_id="TEST", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG
        )
        order = self.rm.generate_order(sig, 50000)
        self.assertIsNone(order)

    def test_balance_check(self):
        """Rule 4.3: Verify Pre-Order Balance Check"""
        # Set cash to very low, but equity high enough for valid size
        self.portfolio._cash = 1.0  
        self.portfolio._equity_cache = 500.0
        
        sig = SignalEvent(
            strategy_id="TEST", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG, 
            atr=500.0
        )
        order = self.rm.generate_order(sig, 50000)
        # Should fail due to insufficient funds (Needed > $5, have $1)
        self.assertIsNone(order)
        
    def test_short_rejection_spot_mode(self):
        """Verify SHORT is rejected if Futures Mode is False"""
        # Mock Config returns
        with patch('risk.risk_manager.Config.BINANCE_USE_FUTURES', False):
            sig = SignalEvent(
                strategy_id="TEST", 
                symbol="BTC/USDT", 
                datetime=datetime.now(timezone.utc), 
                signal_type=SignalType.SHORT
            )
            order = self.rm.generate_order(sig, 50000)
            self.assertIsNone(order)

    def test_atr_volatility_sizing(self):
        """Rule 4.2: Verify ATR-based Position Sizing adjusts position size"""
        # Current code uses micro-account logic with:
        # - Virtual capital cap of $100 for testnet
        # - ATR influences stop distance and position sizing
        
    def test_atr_volatility_sizing(self):
        """Rule 4.2: Verify ATR-based Position Sizing adjusts position size"""
        self.portfolio._equity_cache = 100.0
        self.portfolio._cash = 100.0
        self.rm.peak_capital = 100.0
        
        # Low volatility (small ATR) should allow larger positions
        sig_low_vol = SignalEvent(
            strategy_id="TEST", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG, 
            atr=100.0
        )
        size_low = self.rm.size_position(sig_low_vol, 50000)
        
        # High volatility (large ATR) should reduce position size
        sig_high_vol = SignalEvent(
            strategy_id="TEST", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG, 
            atr=5000.0
        )
        size_high = self.rm.size_position(sig_high_vol, 50000)
        
        # Both should be reasonable sizes >= $5 minimum
        self.assertGreaterEqual(size_low, 5.0)
        self.assertGreaterEqual(size_high, 5.0)
        # Low vol should allow equal or larger size than high vol
        self.assertGreaterEqual(size_low, size_high)

        # 2. Smart Kelly Integration Test (Phase 5)
        self.portfolio._equity_cache = 1000.0
        self.portfolio.strategy_performance['ML_STRAT'] = {'wins': 8, 'losses': 2, 'trades': 10} # 80% WinRate
        sig_kelly = SignalEvent(
            strategy_id="ML_STRAT", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG, 
            atr=None
        )
        
        # Kelly = (p*b - q) / b  => (0.8*1.5 - 0.2) / 1.5 = (1.2 - 0.2) / 1.5 = 1 / 1.5 = 0.66
        # Fractional Kelly (0.5x) = 0.33 => 33% of capital
        # Clamped to 5% Max for safety
        size_kelly = self.rm.size_position(sig_kelly, 50000)
        
        # 5% of 1000 = $50. But with 1.5x ML Boost = $75.
        self.assertLessEqual(size_kelly, 75.1)
        self.assertGreaterEqual(size_kelly, 49.9)

    def test_multi_level_stops(self):
        """Rule 4.4: Verify Stop Loss and Take Profit Logic"""
        # Setup position: Long BTC at 50000, Qty 0.1
        self.portfolio.positions['BTC/USDT'] = {
            'quantity': 0.1,
            'avg_price': 50000.0,
            'current_price': 50000.0,
            'high_water_mark': 50000.0
        }
        
        # 1. Test Stop Loss (-2%)
        # Price drops to 48900 (-2.2%)
        self.portfolio.positions['BTC/USDT']['current_price'] = 48900.0
        stops = self.rm.check_stops(self.portfolio, self.data_provider)
        self.assertEqual(len(stops), 1)
        self.assertIn("RISK_MGR", stops[0].strategy_id)
        
        # 2. Test TP1 (+1% gain -> Trailing at 50% of gain)
        # Price rises to 50600 (+1.2%)
        # HWM = 50600. Gain = 600. Trail = 300. Stop = 50300.
        # But Min Stop = Breakeven + 0.3% = 50000 * 1.003 = 50150.
        # Stop is max(50300, 50150) = 50300.
        self.portfolio.positions['BTC/USDT']['current_price'] = 50600.0
        self.portfolio.positions['BTC/USDT']['high_water_mark'] = 50600.0
        
        # No signal yet
        stops = self.rm.check_stops(self.portfolio, self.data_provider)
        self.assertEqual(len(stops), 0)
        
        # Price drops to 50200 (Below 50300)
        self.portfolio.positions['BTC/USDT']['current_price'] = 50200.0
        stops = self.rm.check_stops(self.portfolio, self.data_provider)
        self.assertEqual(len(stops), 1)
        self.assertIn("TP_MANAGER", stops[0].strategy_id)

if __name__ == '__main__':
    unittest.main()
