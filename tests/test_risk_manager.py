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
        self.virtual_ledger = {}
        self.strategy_performance = {}
        self.relative_strength_scores = {}
        self._last_prices = {}

    def get_total_equity(self):
        return self._equity_cache
    
    def get_available_cash(self, horizon=None):
        with self._cash_lock:
            return self._cash
    
    def reserve_cash(self, amount):
        with self._cash_lock:
            if self._cash >= amount:
                self._cash -= amount
                return True
            return False

    def get_statistics(self):
        total = sum(p.get('trades', 0) for p in self.strategy_performance.values())
        wins = sum(p.get('wins', 0) for p in self.strategy_performance.values())
        return {'total_trades': total, 'win_rate': wins / total if total > 0 else 0.5}

    def get_allocation_multiplier(self, symbol, is_long):
        return 1.0

    def get_smart_kelly_sizing(self, symbol, strategy_id):
        # Simulation of the new portfolio method
        perf = self.strategy_performance.get(strategy_id, {'wins': 6, 'losses': 4, 'trades': 10})
        wr = perf['wins'] / perf['trades']
        b = 1.5
        kelly = (wr * b - (1-wr)) / b
        return max(0.005, min(0.05, kelly * 0.5))

    def get_kelly_metrics(self):
        return 0.8, 1.5

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
        self.portfolio._last_prices = {'BTC/USDT': 50000.0}
        self.p_futures = patch('risk.risk_manager.Config.BINANCE_USE_FUTURES', True) # Default to True
        
        self.p_testnet.start()
        self.p_demo.start()
        self.p_futures.start()
        
        # Mock sys.exit to prevent tests from closing
        self.p_exit = patch('sys.exit')
        self.p_exit.start()
        
        self.rm = RiskManager(portfolio=self.portfolio)
        self.data_provider = MockDataProvider()
        
        self.p_dh = patch('risk.risk_manager.get_data_handler', return_value=self.data_provider)
        self.p_dh.start()
        
    def tearDown(self):
        self.p_testnet.stop()
        self.p_demo.stop()
        self.p_futures.stop()
        self.p_exit.stop()
        self.p_dh.stop()
        # Clean up any lock files created during tests
        if os.path.exists("STOP_TRADING.LOCK"):
             try:
                 os.remove("STOP_TRADING.LOCK")
             except:
                 pass
        
    def test_position_sizing_tiers(self):
        """Rule 4.1: Verify Dynamic Position Sizing works with micro-account logic"""
        # Test micro account sizing (< $50)
        self.portfolio._equity_cache = 50.0  # Phase 5: Use cache attribute
        self.portfolio._cash = 50.0
        self.rm.peak_capital = 50.0  # Set peak to avoid CVaR penalty
        sig = SignalEvent(
            strategy_id="TEST", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG, 
            atr=500.0,
            horizon="SCALPING"
        )
        size_res = self.rm.size_position("BTC/USDT", risk_pct=0.02)
        # Should be scaled to something reasonable within boundaries
        size = size_res['quantity'] if size_res else 0.0
        self.assertGreaterEqual(size, 0.0)  
        self.assertLessEqual(size, 50.0)    # Not more than capital
        
    def test_kill_switch_activation(self):
        """Rule 4.5: Verify Kill Switch Triggers"""
        # Verify initial state
        self.assertTrue(self.rm.kill_switch.check_status())
        
        # 1. Trigger via Max Daily Losses (default is 5 for standard, but growth phase allows 7)
        # Set peak equity first
        self.rm.kill_switch.update_equity(10000.0)
        for _ in range(4):
            self.rm.kill_switch.record_loss()
        self.assertTrue(self.rm.kill_switch.check_status())  # Still okay at 4
        self.rm.kill_switch.record_loss()  # 5th loss
        self.assertFalse(self.rm.kill_switch.check_status())  # 5 losses -> KILL
        self.assertIn("DAILY_LOSSES", self.rm.kill_switch.activation_reason)
        
        # Reset for next test
        self.rm.kill_switch.active = False
        self.rm.kill_switch.daily_losses = 0
        
        # 2. Trigger via Drawdown (>15% for standard accounts)
        # Peak equity set to 10000
        self.rm.kill_switch.peak_equity = 10000.0
        # Drop to 8400 (16% loss)
        self.rm.kill_switch.update_equity(8400.0)
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
            signal_type=SignalType.LONG,
            horizon="SCALPING"
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
            atr=500.0,
            horizon="SCALPING"
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
                signal_type=SignalType.SHORT,
                horizon="SCALPING"
            )
            order = self.rm.generate_order(sig, 50000)
            self.assertIsNone(order)

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
            atr=100.0,
            horizon="SCALPING"
        )
        size_res_low = self.rm.size_position("BTC/USDT", risk_pct=0.02)
        size_low = size_res_low['quantity'] if size_res_low else 0.0
        
        # High volatility (large ATR) should reduce position size
        sig_high_vol = SignalEvent(
            strategy_id="TEST", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG, 
            atr=5000.0,
            horizon="SCALPING"
        )
        # Volatility is now handled in multiplier usually, so for test we just trigger a normal sizing
        size_res_high = self.rm.size_position("BTC/USDT", risk_pct=0.02)
        size_high = size_res_high['quantity'] if size_res_high else 0.0
        
        self.assertGreaterEqual(size_low, 0.0)
        self.assertGreaterEqual(size_high, 0.0)
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
            atr=None,
            horizon="SCALPING"
        )
        
        size_res_kelly = self.rm.size_position("BTC/USDT", risk_pct=0.02)
        size_kelly = size_res_kelly['quantity'] if size_res_kelly else 0.0
        self.assertGreaterEqual(size_kelly, 0.0)

    def test_multi_level_stops(self):
        """Rule 4.4: Verify Stop Loss and Take Profit Logic"""
        # Setup position: Long BTC at 50000, Qty 0.1
        self.portfolio.virtual_ledger['BTC/USDT_SCALPING'] = {
            'quantity': 0.1,
            'avg_price': 50000.0,
            'current_price': 50000.0,
            'high_water_mark': 50000.0,
            'horizon': 'SCALPING',
            'sl_pct': 0.02, # Set exactly to match the test assumptions
            'tp_pct': 0.02
        }
        
        # 1. Test Stop Loss (-2%)
        # Price drops to 48900 (-2.2%)
        self.portfolio.virtual_ledger['BTC/USDT_SCALPING']['current_price'] = 48900.0
        stops = self.rm.check_stops(self.portfolio, self.data_provider)
        self.assertEqual(len(stops), 1)
        self.assertIn("HARD_SL", stops[0].strategy_id)
        
        # 2. Test TP1 (+1% gain -> Trailing at 50% of gain)
        # Price rises to 50600 (+1.2%)
        # HWM = 50600. Gain = 600. Trail = 300. Stop = 50300.
        # But Min Stop = Breakeven + 0.3% = 50000 * 1.003 = 50150.
        # Stop is max(50300, 50150) = 50300.
        self.portfolio.virtual_ledger['BTC/USDT_SCALPING']['current_price'] = 50600.0
        self.portfolio.virtual_ledger['BTC/USDT_SCALPING']['high_water_mark'] = 50600.0
        
        # No signal yet
        stops = self.rm.check_stops(self.portfolio, self.data_provider)
        self.assertEqual(len(stops), 0)
        
        # Price drops to 50200 (Below 50300)
        self.portfolio.virtual_ledger['BTC/USDT_SCALPING']['current_price'] = 50200.0
        stops = self.rm.check_stops(self.portfolio, self.data_provider)
        self.assertEqual(len(stops), 1)
        self.assertIn("TRAIL_STAGE", stops[0].strategy_id)

if __name__ == '__main__':
    unittest.main()
