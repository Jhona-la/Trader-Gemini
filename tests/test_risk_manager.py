import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from risk.risk_manager import RiskManager
from core.events import SignalEvent, OrderEvent
from core.enums import SignalType
from config import Config


class MockPortfolio:
    def __init__(self, cash=10000.0, equity=10000.0):
        self.cash = cash
        self.equity = equity
        self.positions = {} # {symbol: {'quantity': N, 'avg_price': P, ...}}

    def get_total_equity(self):
        return self.equity
    
    def get_available_cash(self):
        return self.cash
    
    def reserve_cash(self, amount):
        if self.cash >= amount:
            self.cash -= amount
            return True
        return False

class TestRiskManager(unittest.TestCase):
    
    def setUp(self):
        # Reset cooldowns and Config overrides if any
        self.portfolio = MockPortfolio()
        self.rm = RiskManager(portfolio=self.portfolio)
        
    def test_position_sizing_tiers(self):
        """Rule 4.1: Verify Dynamic Position Sizing works with micro-account logic"""
        # The current code has micro-account optimization:
        # - For capital < $50: 50% of capital
        # - Virtual capital cap of $100 for testnet
        # - CVaR reduction when in drawdown
        
        # Test micro account sizing (< $50)
        self.portfolio.equity = 50.0
        self.portfolio.cash = 50.0
        self.rm.peak_capital = 50.0  # Set peak to avoid CVaR penalty
        sig = SignalEvent("TEST", "BTC/USDT", datetime.now(timezone.utc), SignalType.LONG, atr=500.0)
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
        self.assertIn("DAILY LOSSES", self.rm.kill_switch.activation_reason)
        
        # Reset for next test
        self.rm.kill_switch.is_active = False
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
        self.rm.kill_switch.is_active = True
        self.rm.kill_switch.activation_reason = "TEST"
        
        sig = SignalEvent("TEST", "BTC/USDT", datetime.now(timezone.utc), SignalType.LONG)
        order = self.rm.generate_order(sig, 50000)
        self.assertIsNone(order)

    def test_balance_check(self):
        """Rule 4.3: Verify Pre-Order Balance Check"""
        # Set cash to very low
        self.portfolio.cash = 1.0  # Below Binance minimum
        self.portfolio.equity = 1.0
        
        sig = SignalEvent("TEST", "BTC/USDT", datetime.now(timezone.utc), SignalType.LONG, atr=500.0)
        order = self.rm.generate_order(sig, 50000)
        # Should fail due to insufficient funds
        self.assertIsNone(order)
        
    def test_short_rejection_spot_mode(self):
        """Verify SHORT is rejected if Futures Mode is False"""
        # Mock Config returns
        with patch('config.Config.BINANCE_USE_FUTURES', False):
            sig = SignalEvent("TEST", "BTC/USDT", datetime.now(timezone.utc), SignalType.SHORT)
            order = self.rm.generate_order(sig, 50000)
            self.assertIsNone(order)

    def test_atr_volatility_sizing(self):
        """Rule 4.2: Verify ATR-based Position Sizing adjusts position size"""
        # Current code uses micro-account logic with:
        # - Virtual capital cap of $100 for testnet
        # - ATR influences stop distance and position sizing
        
        self.portfolio.equity = 100.0
        self.portfolio.cash = 100.0
        self.rm.peak_capital = 100.0
        
        # Low volatility (small ATR) should allow larger positions
        sig_low_vol = SignalEvent("TEST", "BTC/USDT", datetime.now(timezone.utc), SignalType.LONG, atr=100.0)
        size_low = self.rm.size_position(sig_low_vol, 50000)
        
        # High volatility (large ATR) should reduce position size
        sig_high_vol = SignalEvent("TEST", "BTC/USDT", datetime.now(timezone.utc), SignalType.LONG, atr=5000.0)
        size_high = self.rm.size_position(sig_high_vol, 50000)
        
        # Both should be reasonable sizes >= $5 minimum
        self.assertGreaterEqual(size_low, 5.0)
        self.assertGreaterEqual(size_high, 5.0)
        # Low vol should allow equal or larger size than high vol
        self.assertGreaterEqual(size_low, size_high)

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
        stops = self.rm.check_stops(self.portfolio, None)
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
        stops = self.rm.check_stops(self.portfolio, None)
        self.assertEqual(len(stops), 0)
        
        # Price drops to 50200 (Below 50300)
        self.portfolio.positions['BTC/USDT']['current_price'] = 50200.0
        stops = self.rm.check_stops(self.portfolio, None)
        self.assertEqual(len(stops), 1)
        self.assertIn("TP_MANAGER", stops[0].strategy_id)

if __name__ == '__main__':
    unittest.main()
