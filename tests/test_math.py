"""
🧮 TEST MATH - MATHEMATICAL VALIDATION TESTS
=============================================

PROFESSOR METHOD:
- QUÉ: Tests de validación matemática para cálculos financieros.
- POR QUÉ: Garantiza precisión en PnL, Expectancy, y Equity.
- CÓMO: Unit tests con valores conocidos y edge cases.
- CUÁNDO: Se ejecuta con `pytest tests/test_math.py -v`.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.math_helpers import safe_div


class TestSafeDiv(unittest.TestCase):
    """Tests for safe_div utility function."""
    
    def test_scalar_division(self):
        self.assertEqual(safe_div(10, 2), 5.0)
        self.assertEqual(safe_div(10, 0), 0.0)
        self.assertEqual(safe_div(10, 0, fill_value=99), 99)
        
    def test_array_division(self):
        a = np.array([10, 20, 30])
        b = np.array([2, 5, 0])
        expected = np.array([5.0, 4.0, 0.0])
        np.testing.assert_array_equal(safe_div(a, b), expected)
        
    def test_pandas_series_division(self):
        s1 = pd.Series([10, 20])
        s2 = pd.Series([2, 0])
        result = safe_div(s1, s2)
        np.testing.assert_array_equal(result, np.array([5.0, 0.0]))
        
    def test_inf_handling(self):
        a = np.array([1.0])
        b = np.array([0.0])
        result = safe_div(a, b)
        self.assertEqual(result[0], 0.0)


class TestExpectancyCalculation(unittest.TestCase):
    """
    📊 Tests for Expectancy calculation.
    
    Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    """
    
    def calculate_expectancy(self, trades):
        """Helper to calculate expectancy from trades list."""
        if not trades:
            return 0.0
        
        wins = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t < 0]
        
        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        loss_rate = loss_count / total_trades if total_trades > 0 else 0
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        return (win_rate * avg_win) - (loss_rate * avg_loss)
    
    def test_positive_expectancy(self):
        """Test positive expectancy scenario."""
        trades = [10, 15, -5, 20, -8]  # 3 wins, 2 losses
        exp = self.calculate_expectancy(trades)
        
        # Win rate = 3/5 = 0.6
        # Loss rate = 2/5 = 0.4
        # Avg win = (10+15+20)/3 = 15
        # Avg loss = (5+8)/2 = 6.5
        # Expectancy = (0.6 * 15) - (0.4 * 6.5) = 9 - 2.6 = 6.4
        self.assertAlmostEqual(exp, 6.4, places=1)
        self.assertGreater(exp, 0)
    
    def test_negative_expectancy(self):
        """Test negative expectancy scenario."""
        trades = [-10, -15, 5, -20, 8]  # 2 wins, 3 losses
        exp = self.calculate_expectancy(trades)
        
        # Win rate = 2/5 = 0.4
        # Loss rate = 3/5 = 0.6
        # Avg win = (5+8)/2 = 6.5
        # Avg loss = (10+15+20)/3 = 15
        # Expectancy = (0.4 * 6.5) - (0.6 * 15) = 2.6 - 9 = -6.4
        self.assertAlmostEqual(exp, -6.4, places=1)
        self.assertLess(exp, 0)
    
    def test_zero_expectancy(self):
        """Test balanced expectancy scenario."""
        trades = [10, -10, 10, -10]  # 2 wins, 2 losses, equal size
        exp = self.calculate_expectancy(trades)
        
        # Win rate = 0.5, Loss rate = 0.5
        # Avg win = 10, Avg loss = 10
        # Expectancy = (0.5 * 10) - (0.5 * 10) = 5 - 5 = 0
        self.assertEqual(exp, 0.0)
    
    def test_empty_trades(self):
        """Test expectancy with no trades."""
        exp = self.calculate_expectancy([])
        self.assertEqual(exp, 0.0)
    
    def test_all_wins(self):
        """Test expectancy with only winning trades."""
        trades = [10, 20, 30]
        exp = self.calculate_expectancy(trades)
        
        # Win rate = 1.0, Loss rate = 0.0
        # Avg win = 20, Avg loss = 0
        # Expectancy = 1.0 * 20 - 0 = 20
        self.assertEqual(exp, 20.0)
    
    def test_all_losses(self):
        """Test expectancy with only losing trades."""
        trades = [-10, -20, -30]
        exp = self.calculate_expectancy(trades)
        
        # Win rate = 0.0, Loss rate = 1.0
        # Avg win = 0, Avg loss = 20
        # Expectancy = 0 - 1.0 * 20 = -20
        self.assertEqual(exp, -20.0)


class TestEquityCalculation(unittest.TestCase):
    """
    💰 Tests for Equity calculation.
    
    Total Equity = Wallet Balance + Unrealized PnL
    """
    
    def test_equity_with_profit(self):
        """Test equity with unrealized profit."""
        wallet = 5000.0
        unrealized = 100.0
        equity = wallet + unrealized
        
        self.assertEqual(equity, 5100.0)
    
    def test_equity_with_loss(self):
        """Test equity with unrealized loss."""
        wallet = 5000.0
        unrealized = -200.0
        equity = wallet + unrealized
        
        self.assertEqual(equity, 4800.0)
    
    def test_equity_zero_unrealized(self):
        """Test equity with no unrealized PnL."""
        wallet = 5000.0
        unrealized = 0.0
        equity = wallet + unrealized
        
        self.assertEqual(equity, 5000.0)


class TestUnrealizedPnLCalculation(unittest.TestCase):
    """
    📉 Tests for Unrealized PnL calculation.
    
    LONG: Unrealized PnL = (Current Price - Entry Price) * Quantity
    SHORT: Unrealized PnL = (Entry Price - Current Price) * Quantity
    """
    
    def calculate_unrealized_pnl(self, entry_price, current_price, quantity, is_long=True):
        """Calculate unrealized PnL for a position."""
        if is_long:
            return (current_price - entry_price) * quantity
        else:
            return (entry_price - current_price) * quantity
    
    def test_long_profit(self):
        """Test unrealized profit on long position."""
        pnl = self.calculate_unrealized_pnl(
            entry_price=0.55,
            current_price=0.60,
            quantity=100.0,
            is_long=True
        )
        
        # (0.60 - 0.55) * 100 = 5.0
        self.assertAlmostEqual(pnl, 5.0, places=2)
    
    def test_long_loss(self):
        """Test unrealized loss on long position."""
        pnl = self.calculate_unrealized_pnl(
            entry_price=0.55,
            current_price=0.50,
            quantity=100.0,
            is_long=True
        )
        
        # (0.50 - 0.55) * 100 = -5.0
        self.assertAlmostEqual(pnl, -5.0, places=2)
    
    def test_short_profit(self):
        """Test unrealized profit on short position."""
        pnl = self.calculate_unrealized_pnl(
            entry_price=0.55,
            current_price=0.50,
            quantity=100.0,
            is_long=False
        )
        
        # (0.55 - 0.50) * 100 = 5.0
        self.assertAlmostEqual(pnl, 5.0, places=2)
    
    def test_short_loss(self):
        """Test unrealized loss on short position."""
        pnl = self.calculate_unrealized_pnl(
            entry_price=0.55,
            current_price=0.60,
            quantity=100.0,
            is_long=False
        )
        
        # (0.55 - 0.60) * 100 = -5.0
        self.assertAlmostEqual(pnl, -5.0, places=2)
    
    def test_btc_position(self):
        """Test unrealized PnL for BTC position."""
        pnl = self.calculate_unrealized_pnl(
            entry_price=45000.0,
            current_price=46000.0,
            quantity=0.01,
            is_long=True
        )
        
        # (46000 - 45000) * 0.01 = 10.0
        self.assertAlmostEqual(pnl, 10.0, places=2)


class TestMarginCalculation(unittest.TestCase):
    """
    ⚖️ Tests for Margin calculation.
    
    Required Margin = (Notional Value) / Leverage
    Notional Value = Entry Price * Quantity
    """
    
    def test_margin_calculation(self):
        """Test margin requirement calculation."""
        entry_price = 0.55
        quantity = 100.0
        leverage = 3
        
        notional = entry_price * quantity  # 55.0
        margin = notional / leverage  # 18.33
        
        self.assertAlmostEqual(margin, 18.33, places=2)
    
    def test_margin_ratio(self):
        """Test margin ratio calculation."""
        maint_margin = 50.0
        margin_balance = 5000.0
        
        margin_ratio = (maint_margin / margin_balance) * 100
        
        # 50 / 5000 * 100 = 1%
        self.assertEqual(margin_ratio, 1.0)
    
    def test_available_balance(self):
        """Test available balance calculation."""
        wallet_balance = 5000.0
        used_margin = 500.0
        
        available = wallet_balance - used_margin
        
        self.assertEqual(available, 4500.0)


class TestDrawdownCalculation(unittest.TestCase):
    """
    📉 Tests for Drawdown calculation.
    
    Drawdown = (Current Equity - Peak Equity) / Peak Equity
    """
    
    def calculate_drawdown(self, equity_series):
        """Calculate drawdown series."""
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown
    
    def test_drawdown_series(self):
        """Test drawdown calculation over equity series."""
        equity = pd.Series([100, 110, 105, 115, 108, 120])
        dd = self.calculate_drawdown(equity)
        
        # At index 0: peak=100, dd=0
        self.assertEqual(dd.iloc[0], 0.0)
        
        # At index 1: peak=110, dd=0 (new peak)
        self.assertEqual(dd.iloc[1], 0.0)
        
        # At index 2: peak=110, dd=(105-110)/110 = -0.0454
        self.assertAlmostEqual(dd.iloc[2], -0.0454, places=3)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        equity = pd.Series([100, 110, 105, 115, 108, 120])
        dd = self.calculate_drawdown(equity)
        
        max_dd = dd.min()
        
        # Max drawdown is at index 4: (108-115)/115 = -6.09%
        self.assertAlmostEqual(max_dd, -0.0609, places=3)
    
    def test_no_drawdown_uptrend(self):
        """Test no drawdown in pure uptrend."""
        equity = pd.Series([100, 110, 120, 130, 140])
        dd = self.calculate_drawdown(equity)
        
        # All drawdowns should be 0 in uptrend
        self.assertTrue((dd == 0).all())


if __name__ == '__main__':
    unittest.main()

