import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone
import sys
import os

# Ensure the root project directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.math_kernel import kahan_sum
from core.online_learning import OnlineLearner
from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from core.events import FillEvent
from core.enums import OrderSide, EventType

class TestPrecisionAxioma:
    """
    Tests for Protocolo Precisión-Axioma.
    Validates extreme arithmetic constraints and deep tensor audits.
    """
    
    def test_kahan_summation_loss_of_significance(self):
        """
        Verify that kahan summation recovers lost precision compared to naive sum
        when adding wildly different magnitude numbers.
        """
        # Create an array that would cause a float32/float64 catastrophic cancellation
        # 1.0 + 1e16 - 1e16 = ? (Naive could be 0.0, Kahan should be 1.0)
        # Using a slightly less extreme case that reliably breaks naive sum
        arr = np.array([1.0, 1e7, -1e7], dtype=np.float32)
        
        naive_sum = float(np.sum(arr)) # Could be 0.0
        safe_sum = kahan_sum(arr)
        
        assert safe_sum == 1.0, f"Kahan Sum failed! Expected 1.0, got {safe_sum}. Naive: {naive_sum}"
    
    def test_kelly_decimal_precision(self):
        """
        Verify the Kelly Criterion sizing uses decimal logic directly and does not 
        throw float precision errors on extreme inputs (e.g., nano accounts or high wr).
        """
        rm = RiskManager()
        
        # Test extreme edge case: 99.999% win rate, tiny payoff.
        p = 0.99999
        b = 0.01
        
        # Should not crash and should return a clamped 0-40% value
        kelly = rm._compute_kelly_math(p, b)
        assert isinstance(kelly, float)
        assert 0.0 <= kelly <= 0.40, f"Kelly broke bounds: {kelly}"

    def test_portfolio_axioma_drift_log(self, mocker):
        """
        Verify that a tiny floating point divergence in a simulated trade triggers drift 
        accumulation in Portfolio.
        """
        mocker.patch('core.portfolio.Notifier')
        mocker.patch('core.portfolio.safe_append_csv')
        mocker.patch.object(Portfolio, 'log_trade_report')
        try:
             mocker.patch('utils.session_manager.get_session_manager')
        except:
             pass
        mocker.patch('core.portfolio.Config.BINANCE_USE_FUTURES', False)
        
        port = Portfolio(initial_capital=100.0)
        
        fill_event = FillEvent(
            timeindex=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            exchange='BINANCE',
            quantity=1.0,
            direction=OrderSide.BUY,
            fill_cost=50.0,
            commission=0.0# No fee for simple test
        )
        
        # OPEN LONG
        port.update_fill(fill_event)
        
        # INITIAL CASH IS 100 - 50 = 50.0
        assert port.current_cash == 50.0
        
        close_event = FillEvent(
            timeindex=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            exchange='BINANCE',
            quantity=1.0,
            direction=OrderSide.SELL,
            fill_cost=60.0, # Made $10
            commission=0.0 
        )
        
        # Mocking an evil float discrepancy in Python Core
        # The true cash should end at 110.0
        # What if another thread or a float bug changed the cash randomly right before the audit?
        original_cash = port.current_cash
        port.current_cash = 49.99999999  # Simulate a 1e-8 leak from some other module
        
        port.update_fill(close_event)
        
        # The drift tracking should have caught the difference between the pre-sell state and post-sell expected
        drift = port.precision_drift_accumulated
        
        # It's greater than 0
        assert drift > Decimal('0.0'), f"Drift missing! Drift={drift}"

