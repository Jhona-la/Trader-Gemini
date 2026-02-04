
import sys
import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone

# Add the project directory to sys.path
import os
sys.path.append(os.getcwd())

from core.events import SignalEvent
from core.enums import SignalType
from risk.risk_manager import RiskManager
from utils.safe_leverage import safe_leverage_calculator

class TestNoneTypeDivision(unittest.TestCase):
    def test_reproduce_error(self):
        # Mock portfolio and other dependencies
        portfolio = MagicMock()
        portfolio.get_total_equity.return_value = 15.0
        portfolio.get_available_cash.return_value = 15.0
        portfolio.positions = {}
        
        risk_mgr = RiskManager(portfolio=portfolio)
        
        # Create a signal with atr=None (explicitly)
        # In SignalEvent constructor, we can pass atr=None
        signal = SignalEvent(
            strategy_id="TEST",
            symbol="BTC/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type=SignalType.LONG,
            strength=0.8,
            atr=None # CRITICAL: This is the suspected cause
        )
        
        current_price = 50000.0
        
        print(f"Reproducing error with atr={signal.atr} and current_price={current_price}")
        
        try:
            risk_mgr.generate_order(signal, current_price)
            print("Successfully processed signal (No error)")
        except TypeError as e:
            print(f"Caught expected error: {e}")
            if "NoneType" in str(e) and "/" in str(e):
                print("✅ Reproduced the exact error reported by user.")
            else:
                print("❌ Caught a different error.")
        except Exception as e:
            print(f"Caught unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    unittest.main()
