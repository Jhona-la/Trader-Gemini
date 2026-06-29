import os
import sys
from risk.kill_switch import KillSwitch
from utils.logger import logger

class MockPortfolio:
    def __init__(self, equity=13.0):
        self.equity = equity
        self.positions = {}

    def get_total_equity(self):
        return self.equity

def test_kill_switch_resilience():
    print("🚀 Testing Operational Resilience: Kill Switch + Telegram")
    os.environ["TRADER_GEMINI_BACKTEST"] = "true"
    
    # Initialize KillSwitch with a mock portfolio starting at $13.00
    portfolio = MockPortfolio(13.0)
    ks = KillSwitch(portfolio)
    
    # 1. Simulate safe equity ($13.00)
    print("\n[TEST 1] Testing safe equity ($13.00)...")
    portfolio.equity = 13.0
    ks.update_equity(13.0)
    print(f"Status active: {ks.is_active} (Expected: False)")
    assert not ks.is_active, "Should not be active with initial equity"
    
    # 2. Simulate Hard Floor breach ($7.00, below the 35% floor)
    # Note: Config.INITIAL_CAPITAL = 13.0, Config.Risk.MAX_DRAWDOWN = 35.0 => min_capital = 8.45
    print("\n[TEST 2] Testing hard floor breach ($7.00)...")
    portfolio.equity = 7.00
    ks.update_equity(7.00)
    print(f"Status active: {ks.is_active} (Expected: True)")
    print(f"Reason: {ks.activation_reason}")
    
    assert ks.is_active, "Should activate Kill Switch on capital floor breach"
    assert ks.activation_reason in ("CRITICAL_CAPITAL_FLOOR_REACHED", "MAX_DRAWDOWN_EXCEEDED"), f"Expected capital floor or drawdown breach, got {ks.activation_reason}"
    
    print("\n🔍 Verification complete.")

if __name__ == "__main__":
    test_kill_switch_resilience()

