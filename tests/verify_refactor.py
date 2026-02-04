import sys
import os
import time
sys.path.append(os.getcwd())

from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from strategies.sniper_strategy import SniperStrategy
from utils.cooldown_manager import cooldown_manager
from utils.safe_leverage import safe_leverage_calculator
from config import Config

class MockDataProvider:
    def __init__(self):
        self.symbol_list = ['BTC/USDT']

def test_integration():
    print("Testing Integration...")
    
    # 1. Portfolio
    print("Initializing Portfolio...")
    portfolio = Portfolio(initial_capital=12.0, auto_save=False) # Disable auto_save to avoid files
    print(f"Portfolio Capital: ${portfolio.get_total_equity()}")
    assert portfolio.get_total_equity() == 12.0
    
    # 2. Risk Manager
    print("Initializing RiskManager...")
    risk_manager = RiskManager(portfolio=portfolio)
    
    # Test capital delegation
    # SafeLeverageCalculator should automatically pick up portfolio from RiskManager initialization
    assert safe_leverage_calculator.portfolio == portfolio
    print(f"SafeLeverage Capital: ${safe_leverage_calculator.get_capital()}")
    assert safe_leverage_calculator.get_capital() == 12.0
    
    # 3. Sniper Strategy
    print("Initializing SniperStrategy...")
    mock_dp = MockDataProvider()
    sniper = SniperStrategy(mock_dp, None, portfolio=portfolio)
    
    # Test leverage calculation
    atr = 1.0
    price = 100.0 # ATR% = 1%
    # 1% ATR -> Medium correlation
    # In Config: MIN_LEVERAGE=3, MAX=8.
    lev = sniper._calculate_dynamic_leverage(atr, price)
    print(f"Calculated Leverage for 1% Vol: {lev}x")
    assert 3 <= lev <= 8
    
    # 4. Cooldown
    print("Testing CooldownManager...")
    symbol = "BTC/USDT"
    # Reset first
    cooldown_manager.reset()
    
    can_trade, reason = cooldown_manager.can_trade(symbol)
    print(f"Can trade {symbol}? {can_trade}")
    assert can_trade == True
    
    cooldown_manager.record_trade(symbol)
    can_trade, reason = cooldown_manager.can_trade(symbol)
    print(f"Can trade {symbol} after record? {can_trade} ({reason})")
    assert can_trade == False
    
    # 5. Custom Cooldown
    key = "TEST_KEY"
    cooldown_manager.custom_cooldowns = {} # Reset
    c1 = cooldown_manager.check_custom_cooldown(key, 5)
    c2 = cooldown_manager.check_custom_cooldown(key, 5)
    print(f"Custom Cooldown: First={c1}, Second={c2}")
    assert c1 == True
    assert c2 == False

    print("✅ INTEGRATION TEST PASSED")

if __name__ == "__main__":
    test_integration()
