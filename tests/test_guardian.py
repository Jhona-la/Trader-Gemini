
import sys
import os
from unittest.mock import MagicMock
from execution.liquidity_guardian import LiquidityGuardian
from utils.logger import logger

def test_liquidity_guardian_wall():
    print("🚀 Testing Liquidity Guardian: Wall Detection Logic")
    
    # 1. Mock Exchange
    mock_exchange = MagicMock()
    
    # 2. Case A: Order Book with a GIANT Wall
    mock_exchange.fetch_order_book.return_value = {
        'bids': [[99.0, 1.0], [98.5, 1.2], [98.0, 50.0]], # 50.0 is a Wall (avg is ~1.5)
        'asks': [[101.0, 1.0], [101.5, 1.1], [102.0, 100.0]] # 100.0 is a Massive Wall
    }
    
    guardian = LiquidityGuardian(mock_exchange)
    
    # Test BUY with wall ahead
    print("\n[TEST 1] Testing BUY with Sell Wall at 102.0...")
    # Quantity is 0.1, price is 101.0. Wall is at 102.0. 
    # 101.0 * 1.005 = 101.505. The wall at 102.0 is FAR (relative to 0.5% threshold set in code).
    # Let's check 101.1 instead.
    
    mock_exchange.fetch_order_book.return_value['asks'] = [[101.0, 1.0], [101.2, 50.0], [101.5, 1.1]]
    result = guardian.analyze_liquidity("BTC/USDT", 0.5, "BUY")
    print(f"Is Safe: {result['is_safe']} | Reason: {result['reason']}")
    
    # Case B: High Slippage
    print("\n[TEST 2] Testing High Slippage...")
    # Book has low liquidity
    mock_exchange.fetch_order_book.return_value = {
        'bids': [[100.0, 0.01], [99.0, 0.01], [98.0, 0.01]],
        'asks': [[101.0, 0.01], [102.0, 0.01], [103.0, 0.01]]
    }
    result = guardian.analyze_liquidity("BTC/USDT", 0.5, "BUY") # Buying 0.5 when only 0.01 available at top
    print(f"Is Safe: {result['is_safe']} | Reason: {result['reason']} | Slippage: {result['slippage_pct']*100:.2f}%")

if __name__ == "__main__":
    test_liquidity_guardian_wall()
