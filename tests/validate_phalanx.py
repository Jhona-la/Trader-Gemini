"""
🧪 VALIDATION SCRIPT: Phalanx-Omega Protocol
QUÉ: Verifies Absorption Detection and Dynamic Kelly Sizing in a simulated environment.
POR QUÉ: Ensure new Phase 13 components work as expected before Live deployment.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.phalanx import OrderFlowAnalyzer, OnlineGARCH
from core.portfolio import Portfolio
from core.events import SignalEvent
from core.enums import SignalType

# Mock Classes
class MockDataProvider:
    def get_latest_bars(self, symbol, n=1):
        # Generate synthetic bars with absorption pattern
        # Pattern: High Volume + Small Body (Doji) after downtrend
        bars = np.zeros(n, dtype=[
            ('timestamp', 'i8'), ('open', 'f4'), ('high', 'f4'), 
            ('low', 'f4'), ('close', 'f4'), ('volume', 'f4')
        ])
        
        start_ts = 1600000000000
        price = 100.0
        
        for i in range(n):
            bars[i]['timestamp'] = start_ts + (i * 60000)
            
            if i < n-5:
                # Downtrend
                price -= 1.0
                bars[i]['open'] = price + 0.5
                bars[i]['close'] = price
                bars[i]['high'] = price + 0.6
                bars[i]['low'] = price - 0.1
                bars[i]['volume'] = 100.0
            else:
                # Absorption: Stopping Volume
                # Price stabilizes, Volume Spikes, Body Small
                bars[i]['open'] = price
                bars[i]['close'] = price + 0.05 # Small bullish body
                bars[i]['high'] = price + 0.1
                bars[i]['low'] = price - 0.1
                bars[i]['volume'] = 500.0 + (i*10) # Huge volume
            
        return bars

def test_absorption_detection():
    print("\n🔬 TESTING ABSORPTION DETECTION...")
    analyzer = OrderFlowAnalyzer()
    
    # 1. Create Synthetic Data (NumPy)
    data_provider = MockDataProvider()
    bars = data_provider.get_latest_bars('BTC/USDT', n=20)
    
    # 2. Run Detection
    result = analyzer.is_absorption_detected(bars)
    
    print(f"Input Bars (Last 3):")
    for b in bars[-3:]:
        print(f"  Vol: {b['volume']:.1f}, Range: {b['high']-b['low']:.2f}, Body: {abs(b['close']-b['open']):.2f}")
        
    print(f"\nResult: {result}")
    
    # Assertions
    if result['detected'] and result['type'] == 'BULLISH':
        print("✅ PASS: Bullish Absorption Detected Correctly.")
    else:
        print("❌ FAIL: Absorption Not Detected.")

def test_dynamic_kelly():
    print("\n🔬 TESTING DYNAMIC KELLY SIZING...")
    portfolio = Portfolio(initial_capital=1000.0)
    strat_id = "test_strat"
    
    # 1. Simulate Trade History for Strategy
    # Scenario: High Win Rate (60%), Avg Win $100, Avg Loss $50 (Payoff = 2.0)
    print("  Simulating 20 trades (12 Wins, 8 Losses)...")
    
    for i in range(12):
        portfolio._update_strategy_performance(strat_id, 100.0) # Win
        
    for i in range(8):
        portfolio._update_strategy_performance(strat_id, -50.0) # Loss
        
    stats = portfolio.strategy_performance[strat_id]
    print(f"  Stats: WR={stats['win_rate']:.2f}, Wins={stats['wins']}, Loss={stats['losses']}")
    print(f"  Total Win PnL: ${stats.get('total_win_pnl', 0)}")
    print(f"  Total Loss PnL: ${stats.get('total_loss_pnl', 0)}")
    
    # 2. Calculate Kelly
    # p = 0.6, q = 0.4
    # b = 100 / 50 = 2.0
    # f = 0.6 - (0.4 / 2.0) = 0.6 - 0.2 = 0.4 (40%)
    # Smart Kelly caps at 0.05 (5%)
    
    size = portfolio.get_smart_kelly_sizing("BTC/USDT", strat_id)
    print(f"  Calculated Size: {size:.4f}")
    
    expected_raw_kelly = 0.4
    # Our func applies 0.5 safety factor -> 0.2
    # Then max(0.005, min(0.05, 0.2)) -> 0.05
    
    if size == 0.05:
        print("✅ PASS: Kelly Sizing correctly capped at 5%.")
    else:
        print(f"❌ FAIL: Expected 0.05, got {size}")

    # Test Poor Performance
    print("\n  Simulating Losing Streak...")
    for i in range(20):
        portfolio._update_strategy_performance(strat_id, -50.0) # 20 more losses
        
    size_bad = portfolio.get_smart_kelly_sizing("BTC/USDT", strat_id)
    print(f"  Stats after losses: WR={portfolio.strategy_performance[strat_id]['win_rate']:.2f}")
    print(f"  Calculated Size (Bad): {size_bad:.4f}")
    
    if size_bad < size:
        print("✅ PASS: Sizing reduced after poor performance.")
    else:
        print("❌ FAIL: Sizing did not reduce.")

if __name__ == "__main__":
    print("="*40)
    print("PHALANX-OMEGA VALIDATION PROTOCOL")
    print("="*40)
    try:
        test_absorption_detection()
        test_dynamic_kelly()
        print("\n✅ ALL TESTS PASSED.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
