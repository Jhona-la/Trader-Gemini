import time
import sys
import os
import json
from datetime import datetime, timezone
import os
import time
import collections

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import utils.safe_leverage first to break potential cycles
import utils.safe_leverage 
from risk.risk_manager import RiskManager
from core.engine import Engine
from utils.mae_tracker import MAETracker
import traceback

class MockPortfolio:
    def __init__(self):
        self.positions = {}
        self.global_regime_data = {}
        self.global_regime = 'NEUTRAL'
    def get_total_equity(self):
        return 1000.0
    def get_smart_kelly_sizing(self, symbol, strategy_id):
        return 0.20 # Mock 20%

def test_half_kelly():
    print("\n🛡️ TESTING HALF-KELLY SIZING...")
    try:
        # Mock Risk Manager
        portfolio = MockPortfolio()
        rm = RiskManager(portfolio=portfolio)
        
        # Test 1: Standard Calculation
        # We need to bypass get_smart_kelly_sizing if we want to test internal logic
        # OR we verify that the final result is halved mock
        
        # If smart_kelly returns 0.20, half-kelly should be 0.10
        kh = rm.calculate_kelly_fraction(strategy_id="TEST_STRAT")
        
        print(f"  Mock Smart Kelly: 0.20")
        print(f"  Result: {kh:.4f}")
        
        if 0.09 <= kh <= 0.11:
            print("✅ PASS: Half-Kelly Logic Active (approx 0.10).")
        else:
            print(f"❌ FAIL: Expected ~0.10, got {kh}")
            
    except Exception as e:
        print(f"❌ ERROR in Half-Kelly Test: {e}")
        traceback.print_exc()

def test_mae_logic():
    print("\n📉 TESTING MAE TRACKER...")
    try:
        tracker = MAETracker(filepath="tests/temp_mae.json")
        
        # Simulate 10 trades with 1% MAE
        for _ in range(10):
            tracker.record_trade("BTC/USDT", "LONG", 100, 105, 99, 105) # 1% MAE
            
        s = tracker.stats["BTC/USDT"]
        print(f"  Avg MAE: {s['sum_mae']/s['count']:.4f}")
        
        suggested_sl = tracker.get_optimal_stop_loss("BTC/USDT", safety_factor=1.5)
        print(f"  Suggested SL (1.5x): {suggested_sl:.4f}")
        
        if 0.014 <= suggested_sl <= 0.016:
            print("✅ PASS: Optimal SL calculation correct (1.5%).")
        else:
            print(f"❌ FAIL: Expected ~0.015, got {suggested_sl}")
            
    except Exception as e:
        print(f"❌ ERROR in MAE Test: {e}")
        traceback.print_exc()
        
    # Clean up
    if os.path.exists("tests/temp_mae.json"):
        os.remove("tests/temp_mae.json")

def test_latency_breaker():
    print("\n🛑 TESTING LATENCY CIRCUIT BREAKER...")
    try:
        # Mock DataHandler
        class MockDataHandler:
            def get_latency_metrics(self):
                return 200.0, 250.0 # High Latency
                
        engine = Engine()
        engine.data_handlers.append(MockDataHandler())
        
        print("  Simulating 200ms Ping...")
        
        # Create Dummy SIGNAL Event
        from core.events import SignalEvent
        from core.enums import SignalType
        from datetime import datetime
        
        sig = SignalEvent(
            strategy_id="TEST_STRAT", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG, 
            strength=1.0
        )
        
        # Inject into process_event (async)
        import asyncio
        async def run_check():
            engine.metrics['discarded_events'] = 0
            await engine.process_event(sig)
            return engine.metrics['discarded_events']
            
        discarded = asyncio.run(run_check())
        
        print(f"  Discarded Events: {discarded}")
        
        if discarded == 1:
            print("✅ PASS: High Latency Blocked Signal.")
        else:
            print("❌ FAIL: Signal NOT Blocked.")
            
    except Exception as e:
        print(f"❌ ERROR in Latency Test: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_half_kelly()
    test_mae_logic()
    test_latency_breaker()
