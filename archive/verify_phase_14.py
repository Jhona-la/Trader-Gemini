import time
from datetime import datetime, timezone
from core.events import SignalEvent
from core.enums import SignalType
from core.resolution_state import ResolutionState
from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
import logging

# Config logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class MockPortfolio(Portfolio):
    def __init__(self, initial_capital=100.0):
        self.current_cash = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}
        self.strategy_performance = {
            'ML_HIGH_WIN': {'wins': 80, 'losses': 20, 'pnl': 100.0, 'trades': 100},
            'ML_LOW_WIN': {'wins': 40, 'losses': 60, 'pnl': -50.0, 'trades': 100}
        }
    
    def get_total_equity(self):
        return self.current_cash

    # Inherits get_strategy_metrics from real Portfolio

def verify_phase_14():
    print("🧪 [TEST] Verificando Fase 14: Dynamic Capital Allocation...")
    
    # 1. Setup
    portfolio = MockPortfolio(initial_capital=1000.0) # $1000 Capital
    rm = RiskManager(portfolio=portfolio)
    rm.peak_capital = 1000.0
    
    # 2. Test ML Confidence Scaling
    print("\n📊 --- TEST 1: ML Confidence Scaling ---")
    
    # High Confidence Signal
    sig_high = SignalEvent(
        strategy_id="ML_HIGH_WIN",
        symbol="BTC/USDT", 
        datetime=datetime.now(timezone.utc), 
        signal_type=SignalType.LONG,
        strength=0.90 # Super high confidence
    )
    # Low Confidence Signal
    sig_low = SignalEvent(
        strategy_id="ML_HIGH_WIN", 
        symbol="ETH/USDT", 
        datetime=datetime.now(timezone.utc), 
        signal_type=SignalType.LONG, 
        strength=0.55 # Low confidence
    )
    
    size_high = rm.size_position(sig_high, current_price=50000)
    size_low = rm.size_position(sig_low, current_price=3000)
    
    print(f"  Confidence 0.90 -> Size: ${size_high:.2f}")
    print(f"  Confidence 0.55 -> Size: ${size_low:.2f}")
    
    if size_high > size_low:
        print("✅ Confidence Boost Verified")
    else:
        print("❌ Confidence Boost FAILED")
        
    # 3. Test Recovery Mode
    print("\n🛡️ --- TEST 2: Recovery Mode Logic ---")
    
    # Simulate 10% Drawdown
    portfolio.current_cash = 900.0 
    rm.peak_capital = 1000.0 # HWM
    
    # Force update logic
    current_dd = 1 - (portfolio.current_cash / rm.peak_capital)
    rm._update_resolution_state(current_dd)
    
    print(f"  Current DD: {current_dd*100:.1f}%")
    print(f"  Resolution State: {rm.resolution_state}")
    
    if rm.resolution_state == ResolutionState.RECOVERY:
        print("✅ Enters Recovery Mode Correctly")
    else:
        print(f"❌ Failed to enter Recovery Mode (State: {rm.resolution_state})")
        
    # Check sizing in recovery
    size_recovery = rm.size_position(sig_high, current_price=50000)
    print(f"  Recovery Size (Conf 0.90): ${size_recovery:.2f}")
    
    # Expected: Should be half of normal
    # Normal approx size for $900 with boost... hard to calc exact without full Kelly trace
    # But we can verify it's significantly lower than size_high (adjusted for cap diff)
    
    ratio = size_recovery / size_high
    print(f"  Ratio vs Normal: {ratio:.2f}")
    if ratio < 0.6: # Account dropped 10%, size dropped ~50% -> ratio approx 0.45
        print("✅ Defensive Sizing Verified")
    else:
        print("❌ Defensive Sizing Failed")

    # 4. Test Strategy Specific Kelly
    print("\n🧠 --- TEST 3: Strategy Specific Kelly ---")
    sig_bad = SignalEvent(
        strategy_id="ML_LOW_WIN", # 40% Win Rate
        symbol="LTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=0.7
    )
    
    # Reset capital for clean comparison
    portfolio.current_cash = 1000.0
    rm.peak_capital = 1000.0
    rm.resolution_state = ResolutionState.STABLE
    
    size_bad_strat = rm.size_position(sig_bad, current_price=100)
    print(f"  Bad Strategy (40% WR) Size: ${size_bad_strat:.2f}")
    
    # Compare with Good Strat (80% WR)
    size_good_strat = rm.size_position(sig_high, current_price=50000)
    print(f"  Good Strategy (80% WR) Size: ${size_good_strat:.2f}")
    
    if size_good_strat > size_bad_strat * 1.5:
        print("✅ Kelly Differentiation Verified")
    else:
        print("❌ Kelly Differentiation Failed")

if __name__ == "__main__":
    verify_phase_14()
