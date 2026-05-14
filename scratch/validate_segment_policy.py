import sys
import os
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.DEBUG)

from core.engine import SegmentPolicyEngine, AssetClassifier
from core.events import SignalEvent, SignalType
from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from config import Config

def test_segment_policy_flow():
    print("🚀 Running Segment Policy Flow Validation...")
    
    # 1. Setup Environment
    portfolio = Portfolio(initial_capital=13.0, csv_path='scratch/test_trades.csv')
    risk_manager = RiskManager(portfolio=portfolio)
    print(f"RiskManager Portfolio: {risk_manager.portfolio}")
    print(f"Bool RiskManager Portfolio: {bool(risk_manager.portfolio)}")
    classifier = AssetClassifier()
    spe = SegmentPolicyEngine(classifier)
    
    # 2. Test MEMECOIN (WIF/USDT)
    meme_symbol = "WIF/USDT"
    print(f"\n--- Testing MEMECOIN: {meme_symbol} ---")
    segment = spe.classifier.get_class(meme_symbol)
    policy = spe._matrix_lookup(segment, "TRENDING", "SCALPING")
    print(f"Classification: {segment.name}")
    print(f"Policy: Alloc={policy.capital_allocation_pct}, Exec={policy.execution_type}, Trail={policy.trailing_aggression}")
    
    # Simulate Signal
    signal_meme = SignalEvent(
        strategy_id="TEST",
        symbol=meme_symbol,
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=0.8,
        horizon="SCALPING"
    )
    # Inject policy (usually done in Engine)
    object.__setattr__(signal_meme, 'segment_policy', policy)
    
    # Generate Order
    order_meme = risk_manager.generate_order(signal_meme, current_price=2.5)
    
    if order_meme:
        print(f"✅ Order Generated. Type: {order_meme.order_type}")
        print(f"   Reserved Dollar Size: ${order_meme.metadata.get('dollar_size', 0):.2f}")
        print(f"   Expected Max Alloc: ${13.0 * policy.capital_allocation_pct:.2f}")
        print(f"   Order Metadata Exec Reason: {order_meme.metadata.get('routing_reason')}")
        assert order_meme.order_type.name == "MARKET", "MEMECOIN should override to MARKET!"
        assert order_meme.metadata['segment_policy'] == policy, "Policy must propagate to OrderEvent metadata!"
    else:
        print("❌ MEME Order Rejected unexpectedly.")

    # 3. Test MAJOR (BTC/USDT)
    major_symbol = "BTC/USDT"
    print(f"\n--- Testing MAJOR: {major_symbol} ---")
    segment = spe.classifier.get_class(major_symbol)
    policy = spe._matrix_lookup(segment, "TRENDING", "SCALPING")
    print(f"Classification: {segment.name}")
    print(f"Policy: Alloc={policy.capital_allocation_pct}, Exec={policy.execution_type}, Trail={policy.trailing_aggression}")
    
    signal_major = SignalEvent(
        strategy_id="TEST",
        symbol=major_symbol,
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=0.8,
        horizon="SCALPING"
    )
    object.__setattr__(signal_major, 'segment_policy', policy)
    
    order_major = risk_manager.generate_order(signal_major, current_price=65000.0)
    
    if order_major:
        print(f"✅ Order Generated. Type: {order_major.order_type}")
        print(f"   Reserved Dollar Size: ${order_major.metadata.get('dollar_size', 0):.2f}")
        print(f"   Expected Max Alloc: ${13.0 * policy.capital_allocation_pct:.2f}")
        print(f"   Order Metadata Exec Reason: {order_major.metadata.get('routing_reason')}")
        # Assuming not exploding, it should be LIMIT
        assert order_major.order_type.name == "LIMIT", "MAJOR should use LIMIT (Maker) default!"
    else:
        print("❌ MAJOR Order Rejected unexpectedly.")

    print("\n✅ VALIDATION COMPLETE.")

if __name__ == "__main__":
    test_segment_policy_flow()
