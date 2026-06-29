import os
import time
from datetime import datetime, timezone

os.environ["ENVIRONMENT"] = "BACKTEST"
os.environ["OMNISCIENT_NO_DB"] = "1"

from core.engine import Engine, PriorityBoundedQueue
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from core.events import SignalEvent
from core.enums import SignalType

def main():
    print("Initializing...")
    event_queue = PriorityBoundedQueue()
    portfolio = Portfolio(13.0)
    portfolio.events = event_queue
    risk_manager = RiskManager(portfolio=portfolio)
    
    signal = SignalEvent(
        symbol="AAVE/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=0.9,
        strategy_id="TEST_SCALPING",
        ml_confidence=0.8,
        horizon="SCALPING",
        current_price=100.0,
        metadata={"atr_pct": 0.05, "volatility": "HIGH"}
    )
    
    print("Generating order...")
    order = risk_manager.generate_order(signal, 100.0)
    
    if order is None:
        print(f"Order rejected. Rejection reason: {signal.metadata.get('rejection_reason', 'UNKNOWN')}")
    elif isinstance(order, list):
        print(f"Order generated successfully! Quantity: {[o.quantity for o in order]}")
    else:
        print(f"Order generated successfully! Quantity: {order.quantity}")

if __name__ == "__main__":
    main()
