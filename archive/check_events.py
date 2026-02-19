from core.events import SignalEvent
from core.enums import SignalType
from datetime import datetime, timezone
import time

def check_event():
    print("🐘 Checking SignalEvent initialization...")
    s = SignalEvent(
        strategy_id="TEST",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG
    )
    print(f"  Signal timestamp_ns: {s.timestamp_ns} (Type: {type(s.timestamp_ns)})")
    
    if s.timestamp_ns is None:
        print("❌ ERROR: timestamp_ns is None!")
    else:
        print("✅ OK: timestamp_ns is set.")

if __name__ == "__main__":
    check_event()
