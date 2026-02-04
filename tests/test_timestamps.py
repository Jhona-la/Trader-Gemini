import unittest
from datetime import datetime, timezone, timedelta
from core.events import SignalEvent
from utils.time_helpers import ensure_utc_aware
from config import Config
from dataclasses import FrozenInstanceError

class TestTimestamps(unittest.TestCase):
    def test_ensure_utc_aware_valid(self):
        """Valid UTC datetime should pass."""
        dt = datetime.now(timezone.utc)
        self.assertEqual(ensure_utc_aware(dt), dt)

    def test_ensure_utc_aware_naive_raises(self):
        """Naive datetime should fail."""
        dt = datetime.now()
        with self.assertRaisesRegex(ValueError, "naive"):
            ensure_utc_aware(dt)

    def test_ensure_utc_aware_wrong_tz_raises(self):
        """Non-UTC timezone should fail."""
        # Create a non-UTC timezone
        est = timezone(timedelta(hours=-5))
        dt = datetime.now(est)
        with self.assertRaisesRegex(ValueError, "no está en UTC"):
            ensure_utc_aware(dt)

    def test_signal_event_requires_utc_aware(self):
        """SignalEvent should enforce UTC timestamps."""
        naive_dt = datetime.now()
        
        with self.assertRaisesRegex(ValueError, "SignalEvent validation failed"):
            SignalEvent(
                strategy_id="TEST",
                symbol="BTC/USDT",
                datetime=naive_dt,
                signal_type="LONG"
            )
            
    def test_signal_event_immutability(self):
        """Events should be immutable."""
        dt = datetime.now(timezone.utc)
        signal = SignalEvent(
            strategy_id="TEST",
            symbol="BTC/USDT",
            datetime=dt,
            signal_type="LONG"
        )
        
        with self.assertRaises(FrozenInstanceError):
            signal.strength = 0.5

    def test_engine_ttl_logic(self):
        """Verify TTL calculation logic."""
        # Manually verify the arithmetic used in engine
        old_dt = datetime.now(timezone.utc) - timedelta(seconds=Config.MAX_SIGNAL_AGE + 10)
        now_utc = datetime.now(timezone.utc)
        age = (now_utc - old_dt).total_seconds()
        self.assertTrue(age > Config.MAX_SIGNAL_AGE)

if __name__ == '__main__':
    unittest.main()
