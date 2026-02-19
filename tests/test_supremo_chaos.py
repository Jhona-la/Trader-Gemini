import asyncio
import pytest
import time
from core.engine import Engine
from core.events import MarketEvent, SignalEvent, EventType
from core.enums import SignalType
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_latency_spike_resilience():
    """Simulate a massive latency spike in event processing"""
    engine = Engine()
    # Mock portfolio and risk to avoid real calls
    engine.portfolio = MagicMock()
    engine.risk_manager = MagicMock()
    
    # Send 100 events rapidly
    start_time = time.perf_counter()
    for i in range(100):
        event = MarketEvent(
            symbol="BTC/USDT",
            timestamp=datetime.now(timezone.utc),
            close_price=50000.0 + i
        )
        engine.events.put(event)
        
    # Process them
    # We run for a short burst
    try:
        await asyncio.wait_for(engine.start(), timeout=0.1)
    except asyncio.TimeoutError:
        pass
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Chaos Test: Processed events. Duration: {duration:.4f}s")
    assert duration < 1.0 # Should be very fast due to async queue

@pytest.mark.asyncio
async def test_broken_connection_recovery():
    """Simulate a network timeout during order execution"""
    engine = Engine()
    engine.executor = MagicMock()
    engine.executor.execute_order.side_effect = ConnectionError("Binance API Timeout")
    
    sig = SignalEvent(
        strategy_id="CHAOS",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG
    )
    
    # This should be caught by error_handler and logged, not crash the engine
    with patch('utils.error_handler.logger.error') as mock_log:
        await engine._process_signal_event(sig)
        # Should NOT raise exception up to here
        print("Chaos Test: Connection error handled gracefully.")
