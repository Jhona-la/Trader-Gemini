import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta
from core.events import OrderEvent, FillEvent, OrderType, OrderSide
from core.order_manager import OrderManager

class TestF31_PartialFill(unittest.IsolatedAsyncioTestCase):
    async def test_partial_fill_does_not_remove_order(self):
        # Setup
        executor = MagicMock()
        om = OrderManager(executor)
        
        order_id = "ORD-123"
        om.track_order(order_id, "BTCUSDT", "LIMIT", "BUY", 10000.0, 1.0)
        
        # Simulate Partial Fill Event (50%)
        fill_event = FillEvent(
            timeindex=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            exchange="BINANCE",
            quantity=0.5,
            direction=OrderSide.BUY,
            fill_cost=5000.0,
            order_id=order_id,
            is_closed=False # Explicit Partial Fill
        )
        
        # Act
        om.remove_order(order_id, event=fill_event)
        
        # Assert
        self.assertIn(order_id, om.open_orders)
        
    async def test_full_fill_removes_order(self):
        # Setup
        executor = MagicMock()
        om = OrderManager(executor)
        
        order_id = "ORD-456"
        om.track_order(order_id, "BTCUSDT", "LIMIT", "BUY", 10000.0, 1.0)
        
        # Simulate Full Fill Event
        fill_event = FillEvent(
            timeindex=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            exchange="BINANCE",
            quantity=1.0,
            direction=OrderSide.BUY,
            fill_cost=10000.0,
            order_id=order_id,
            is_closed=True # Full Fill
        )
        
        # Act
        om.remove_order(order_id, event=fill_event)
        
        # Assert
        self.assertNotIn(order_id, om.open_orders)

class TestF32_ChaseLogic(unittest.IsolatedAsyncioTestCase):
    async def test_chase_increments_count(self):
        # Setup
        executor = MagicMock()
        executor.cancel_order = AsyncMock(return_value=True)
        # Mock execute_order to capture the new event
        captured_event = None
        def capture(event):
            nonlocal captured_event
            captured_event = event
        executor.execute_order = MagicMock(side_effect=capture)
        
        data_provider = MagicMock()
        data_provider.get_last_price.return_value = 10100.0
        
        om = OrderManager(executor, data_provider)
        
        # 1. Track initial order with Chase=0 and immediate timeout
        order_id = "ORD-ORIG"
        # We simulate that this order was created a minute ago
        om.track_order(order_id, "BTCUSDT", "LIMIT", "BUY", 10000.0, 1.0, ttl=5)
        om.open_orders[order_id]['timestamp'] = datetime.now(timezone.utc) - timedelta(seconds=10)
        
        # 2. Force lifecycle check (should detect stale order)
        await om.monitor_lifecycle()
        
        # Assert - Verify new order event was generated with incremented chase count
        self.assertIsNotNone(captured_event)
        self.assertEqual(captured_event.price, 10100.0)
        self.assertIn('chase_count', captured_event.metadata)
        self.assertEqual(captured_event.metadata['chase_count'], 1)
        
        # 3. Simulate re-tracking the NEW order (what binance_executor would do)
        new_oid = "ORD-CHASE-1"
        om.track_order(
            new_oid, 
            captured_event.symbol, 
            "LIMIT", 
            "BUY", 
            captured_event.price, 
            captured_event.quantity, 
            captured_event.strategy_id, 
            metadata=captured_event.metadata
        )
        
        # Verify stored state
        self.assertEqual(om.open_orders[new_oid]['chase_count'], 1)
        
    async def test_max_chase_limit(self):
        # Setup
        executor = MagicMock()
        executor.cancel_order = AsyncMock(return_value=True)
        executor.execute_order = MagicMock()
        
        data_provider = MagicMock()
        data_provider.get_last_price.return_value = 10200.0
        
        om = OrderManager(executor, data_provider)
        om.max_chase_count = 3
        
        # 1. Track order ALREADY at max chase count
        order_id = "ORD-MAX"
        om.track_order(order_id, "BTCUSDT", "LIMIT", "BUY", 10000.0, 1.0, metadata={'chase_count': 3})
        om.open_orders[order_id]['timestamp'] = datetime.now(timezone.utc) - timedelta(seconds=60)
        
        # 2. Force check
        await om.monitor_lifecycle()
        
        # Assert - Should CANCEL but NOT CHASE
        executor.cancel_order.assert_called_once()
        executor.execute_order.assert_not_called()
        self.assertNotIn(order_id, om.open_orders)

if __name__ == '__main__':
    unittest.main()
