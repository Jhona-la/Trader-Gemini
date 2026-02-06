"""
Order Manager - Active Order Lifecycle Management (Phase 9)
==========================================================
- Monitors open limit orders.
- Implements Time-To-Live (TTL) protection.
- Orchestrates cancellation and repricing instructions.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from utils.logger import logger
from config import Config

class OrderManager:
    def __init__(self, executor):
        self.executor = executor
        self.open_orders = {}  # {order_id: {'timestamp': dt, 'symbol': s, 'type': t, 'ttl': int}}
        self.default_ttl = 60  # Phase 9.1: Balanced TTL

    def track_order(self, order_id: str, symbol: str, order_type: str, ttl: Optional[int] = None):
        """Registers a new order for monitoring."""
        self.open_orders[order_id] = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'type': order_type,
            'ttl': ttl or self.default_ttl
        }
        logger.debug(f"🎯 [OrderMgr] Tracking order {order_id} ({symbol}) | TTL: {ttl or self.default_ttl}s")

    async def monitor_lifecycle(self):
        """
        Background task to clean up stale orders.
        PROFESSOR: Protege contra el 'sniping' eliminando órdenes viejas.
        """
        now = datetime.now(timezone.utc)
        to_cancel = []

        for oid, info in self.open_orders.items():
            ttl = info.get('ttl', self.default_ttl)
            if now - info['timestamp'] > timedelta(seconds=ttl):
                to_cancel.append((oid, info['symbol'], ttl))

        for oid, symbol, ttl in to_cancel:
            logger.warning(f"🛡️ [OrderMgr] Order {oid} is STALE (> {ttl}s). Cancelling...")
            success = await self.executor.cancel_order(symbol, oid)
            if success:
                del self.open_orders[oid]
            else:
                logger.error(f"❌ [OrderMgr] Failed to cancel stale order {oid}")

    def remove_order(self, order_id: str):
        """Call this when a FILL event is received."""
        if order_id in self.open_orders:
            del self.open_orders[order_id]
            logger.debug(f"✅ [OrderMgr] Order {order_id} removed (Filled/Cancelled externally)")
