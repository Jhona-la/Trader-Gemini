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
    def __init__(self, executor, data_provider=None):
        self.executor = executor
        self.data_provider = data_provider
        self.open_orders = {}  # {order_id: {'timestamp': dt, 'symbol': s, 'type': t, 'ttl': int, 'side': s, 'price': p}}
        self.default_ttl = 30  # Phase 41: Aggressive HFT TTL (30s)
        self.max_chase_count = 3 # Avoid endless chasing

    def track_order(self, order_id: str, symbol: str, order_type: str, side: str = 'BUY', 
                    price: float = 0.0, quantity: float = 0.0, strategy_id: str = None, 
                    ttl: Optional[int] = None, metadata: Optional[Dict] = None):
        """Registers a new order for monitoring (Phase 41 Enhanced)."""
        # Phase 32: Persist Chase Count via Metadata
        chase_count = 0
        if metadata and 'chase_count' in metadata:
            chase_count = metadata.get('chase_count', 0)
            
        self.open_orders[order_id] = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'type': order_type,
            'side': side.upper(),
            'price': price,
            'quantity': quantity,
            'strategy_id': strategy_id,
            'ttl': ttl or self.default_ttl,
            'chase_count': chase_count,
            'metadata': metadata or {}
        }
        logger.debug(f"🎯 [OrderMgr] Tracking {side} order {order_id} ({symbol}) | Qty: {quantity} | Price: {price} | TTL: {ttl or self.default_ttl}s | Chase: {chase_count}")

    async def monitor_lifecycle(self):
        """
        Phase 41: Smart Chasing & Anti-Sniping.
        If a LIMIT order is stale, check if we should "chase" the price or just cancel.
        """
        now = datetime.now(timezone.utc)
        to_process = []

        for oid, info in self.open_orders.items():
            ttl = info.get('ttl', self.default_ttl)
            if now - info['timestamp'] > timedelta(seconds=ttl):
                to_process.append((oid, info))

        for oid, info in to_process:
            symbol = info['symbol']
            side = info['side']
            chase_count = info.get('chase_count', 0)
            
            # 1. CANCEL THE STALE ORDER
            logger.warning(f"🛡️ [OrderMgr] Order {oid} ({symbol}) is STALE (> {info['ttl']}s). Cancelling...")
            success = await self.executor.cancel_order(symbol, oid)
            if success:
                # Remove old order first to avoid tracking confusion
                if oid in self.open_orders:
                    del self.open_orders[oid]
                
                # 2. DECIDE IF WE CHASE
                if chase_count < self.max_chase_count and self.data_provider:
                    await self._attempt_chase(symbol, side, chase_count, info)
                else:
                    logger.info(f"🛑 [OrderMgr] Max chase reached ({chase_count}/{self.max_chase_count}) or data missing for {symbol}. Order terminated.")
            else:
                logger.error(f"❌ [OrderMgr] Failed to cancel stale order {oid}")

    async def _attempt_chase(self, symbol: str, side: str, chase_count: int, old_info: dict):
        """
        Logic to replace the cancelled order at new BBO.
        """
        try:
            # Get latest BBO/Price
            current_price = 0.0
            if hasattr(self.data_provider, 'get_last_price'):
                current_price = self.data_provider.get_last_price(symbol)
            
            if current_price <= 0:
                logger.warning(f"⚠️ [OrderMgr] Cannot chase {symbol}: price unknown.")
                return

            # RE-PRICE: Ensure we are at BBO
            new_price = current_price
            qty = old_info.get('quantity', 0)
            if qty <= 0:
                logger.warning(f"⚠️ [OrderMgr] Cannot chase {symbol}: quantity missing.")
                return

            new_chase_count = chase_count + 1
            logger.info(f"🏹 [OrderMgr] CHASING {symbol}: Re-placing {side} at ${new_price:.2f} (Chase #{new_chase_count})")
            
            from core.events import OrderEvent
            from core.enums import OrderType, OrderSide
            
            # Create a virtual OrderEvent to trigger the re-entry
            # Note: We reuse the original strategy_id if available
            new_event = OrderEvent(
                symbol=symbol,
                order_type=OrderType.LIMIT,
                quantity=qty,
                direction=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                price=new_price,
                strategy_id=old_info.get('strategy_id', 'ORDER_CHASER'),
                # Pass incremented chase count via metadata
                metadata={'chase_count': new_chase_count}
            )
            
            # Execute directly via executor to bypass queue delays in HFT
            # The executor.execute_order will handle the actual API call
            self.executor.execute_order(new_event)
            
        except Exception as e:
            logger.error(f"Error during chase: {e}")

    def remove_order(self, order_id: str, event=None):
        """
        Call this when a FILL event is received.
        Phase 31: Support Partial Fills via 'event.is_closed'.
        """
        if order_id in self.open_orders:
            # Check for partial fill
            if event and hasattr(event, 'is_closed') and not event.is_closed:
                # Order is PARTIALLY filled, keep tracking
                # Ideally update remaining quantity, but tracking logic is minimal here.
                # Just LOG it.
                logger.info(f"⏳ [OrderMgr] Partial Fill for {order_id}. Order remains active.")
                return

            del self.open_orders[order_id]
            logger.debug(f"✅ [OrderMgr] Order {order_id} removed (Filled/Cancelled/Closed)")
