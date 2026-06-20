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
    """
    BBO ARCHITECTURE: Order Lifecycle Manager
    QUÉ: Gestiona el ciclo de vida de órdenes LIMIT (entries y exits).
    POR QUÉ: Con BBO architecture, exits LIMIT necesitan chase/fallback.
    PARA QUÉ: Garantizar que exits nunca queden huérfanos (unfilled).
    CÓMO: EXIT orders que agotan chases → MARKET fallback nuclear.
         ENTRY orders que agotan chases → cancel (oportunidad perdida, no pérdida).
    """
    def __init__(self, executor, data_provider=None):
        self.executor = executor
        self.data_provider = data_provider
        self.open_orders = {}  # {order_id: {'timestamp': dt, 'symbol': s, 'type': t, 'ttl': int, 'side': s, 'price': p}}
        self.default_ttl = 30  # Phase 41: Aggressive HFT TTL (30s)
        self.max_chase_count = getattr(Config, 'Execution', None) and getattr(Config.Execution, 'MAX_CHASE_ATTEMPTS', 3) or 3

    def track_order(self, order_id: str, symbol: str, order_type: str, side: str = 'BUY', 
                    price: float = 0.0, quantity: float = 0.0, strategy_id: str = None, 
                    ttl: Optional[int] = None, metadata: Optional[Dict] = None):
        """Registers a new order for monitoring (Phase 41 Enhanced)."""
        # Phase 32: Persist Chase Count via Metadata
        chase_count = 0
        if metadata and 'chase_count' in metadata:
            chase_count = metadata['chase_count']
        
        # BBO: Extract exit order flag for chase/fallback behavior
        is_exit_order = False
        if metadata and 'is_exit_order' in metadata:
            is_exit_order = metadata['is_exit_order']
            
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
            'is_exit_order': is_exit_order,  # BBO: Determines MARKET fallback behavior
            'metadata': metadata or {}
        }
        logger.debug(f"🎯 [OrderMgr] Tracking {side} {'EXIT' if is_exit_order else 'ENTRY'} order {order_id} ({symbol}) | Qty: {quantity} | Price: {price} | TTL: {ttl or self.default_ttl}s | Chase: {chase_count}")

    async def monitor_lifecycle(self):
        """
        Phase 41: Smart Chasing & Anti-Sniping.
        If a LIMIT order is stale, check if we should "chase" the price or just cancel.
        """
        now = datetime.now(timezone.utc)
        to_process = []

        for oid, info in self.open_orders.items():
            ttl = info['ttl']
            if now - info['timestamp'] > timedelta(seconds=ttl):
                to_process.append((oid, info))

        for oid, info in to_process:
            symbol = info['symbol']
            side = info['side']
            chase_count = info['chase_count']
            is_exit_order = info['is_exit_order']
            
            # 1. CANCEL THE STALE ORDER
            logger.warning(f"🛡️ [OrderMgr] Order {oid} ({symbol}) is STALE (>{info['ttl']}s). {'EXIT' if is_exit_order else 'ENTRY'} order. Cancelling...")
            success = await self.executor.cancel_order(symbol, oid)
            if success:
                # Remove old order first to avoid tracking confusion
                if oid in self.open_orders:
                    del self.open_orders[oid]
                
                # 2. DECIDE IF WE CHASE OR FALLBACK
                if chase_count < self.max_chase_count and self.data_provider:
                    await self._attempt_chase(symbol, side, chase_count, info)
                elif is_exit_order:
                    # ══════════════════════════════════════════════════════════
                    # BBO: EXIT MARKET FALLBACK ("TAKER PANIC")
                    # QUÉ: Exits que agotaron todos los chases LIMIT.
                    # POR QUÉ: Un exit que NO se ejecuta = exposición abierta.
                    # PARA QUÉ: Cerrar la posición a cualquier costo.
                    # CÓMO: Emitir MARKET order nuclear via executor.
                    # ══════════════════════════════════════════════════════════
                    logger.warning(f"🔴 [BBO-FALLBACK] EXIT {symbol}: Max chase reached ({chase_count}/{self.max_chase_count}). MARKET NUCLEAR fallback!")
                    await self._market_fallback_exit(symbol, side, info)
                else:
                    logger.info(f"🛑 [OrderMgr] ENTRY {symbol}: Max chase reached ({chase_count}/{self.max_chase_count}). Order terminated (opportunity lost, not capital lost).")
            else:
                logger.error(f"❌ [OrderMgr] Failed to cancel stale order {oid}")

    async def _attempt_chase(self, symbol: str, side: str, chase_count: int, old_info: dict):
        """
        BBO ARCHITECTURE: Replace cancelled order at new BBO price.
        QUÉ: Re-place la orden LIMIT cancelada al nuevo mejor precio.
        POR QUÉ: El precio se movió y la orden original está demasiado lejos.
        PARA QUÉ: Mantener Maker fee mientras perseguimos el fill.
        """
        try:
            # Get latest BBO/Price
            current_price = 0.0
            if hasattr(self.data_provider, 'get_last_price'):
                current_price = self.data_provider.get_last_price(symbol)
            
            if current_price <= 0:
                is_exit = old_info['is_exit_order']
                if is_exit:
                    logger.warning(f"⚠️ [OrderMgr] Cannot chase EXIT {symbol}: price unknown. MARKET fallback!")
                    await self._market_fallback_exit(symbol, side, old_info)
                else:
                    logger.warning(f"⚠️ [OrderMgr] Cannot chase ENTRY {symbol}: price unknown.")
                return

            # RE-PRICE: Ensure we are at BBO
            new_price = current_price
            qty = old_info['quantity']
            if qty <= 0:
                logger.warning(f"⚠️ [OrderMgr] Cannot chase {symbol}: quantity missing.")
                return

            new_chase_count = chase_count + 1
            is_exit = old_info['is_exit_order']
            logger.info(f"🏹 [OrderMgr] CHASING {'EXIT' if is_exit else 'ENTRY'} {symbol}: Re-placing {side} at ${new_price:.2f} (Chase #{new_chase_count})")
            
            from core.events import OrderEvent
            from core.enums import OrderType, OrderSide
            
            # Preserve exit metadata for the chase
            chase_metadata = dict(old_info['metadata'])
            chase_metadata['chase_count'] = new_chase_count
            
            new_event = OrderEvent(
                symbol=symbol,
                order_type=OrderType.LIMIT,
                quantity=qty,
                direction=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                price=new_price,
                strategy_id=old_info['strategy_id'],
                horizon=old_info['metadata'].get('horizon', 'SCALPING'),
                is_exit=is_exit,
                is_close=is_exit,
                metadata=chase_metadata
            )
            
            # Execute directly via executor to bypass queue delays in HFT
            await self.executor.execute_order(new_event)
            
        except Exception as e:
            logger.error(f"Error during chase: {e}")

    async def _market_fallback_exit(self, symbol: str, side: str, info: dict):
        """
        BBO ARCHITECTURE: MARKET Nuclear Fallback for unfilled exits.
        QUÉ: Orden MARKET de emergencia para cerrar posición.
        POR QUÉ: Un exit LIMIT que no se llena deja la posición expuesta.
        PARA QUÉ: Garantizar cierre a cualquier costo (Taker fee acceptable).
        """
        try:
            from core.events import OrderEvent
            from core.enums import OrderType, OrderSide
            
            qty = info['quantity']
            if qty <= 0:
                logger.error(f"❌ [BBO-FALLBACK] Cannot do MARKET fallback for {symbol}: no quantity")
                return
            
            # Get current price for reference
            current_price = 0.0
            if self.data_provider and hasattr(self.data_provider, 'get_last_price'):
                current_price = self.data_provider.get_last_price(symbol)
            
            fallback_event = OrderEvent(
                symbol=symbol,
                order_type=OrderType.MARKET,  # 🔴 TAKER PANIC: MARKET nuclear
                quantity=qty,
                direction=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                price=current_price if current_price > 0 else None,
                strategy_id='BBO_MARKET_FALLBACK',
                horizon=info['metadata'].get('horizon', 'SCALPING'),
                is_exit=True,
                is_close=True,
                metadata={'exit_mode': 'TAKER_PANIC', 'chase_exhausted': True}
            )
            
            logger.warning(f"🔴 [BBO-FALLBACK] Executing MARKET exit for {symbol} | Qty: {qty} | Side: {side}")
            await self.executor.execute_order(fallback_event)
            
        except Exception as e:
            logger.error(f"❌ [BBO-FALLBACK] MARKET fallback failed for {symbol}: {e}")

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
