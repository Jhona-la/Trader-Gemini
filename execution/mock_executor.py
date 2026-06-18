import asyncio
import time
import random
from typing import Dict, Any, List
from datetime import datetime, timezone

from utils.logger import logger
from config import Config
from core.events import OrderEvent, FillEvent
from core.enums import OrderType, OrderSide

class MockExecutor:
    """
    PHASE 36: NATIVE NANOSECOND PAPER TRADING SIMULATOR
    Replaces Binance Testnet completely. Operates locally in memory,
    simulating infinite liquidity and realistic slippage.
    """

    def __init__(self, leverage: int = 10):
        self.leverage = leverage
        self.mock_balances = {
            "USDT": Config.INITIAL_CAPITAL
        }
        self.mock_positions = {}
        self.order_id_counter = 1000
        self.latency_sim_ms = 5  # Simulate 5ms local execution latency
        self.events_queue = None
        self.portfolio = None
        
        logger.info(f"⚡ [MOCK EXECUTOR] Initialized with ${self.mock_balances['USDT']} USD. Infinite Liquidity Engine Active.")

    def set_leverage(self, symbol: str, leverage: int):
        self.leverage = leverage
        logger.info(f"  [MOCK] Leverage set to {leverage}x for {symbol}")

    def get_account_balance(self) -> float:
        return self.mock_balances.get("USDT", 0.0)

    async def start_zmq_loop(self):
        """Phase OMNI: Decoupled OS-Level Event Ingestion via ZMQ or local Queue."""
        logger.info("🔄 [MOCK ZMQ] Starting Mock Executor PULL Loop...")
        try:
            while True:
                if self.zmq_pull:
                    event = await self.zmq_pull.pull()
                elif self.events_queue:
                    # Fallback to local asyncio queue if no ZMQ
                    event = await self.events_queue.get()
                else:
                    await asyncio.sleep(1)
                    continue
                    
                if isinstance(event, OrderEvent):
                    await self.execute_order(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"MockExecutor Pull Error: {e}")

    async def execute_order(self, event: OrderEvent):
        """
        Executes a simulated OrderEvent instantly with calculated slippage
        and emits a FillEvent back to the engine.
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_sim_ms / 1000.0)
        
        symbol = event.symbol
        side = event.direction
        quantity = event.quantity
        price = event.price if event.price else getattr(event, 'close_price', 0.0)
        order_type = event.order_type

        if not price or price <= 0:
            logger.error(f"[MOCK] Market orders need a reference price. Symbol: {symbol}")
            return None

        # Apply slippage dynamically
        slip_pct = 0.0 if order_type == OrderType.LIMIT else 0.00015
        executed_price = price * (1 + slip_pct) if side == OrderSide.BUY else price * (1 - slip_pct)
        
        fill_cost = executed_price * quantity
        required_margin = fill_cost / self.leverage

        if not getattr(event, 'is_exit', False):
            if required_margin > self.mock_balances["USDT"]:
                logger.warning(f"🛑 [MOCK INSUFFICIENT MARGIN] Need ${required_margin:.2f}, Have ${self.mock_balances['USDT']:.2f}")
                return None

            # Deduct mock balance
            self.mock_balances["USDT"] -= required_margin
        
        self.order_id_counter += 1
        commission = fill_cost * (0.0002 if order_type == OrderType.LIMIT else 0.0004)
        
        metadata = dict(event.metadata) if getattr(event, 'metadata', None) else {}
        metadata["is_exit"] = getattr(event, 'is_exit', False)

        fill_event = FillEvent(
            timeindex=datetime.now(timezone.utc),
            symbol=symbol,
            exchange="MOCK_BINANCE",
            quantity=quantity,
            direction=side,
            fill_cost=fill_cost,
            commission=commission,
            strategy_id=getattr(event, 'strategy_id', 'UNKNOWN'),
            fill_price=executed_price,
            order_id=f"MOCK_{self.order_id_counter}",
            sl_pct=event.sl_pct,
            tp_pct=event.tp_pct,
            horizon=event.horizon,
            leverage=getattr(event, 'leverage', self.leverage),
            metadata=metadata,
            trade_id=getattr(event, 'trade_id', None),
            setup_type=getattr(event, 'setup_type', None),
            exit_reason=getattr(event, 'exit_reason', None)
        )

        logger.info(f"🟢 [MOCK EXECUTION] {side.value} {quantity} {symbol} @ {executed_price:.4f} (Margin: ${required_margin:.2f})")
        
        # Emit FillEvent
        if getattr(self, 'events_queue', None):
            self.events_queue.put(fill_event)
        elif self.events_queue:
            await self.events_queue.put(fill_event)
            
        return {"status": "FILLED", "orderId": f"MOCK_{self.order_id_counter}"}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        logger.info(f"  [MOCK] Cancel order requested for {order_id}")
        return True

    async def update_leverage_and_margin(self, symbol: str, leverage: int = 10, margin_type: str = 'ISOLATED'):
        self.set_leverage(symbol, leverage)
        return True
        
    def sync_portfolio_state(self, portfolio):
        logger.info("  [MOCK] sync_portfolio_state called.")
        pass

    def sync_execute_order(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_order(*args, **kwargs))
