import asyncio
import time
from typing import Dict, Any, List
from utils.logger import logger
from config import Config

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
        
        # We need an order manager reference later
        self.order_manager = None
        
        logger.info(f"⚡ [MOCK EXECUTOR] Initialized with ${self.mock_balances['USDT']} USD. Infinite Liquidity Engine Active.")

    def set_leverage(self, symbol: str, leverage: int):
        self.leverage = leverage
        logger.info(f"  [MOCK] Leverage set to {leverage}x for {symbol}")

    def get_account_balance(self) -> float:
        return self.mock_balances.get("USDT", 0.0)

    async def start_zmq_loop(self):
        # We don't need a real ZMQ loop for local mock. Just stay alive.
        while True:
            await asyncio.sleep(3600)

    async def execute_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, reduce_only: bool = False, horizon: str = "SCALPING") -> Dict[str, Any]:
        """
        Executes a simulated order instantly with calculated slippage.
        """
        if not price:
            logger.error(f"[MOCK] Market orders need a reference price for the mock to work accurately.")
            return None

        # Simulate network latency
        await asyncio.sleep(self.latency_sim_ms / 1000.0)

        # Apply slippage dynamically
        slippage_pct = getattr(Config.Risk, 'MAX_SLIPPAGE_PCT', 0.001)
        executed_price = price * (1 + slippage_pct) if side.upper() == 'BUY' else price * (1 - slippage_pct)
        
        notional = executed_price * quantity
        required_margin = notional / self.leverage

        if not reduce_only:
            if required_margin > self.mock_balances["USDT"]:
                logger.warning(f"🛑 [MOCK INSUFFICIENT MARGIN] Need ${required_margin:.2f}, Have ${self.mock_balances['USDT']:.2f}")
                return None

            # Deduct mock balance
            self.mock_balances["USDT"] -= required_margin
        
        self.order_id_counter += 1
        
        fill_record = {
            "symbol": symbol,
            "orderId": f"MOCK_{self.order_id_counter}",
            "clientOrderId": f"mock_client_{self.order_id_counter}",
            "price": executed_price,
            "origQty": quantity,
            "executedQty": quantity,
            "status": "FILLED",
            "type": order_type,
            "side": side,
            "reduceOnly": reduce_only,
            "timestamp": int(time.time() * 1000)
        }

        logger.info(f"🟢 [MOCK EXECUTION] {side} {quantity} {symbol} @ {executed_price:.4f} (Margin: ${required_margin:.2f}) [Horizon: {horizon}]")
        return fill_record

    async def place_tp_sl_orders(self, symbol: str, quantity: float, side: str, entry_price: float, horizon: str = "SCALPING") -> Dict[str, str]:
        """
        Simulate placing TP/SL orders locally.
        """
        self.order_id_counter += 2
        return {
            "tp_order_id": f"MOCK_TP_{self.order_id_counter - 1}",
            "sl_order_id": f"MOCK_SL_{self.order_id_counter}"
        }

    async def update_leverage_and_margin(self, symbol: str, leverage: int = 10, margin_type: str = 'ISOLATED'):
        self.set_leverage(symbol, leverage)
        return True

    def sync_execute_order(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_order(*args, **kwargs))
