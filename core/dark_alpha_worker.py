import json
import asyncio
import websockets
import threading
from utils.logger import logger
from core.dark_alpha_queue import DarkAlphaQueue

class DarkAlphaWorker:
    """
    Connects to Hyperliquid WebSocket to feed the Zero-Copy Cython Queue.
    Detects market liquidations (whale sweeps) ahead of Binance CEX.
    """
    def __init__(self):
        self.queue = DarkAlphaQueue(halflife=15.0)
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self._thread = None
        self._loop = None
        self.is_running = False
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="DarkAlphaWorker")
        self._thread.start()
        logger.info("🛰️ [DARK ALPHA] Hyperliquid Zero-Copy Worker started.")
        
    def stop(self):
        self.is_running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_loop())
        except Exception as e:
            logger.error(f"❌ [DARK ALPHA] Loop crashed: {e}")
        
    async def _ws_loop(self):
        while self.is_running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("⚡ [DARK ALPHA] Connected to Hyperliquid DEX.")
                    # Subscribe to trades to infer liquidations or large sweeps
                    req = {
                        "method": "subscribe",
                        "subscription": {"type": "trades", "coin": "BTC"}
                    }
                    await ws.send(json.dumps(req))
                    
                    while self.is_running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        if "data" in data and isinstance(data["data"], list):
                            for trade in data["data"]:
                                sz = float(trade.get("sz", 0))
                                px = float(trade.get("px", 0))
                                notional = sz * px
                                
                                # High frequency whales or liquidation sweeps > $250k
                                if notional > 250_000:
                                    side_str = trade.get("side", "")
                                    # "B" implies Short getting liquidated (Buy pressure)
                                    # "A" implies Long getting liquidated (Sell pressure)
                                    side_int = -1 if side_str == "B" else 1
                                    
                                    # Push to Lock-Free Cython Queue
                                    self.queue.push_liquidation(side_int, notional)
                                    
            except Exception as e:
                if self.is_running:
                    logger.warning(f"⚠️ [DARK ALPHA] WS Disconnected. Reconnecting in 2s... ({e})")
                    await asyncio.sleep(2)
                
    def get_net_pressure(self):
        """
        Gets the exponential time-decayed liquidation pressure.
        Calculated entirely in C++ without GIL interference.
        """
        return self.queue.get_net_pressure()

# Singleton instance
dark_alpha_worker = DarkAlphaWorker()
