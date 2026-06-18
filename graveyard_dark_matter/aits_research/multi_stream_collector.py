"""
AITS Phase 2: Data Nervous System
Multi-Stream Collector

Ingests high-value market events (Liquidations) from Binance Futures WebSockets.
Simulates Macro/On-Chain feeds for the Plug-and-Play architecture.
Pushes all captured raw events into a Redis Stream for the Feature Warehouse.
"""

import asyncio
import json
import time
import logging
try:
    import aiohttp
    import redis.asyncio as redis
except ImportError:
    aiohttp = None
    redis = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REDIS_URL = "redis://localhost:6379"
STREAM_KEY = "aits:raw:events"

class MultiStreamCollector:
    def __init__(self):
        self.redis = None
        self.running = False
        self.binance_ws_url = "wss://fstream.binance.com/ws/!forceOrder@arr"

    async def init_redis(self):
        if not redis:
            logging.error("redis package not installed.")
            return False
        try:
            self.redis = await redis.from_url(REDIS_URL)
            await self.redis.ping()
            logging.info("✅ Connected to Redis for Stream Publishing.")
            return True
        except Exception as e:
            logging.error(f"Redis Connection Failed: {e}")
            return False

    async def publish_event(self, event_type: str, payload: dict):
        if self.redis:
            # Append standard metadata
            payload["_event_type"] = event_type
            payload["_ingest_time"] = time.time()
            try:
                await self.redis.xadd(STREAM_KEY, {"payload": json.dumps(payload)})
            except Exception as e:
                logging.error(f"Redis Publish Error: {e}")

    async def consume_binance_liquidations(self):
        """Connects to Binance Futures to catch liquidation orders in real-time."""
        if not aiohttp:
            logging.error("aiohttp not installed.")
            return

        logging.info("🌊 Starting Binance Liquidation Sniper...")
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.binance_ws_url) as ws:
                        logging.info("🔗 Connected to Binance @forceOrder stream.")
                        async for msg in ws:
                            if not self.running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                # Extract liquidation payload
                                o = data.get('o', {})
                                payload = {
                                    "symbol": o.get("s"),
                                    "side": o.get("S"),
                                    "order_type": o.get("o"),
                                    "quantity": float(o.get("q", 0)),
                                    "price": float(o.get("p", 0)),
                                    "avg_price": float(o.get("ap", 0)),
                                    "status": o.get("X"),
                                    "liquidation_time": o.get("T")
                                }
                                logging.info(f"💥 LIQUIDATION: {payload['side']} {payload['symbol']} @ {payload['price']} (Qty: {payload['quantity']})")
                                await self.publish_event("LIQUIDATION", payload)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as e:
                logging.error(f"WebSocket Error: {e}")
                await asyncio.sleep(5)  # Reconnection backoff

    async def simulate_macro_onchain_stream(self):
        """Simulates incoming Whale Alerts and DXY Macro movements."""
        logging.info("🌍 Starting Macro/On-Chain Mock Stream...")
        while self.running:
            await asyncio.sleep(15)  # Simulate infrequent events
            
            # Simulated Whale Transfer Alert
            whale_payload = {
                "asset": "BTC",
                "amount": 1500,
                "from_address": "Unknown Wallet",
                "to_address": "Binance Hot Wallet",
                "severity": "HIGH_EXCHANGE_INFLOW"
            }
            logging.info(f"🐋 WHALE ALERT: {whale_payload['amount']} {whale_payload['asset']} transferred to Exchange.")
            await self.publish_event("ONCHAIN_WHALE_ALERT", whale_payload)

    async def start(self):
        self.running = True
        if not await self.init_redis():
            logging.warning("Running without Redis (Dry Run).")
            
        tasks = [
            asyncio.create_task(self.consume_binance_liquidations()),
            asyncio.create_task(self.simulate_macro_onchain_stream())
        ]
        
        logging.info("🚀 AITS Multi-Stream Collector is active.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logging.info("Shutting down streams...")
        finally:
            self.running = False
            if self.redis:
                await self.redis.close()

if __name__ == "__main__":
    collector = MultiStreamCollector()
    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
