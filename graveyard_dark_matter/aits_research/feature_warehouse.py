"""
AITS Phase 2: Feature Warehouse
Polars-based Real-Time Processing Engine

This engine consumes raw streams from Redis (e.g., L2 Order Book snapshots from Phase 1 
and Liquidations/Macro events from Phase 2).
It calculates high-level institutional features like 'Liquidation Density' and 
'Volatility Bursts', then publishes them back to a new Redis stream for the 
Predictive Layer (AITS Layer 3) to consume.
"""

import asyncio
import json
import logging
import time

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    import polars as pl
except ImportError:
    pl = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REDIS_URL = "redis://localhost:6379"
STREAM_IN = "aits:raw:events"
STREAM_OUT = "aits:features:computed"
GROUP_NAME = "feature_warehouse_workers"
CONSUMER_NAME = "fw_worker_1"

class FeatureWarehouse:
    def __init__(self):
        self.redis = None
        self.running = False
        
        # State memory for calculating time-based features
        self.liquidation_window = []  # Store recent liquidations for density calc
        self.window_seconds = 60.0    # 1-minute rolling window

    async def init_redis(self):
        if not redis:
            logging.error("redis package not installed.")
            return False
        try:
            self.redis = await redis.from_url(REDIS_URL)
            await self.redis.ping()
            
            # Create consumer group
            try:
                await self.redis.xgroup_create(STREAM_IN, GROUP_NAME, mkstream=True)
            except Exception as e:
                if "BUSYGROUP" not in str(e):
                    logging.warning(f"Consumer group error: {e}")
                    
            logging.info("✅ Feature Warehouse connected to Redis.")
            return True
        except Exception as e:
            logging.error(f"Redis Connection Failed: {e}")
            return False

    def clean_liquidation_window(self, current_time):
        """Removes liquidations older than the window_seconds."""
        self.liquidation_window = [
            liq for liq in self.liquidation_window 
            if current_time - liq["_ingest_time"] <= self.window_seconds
        ]

    def compute_liquidation_density(self, current_time) -> float:
        """Calculates the total dollar volume of liquidations in the rolling window."""
        self.clean_liquidation_window(current_time)
        density = sum(liq.get("quantity", 0) * liq.get("price", 0) for liq in self.liquidation_window)
        return density

    async def process_batch(self, batch):
        """Processes a batch of raw events into structured features."""
        now = time.time()
        features = {}
        
        for msg_id, msg_data in batch:
            try:
                payload = json.loads(msg_data[b'payload'].decode('utf-8'))
                event_type = payload.get("_event_type")
                
                if event_type == "LIQUIDATION":
                    self.liquidation_window.append(payload)
                    density = self.compute_liquidation_density(now)
                    
                    logging.info(f"📊 [Feature Computed] Liquidation Density (1m): ${density:,.2f}")
                    
                    # If density is extreme, mark as Volatility Burst
                    if density > 1_000_000: # $1M liquidated in 1min
                        features["VOLATILITY_BURST"] = True
                        features["burst_magnitude"] = density
                        logging.warning("⚠️ VOLATILITY BURST DETECTED!")
                        
                elif event_type == "ONCHAIN_WHALE_ALERT":
                    features["macro_regime_shift_prob"] = 0.85 # Heuristic bump
                    
                # Acknowledge the message
                await self.redis.xack(STREAM_IN, GROUP_NAME, msg_id)
                
            except Exception as e:
                logging.error(f"Error processing message {msg_id}: {e}")
                
        # Emit computed features
        if features:
            features["_timestamp"] = now
            await self.redis.xadd(STREAM_OUT, {"features": json.dumps(features)})

    async def start(self):
        self.running = True
        if not await self.init_redis():
            return
            
        logging.info("🚀 Feature Warehouse is listening for raw events...")
        while self.running:
            try:
                # Read from Redis Stream (blocking for 2s)
                messages = await self.redis.xreadgroup(
                    GROUP_NAME, CONSUMER_NAME, {STREAM_IN: ">"}, count=100, block=2000
                )
                
                if messages:
                    for stream, msgs in messages:
                        await self.process_batch(msgs)
                else:
                    # Periodic cleanup even if no new messages
                    self.clean_liquidation_window(time.time())
            except Exception as e:
                logging.error(f"Stream Read Error: {e}")
                await asyncio.sleep(2)

if __name__ == "__main__":
    warehouse = FeatureWarehouse()
    try:
        asyncio.run(warehouse.start())
    except KeyboardInterrupt:
        logging.info("Warehouse Shutdown.")
