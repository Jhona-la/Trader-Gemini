"""
AITS Phase 0: Proof of Concept - Redis Streams
This script demonstrates ultra-low latency Pub/Sub event streaming using Redis.
This will replace the in-memory Python `asyncio.Queue` for the event bus, 
allowing multiple distributed processes (ML Workers, Execution Engine, Risk Governor)
to communicate simultaneously.

Dependencies: pip install redis
"""

import asyncio
import json
import time
import logging

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STREAM_KEY = "aits:market:events"
GROUP_NAME = "aits_processors"
CONSUMER_NAME = "worker_1"

async def publisher(r: redis.Redis):
    """Simulates the Data Nervous System publishing events."""
    logging.info("Publisher started.")
    for i in range(5):
        event = {
            "type": "MARKET_TICK",
            "symbol": "BTC/USDT",
            "price": 50000 + i * 10,
            "volume": 2.5 + i,
            "timestamp": time.time()
        }
        # XADD adds a new entry to the stream
        msg_id = await r.xadd(STREAM_KEY, {"payload": json.dumps(event)})
        logging.info(f"Published event: {msg_id.decode()} -> {event['price']}")
        await asyncio.sleep(1)

async def consumer(r: redis.Redis):
    """Simulates an AITS Layer (e.g., ML Predictor) consuming events."""
    logging.info("Consumer started.")
    
    # Create consumer group (ignore if exists)
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logging.error(f"Group create error: {e}")

    processed = 0
    while processed < 5:
        # XREADGROUP reads from the stream as part of a consumer group
        messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, {STREAM_KEY: ">"}, count=1, block=2000)
        
        if messages:
            for stream, msgs in messages:
                for msg_id, msg_data in msgs:
                    payload = json.loads(msg_data[b'payload'].decode('utf-8'))
                    logging.info(f"Consumed event: {msg_id.decode()} -> Processing Tick @ {payload['price']}")
                    
                    # Acknowledge the message so it's removed from pending entries
                    await r.xack(STREAM_KEY, GROUP_NAME, msg_id)
                    processed += 1
        else:
            logging.info("Waiting for new events...")

async def main():
    if not redis:
        logging.error("redis-py not installed. Run: pip install redis")
        return

    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        await r.ping()
        logging.info("✅ Redis connection successful.")
        
        # Run publisher and consumer concurrently
        await asyncio.gather(
            publisher(r),
            consumer(r)
        )
        
        await r.close()
    except Exception as e:
        logging.error(f"Redis connection failed: {e}. Is the Docker container running?")

if __name__ == "__main__":
    logging.info("Starting AITS Redis Streams PoC...")
    asyncio.run(main())
