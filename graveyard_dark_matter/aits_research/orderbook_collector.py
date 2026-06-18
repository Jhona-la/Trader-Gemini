"""
AITS Phase 1: Market Microstructure Foundation
Multi-Exchange Order Book Collector

This daemon connects to Binance, Bybit, and OKX via ccxt.pro (WebSockets).
It captures real-time Level 2 Order Book updates, computes institutional features 
like the Spread and Order Flow Imbalance (OFI), and flushes the data in batches 
to TimescaleDB using asyncpg for ultra-low latency.
"""

import asyncio
import time
import logging
import traceback
from datetime import datetime, timezone

try:
    import ccxt.pro as ccxtpro
except ImportError:
    ccxtpro = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# TimescaleDB Connection string
DB_DSN = "postgresql://aits_user:aits_pass@localhost:5432/aits_market_data"
SYMBOL = "BTC/USDT"
BATCH_SIZE = 100
FLUSH_INTERVAL = 1.0  # seconds

class OrderBookCollector:
    def __init__(self):
        self.exchanges = {
            'binance': ccxtpro.binance({'enableRateLimit': True}),
            'bybit': ccxtpro.bybit({'enableRateLimit': True}),
            'okx': ccxtpro.okx({'enableRateLimit': True})
        }
        self.db_pool = None
        self.queue = asyncio.Queue()
        self.running = False
        
        # State to compute OFI
        self.prev_state = {
            'binance': None,
            'bybit': None,
            'okx': None
        }

    async def init_db(self):
        if not asyncpg:
            logging.error("asyncpg is not installed. Run: pip install asyncpg")
            return False
            
        try:
            self.db_pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=2, max_size=10)
            
            async with self.db_pool.acquire() as conn:
                # Create advanced hypertable for L2 Snapshots
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS l2_orderbook_snapshots (
                        time TIMESTAMPTZ NOT NULL,
                        exchange TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        best_bid DOUBLE PRECISION,
                        best_ask DOUBLE PRECISION,
                        spread DOUBLE PRECISION,
                        bid_volume_top5 DOUBLE PRECISION,
                        ask_volume_top5 DOUBLE PRECISION,
                        ofi DOUBLE PRECISION
                    );
                """)
                # TimescaleDB hypertable conversion (ignores if already converted)
                try:
                    await conn.execute("SELECT create_hypertable('l2_orderbook_snapshots', 'time', if_not_exists => TRUE);")
                except asyncpg.exceptions.UniqueViolationError:
                    pass
                except Exception as e:
                    if "already a hypertable" not in str(e):
                        logging.warning(f"Hypertable creation msg: {e}")
            logging.info("✅ Database initialized successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Database: {e}")
            return False

    def calculate_ofi(self, exchange_id, best_bid, best_ask, bid_vol, ask_vol):
        """Calculates a simplified top-of-book Order Flow Imbalance."""
        prev = self.prev_state[exchange_id]
        if not prev:
            self.prev_state[exchange_id] = (best_bid, best_ask, bid_vol, ask_vol)
            return 0.0
            
        prev_bid, prev_ask, prev_bid_vol, prev_ask_vol = prev
        
        # Bid Imbalance
        if best_bid > prev_bid:
            bid_imb = bid_vol
        elif best_bid == prev_bid:
            bid_imb = bid_vol - prev_bid_vol
        else:
            bid_imb = -prev_bid_vol
            
        # Ask Imbalance
        if best_ask < prev_ask:
            ask_imb = ask_vol
        elif best_ask == prev_ask:
            ask_imb = ask_vol - prev_ask_vol
        else:
            ask_imb = -prev_ask_vol
            
        ofi = bid_imb - ask_imb
        self.prev_state[exchange_id] = (best_bid, best_ask, bid_vol, ask_vol)
        return ofi

    async def fetch_order_book(self, exchange_id: str, exchange: ccxtpro.Exchange):
        logging.info(f"Starting {exchange_id} L2 WebSocket stream...")
        while self.running:
            try:
                # CCXT automatically handles WebSocket connection and reconnection
                orderbook = await exchange.watch_order_book(SYMBOL)
                
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if not bids or not asks:
                    continue
                    
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                spread = best_ask - best_bid
                
                # Aggregate Top 5 levels for volume pressure
                bid_vol_top5 = sum(b[1] for b in bids[:5])
                ask_vol_top5 = sum(a[1] for a in asks[:5])
                
                ofi = self.calculate_ofi(exchange_id, best_bid, best_ask, bids[0][1], asks[0][1])
                
                record = (
                    datetime.now(timezone.utc),
                    exchange_id,
                    SYMBOL,
                    best_bid,
                    best_ask,
                    spread,
                    bid_vol_top5,
                    ask_vol_top5,
                    ofi
                )
                
                await self.queue.put(record)
                
            except Exception as e:
                logging.error(f"{exchange_id} WebSocket Error: {e}")
                await asyncio.sleep(5)  # Backoff before reconnecting

    async def db_flusher(self):
        """Asynchronously writes batches to TimescaleDB to prevent I/O blocking."""
        logging.info("DB Flusher started.")
        batch = []
        last_flush = time.time()
        
        while self.running or not self.queue.empty():
            try:
                # Try to get item from queue with timeout
                record = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                batch.append(record)
            except asyncio.TimeoutError:
                pass
                
            now = time.time()
            if len(batch) >= BATCH_SIZE or (batch and now - last_flush > FLUSH_INTERVAL):
                try:
                    async with self.db_pool.acquire() as conn:
                        await conn.copy_records_to_table(
                            'l2_orderbook_snapshots',
                            records=batch,
                            columns=['time', 'exchange', 'symbol', 'best_bid', 'best_ask', 'spread', 'bid_volume_top5', 'ask_volume_top5', 'ofi']
                        )
                    logging.debug(f"Flushed {len(batch)} records to DB.")
                    batch.clear()
                    last_flush = now
                except Exception as e:
                    logging.error(f"DB Flush Error: {e}")
                    await asyncio.sleep(1)

    async def start(self):
        if not ccxtpro:
            logging.error("ccxt.pro is not installed. Run: pip install ccxt")
            return
            
        if not await self.init_db():
            logging.warning("Continuing without DB connection (Dry Run Mode).")
            
        self.running = True
        
        tasks = [
            asyncio.create_task(self.fetch_order_book(ex_id, ex)) 
            for ex_id, ex in self.exchanges.items()
        ]
        
        if self.db_pool:
            tasks.append(asyncio.create_task(self.db_flusher()))
            
        logging.info(f"🚀 AITS Order Book Collector running for {SYMBOL} across {len(self.exchanges)} exchanges.")
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logging.info("Collector stopping...")
        finally:
            self.running = False
            for ex in self.exchanges.values():
                await ex.close()
            if self.db_pool:
                await self.db_pool.close()

if __name__ == "__main__":
    collector = OrderBookCollector()
    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
