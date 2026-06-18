"""
AITS Phase 0: Proof of Concept - TimescaleDB
This script demonstrates connecting to PostgreSQL/TimescaleDB, creating a hypertable
for high-frequency market data (OHLCV + Order Book Imbalance), and inserting/querying data.

Dependencies: pip install psycopg2-binary
"""

import asyncio
import datetime
import random
import logging

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PARAMS = {
    "dbname": "aits_market_data",
    "user": "aits_user",
    "password": "aits_pass",
    "host": "localhost",
    "port": "5432"
}

def setup_timescale():
    if not psycopg2:
        logging.error("psycopg2-binary not installed. Run: pip install psycopg2-binary")
        return False

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # 1. Create standard PostgreSQL table
        logging.info("Creating market_data table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                order_flow_imbalance DOUBLE PRECISION
            );
        """)
        
        # 2. Convert to TimescaleDB Hypertable
        logging.info("Converting to TimescaleDB Hypertable...")
        cursor.execute("""
            SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
        """)
        
        # 3. Create index for fast symbol querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS ix_symbol_time ON market_data (symbol, time DESC);
        """)
        
        cursor.close()
        conn.close()
        logging.info("✅ TimescaleDB setup successful.")
        return True
    except Exception as e:
        logging.error(f"Failed to setup TimescaleDB: {e}")
        return False

def insert_mock_data():
    if not psycopg2: return
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        now = datetime.datetime.now(datetime.timezone.utc)
        records = []
        for i in range(100):
            timestamp = now - datetime.timedelta(minutes=100-i)
            records.append((
                timestamp,
                "BTC/USDT",
                50000 + random.uniform(-10, 10),
                50050 + random.uniform(-10, 10),
                49950 + random.uniform(-10, 10),
                50010 + random.uniform(-10, 10),
                random.uniform(1, 50),
                random.uniform(-1, 1)  # OFI
            ))
            
        insert_query = """
            INSERT INTO market_data (time, symbol, open, high, low, close, volume, order_flow_imbalance)
            VALUES %s
        """
        execute_values(cursor, insert_query, records)
        conn.commit()
        
        cursor.close()
        conn.close()
        logging.info(f"✅ Inserted {len(records)} mock rows into TimescaleDB.")
    except Exception as e:
        logging.error(f"Insert failed: {e}")

if __name__ == "__main__":
    logging.info("Starting AITS TimescaleDB PoC...")
    if setup_timescale():
        insert_mock_data()
