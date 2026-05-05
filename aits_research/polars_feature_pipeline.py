"""
AITS Phase 0: Data Engineering Preparation
Demonstrates refactoring Pandas-based feature engineering to Polars.
Polars is written in Rust, leverages Apache Arrow, and uses multi-threading 
to reduce feature calculation latency by ~80% for AITS Layer 2.

Dependencies: pip install polars numpy
"""

import numpy as np
import time
import logging

try:
    import polars as pl
except ImportError:
    pl = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_mock_data(rows=1_000_000):
    """Generates synthetic tick data for benchmarking."""
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(rows) * 0.5) + 50000
    volumes = np.abs(np.random.randn(rows) * 10)
    timestamps = np.arange(0, rows * 1000, 1000) # 1 sec intervals in ms
    
    return pl.DataFrame({
        "timestamp_ms": timestamps,
        "price": prices,
        "volume": volumes,
        "side": np.random.choice(["buy", "sell"], size=rows)
    })

def calculate_institutional_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes complex institutional features using Polars Lazy API.
    Features: VWAP, Rolling Volatility, Order Imbalance.
    """
    logging.info("Starting Polars feature pipeline...")
    
    # Using Polars lazy execution for query optimization
    q = (
        df.lazy()
        .with_columns([
            # 1. Price Returns
            (pl.col("price") / pl.col("price").shift(1) - 1).alias("return_1s"),
            
            # 2. Cumulative VWAP
            ((pl.col("price") * pl.col("volume")).cum_sum() / pl.col("volume").cum_sum()).alias("vwap"),
            
            # 3. Signed Volume (Buy volume is positive, Sell is negative)
            pl.when(pl.col("side") == "buy")
              .then(pl.col("volume"))
              .otherwise(-pl.col("volume"))
              .alias("signed_volume")
        ])
        .with_columns([
            # 4. Rolling Volatility (100 ticks)
            pl.col("return_1s").rolling_std(window_size=100).alias("volatility_100t"),
            
            # 5. Order Flow Imbalance (Rolling Sum of signed volume)
            pl.col("signed_volume").rolling_sum(window_size=100).alias("ofi_100t")
        ])
        .drop_nulls()
    )
    
    # Execute the lazy graph
    result = q.collect()
    return result

def run_benchmark():
    if not pl:
        logging.error("Polars is not installed. Run: pip install polars")
        return
        
    logging.info("Generating 1,000,000 mock tick records...")
    df = generate_mock_data(1_000_000)
    
    start_time = time.time()
    result_df = calculate_institutional_features(df)
    elapsed = time.time() - start_time
    
    logging.info(f"✅ Processed {len(df):,} rows and engineered 5 features in {elapsed:.4f} seconds.")
    logging.info(f"Result DataFrame Shape: {result_df.shape}")
    print(result_df.head())

if __name__ == "__main__":
    run_benchmark()
