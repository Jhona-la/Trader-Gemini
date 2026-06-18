import sys
import os
import time
import numpy as np
import pandas as pd

# Add the directory containing the nano_core.pyd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/rust_core')))
import nano_core

def compute_pandas_indicators(prices: np.ndarray, period_ema=20, period_rsi=14):
    df = pd.DataFrame({'close': prices})
    
    # EMA Calculation
    df['ema'] = df['close'].ewm(span=period_ema, adjust=False).mean()
    
    # RSI Calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Uses Wilder's smoothing method (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period_rsi, adjust=False, min_periods=period_rsi).mean()
    avg_loss = loss.ewm(alpha=1/period_rsi, adjust=False, min_periods=period_rsi).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    
    return df['ema'].values, df['rsi'].values

def test_precision_and_latency():
    # 1. Generate 1000 simulated close prices
    n_candles = 1000
    np.random.seed(42)
    # Start at 90000, random walk
    prices = 90000.0 + np.cumsum(np.random.normal(0, 10, n_candles))
    
    period_ema = 20
    period_rsi = 14
    
    # 2. Compute Pandas (O(N) vectorized)
    print("Computing Pandas O(N) baseline...")
    t0_pd = time.perf_counter()
    pandas_ema, pandas_rsi = compute_pandas_indicators(prices, period_ema, period_rsi)
    t1_pd = time.perf_counter()
    print(f"Pandas Time: {(t1_pd - t0_pd)*1000:.4f} ms")
    
    # 3. Compute Rust (O(1) streaming)
    print("\nComputing Rust O(1) Stateful Engine...")
    
    # We must seed the first period_rsi candles, then iterate the rest
    engine = nano_core.StatefulEngine(period_ema, period_rsi, 1000)
    
    seed_length = period_ema + 10
    seed_prices = prices[:seed_length]
    engine.seed_history(seed_prices)
    
    rust_ema = np.zeros(n_candles, dtype=np.float64)
    rust_rsi = np.zeros(n_candles, dtype=np.float64)
    
    # For seed period, Rust doesn't return intermediate outputs easily without calling update_and_inject,
    # but we just care about the values after seeding.
    
    # Dummy arrays for QuantumArena injection
    features = np.zeros((1000, 200), dtype=np.float32)
    version = np.zeros(8, dtype=np.int64)
    reader_head = np.array([-1], dtype=np.int64)
    feature_vec = np.zeros(200, dtype=np.float32)
    
    t0_rust = time.perf_counter()
    
    # We will test the raw update logic through the injected array (to see what is written)
    for i in range(seed_length, n_candles):
        idx = engine.update_and_inject(prices[i], version, reader_head, features, feature_vec)
        # Assuming the Rust engine injects ema and rsi at the last two positions:
        # row[row_len - 2] = self.ema_val as f32;
        # row[row_len - 1] = rsi as f32;
        rust_ema[i] = features[idx, 198]
        rust_rsi[i] = features[idx, 199]
        
    t1_rust = time.perf_counter()
    
    ops = n_candles - period_rsi
    ns_per_op = (t1_rust - t0_rust) * 1e9 / ops
    
    print(f"Rust Total Time: {(t1_rust - t0_rust)*1000:.4f} ms")
    print(f"Rust Latency per tick: {ns_per_op:.2f} ns")
    
    # 4. Compare Precision
    print("\nVerifying Mathematical Parity (Precision Drift)...")
    max_ema_diff = 0.0
    max_rsi_diff = 0.0
    
    # We compare from seed_length to avoid initial seeding differences 
    # (Pandas uses min_periods=14, Wilder's smoothing starts exactly at 14 but diffs exist until 15)
    for i in range(seed_length + 5, n_candles):
        # We must cast Pandas to float32 to match Rust's memory mapped precision
        pd_ema_f32 = np.float32(pandas_ema[i])
        pd_rsi_f32 = np.float32(pandas_rsi[i])
        
        ema_diff = abs(pd_ema_f32 - rust_ema[i])
        rsi_diff = abs(pd_rsi_f32 - rust_rsi[i])
        
        if ema_diff > max_ema_diff: max_ema_diff = ema_diff
        if rsi_diff > max_rsi_diff: max_rsi_diff = rsi_diff
        
    print(f"Max EMA Difference: {max_ema_diff:.8e}")
    print(f"Max RSI Difference: {max_rsi_diff:.8e}")
    
    if max_ema_diff < 1e-4 and max_rsi_diff < 1e-4:
        print("\n✅ PARIDAD CONFIRMADA: El motor O(1) de Rust coincide con el O(N) de Pandas dentro del error de coma flotante.")
    else:
        print("\n❌ DESVIACIÓN DETECTADA: Se superó el límite termodinámico de error.")
        print("Muestra (Pandas vs Rust):")
        for i in range(n_candles - 5, n_candles):
            print(f"Idx {i} - RSI Pd: {pandas_rsi[i]:.4f} | RSI Rust: {rust_rsi[i]:.4f}")

if __name__ == "__main__":
    test_precision_and_latency()
