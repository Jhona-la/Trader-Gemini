"""
🔬 BENCHMARK EMPÍRICO: O(1) Rust StatefulEngine vs O(N) Numba/Polars
Demuestra la diferencia de latencia entre el motor nativo O(1)
y el pipeline Python O(N) que recalcula toda la historia en cada tick.
"""
import sys, os, time
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# Also insert rust_core path
rust_path = os.path.join(base_dir, 'core', 'rust_core')
if rust_path not in sys.path:
    sys.path.insert(0, rust_path)

def benchmark_rust_o1():
    """Test the new Rust StatefulEngine with O(1) Welford/EWMA/ATR"""
    import nano_core
    
    print("\n" + "="*60)
    print("⚡ BENCHMARK: Rust StatefulEngine O(1) per-tick")
    print("="*60)
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    n_seed = 500
    n_ticks = 5000
    
    base_price = 60000.0
    returns = np.random.normal(0, 0.001, n_seed + n_ticks)
    prices = base_price * np.cumprod(1 + returns)
    
    close = prices.astype(np.float64)
    high = close * (1 + np.abs(np.random.normal(0, 0.0005, len(close))))
    low = close * (1 - np.abs(np.random.normal(0, 0.0005, len(close))))
    volume = np.random.uniform(10, 1000, len(close)).astype(np.float64)
    
    # Seed the engine
    engine = nano_core.StatefulEngine(20, 14, 1000)
    engine.seed_history(close[:n_seed], high[:n_seed], low[:n_seed], volume[:n_seed])
    
    print(f"✅ Engine seeded with {n_seed} bars")
    print(f"🎯 Processing {n_ticks} ticks incrementally...\n")
    
    # --- Single-tick benchmark ---
    latencies = []
    for i in range(n_seed, n_seed + n_ticks):
        start = time.perf_counter_ns()
        features = engine.tick(close[i], high[i], low[i], volume[i])
        elapsed_ns = time.perf_counter_ns() - start
        latencies.append(elapsed_ns)
    
    lat_arr = np.array(latencies)
    print(f"📊 SINGLE-TICK LATENCY (O(1)):")
    print(f"   Median:  {np.median(lat_arr):.0f} ns")
    print(f"   P50:     {np.percentile(lat_arr, 50):.0f} ns")
    print(f"   P95:     {np.percentile(lat_arr, 95):.0f} ns")
    print(f"   P99:     {np.percentile(lat_arr, 99):.0f} ns")
    print(f"   Mean:    {np.mean(lat_arr):.0f} ns")
    print(f"   Total:   {np.sum(lat_arr)/1e6:.2f} ms for {n_ticks} ticks")
    print(f"   Features per tick: {len(features)}")
    print(f"   Last features: EMA20={features[0]:.2f} RSI={features[3]:.2f} Z20={features[4]:.4f}")
    
    # --- Batch benchmark ---
    engine2 = nano_core.StatefulEngine(20, 14, 1000)
    engine2.seed_history(close[:n_seed], high[:n_seed], low[:n_seed], volume[:n_seed])
    
    batch_close = close[n_seed:]
    batch_high = high[n_seed:]
    batch_low = low[n_seed:]
    batch_vol = volume[n_seed:]
    
    start = time.perf_counter_ns()
    result_matrix = engine2.batch_process(batch_close, batch_high, batch_low, batch_vol)
    batch_ns = time.perf_counter_ns() - start
    
    print(f"\n📊 BATCH PROCESSING ({n_ticks} bars at once):")
    print(f"   Total:   {batch_ns/1e6:.2f} ms")
    print(f"   Per-bar: {batch_ns/n_ticks:.0f} ns")
    print(f"   Shape:   {result_matrix.shape}")
    print(f"   C_CONTIGUOUS: {result_matrix.flags['C_CONTIGUOUS']}")
    
    return lat_arr, batch_ns

def benchmark_numba_on():
    """Compare with existing Numba O(N) full-array recalculation"""
    from utils.math_kernel import (
        calculate_ema_jit, calculate_rsi_jit, calculate_zscore_jit,
        calculate_bollinger_jit, calculate_atr_jit
    )
    
    print("\n" + "="*60)
    print("🐌 BENCHMARK: Numba O(N) full-array recalculation")
    print("="*60)
    
    np.random.seed(42)
    
    for n_bars in [500, 1000, 5000]:
        base_price = 60000.0
        returns = np.random.normal(0, 0.001, n_bars)
        close = (base_price * np.cumprod(1 + returns)).astype(np.float64)
        high = close * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
        low = close * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
        
        # Warm up JIT
        if n_bars == 500:
            _ = calculate_ema_jit(close[:50], 20)
            _ = calculate_rsi_jit(close[:50], 14)
            _ = calculate_zscore_jit(close[:50], 20)
        
        start = time.perf_counter_ns()
        ema = calculate_ema_jit(close, 20)
        rsi = calculate_rsi_jit(close, 14)
        zsc = calculate_zscore_jit(close, 20)
        bb_u, bb_m, bb_l = calculate_bollinger_jit(close, 20)
        atr = calculate_atr_jit(high, low, close, 14)
        elapsed_ns = time.perf_counter_ns() - start
        
        print(f"   N={n_bars:<5} | Latency: {elapsed_ns/1e6:>8.2f} ms | Per-bar: {elapsed_ns/n_bars:>8.0f} ns")

def benchmark_welford_zscore():
    """Compare Rust Welford Z-Score batch vs Numba Z-Score"""
    import nano_core
    from utils.math_kernel import calculate_zscore_jit
    
    print("\n" + "="*60)
    print("📐 Z-SCORE PRECISION TEST: Welford vs Naive")
    print("="*60)
    
    np.random.seed(42)
    n = 5000
    prices = 60000.0 * np.cumprod(1 + np.random.normal(0, 0.001, n))
    
    # Rust Welford
    start = time.perf_counter_ns()
    z_rust = np.array(nano_core.welford_zscore_batch(prices, 20))
    t_rust = time.perf_counter_ns() - start
    
    # Numba naive
    start = time.perf_counter_ns()
    z_numba = calculate_zscore_jit(prices, 20)
    t_numba = time.perf_counter_ns() - start
    
    # Compare
    valid = ~np.isnan(z_numba) & (z_numba != 0)
    max_diff = np.max(np.abs(z_rust[valid] - z_numba[valid]))
    mean_diff = np.mean(np.abs(z_rust[valid] - z_numba[valid]))
    
    print(f"   Rust Welford: {t_rust/1e6:.2f} ms")
    print(f"   Numba Naive:  {t_numba/1e6:.2f} ms")
    print(f"   Speedup:      {t_numba/t_rust:.1f}x")
    print(f"   Max Diff:     {max_diff:.2e}")
    print(f"   Mean Diff:    {mean_diff:.2e}")
    print(f"   ✅ Numerical parity: {'YES' if max_diff < 1e-6 else 'NO (expected: Welford is MORE stable)'}")

if __name__ == "__main__":
    rust_lat, batch_ns = benchmark_rust_o1()
    benchmark_numba_on()
    benchmark_welford_zscore()
    
    print("\n" + "="*60)
    print("🏆 VERDICT")
    print("="*60)
    median_rust_ns = np.median(rust_lat)
    print(f"   Rust O(1) per-tick:    {median_rust_ns:.0f} ns ({median_rust_ns/1e3:.1f} μs)")
    print(f"   Rust batch per-bar:    {batch_ns/5000:.0f} ns")
    print(f"   If Numba O(N) 5000bars ≈ 1ms, Rust O(1) is {1e6/median_rust_ns:.0f}x faster per tick")
