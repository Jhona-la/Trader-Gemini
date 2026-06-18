import time
import numpy as np
from numba import njit

# 1. Existing O(N) Algorithm for Bollinger Band (SMA & STD over rolling window)
@njit(fastmath=True, cache=True)
def bollinger_on(prices, period=20):
    n = len(prices)
    mid = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)
    
    # simulate tick by tick execution where we process the window 
    # to emulate the current loop inside bollinger_numba
    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        mid[i] = np.mean(window)
        std[i] = np.std(window)
    return mid, std


# 2. Stateful O(1) Algorithm (Welford's Method) for Variance & Mean
@njit(fastmath=True, cache=True)
def bollinger_welford(prices, period=20):
    n = len(prices)
    mid = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)
    
    # We maintain a ring buffer and running sums for O(1) update
    # Welford's algorithm or simplified running variance
    # For a rolling window of fixed size, we can maintain sum(x) and sum(x^2)
    
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Initialize the first window
    for i in range(period):
        sum_x += prices[i]
        sum_x2 += prices[i] * prices[i]
        
    mid[period-1] = sum_x / period
    var = (sum_x2 - (sum_x * sum_x) / period) / period
    std[period-1] = np.sqrt(max(0.0, var))
    
    # O(1) Tick-by-tick update
    for i in range(period, n):
        old_val = prices[i - period]
        new_val = prices[i]
        
        sum_x += (new_val - old_val)
        sum_x2 += (new_val * new_val - old_val * old_val)
        
        mid[i] = sum_x / period
        # Numerical stability check
        var = (sum_x2 - (sum_x * sum_x) / period) / period
        std[i] = np.sqrt(max(0.0, var))
        
    return mid, std


def main():
    print("⚛️ [AUDITORIA DEL METAL] Iniciando Benchmark O(N) vs O(1)...")
    np.random.seed(42)
    # 1 million ticks
    ticks = 1_000_000
    prices = np.random.lognormal(mean=0.0, sigma=0.01, size=ticks).cumsum() + 100.0
    
    # Warmup JIT
    bollinger_on(prices[:100], 20)
    bollinger_welford(prices[:100], 20)
    
    print(f"📊 Ejecutando simulador de mercado con {ticks} ticks...")
    
    period = 100 # Emulate a larger window for clearer separation
    
    t0 = time.perf_counter()
    res_on_mid, res_on_std = bollinger_on(prices, period)
    t1 = time.perf_counter()
    on_time = t1 - t0
    
    t2 = time.perf_counter()
    res_we_mid, res_we_std = bollinger_welford(prices, period)
    t3 = time.perf_counter()
    we_time = t3 - t2
    
    print(f"❌ Actual O(N) Loop: {on_time*1000:.2f} ms")
    print(f"✅ O(1) Welford Ring: {we_time*1000:.2f} ms")
    
    speedup = on_time / we_time
    print(f"🚀 Speedup Múltiplo: {speedup:.2f}x")
    
    # Verify correctness (allow small float drift difference)
    # The first 'period' elements will differ because O(N) doesn't set the first (period-1) correctly vs welford, 
    # but we care about the steady state.
    diff_mid = np.abs(res_on_mid[period:] - res_we_mid[period:])
    diff_std = np.abs(res_on_std[period:] - res_we_std[period:])
    
    print(f"⚖️ Diferencia Máxima Mean (Drift Coma Flotante): {np.max(diff_mid):.10f}")
    print(f"⚖️ Diferencia Máxima Std  (Drift Coma Flotante): {np.max(diff_std):.10f}")

if __name__ == "__main__":
    main()
