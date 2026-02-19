
import numpy as np
import time
from utils.logger import logger
from utils.math_kernel import vector_zscore, vector_rsi, vector_volatility
from numba import jit

@jit(nopython=True, cache=True)
def prefetch_array(arr):
    """
    [PHASE 18] L3 Cache Warmer.
    Iterates array to force pre-fetching into CPU Cache.
    """
    s = 0.0
    for i in range(arr.shape[0]):
        s += arr[i]
    return s

def warmup_numba_kernels():
    """
    [PHASE 11] JIT-Warmup Protocol.
    Runs dummy data through Numba functions to trigger compilation 
    BEFORE the first real tick arrives.
    """
    logger.info("🔥 [JIT-Warmup] Compiling Numba Kernels...")
    start_t = time.perf_counter()
    
    # 1. Dummy Data
    dummy_prices = np.random.random(1000).astype(np.float64) * 10000
    dummy_returns = np.diff(dummy_prices) / dummy_prices[:-1]
    
    # 2. Warmup Z-Score
    _ = vector_zscore(dummy_prices, window=20)
    
    # 3. Warmup RSI
    _ = vector_rsi(dummy_prices, period=14)
    
    # 4. Warmup Volatility
    _ = vector_volatility(dummy_returns, window=20)
    
    duration = (time.perf_counter() - start_t) * 1000
    logger.info(f"✅ [JIT-Warmup] Compilation Complete in {duration:.2f}ms")

if __name__ == "__main__":
    warmup_numba_kernels()
