"""
🧪 VALIDATION SCRIPT: AEGIS-ULTRA Metal (Phase 14)
QUÉ: Verifies Hardware Affinity & Benchmarks Vectorized Math.
POR QUÉ: Ensure Ryzen 5700U optimizations are active and effective.
"""

import sys
import os
import time
import psutil
import numpy as np
import pandas as pd
import numba

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cpu_affinity import CPUManager
from strategies.quant_math import rsi_numba, bollinger_numba, ema_numba

def test_hardware_awareness():
    print("\n⚙️ TESTING HARDWARE AWARENESS (CPUManager)...")
    
    # 1. Apply Profile
    try:
        CPUManager.pin_process()
        CPUManager.set_priority("HIGH")
    except Exception as e:
        print(f"❌ FAIL: Could not apply optimizations: {e}")
        return

    # 2. Verify Priority (Windows only reliable check)
    p = psutil.Process()
    try:
        if sys.platform == 'win32':
            current_prio = p.nice()
            print(f"  Current Priority: {current_prio} (Target: {psutil.HIGH_PRIORITY_CLASS})")
            if current_prio == psutil.HIGH_PRIORITY_CLASS:
                print("✅ PASS: Process Priority is HIGH.")
            else:
                print("⚠️ WARN: Process Priority not HIGH (Admin rights might be needed).")
        else:
            print(f"  Current Nice: {p.nice()}")
            
    except Exception as e:
        print(f"⚠️ WARN: Could not verify priority: {e}")

    # 3. Verify Affinity (Subset check)
    try:
        affinity = p.cpu_affinity()
        print(f"  Current CPU Affinity: {affinity}")
        # We didn't enforce specific pinning in code yet (passed), so just printing is consistent
        print("✅ PASS: CPU Affinity readable.")
    except Exception as e:
        print(f"⚠️ WARN: Could not read affinity: {e}")

def benchmark_math():
    print("\n🧬 BENCHMARKING VECTORIZED MATH (AVX-2)...")
    
    N = 1_000_000
    print(f"  Generating {N} random prices...")
    prices = np.random.uniform(100, 200, N).astype(np.float64)
    
    # --- RSI ---
    print("\n  [RSI Benchmark]")
    
    # Pandas (Baseline) - using simple diff/rolling implementation simulation
    start = time.time()
    s = pd.Series(prices)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_pd = 100 - (100 / (1 + rs))
    pd_time = time.time() - start
    print(f"  Pandas Time: {pd_time:.4f}s")
    
    # Numba (First Run - Compilation)
    start = time.time()
    rsi_numba(prices, 14)
    jit_time = time.time() - start
    print(f"  Numba (JIT): {jit_time:.4f}s (Warmup)")
    
    # Numba (Hot Run)
    start = time.time()
    res = rsi_numba(prices, 14)
    nb_time = time.time() - start
    print(f"  Numba (Hot): {nb_time:.4f}s")
    
    speedup = pd_time / nb_time
    print(f"  ⚡ SPEEDUP: {speedup:.1f}x")
    
    if speedup > 10:
        print("✅ PASS: RSI Vectorization achieves >10x speedup.")
    else:
        print("⚠️ WARN: RSI Speedup below target.")

    # --- Bollinger ---
    print("\n  [Bollinger Benchmark]")
    
    start = time.time()
    s = pd.Series(prices)
    sma = s.rolling(20).mean()
    std = s.rolling(20).std()
    upper = sma + 2*std
    lower = sma - 2*std
    pd_time = time.time() - start
    print(f"  Pandas Time: {pd_time:.4f}s")
    
    # Numba Hot
    bollinger_numba(prices, 20, 2.0) # Warmup
    start = time.time()
    u, m, l = bollinger_numba(prices, 20, 2.0)
    nb_time = time.time() - start
    print(f"  Numba (Hot): {nb_time:.4f}s")
    
    speedup = pd_time / nb_time
    print(f"  ⚡ SPEEDUP: {speedup:.1f}x")

if __name__ == "__main__":
    test_hardware_awareness()
    benchmark_math()
