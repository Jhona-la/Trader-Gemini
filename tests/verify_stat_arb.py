"""
🧪 VALIDATION SCRIPT: AEGIS-ULTRA StatArb (Phase 15)
QUÉ: Verifies Lite-EG Cointegration & Dynamic Correlation.
POR QUÉ: Ensure math is correct without 'statsmodels'.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.stat_arb import StatArbEngine

def test_correlation_matrix():
    print("\n📊 TESTING CORRELATION MATRIX...")
    
    # 1. Generate Correlated Data (Fleet Move)
    # 20 assets, 100 samples
    # All moving up together (0.9 corr)
    t = np.linspace(0, 10, 100)
    trend = t * 1.0 # Strong trend
    
    returns_list = []
    for i in range(20):
        noise = np.random.normal(0, 0.1, 100)
        returns_list.append(trend + noise) # All correlated to 'trend'
        
    returns_matrix = np.array(returns_list).T # [Samples, Assets]
    
    start = time.time()
    corr_mat = StatArbEngine.calculate_correlation_matrix(returns_matrix)
    avg_corr = StatArbEngine.get_systemic_risk(corr_mat)
    duration = time.time() - start
    
    print(f"  Avg Fleet Correlation: {avg_corr:.4f}")
    if avg_corr > 0.8:
        print(f"✅ PASS: Detected High Correlation (>0.8).")
    else:
        print(f"❌ FAIL: Correlation logic suspect (Expected >0.8, got {avg_corr})")
        
    print(f"  Compute Time (20 assets): {duration*1000:.2f}ms")
    if duration < 0.05:
        print("✅ PASS: Latency < 50ms")
    else:
        print("⚠️ WARN: Latency High")

def test_lite_eg():
    print("\n📉 TESTING LITE-EG COINTEGRATION...")
    
    # 1. Generate Cointegrated Pair (X, Y)
    # Y = 0.5 * X + Noise
    np.random.seed(42)
    n = 200
    x = np.cumsum(np.random.normal(0, 1, n)) # Random Walk
    noise = np.random.normal(0, 0.5, n)
    y = 0.5 * x + noise # Cointegrated
    
    start = time.time()
    res = StatArbEngine.lite_engle_granger(y, x)
    duration = time.time() - start
    
    print(f"  [Cointegrated Case]")
    print(f"  P-Value: {res.p_value:.4f}")
    print(f"  Beta: {res.beta:.4f} (Expected 0.5)")
    print(f"  Time: {duration*1000:.2f}ms")
    
    if res.p_value < 0.05 and 0.4 < res.beta < 0.6:
        print("✅ PASS: Correctly identified cointegration.")
    else:
        print("❌ FAIL: Failed to identify cointegration.")

    # 2. Generate Uncorrelated Pair
    y_random = np.cumsum(np.random.normal(0, 1, n))
    res2 = StatArbEngine.lite_engle_granger(y_random, x)
    
    print(f"  [Uncorrelated Case]")
    print(f"  P-Value: {res2.p_value:.4f}")
    
    if res2.p_value > 0.10:
        print("✅ PASS: Correctly rejected random pair.")
    else:
        print("❌ FAIL: False Positive on random pair.")

if __name__ == "__main__":
    test_correlation_matrix()
    test_lite_eg()
