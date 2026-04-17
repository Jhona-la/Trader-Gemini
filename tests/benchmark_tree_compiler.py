"""
🔬 BENCHMARK: Scikit-learn Tree Models vs Numba JIT Nanosecond Compiler
Validates correctness and measures performance improvement.
"""
import numpy as np
import time
import sys
import warnings

# Suppress sklearn warnings if any
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification

sys.path.insert(0, '.')

from ml.tree_compiler import compile_rf_to_numpy_batch, compile_gb_to_numpy_batch
from core.fused_strategy_kernel import predict_rf_jit, predict_gb_jit

def main():
    print("=" * 70)
    print("🔬 NUMBA TREE COMPILER BENCHMARK (RF & GB)")
    print("=" * 70)
    
    # 1. Generate realistic dataset
    X, y = make_classification(n_samples=5000, n_features=25, n_informative=15, 
                               n_redundant=5, random_state=42)
    
    X_test = X[-1000:].astype(np.float32)
    X_single = X_test[0] # Single tick inference
    
    # =================================================================
    # TEST 1: RANDOM FOREST
    # =================================================================
    print("\n🌲 Training RandomForest (n_estimators=100)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1, random_state=42)
    rf.fit(X, y)
    
    print("🔨 Compiling RF to Numba Matrices...")
    rf_arrays = compile_rf_to_numpy_batch(rf)
    
    cl = rf_arrays['children_left']
    cr = rf_arrays['children_right']
    ft = rf_arrays['feature']
    th = rf_arrays['threshold']
    val = rf_arrays['value']
    offs = rf_arrays['tree_offsets']
    
    print("⏳ Warming up JIT...")
    _ = predict_rf_jit(X_single, cl, cr, ft, th, val, offs)
    
    # Benchmark Sklearn
    sk_iterations = 50
    sk_rf_times = []
    
    for _ in range(sk_iterations):
        t0 = time.perf_counter()
        _ = rf.predict_proba(X_single.reshape(1, -1))[0, 1]
        t1 = time.perf_counter()
        sk_rf_times.append(t1 - t0)
    
    avg_sk_rf = np.mean(sk_rf_times) * 1_000_000 # in microseconds
    
    # Benchmark JIT
    jit_iterations = 5000
    jit_rf_times = []
    for _ in range(jit_iterations):
        t0 = time.perf_counter()
        _ = predict_rf_jit(X_single, cl, cr, ft, th, val, offs)
        t1 = time.perf_counter()
        jit_rf_times.append(t1 - t0)
        
    avg_jit_rf = np.mean(jit_rf_times) * 1_000_000 # in microseconds
    
    # Validate
    idx = 42
    test_vec = X_test[idx]
    sk_prob = rf.predict_proba(test_vec.reshape(1, -1))[0, 1]
    jit_prob = predict_rf_jit(test_vec, cl, cr, ft, th, val, offs)
    is_rf_correct = np.allclose(sk_prob, jit_prob, atol=1e-5)
    
    # =================================================================
    # TEST 2: GRADIENT BOOSTING
    # =================================================================
    print("\n🚀 Training GradientBoosting (n_estimators=100)...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    gb.fit(X, y)
    
    print("🔨 Compiling GB to Numba Matrices...")
    gb_arrays = compile_gb_to_numpy_batch(gb)
        
    cl_g = gb_arrays['children_left']
    cr_g = gb_arrays['children_right']
    ft_g = gb_arrays['feature']
    th_g = gb_arrays['threshold']
    val_g = gb_arrays['value']
    offs_g = gb_arrays['tree_offsets']
    i_score = gb_arrays['init_score']
    lr = gb_arrays['learning_rate']
    
    print("⏳ Warming up GB JIT...")
    _ = predict_gb_jit(X_single, cl_g, cr_g, ft_g, th_g, val_g, offs_g, i_score, lr)
    
    sk_gb_times = []
    for _ in range(sk_iterations):
        t0 = time.perf_counter()
        _ = gb.predict_proba(X_single.reshape(1, -1))[0, 1]
        t1 = time.perf_counter()
        sk_gb_times.append(t1 - t0)
        
    avg_sk_gb = np.mean(sk_gb_times) * 1_000_000 # micro
    
    jit_gb_times = []
    for _ in range(jit_iterations):
        t0 = time.perf_counter()
        _ = predict_gb_jit(X_single, cl_g, cr_g, ft_g, th_g, val_g, offs_g, i_score, lr)
        t1 = time.perf_counter()
        jit_gb_times.append(t1 - t0)
        
    avg_jit_gb = np.mean(jit_gb_times) * 1_000_000 # micro
    
    sk_prob_gb = gb.predict_proba(test_vec.reshape(1, -1))[0, 1]
    jit_prob_gb = predict_gb_jit(test_vec, cl_g, cr_g, ft_g, th_g, val_g, offs_g, i_score, lr)
    is_gb_correct = np.allclose(sk_prob_gb, jit_prob_gb, atol=1e-5)
    
    print(f"\n{'='*70}")
    print(f"📈 RESULTS (1 Tick Inference - NANO LATENCY TEST):")
    print(f"{'='*70}")
    
    print(f"🌲 RANDOM FOREST (100 Trees):")
    print(f"  🐢 OLD (Sklearn Python Obj): {avg_sk_rf:8.2f} μs")
    print(f"  🚀 NEW (Numba C-Level Matrix): {avg_jit_rf:8.2f} μs")
    print(f"  ⚡ SPEEDUP:                    {avg_sk_rf/avg_jit_rf:8.1f}x FASTER")
    print(f"  🔍 CORRECTNESS:                {'✅ PASS' if is_rf_correct else '❌ FAIL'} (sk: {sk_prob:.4f}, jit: {jit_prob:.4f})")
    
    print(f"\n🚀 GRADIENT BOOSTING (100 Trees):")
    print(f"  🐢 OLD (Sklearn Python Obj): {avg_sk_gb:8.2f} μs")
    print(f"  🚀 NEW (Numba C-Level Matrix): {avg_jit_gb:8.2f} μs")
    print(f"  ⚡ SPEEDUP:                    {avg_sk_gb/avg_jit_gb:8.1f}x FASTER")
    print(f"  🔍 CORRECTNESS:                {'✅ PASS' if is_gb_correct else '❌ FAIL'} (sk: {sk_prob_gb:.4f}, jit: {jit_prob_gb:.4f})")
    
    print(f"{'='*70}")
    if is_rf_correct and is_gb_correct and (avg_sk_rf/avg_jit_rf) > 10:
        print("🏆 NANO LATENCY ARCHITECTURE VERIFIED & APPROVED")
    else:
        print("⚠️ ARCHITECTURE NEEDS REVIEW")

if __name__ == "__main__":
    main()
