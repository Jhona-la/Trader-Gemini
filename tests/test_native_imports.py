"""Verify which module file is actually loaded for each native module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

modules = [
    "core.nano_core",
    "core.nano_portfolio", 
    "core.dark_alpha_queue",
    "core.mev_rbf_engine",
    "data.fast_lob",
    "strategies.math_core",
    "risk.c_risk",
    "execution.c_executor",
]

print("=" * 70)
print("🔬 MODULE RESOLUTION: .pyd vs .py")
print("=" * 70)

for mod_name in modules:
    try:
        m = __import__(mod_name, fromlist=[""])
        f = getattr(m, "__file__", "unknown")
        is_native = f.endswith(".pyd") or f.endswith(".so")
        tag = "🟢 NATIVE" if is_native else "🔴 PYTHON"
        print(f"  {tag} {mod_name}")
        print(f"         → {f}")
    except Exception as e:
        print(f"  ❌ DEAD  {mod_name}: {e}")

# Also check if math_core functions are used by ml_strategy
print("\n" + "=" * 70)
print("🔬 CHECKING math_core FUNCTIONS")
print("=" * 70)
try:
    from strategies.math_core import fast_ema, fast_rsi, fast_sma, fast_std
    import numpy as np
    import time
    
    data = np.random.uniform(60000, 70000, 1000).astype(np.float64)
    
    t0 = time.perf_counter_ns()
    result_ema = fast_ema(data, 14)
    t1 = time.perf_counter_ns()
    print(f"  fast_ema(1000, 14): {t1-t0:,} ns → result[-1]={result_ema[-1]:.2f}")
    
    t0 = time.perf_counter_ns()
    result_rsi = fast_rsi(data, 14)
    t1 = time.perf_counter_ns()
    print(f"  fast_rsi(1000, 14): {t1-t0:,} ns → result[-1]={result_rsi[-1]:.2f}")
    
except Exception as e:
    print(f"  ❌ math_core test failed: {e}")

# Check c_risk functions
print("\n" + "=" * 70)
print("🔬 CHECKING c_risk FUNCTIONS")
print("=" * 70)
try:
    from risk.c_risk import compute_kelly_fraction, compute_dynamic_sizing, check_drawdown_limit
    import time
    
    t0 = time.perf_counter_ns()
    k = compute_kelly_fraction(0.6, 2.0, 0.02)
    t1 = time.perf_counter_ns()
    print(f"  compute_kelly_fraction(0.6, 2.0, 0.02): {t1-t0:,} ns → {k:.4f}")
    
    t0 = time.perf_counter_ns()
    ok = check_drawdown_limit(0.95, 0.90)
    t1 = time.perf_counter_ns()
    print(f"  check_drawdown_limit(0.95, 0.90): {t1-t0:,} ns → {ok}")
    
except Exception as e:
    print(f"  ❌ c_risk test failed: {e}")
