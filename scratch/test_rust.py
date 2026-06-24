import time
try:
    import tg_core_fast
    print("SUCCESS: tg_core_fast imported!")
    
    # Test kelly
    t0 = time.perf_counter_ns()
    kelly_f, status = tg_core_fast.calculate_kelly(0.55, 1.5)
    t1 = time.perf_counter_ns()
    print(f"Kelly calc: {kelly_f:.4f}, Status: {status} in {t1-t0} ns")
    
    # Test micro sizing
    t0 = time.perf_counter_ns()
    risk_amt, notional, status = tg_core_fast.apply_micro_sizing(1.0, 13.0, 5.0, 5.5, 0.25)
    t1 = time.perf_counter_ns()
    print(f"Micro Sizing: risk=${risk_amt:.2f}, notional=${notional:.2f}, Status: {status} in {t1-t0} ns")
    
except ImportError as e:
    print(f"FAILED to import tg_core_fast: {e}")
except Exception as e:
    print(f"ERROR: {e}")
