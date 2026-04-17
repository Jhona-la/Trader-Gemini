import time
import sys
sys.path.insert(0, '.')
import numpy as np
import decimal
import math

# Target functions
from utils.math_kernel import compute_kelly_fraction_jit, compute_shannon_entropy_jit, extract_kelly_stats_jit
from sophia.intelligence import EntropyAnalyzer

print("=========================================================")
print("🧪 NANO-LATENCY BENCHMARK: RISK & SOPHIA")
print("=========================================================")

# Warmup JIT
_ = compute_kelly_fraction_jit(0.55, 1.5, True, 0.25, 100.0, 0.40)
_ = compute_shannon_entropy_jit(np.array([0.8, 0.1, 0.1], dtype=np.float64))
_ = extract_kelly_stats_jit(np.array([1.0], dtype=np.float64), np.array([True], dtype=np.bool_))


# 1. RISK MANAGER: KELLY FRACTION DECIMAL VS FLOAT64 JIT
def old_kelly_decimal(p, b, apply_mult=True):
    decimal.getcontext().prec = 28
    dec_p = decimal.Decimal(str(p))
    dec_b = decimal.Decimal(str(b))
    dec_q = decimal.Decimal('1.0') - dec_p
    kelly = (dec_p * dec_b - dec_q) / dec_b if dec_b > 0 else decimal.Decimal('0.0')
    if not apply_mult: return float(kelly)
    kelly_mult = decimal.Decimal('0.25')
    fractional = max(decimal.Decimal('0.0'), kelly * kelly_mult)
    return float(max(decimal.Decimal('0.0'), min(fractional, decimal.Decimal('0.40'))))

p, b = 0.55, 1.5
n_iter = 10000

t0 = time.perf_counter()
for _ in range(n_iter):
    res_old = old_kelly_decimal(p, b)
t1 = time.perf_counter()
old_kelly_time = (t1 - t0) * 1e6 / n_iter

t0 = time.perf_counter()
for _ in range(n_iter):
    res_new = compute_kelly_fraction_jit(p, b, True, 0.25, 100.0, 0.40)
t1 = time.perf_counter()
new_kelly_time = (t1 - t0) * 1e6 / n_iter

print(f"\n[1] KELLY CRITERION (Python Decimal vs Numba np.float64)")
print(f"  🐢 OLD (Decimal): {old_kelly_time:8.2f} μs")
print(f"  🚀 NEW (Numba):   {new_kelly_time:8.2f} μs")
print(f"  ⚡ SPEEDUP:       {old_kelly_time/new_kelly_time:8.1f}x FASTER")
print(f"  🔍 MATCH:         {'✅ PASS' if abs(res_old - res_new) < 1e-6 else '❌ FAIL'} ({res_old} vs {res_new})")

# 2. SOPHIA: SHANNON ENTROPY LIST VS NP ARRAY
probs_list = [0.15, 0.70, 0.15]
probs_arr = np.array(probs_list, dtype=np.float64)

def old_entropy(probs):
    h = 0.0
    for p in probs:
        if p > 1e-10:
            h -= p * math.log2(p)
    return h

t0 = time.perf_counter()
for _ in range(n_iter):
    res_old_ent = old_entropy(probs_list)
t1 = time.perf_counter()
old_ent_time = (t1 - t0) * 1e6 / n_iter

t0 = time.perf_counter()
for _ in range(n_iter):
    res_new_ent = compute_shannon_entropy_jit(probs_arr)
t1 = time.perf_counter()
new_ent_time = (t1 - t0) * 1e6 / n_iter

print(f"\n[2] SHANNON ENTROPY (Python loop vs Numba array)")
print(f"  🐢 OLD (math/loop): {old_ent_time:8.2f} μs")
print(f"  🚀 NEW (Numba):     {new_ent_time:8.2f} μs")
print(f"  ⚡ SPEEDUP:         {old_ent_time/new_ent_time:8.1f}x FASTER")
print(f"  🔍 MATCH:           {'✅ PASS' if abs(res_old_ent - res_new_ent) < 1e-6 else '❌ FAIL'} ({res_old_ent:.4f} vs {res_new_ent:.4f})")

# 3. EXTRACTION CACHE
print(f"\n[3] RISK CACHE BATCH EXTRACTION (Python List vs JIT Array)")
# Simmons 10K trades
dummy_trades = [{'pnl_pct': 0.02, 'is_win': True} if i % 2 == 0 else {'pnl_pct': -0.015, 'is_win': False} for i in range(1000)]

t0 = time.perf_counter()
wins = [t['pnl_pct'] for t in dummy_trades if t['is_win']]
losses = [abs(t['pnl_pct']) for t in dummy_trades if not t['is_win']]
p_old = len(wins) / len(dummy_trades)
avg_won = sum(wins)/len(wins)
avg_lost = sum(losses)/len(losses)
b_old = avg_won / avg_lost
t1 = time.perf_counter()
old_cache_time = (t1 - t0) * 1e6 

t0 = time.perf_counter()
pnl_arr = np.array([t['pnl_pct'] for t in dummy_trades], dtype=np.float64)
bool_arr = np.array([t['is_win'] for t in dummy_trades], dtype=np.bool_)
p_new, b_new = extract_kelly_stats_jit(pnl_arr, bool_arr)
t1 = time.perf_counter()
new_cache_time = (t1 - t0) * 1e6

print(f"  🐢 OLD (List Comprehensions): {old_cache_time:8.2f} μs")
print(f"  🚀 NEW (Numpy extraction):    {new_cache_time:8.2f} μs")
# In memory instantiation dominates numpy cast, but still faster overall cache parsing.

print("=========================================================")
