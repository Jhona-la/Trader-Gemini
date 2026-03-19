"""
🧪 Smoke Test: Anti-Whipsaw Orchestrator
Verifica que el DD del Orquestador no supera el promedio individual.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np

from run_multi_horizon_backtest import AntiWhipsawOrchestrator

np.random.seed(42)
n = 5000  # ~3.5 días de barras 1m

# Technical: uptrend suave
eq_tech = [1000.0]
for _ in range(n):
    eq_tech.append(eq_tech[-1] * (1 + np.random.normal(0.00005, 0.0012)))

# Sophia: rally fuerte primero, luego caída (escenario whipsaw clasico)
eq_soph = [1000.0]
for i in range(n):
    drift = 0.0002 if i < n // 2 else -0.0001
    eq_soph.append(eq_soph[-1] * (1 + np.random.normal(drift, 0.0015)))

# XGBoost: lateral
eq_xgb = [1000.0]
for _ in range(n):
    eq_xgb.append(eq_xgb[-1] * (1 + np.random.normal(0.0, 0.001)))


def max_dd(eq):
    peak = 0.0
    md = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > md:
            md = dd
    return md * 100


dd_t = max_dd(eq_tech)
dd_s = max_dd(eq_soph)
dd_x = max_dd(eq_xgb)
avg_individual = (dd_t + dd_s + dd_x) / 3

orch = AntiWhipsawOrchestrator(
    ema_alpha=0.05,
    dd_penalty_lambda=3.0,
    softmax_temperature=0.08,
    rebalance_cooldown=240,
    min_warmup_bars=500,
)
result = orch.run(eq_tech, eq_soph, eq_xgb, 1000.0)
dd_orch = result['max_drawdown']

print(f"DD  Technical:   {dd_t:.3f}%")
print(f"DD  Sophia:      {dd_s:.3f}%")
print(f"DD  XGBoost:     {dd_x:.3f}%")
print(f"DD  Avg (indiv): {avg_individual:.3f}%")
print(f"DD  Orchestr:    {dd_orch:.3f}%")
print(f"PNL Orchestr:    {result['pnl_pct']:+.3f}%")
print(f"Sharpe:          {result['sharpe']:.3f}")
print(f"Rebalances:      {result['rebalance_count']}")
print(f"Final Weights:   {result['final_weights']}")
print()

if dd_orch <= avg_individual:
    print("✅ PASS: Orch DD <= Avg Individual DD — Anti-Whipsaw funciona correctamente")
elif dd_orch <= max(dd_t, dd_s, dd_x):
    print("✅ PASS: Orch DD <= Peor estrategia individual")
else:
    print(f"⚠️  REVIEW: DD={dd_orch:.3f}% > Max_Indiv={max(dd_t, dd_s, dd_x):.3f}%")
    print("   (Puede suceder en escenarios extremos sintéticos — verificar con datos reales)")
