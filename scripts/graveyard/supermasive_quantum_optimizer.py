#!/usr/bin/env python3
"""
🌌 SUPERMASIVE QUANTUM OPTIMIZER v2.0 — PARALLEL TURBO
═══════════════════════════════════════════════════════════════════════════
QUÉ:    Motor de optimización masiva que barre MILES de combinaciones
        usando Numba paralelo sobre TODOS los hilos del CPU simultáneamente.
POR QUÉ: v1.0 tardó 12s. El usuario exige nanosegundos.
CÓMO:   - Pre-computa indicadores UNA VEZ por moneda (SMA, RSI)
        - Aplana la malla de params a un solo array de configuraciones
        - Evalúa TODAS las configs en un solo prange() paralelo
        - Resultado: 100,000+ evaluaciones en < 2 segundos
DÓNDE:  scripts/supermasive_quantum_optimizer.py
QUIÉN:  Quant Developer + SRE/DevOps
"""

import os, sys, time, json
import numpy as np
import polars as pl
from numba import njit, prange

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR CUÁNTICO v2: EVALUACIÓN PARALELA DE MALLA COMPLETA
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _compute_sma(closes, period):
    n = len(closes)
    sma = np.full(n, np.nan)
    if n < period:
        return sma
    s = 0.0
    for i in range(period):
        s += closes[i]
    sma[period - 1] = s / period
    for i in range(period, n):
        s += closes[i] - closes[i - period]
        sma[i] = s / period
    return sma


@njit(cache=True)
def _compute_rsi(closes, period):
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    avg_g = 0.0
    avg_l = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i-1]
        if d > 0:
            avg_g += d
        else:
            avg_l -= d
    avg_g /= period
    avg_l /= period
    if avg_l > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        d = closes[i] - closes[i-1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        if avg_l > 0:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
        else:
            rsi[i] = 100.0
    return rsi


@njit(cache=True)
def _single_eval(opens, highs, lows, closes,
                 sma_fast, sma_slow, rsi_arr,
                 capital, tp, sl, lev, rf, fee):
    """Evalúa UNA configuración. Compilado a ASM nativo."""
    n = len(closes)
    cap = capital
    pos = 0
    epx = 0.0
    qty = 0.0
    tr = 0
    w = 0
    mx = cap
    mdd = 0.0
    gp = 0.0
    gl = 0.0
    streak = 0  # Anti-Martingale win streak counter

    for i in range(1, n):
        if cap <= 1.5:  # 🚨 Absolute liquidation threshold for $13
            mdd = 1.0
            cap = 0.0
            break

        if cap > mx:
            mx = cap
        dd = (mx - cap) / mx if mx > 0 else 0.0
        if dd > mdd:
            mdd = dd
        if mdd > 0.99:  # Allow 99% DD for ALL-IN compounding
            break

        if pos != 0:
            if pos == 1:
                hp = (highs[i] - epx) / epx
                lp = (lows[i] - epx) / epx
            else:
                hp = (epx - lows[i]) / epx
                lp = (epx - highs[i]) / epx

            if hp >= tp:
                xp = epx * (1.0 + tp) if pos == 1 else epx * (1.0 - tp)
                pnl = (xp - epx) * qty * pos - xp * qty * fee
                cap += pnl
                tr += 1
                if pnl > 0:
                    w += 1
                    gp += pnl
                    streak += 1  # Infinite streak multiplier
                else:
                    gl -= pnl
                    streak = 0
                pos = 0
            elif lp <= -sl:
                xp = epx * (1.0 - sl) if pos == 1 else epx * (1.0 + sl)
                pnl = (xp - epx) * qty * pos - xp * qty * fee
                cap += pnl
                tr += 1
                if pnl > 0:
                    w += 1
                    gp += pnl
                    streak += 1
                else:
                    gl -= pnl
                    streak = 0
                pos = 0

        if pos == 0 and i + 1 < n:
            sf = sma_fast[i]
            ss = sma_slow[i]
            rv = rsi_arr[i]
            if sf != sf or ss != ss or rv != rv:
                continue
            sig = 0
            if sf > ss and rv < 65.0:
                sig = 1
            elif sf < ss and rv > 35.0:
                sig = -1
            if sig != 0:
                epx = opens[i + 1]
                if epx <= 0:
                    continue
                
                # 🚀 ALL-IN ANTI-MARTINGALE SIZING
                # Risk grows massively with streak, capped at 1.0 (100% of capital)
                dynamic_rf = min(rf * (1.0 + (streak * 0.50)), 1.0)
                notional = cap * dynamic_rf * lev
                
                if notional < 5.05:
                    continue
                pos = sig
                qty = notional / epx
                cap -= notional * fee

    wr = (w / tr * 100.0) if tr > 0 else 0.0
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    ret = ((cap - capital) / capital) * 100.0
    return cap, tr, wr, mdd * 100.0, pf, ret


@njit(parallel=True, cache=True)
def _sweep_grid(opens, highs, lows, closes,
                sma_banks, sma_pair_idx,
                rsi_arr,
                param_grid, capital, fee):
    """
    Evalúa TODA la malla de parámetros en PARALELO.
    param_grid: (N, 4) → [tp, sl, lev, rf]
    sma_pair_idx: (M, 2) → índices en sma_banks
    Resultado: (M * N, 6) → [cap, trades, wr, dd, pf, ret]
    """
    n_sma = sma_pair_idx.shape[0]
    n_params = param_grid.shape[0]
    total = n_sma * n_params
    results = np.zeros((total, 6), dtype=np.float64)

    for idx in prange(total):
        sma_i = idx // n_params
        par_i = idx % n_params

        fast_idx = sma_pair_idx[sma_i, 0]
        slow_idx = sma_pair_idx[sma_i, 1]
        sma_f = sma_banks[fast_idx]
        sma_s = sma_banks[slow_idx]

        tp = param_grid[par_i, 0]
        sl = param_grid[par_i, 1]
        lev = param_grid[par_i, 2]
        rf = param_grid[par_i, 3]

        cap, tr, wr, dd, pf, ret = _single_eval(
            opens, highs, lows, closes,
            sma_f, sma_s, rsi_arr,
            capital, tp, sl, lev, rf, fee
        )
        results[idx, 0] = cap
        results[idx, 1] = tr
        results[idx, 2] = wr
        results[idx, 3] = dd
        results[idx, 4] = pf
        results[idx, 5] = ret

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER GRID
# ═══════════════════════════════════════════════════════════════════════════════

SCALPING_PARAMS = {
    'tp': np.array([0.005, 0.010, 0.020, 0.030, 0.050, 0.100]),
    'sl': np.array([0.002, 0.003, 0.004, 0.005, 0.008]),
    'lev': np.array([20.0, 50.0, 75.0, 100.0, 125.0]),
    'rf': np.array([0.25, 0.50, 0.75, 0.95]),
    'sma_periods': np.array([5, 8, 12, 20, 30, 50]),
}

SWING_PARAMS = {
    'tp': np.array([0.030, 0.050, 0.100, 0.150, 0.200, 0.300]),
    'sl': np.array([0.005, 0.008, 0.010, 0.015, 0.020]),
    'lev': np.array([20.0, 50.0, 75.0, 100.0, 125.0]),
    'rf': np.array([0.25, 0.50, 0.75, 0.95]),
    'sma_periods': np.array([10, 20, 50, 100, 200]),
}

TOP_10 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT",
]

CAPITAL = 13.0
FEE = 0.000375


def build_param_grid(params):
    """Construye la malla aplanada de (tp, sl, lev, rf) filtrando sl >= tp."""
    combos = []
    for tp in params['tp']:
        for sl in params['sl']:
            if sl >= tp:
                continue
            for lev in params['lev']:
                for rf in params['rf']:
                    combos.append([tp, sl, lev, rf])
    return np.array(combos, dtype=np.float64)


def build_sma_pairs(periods, n_bars):
    """Pre-computa todos los SMAs y devuelve pares válidos (fast < slow)."""
    banks = []
    period_list = []
    for p in periods:
        if p >= n_bars:
            continue
        # Placeholder — será llenado por moneda
        banks.append(p)
        period_list.append(p)
    
    pairs = []
    for i, pf in enumerate(period_list):
        for j, ps in enumerate(period_list):
            if ps > pf:
                pairs.append([i, j])
    return period_list, np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


def optimize_symbol_v2(symbol, data, params, horizon_name):
    """Optimiza un símbolo usando el sweep paralelo."""
    closes = data['close']
    opens = data['open']
    highs = data['high']
    lows = data['low']
    n = len(closes)

    # 1. Build param grid
    pgrid = build_param_grid(params)
    if pgrid.shape[0] == 0:
        return None

    # 2. Pre-compute all SMAs
    period_list, sma_pair_idx = build_sma_pairs(params['sma_periods'], n)
    if sma_pair_idx.shape[0] == 0:
        return None

    sma_banks_list = []
    for p in period_list:
        sma_banks_list.append(_compute_sma(closes, int(p)))
    sma_banks = np.stack(sma_banks_list)

    # 3. RSI
    rsi_arr = _compute_rsi(closes, 14)

    # 4. SWEEP PARALELO
    results = _sweep_grid(opens, highs, lows, closes,
                          sma_banks, sma_pair_idx,
                          rsi_arr,
                          pgrid, CAPITAL, FEE)

    total_evals = results.shape[0]

    # 5. Scoring: priorizar WR > 55%, retorno, penalizar DD
    scores = np.zeros(total_evals)
    for i in range(total_evals):
        tr = results[i, 1]
        wr = results[i, 2]
        dd = results[i, 3]
        pf = results[i, 4]
        ret = results[i, 5]
        if tr < 3:
            scores[i] = -999.0
        else:
            wr_bonus = max(0.0, wr - 55.0) * 2.0
            dd_pen = max(0.0, dd - 5.0) * 3.0
            scores[i] = ret + wr_bonus - dd_pen + pf * 2.0

    best_idx = np.argmax(scores)
    sma_i = best_idx // pgrid.shape[0]
    par_i = best_idx % pgrid.shape[0]

    best_p = pgrid[par_i]
    best_sma_pair = sma_pair_idx[sma_i]
    best_r = results[best_idx]

    return {
        'symbol': symbol,
        'horizon': horizon_name,
        'evals': total_evals,
        'params': {
            'tp_pct': round(float(best_p[0]), 4),
            'sl_pct': round(float(best_p[1]), 4),
            'leverage': int(best_p[2]),
            'risk_frac': round(float(best_p[3]), 2),
            'sma_fast': int(period_list[int(best_sma_pair[0])]),
            'sma_slow': int(period_list[int(best_sma_pair[1])]),
        },
        'metrics': {
            'capital': round(float(best_r[0]), 4),
            'trades': int(best_r[1]),
            'win_rate': round(float(best_r[2]), 2),
            'max_dd': round(float(best_r[3]), 2),
            'profit_factor': round(float(best_r[4]), 2),
            'return_pct': round(float(best_r[5]), 2),
        },
        'score': round(float(scores[best_idx]), 2),
    }


def load_data(data_dir, symbol, bars=43200):
    fname = symbol.replace("/", "_") + "_1m.csv"
    fp = os.path.join(data_dir, fname)
    if not os.path.exists(fp):
        return None
    df = pl.read_csv(fp).tail(bars)
    return {
        'open':   df['open'].cast(pl.Float64).to_numpy(),
        'high':   df['high'].cast(pl.Float64).to_numpy(),
        'low':    df['low'].cast(pl.Float64).to_numpy(),
        'close':  df['close'].cast(pl.Float64).to_numpy(),
        'volume': df['volume'].cast(pl.Float64).to_numpy(),
    }


def run_supermasive_optimizer():
    data_dir = os.path.join(_project_root, "data", "historical")

    print("=" * 80)
    print("🌌 SUPERMASIVE QUANTUM OPTIMIZER v2.0 — PARALLEL TURBO")
    print("=" * 80)

    # Count combos
    sg = build_param_grid(SCALPING_PARAMS)
    wg = build_param_grid(SWING_PARAMS)
    _, sp = build_sma_pairs(SCALPING_PARAMS['sma_periods'], 99999)
    _, wp = build_sma_pairs(SWING_PARAMS['sma_periods'], 99999)
    total_per_sym = sg.shape[0] * sp.shape[0] + wg.shape[0] * wp.shape[0]

    print(f"  📊 Monedas:       {len(TOP_10)} (Top 10)")
    print(f"  💰 Capital:       ${CAPITAL}")
    print(f"  📅 Datos:         Máx 30 días (43,200 barras/moneda)")
    print(f"  🔬 Horizontes:    SCALPING + SWING")
    print(f"  🧬 Evals/moneda:  {total_per_sym:,} (Scalp: {sg.shape[0]*sp.shape[0]:,} + Swing: {wg.shape[0]*wp.shape[0]:,})")
    print(f"  🔢 Total evals:   {total_per_sym * len(TOP_10):,}")
    print("=" * 80)

    # Warm-up JIT
    print("\n⚙️  [JIT] Compilando motor cuántico a ASM nativo (paralelo)...")
    t_jit = time.perf_counter()
    warmup = np.random.random(200).astype(np.float64)
    ws = _compute_sma(warmup, 5)
    wr = _compute_rsi(warmup, 14)
    bank = np.stack([ws, ws])
    pairs = np.array([[0, 1]], dtype=np.int64)
    pg = np.array([[0.01, 0.005, 10.0, 0.05]], dtype=np.float64)
    _ = _sweep_grid(warmup, warmup, warmup, warmup, bank, pairs, wr, pg, 13.0, 0.0005)
    t_jit_end = time.perf_counter()
    print(f"  ✅ Compilado en {(t_jit_end - t_jit)*1000:.0f} ms\n")

    # Load data
    print("📂 Cargando datos...")
    t_load = time.perf_counter()
    all_data = {}
    for sym in TOP_10:
        d = load_data(data_dir, sym)
        if d:
            all_data[sym] = d
            print(f"  ✅ {sym}: {len(d['close']):,} barras")
    t_load_end = time.perf_counter()
    print(f"  📂 Cargado en {(t_load_end - t_load)*1000:.0f} ms\n")

    # OPTIMIZE
    print("🚀 INICIANDO BARRIDO CUÁNTICO PARALELO...\n")
    results = {'scalping': {}, 'swing': {}}
    total_evals = 0

    t_start = time.perf_counter()

    for sym, data in all_data.items():
        t_sym = time.perf_counter()

        rs = optimize_symbol_v2(sym, data, SCALPING_PARAMS, "SCALPING")
        if rs:
            results['scalping'][sym] = rs
            total_evals += rs['evals']

        rw = optimize_symbol_v2(sym, data, SWING_PARAMS, "SWING")
        if rw:
            results['swing'][sym] = rw
            total_evals += rw['evals']

        t_sym_end = time.perf_counter()
        ms = (t_sym_end - t_sym) * 1000

        sm = rs['metrics'] if rs else {}
        wm = rw['metrics'] if rw else {}
        print(f"  🔬 {sym:12} | "
              f"SCL: WR={sm.get('win_rate',0):5.1f}% ${sm.get('capital',13):7.2f} T={sm.get('trades',0):3d} | "
              f"SWG: WR={wm.get('win_rate',0):5.1f}% ${wm.get('capital',13):7.2f} T={wm.get('trades',0):3d} | "
              f"⏱️ {ms:.0f}ms")

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    total_s = total_ms / 1000.0

    # SUMMARY
    print(f"\n{'=' * 80}")
    print("🏆 CONFIGURACIÓN ÓPTIMA POR MONEDA")
    print(f"{'=' * 80}")

    for horizon in ['scalping', 'swing']:
        print(f"\n{'─' * 40}")
        print(f"  📊 HORIZONTE: {horizon.upper()}")
        print(f"{'─' * 40}")
        print(f"  {'Moneda':12} {'TP%':>6} {'SL%':>6} {'Lev':>4} {'Rsk%':>5} {'SMA':>7} {'WR%':>6} {'Ret%':>7} {'DD%':>5} {'PF':>5} {'#Tr':>4}")
        print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*4} {'─'*5} {'─'*7} {'─'*6} {'─'*7} {'─'*5} {'─'*5} {'─'*4}")

        for sym in TOP_10:
            if sym not in results[horizon]:
                continue
            r = results[horizon][sym]
            p = r['params']
            m = r['metrics']
            print(f"  {sym:12} "
                  f"{p['tp_pct']*100:5.2f}% "
                  f"{p['sl_pct']*100:5.2f}% "
                  f"{p['leverage']:3d}x "
                  f"{p['risk_frac']*100:4.0f}% "
                  f"{p['sma_fast']}/{p['sma_slow']:>3} "
                  f"{m['win_rate']:5.1f}% "
                  f"{m['return_pct']:6.1f}% "
                  f"{m['max_dd']:4.1f}% "
                  f"{m['profit_factor']:4.1f} "
                  f"{m['trades']:4d}")

    print(f"\n{'=' * 80}")
    print(f"⚡ OPTIMIZACIÓN COMPLETADA")
    print(f"  🔢 Evaluaciones totales: {total_evals:,}")
    print(f"  ⏱️  Tiempo PURO (sin JIT): {total_s:.3f} s ({total_ms:.0f} ms)")
    print(f"  🚀 Velocidad:             {total_evals / total_s:,.0f} evaluaciones/seg")
    print(f"  📊 Monedas:               {len(all_data)}")
    print(f"{'=' * 80}")

    # Save
    output = os.path.join(_project_root, "data", "optimal_config.json")
    ser = {}
    for h in ['scalping', 'swing']:
        ser[h] = {}
        for sym, r in results[h].items():
            ser[h][sym] = {'params': r['params'], 'metrics': r['metrics'], 'score': r['score']}
    with open(output, 'w') as f:
        json.dump(ser, f, indent=2)
    print(f"\n💾 Guardado en: {output}")

    return results


if __name__ == "__main__":
    run_supermasive_optimizer()
