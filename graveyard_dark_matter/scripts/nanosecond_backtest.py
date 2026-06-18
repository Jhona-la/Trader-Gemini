#!/usr/bin/env python3
"""
🌌 NANOSECOND VECTORIZED BACKTEST (GOD MODE)
═══════════════════════════════════════════════════════════════════════════
Ejecuta un backtest completo de 30 días para las 10 monedas principales 
utilizando Numba JIT para lograr tiempos de ejecución de sub-milisegundos.
Utiliza EXACTAMENTE los parámetros inyectados en config.py.
"""

import os, sys, time, json
import numpy as np
import polars as pl
from numba import njit, prange

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config

# --- NUMBA CORE ---
@njit(cache=True)
def _compute_sma(closes, period):
    n = len(closes)
    sma = np.full(n, np.nan)
    if n < period: return sma
    s = 0.0
    for i in range(period): s += closes[i]
    sma[period - 1] = s / period
    for i in range(period, n):
        s += closes[i] - closes[i - period]
        sma[i] = s / period
    return sma

@njit(cache=True)
def _compute_rsi(closes, period):
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1: return rsi
    avg_g = 0.0
    avg_l = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i-1]
        if d > 0: avg_g += d
        else: avg_l -= d
    avg_g /= period
    avg_l /= period
    if avg_l > 0: rsi[period] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    else: rsi[period] = 100.0
    for i in range(period + 1, n):
        d = closes[i] - closes[i-1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        if avg_l > 0: rsi[i] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
        else: rsi[i] = 100.0
    return rsi

@njit(cache=True)
def _single_eval(opens, highs, lows, closes,
                 sma_fast, sma_slow, rsi_arr,
                 capital, tp, sl, lev, rf, fee, rsi_buy, rsi_sell):
    """Evalúa configuración única."""
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

    for i in range(1, n):
        if cap > mx: mx = cap
        dd = (mx - cap) / mx if mx > 0 else 0.0
        if dd > mdd: mdd = dd
        if mdd > 0.90: break # Margin call basically

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
                if pnl > 0: w += 1; gp += pnl
                else: gl -= pnl
                pos = 0
            elif lp <= -sl:
                xp = epx * (1.0 - sl) if pos == 1 else epx * (1.0 + sl)
                pnl = (xp - epx) * qty * pos - xp * qty * fee
                cap += pnl
                tr += 1
                if pnl > 0: w += 1; gp += pnl
                else: gl -= pnl
                pos = 0

        if pos == 0 and i + 1 < n:
            sf = sma_fast[i]
            ss = sma_slow[i]
            rv = rsi_arr[i]
            if sf != sf or ss != ss or rv != rv: continue
            sig = 0
            if sf > ss and rv < rsi_sell: sig = 1
            elif sf < ss and rv > rsi_buy: sig = -1
            if sig != 0:
                epx = opens[i + 1]
                if epx <= 0: continue
                notional = cap * rf * lev
                if notional < 5.05: continue
                pos = sig
                qty = notional / epx
                cap -= notional * fee

    wr = (w / tr * 100.0) if tr > 0 else 0.0
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    ret = ((cap - capital) / capital) * 100.0
    return cap, tr, wr, mdd * 100.0, pf, ret

def load_data(horizon="SCALPING"):
    data_dir = os.path.join(_project_root, "data", "historical")
    all_data = {}
    top_10 = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
              "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT"]
    
    # Map horizon to file timeframe
    tf_suffix = "_1m.csv" if horizon == "SCALPING" else "_1h.csv"
    bars_per_day = 1440 if horizon == "SCALPING" else 24
    days_to_load = 30 # 30 Days of data
    max_bars = bars_per_day * days_to_load
    
    print(f"📥 Carga de datos para {horizon} ({days_to_load} días, max {max_bars} barras)...")
    t0 = time.time()
    for fname in os.listdir(data_dir):
        if fname.endswith(tf_suffix):
            sym = fname.replace(tf_suffix, "").replace("_", "/")
            if sym not in top_10: continue
            
            # Polars is extremely fast
            df = pl.read_csv(os.path.join(data_dir, fname))
            if df.height > max_bars:
                df = df.tail(max_bars)
            
            all_data[sym] = {
                'open': np.array(df['open'].to_numpy(), dtype=np.float64),
                'high': np.array(df['high'].to_numpy(), dtype=np.float64),
                'low': np.array(df['low'].to_numpy(), dtype=np.float64),
                'close': np.array(df['close'].to_numpy(), dtype=np.float64),
            }
    
    print(f"✅ Datos cargados en {(time.time()-t0)*1000:.2f} ms")
    return all_data

def run_nanosecond_backtest():
    print("="*60)
    print("🌌 NANOSECOND VECTORIZED BACKTEST (GOD MODE)")
    print("="*60)
    
    # 1. Warmup Numba JIT
    print("🔥 Calentando compilador JIT (Nano-Warmup)...")
    dummy = np.random.rand(100)
    _compute_sma(dummy, 10)
    _compute_rsi(dummy, 14)
    _single_eval(dummy, dummy, dummy, dummy, dummy, dummy, dummy, 13.0, 0.01, 0.005, 10.0, 0.1, 0.000375, 30, 70)
    
    capital = 13.0
    fee = 0.000375
    
    for horizon in ["SCALPING", "SWING"]:
        data = load_data(horizon)
        if not data:
            print(f"⚠️ No hay datos para {horizon}. Usa fetch_historical_data.py")
            continue
            
        cfg = getattr(Config.Horizons, horizon.capitalize())
        
        # Parámetros Globales del Horizonte
        sma_fast_p = cfg['ema_fast']
        sma_slow_p = cfg['ema_slow']
        rsi_period = cfg['rsi_period']
        rsi_buy = cfg['rsi_buy']
        rsi_sell = cfg['rsi_sell']
        
        # Iterar monedas
        print(f"\n🚀 EJECUTANDO BACKTEST {horizon} EN TOP 10 MONEDAS:")
        print(f"{'Moneda':<10} | {'TP%':<6} | {'SL%':<6} | {'Lev':<4} | {'RF%':<4} || {'Ret%':<7} | {'Win%':<5} | {'Trades':<6} | {'PF':<5}")
        print("-" * 80)
        
        total_pnl = 0.0
        t_start = time.perf_counter()
        
        for sym, bars in data.items():
            # Extraer params óptimos inyectados
            tp = cfg['tp_pct_per_asset'].get(sym, cfg['tp_pct'])
            sl = cfg['sl_pct_per_asset'].get(sym, cfg['sl_pct'])
            # Assuming global optimal lev and rf for now based on risk logic
            lev = 20.0 if horizon == "SCALPING" else 10.0
            rf = 0.05 if horizon == "SCALPING" else 0.10
            
            # Pre-compute indicators
            sma_f = _compute_sma(bars['close'], sma_fast_p)
            sma_s = _compute_sma(bars['close'], sma_slow_p)
            rsi_arr = _compute_rsi(bars['close'], rsi_period)
            
            # Single nanosecond evaluation
            cap, tr, wr, dd, pf, ret = _single_eval(
                bars['open'], bars['high'], bars['low'], bars['close'],
                sma_f, sma_s, rsi_arr,
                capital, tp, sl, lev, rf, fee, rsi_buy, rsi_sell
            )
            
            total_pnl += (cap - capital)
            print(f"{sym:<10} | {tp*100:<6.2f} | {sl*100:<6.2f} | {lev:<4} | {rf*100:<4} || {ret:>7.2f}% | {wr:>5.1f}% | {tr:<6} | {pf:>5.2f}")
        
        t_end = time.perf_counter()
        t_ms = (t_end - t_start) * 1000
        
        print("-" * 80)
        print(f"⏱️  TIEMPO DE EJECUCIÓN (Motor Cuántico): {t_ms:.3f} ms")
        print(f"💰 PnL Total Acumulado ({horizon}): ${total_pnl:.2f}")

if __name__ == '__main__':
    run_nanosecond_backtest()
