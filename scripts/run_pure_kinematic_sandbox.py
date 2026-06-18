import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import optuna
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_bridge.quantum_bridge import run_sandbox_trial_py
from core.quantum.mmap_storage import QuantumMMAP
from config import Config
from numba import njit

@njit(fastmath=True, nogil=True)
def calculate_sma(arr, window):
    n = len(arr)
    res = np.zeros(n, dtype=np.float64)
    if n > window:
        # Simple SMA
        for i in range(window - 1, n):
            suma = 0.0
            for j in range(window):
                suma += arr[i - j]
            res[i] = suma / window
    return res

@njit(fastmath=True, nogil=True)
def calculate_bollinger_bands(closes, window, num_std):
    n = len(closes)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    sma = calculate_sma(closes, window)
    
    if n > window:
        for i in range(window - 1, n):
            suma_sq = 0.0
            for j in range(window):
                diff = closes[i - j] - sma[i]
                suma_sq += diff * diff
            std = np.sqrt(suma_sq / window)
            upper[i] = sma[i] + (num_std * std)
            lower[i] = sma[i] - (num_std * std)
            
    return upper, lower

@njit(fastmath=True, nogil=True)
def breakout_signals(highs, lows, closes, window=40, num_std=2.0):
    n = len(closes)
    signals = np.zeros(n, dtype=np.int8)
    
    upper, lower = calculate_bollinger_bands(closes, window, num_std)
    
    for i in range(2, n):
        if upper[i-1] == 0.0:
            continue
            
        # Breakout LONG
        if closes[i-1] > upper[i-2] and closes[i-2] <= upper[i-3]:
            signals[i] = 1
        # Breakout SHORT
        elif closes[i-1] < lower[i-2] and closes[i-2] >= lower[i-3]:
            signals[i] = -1
            
    return signals

def load_real_data(symbol):
    mmap = QuantumMMAP(symbol)
    df = mmap.to_dataframe()
    
    if len(df) < 110000:
        return None, None
        
    df = df.sort_index()
    
    highs = np.ascontiguousarray(df['high'].values, dtype=np.float64)
    lows = np.ascontiguousarray(df['low'].values, dtype=np.float64)
    closes = np.ascontiguousarray(df['close'].values, dtype=np.float64)
    
    # Generate pure Kinematic signals (Bollinger Breakout delayed by 1 to avoid leakage)
    signals = breakout_signals(highs, lows, closes, window=40, num_std=2.0)
    
    # 70% Train, 30% Val
    split_idx = int(len(closes) * 0.70)
    
    train_arrays = (highs[:split_idx], lows[:split_idx], closes[:split_idx], signals[:split_idx])
    val_arrays = (highs[split_idx:], lows[split_idx:], closes[split_idx:], signals[split_idx:])
    
    return train_arrays, val_arrays

def objective(trial, highs, lows, closes, signals):
    # En el modo termodinámico dependemos 100% de la asimetría del Risk Manager
    # Vamos a buscar Stop Loss ultra cortos y Trailing altamente agresivo
    sl_pct = trial.suggest_float("sl_pct", 0.002, 0.02) # SL entre 0.2% y 2.0%
    kinematic_umbral = trial.suggest_float("kinematic_umbral", -0.05, -0.001) # Trailing Rate
    
    res = run_sandbox_trial_py(highs, lows, closes, signals, sl_pct, kinematic_umbral)
    
    trades = res["trades"]
    if trades < 30:
        return -9999.0
        
    if res["is_pruned"]:
        return -9999.0
        
    pnls = res["pnl"]
    total_pnl = np.sum(pnls)
    return total_pnl

def run_thermodynamic_sandbox():
    print("🔥 INICIANDO GÉNESIS TERMOCINEMÁTICA PURA (ZERO ML) 🔥")
    
    files = glob.glob(os.path.join(Config.BASE_DIR, "data/quantum_lake/*.qbin"))
    symbols = [os.path.basename(f).replace(".qbin", "") for f in files]
    
    matrix = {}
    
    for symbol in symbols:
        print(f"\n⚙️ Optimizando {symbol} (Kinematic Breakout)...")
        train_arrays, val_arrays = load_real_data(symbol)
        
        if train_arrays is None:
            print(f"⚠️ Saltando {symbol} (Falta de datos)")
            continue
            
        t_highs, t_lows, t_closes, t_signals = train_arrays
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, t_highs, t_lows, t_closes, t_signals), n_trials=100)
        
        best = study.best_params
        best_pnl = study.best_value
        
        if best_pnl <= 0:
            print(f"❌ {symbol}: No se halló Edge en Train (PnL: {best_pnl:.2f}%)")
            continue
            
        print(f"✅ {symbol} Train PnL: {best_pnl:.2f}% | Validando OOS...")
        
        v_highs, v_lows, v_closes, v_signals = val_arrays
        v_res = run_sandbox_trial_py(v_highs, v_lows, v_closes, v_signals, best["sl_pct"], best["kinematic_umbral"])
        
        v_pnl = np.sum(v_res["pnl"])
        v_trades = v_res["trades"]
        
        print(f"   OOS PnL: {v_pnl:.2f}% | Trades: {v_trades} | WinRate: {v_res['win_rate']:.2f}%")
        
        if v_pnl > 0 and v_trades > 10:
            print(f"🚀 EDGE CONFIRMADO para {symbol}")
            matrix[symbol] = {
                "horizon": "SCALPING",
                "sl_pct": best["sl_pct"],
                "kinematic_umbral": best["kinematic_umbral"],
                "oos_pnl": float(v_pnl),
                "status": "ACTIVE"
            }
        else:
            print(f"💀 SOBREAJUSTE OOS para {symbol}. Ruido.")
            
    with open("config/quantum_kinematic_matrix.json", "w") as f:
        json.dump(matrix, f, indent=4)
        
    print(f"\n🎉 GÉNESIS COMPLETADA. {len(matrix)} Edges Termodinámicos puros hallados.")

if __name__ == "__main__":
    run_thermodynamic_sandbox()
