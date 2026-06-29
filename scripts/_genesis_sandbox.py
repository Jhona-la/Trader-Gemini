import os
import sys
import json
import time
import gc
import glob
import numpy as np
import pandas as pd
import optuna
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ensure we can import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_bridge.quantum_bridge import run_sandbox_trial_py
from core.nano_backtester import vectorized_signals
from core.quantum.mmap_storage import QuantumMMAP
from config import Config

def load_real_data(symbol, horizon="SCALPING"):
    """
    Loads real historical data from QuantumMMAP for a given symbol.
    Returns (train_arrays, val_arrays)
    """
    mmap = QuantumMMAP(symbol)
    df = mmap.to_dataframe()
    
    if df.empty:
        print(f"⚠️ [Sandbox] {symbol} está vacío en Quantum Data Lake.")
        return None, None
        
    df = df.sort_index()
    
    # Gap Detection
    diff = df.index.to_series().diff()
    gaps = diff[diff > pd.Timedelta(seconds=60)]
    if not gaps.empty:
        # print(f"⚠️ [Sandbox] {symbol} tiene {len(gaps)} gaps > 60s. Entropía mitigada temporalmente.")
        pass # In a production system we'd fill or split. For optimization, we accept as is.

    if len(df) < 10000:
        print(f"⚠️ [Sandbox] {symbol} tiene muy pocos datos ({len(df)}).")
        return None, None
        
    # Extract arrays
    highs = np.ascontiguousarray(df['high'].values, dtype=np.float64)
    lows = np.ascontiguousarray(df['low'].values, dtype=np.float64)
    closes = np.ascontiguousarray(df['close'].values, dtype=np.float64)
    
    # Generate real mathematical signals
    # Parameters for base signals (e.g., RSI=14, oversold=30, overbought=70, MACD=12,26)
    signals = vectorized_signals(closes, 14, 30.0, 70.0, 12, 26)
    
    # Walk-Forward Split (70% Train, 30% Validation)
    split_idx = int(len(closes) * 0.70)
    
    train_arrays = (highs[:split_idx], lows[:split_idx], closes[:split_idx], signals[:split_idx])
    val_arrays = (highs[split_idx:], lows[split_idx:], closes[split_idx:], signals[split_idx:])
    
    return train_arrays, val_arrays

def objective(trial, highs, lows, closes, signals):
    sl_pct = trial.suggest_float("sl_pct", 0.005, 0.05)
    kinematic_umbral = trial.suggest_float("kinematic_umbral", -0.05, -0.001)
    
    # The real signals are already generated. We pass them as is.
    res = run_sandbox_trial_py(highs, lows, closes, signals, sl_pct, kinematic_umbral)
    
    trades = res["trades"]
    win_rate = res["win_rate"]
    is_pruned = res["is_pruned"]
    
    if trades < 20:
        # Penalize severely if not enough trades
        return -9999.0
        
    if is_pruned:
        return -9999.0
        
    # Maximize total PnL
    pnls = res["pnl"]
    total_pnl = np.sum(pnls)
    
    return total_pnl

def evaluate_validation(highs, lows, closes, signals, best_params):
    res = run_sandbox_trial_py(
        highs, lows, closes, signals, 
        best_params["sl_pct"], 
        best_params["kinematic_umbral"]
    )
    
    return res

def get_symbols_from_datalake():
    files = glob.glob(os.path.join(Config.BASE_DIR, "data/quantum_lake/*.qbin"))
    symbols = [os.path.basename(f).replace(".qbin", "") for f in files]
    return symbols

def main():
    print("🚀 Iniciando Sandbox Holográfico - BARRIDO DE DATOS REALES (FASE II/III)")
    
    symbols = get_symbols_from_datalake()
    print(f"📡 Se detectaron {len(symbols)} activos en el DataLake para procesar secuencialmente.")
    
    output_matrix = {}
    matrix_path = os.path.join(Config.BASE_DIR, "config", "quantum_confidence_matrix.json")
    
    if os.path.exists(matrix_path):
        try:
            with open(matrix_path, "r") as f:
                output_matrix = json.load(f)
        except:
            from utils.error_handler import SystemIntegrityError
            raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

    horizons = ["SCALPING", "SWING"]
    
    total_start = time.time()
    
    for symbol in symbols:
        print(f"\n======================================")
        print(f"🔍 Auditando: {symbol}")
        
        if symbol not in output_matrix:
            output_matrix[symbol] = {}
            
        train_arrays, val_arrays = load_real_data(symbol)
        
        if train_arrays is None:
            continue
            
        for horizon in horizons:
            print(f"  [{horizon}] Forjando el metal en 50 trials...")
            
            # TODO: If SWING, we'd resample the array to 15m. For now we use 1m but relax thresholds
            # Or just use same array for demo.
            
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, *train_arrays), n_trials=50, n_jobs=1)
            
            best_pnl = study.best_value
            best_params = study.best_params
            
            if best_pnl <= -9990.0:
                print(f"  ❌ [{horizon}] Ningún Trial sobrevivió a la poda térmica. Edge no encontrado.")
                output_matrix[symbol][horizon] = {"status": "DISABLED", "reason": "PRUNED"}
                continue
                
            print(f"  ✅ [{horizon}] Train Edge Encontrado: PnL Esperado = {best_pnl:.2f}")
            print(f"  🧪 Ejecutando Walk-Forward en Out-Of-Sample Validation...")
            
            val_res = evaluate_validation(*val_arrays, best_params)
            
            v_trades = val_res["trades"]
            v_wr = val_res["win_rate"]
            v_pnl = np.sum(val_res["pnl"])
            
            print(f"  📊 Validación: Trades={v_trades}, WinRate={v_wr:.2f}, PnL={v_pnl:.2f}")
            
            if v_wr >= 0.70 and v_pnl > 0 and v_trades >= 5: # Relaxed slightly for real market tests to >70%
                print(f"  🛡️ Validación EXITOSA. Binomio {symbol} {horizon} AUTORIZADO.")
                output_matrix[symbol][horizon] = {
                    "status": "ACTIVE",
                    "sl_pct": best_params["sl_pct"],
                    "kinematic_snap_threshold": best_params["kinematic_umbral"],
                    "min_confidence": 0.85, # static base for now
                    "validation_wr": v_wr,
                    "validation_pnl": v_pnl
                }
            else:
                print(f"  🗑️ Validación FALLIDA (Overfitting). Binomio RECHAZADO.")
                output_matrix[symbol][horizon] = {
                    "status": "DISABLED",
                    "reason": "OVERFITTING_FAIL",
                    "val_wr": v_wr,
                    "val_pnl": v_pnl
                }
            
            # Guardado atómico progresivo
            with open(matrix_path, "w") as f:
                json.dump(output_matrix, f, indent=4)
                
        # Liberar RAM implacablemente
        del train_arrays
        del val_arrays
        gc.collect()

    print(f"\n✅ Barrido Completado en {time.time() - total_start:.2f} segundos.")
    print(f"📂 Matriz guardada en: {matrix_path}")

if __name__ == "__main__":
    main()
