import os
import sys
import numpy as np
import time
from numba import njit
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from core.quantum.mmap_storage import QuantumMMAP

# ==============================================================================
# 🚀 NUMBA CORE: ZERO-ALLOCATION KINEMATIC ENGINE
# ==============================================================================
@njit(fastmath=True, nogil=True)
def calculate_sma_numba(arr, window):
    n = len(arr)
    res = np.zeros(n, dtype=np.float64)
    if n > window:
        for i in range(window - 1, n):
            suma = 0.0
            for j in range(window):
                suma += arr[i - j]
            res[i] = suma / window
    return res

@njit(fastmath=True, nogil=True)
def calculate_bollinger_bands_numba(closes, window, num_std):
    n = len(closes)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    sma = calculate_sma_numba(closes, window)
    
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
def backtest_kinematic_numba(closes, highs, lows, window, num_std, sl_pct, kinematic_umbral):
    upper, lower = calculate_bollinger_bands_numba(closes, window, num_std)
    
    n = len(closes)
    in_position = False
    direction = 0 # 1=LONG, -1=SHORT
    entry_price = 0.0
    trailing_stop = 0.0
    
    trades = 0
    wins = 0
    total_pnl = 0.0
    
    for i in range(window + 1, n):
        # Mantenimiento de posición abierta
        if in_position:
            if direction == 1:
                # Actualizar trailing stop (Trailing cinemático - toma el umbral % de ganancia max)
                potential_ts = closes[i] * (1.0 - kinematic_umbral)
                if potential_ts > trailing_stop:
                    trailing_stop = potential_ts
                    
                # Chequeo de Stop Loss / Trailing
                if lows[i] <= trailing_stop:
                    pnl = (trailing_stop - entry_price) / entry_price
                    total_pnl += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                    in_position = False
            else:
                # SHORT
                potential_ts = closes[i] * (1.0 + kinematic_umbral)
                if potential_ts < trailing_stop:
                    trailing_stop = potential_ts
                    
                # Chequeo de Stop Loss / Trailing
                if highs[i] >= trailing_stop:
                    pnl = (entry_price - trailing_stop) / entry_price
                    total_pnl += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                    in_position = False
            continue

        # Evaluación de nueva entrada (Breakout puro, -2 vs -3 respecto a i, pero en simulación 
        # miramos i-1 e i-2 de forma iterativa)
        # i es "hoy" (cierre actual). Breakout se dio ayer.
        
        # Breakout LONG (cierre ayer rompió arriba, y antier estaba debajo)
        if closes[i-1] > upper[i-2] and closes[i-2] <= upper[i-3]:
            in_position = True
            direction = 1
            entry_price = closes[i] # Ejecución al cierre de esta vela o apertura siguiente
            trailing_stop = entry_price * (1.0 - sl_pct)
            continue
            
        # Breakout SHORT
        if closes[i-1] < lower[i-2] and closes[i-2] >= lower[i-3]:
            in_position = True
            direction = -1
            entry_price = closes[i]
            trailing_stop = entry_price * (1.0 + sl_pct)
            continue
            
    wr = (wins / trades) if trades > 0 else 0.0
    return total_pnl, trades, wr

# ==============================================================================
# EXPERIMENTO WALK-FORWARD
# ==============================================================================
def grid_search_train(closes, highs, lows):
    best_pnl = -999.0
    best_params = None
    
    windows = [20, 30, 40, 50]
    stds = [2.0, 2.5, 3.0]
    sls = [0.005, 0.01, 0.02]
    ts_umbrals = [0.005, 0.01, 0.015]
    
    # Pre-compilar Numba
    backtest_kinematic_numba(closes[:100], highs[:100], lows[:100], 20, 2.0, 0.01, 0.01)
    
    for w in windows:
        for s in stds:
            for sl in sls:
                for ts in ts_umbrals:
                    pnl, trades, wr = backtest_kinematic_numba(closes, highs, lows, w, s, sl, ts)
                    # Exigir un mínimo de trades para significancia estadística
                    if trades > 50 and pnl > best_pnl:
                        best_pnl = pnl
                        best_params = (w, s, sl, ts)
                        
    return best_params, best_pnl

def evaluate_symbol(symbol: str):
    print(f"\n==============================================")
    print(f"🧬 AUDITORÍA CINEMÁTICA: {symbol}")
    print(f"==============================================")
    
    mmap = QuantumMMAP(symbol)
    df = mmap.to_dataframe()
    if df.empty or len(df) < 5000:
        print(f"[{symbol}] Data insuficiente ({len(df)} bars)")
        return
        
    print(f"[{symbol}] Cargando {len(df)} velas tick-by-tick...")
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    
    # 70/30 Split Cronológico
    split_idx = int(len(closes) * 0.70)
    
    c_train, h_train, l_train = closes[:split_idx], highs[:split_idx], lows[:split_idx]
    c_test, h_test, l_test = closes[split_idx:], highs[split_idx:], lows[split_idx:]
    
    print(f"[{symbol}] FASE I: Entrenando en {len(c_train)} velas...")
    t0 = time.time()
    best_params, best_pnl = grid_search_train(c_train, h_train, l_train)
    t1 = time.time()
    
    if best_params is None:
        print(f"❌ [{symbol}] FAIL: Ningún set de parámetros logró rentabilidad con significancia estadística en TRAIN.")
        return
        
    w, s, sl, ts = best_params
    _, tr_trades, tr_wr = backtest_kinematic_numba(c_train, h_train, l_train, w, s, sl, ts)
    print(f"[{symbol}] TRAIN OPTIMAL: PnL={best_pnl*100:.2f}% | WR={tr_wr*100:.2f}% | Trades={tr_trades} | Time={t1-t0:.2f}s")
    print(f"         Params -> Window={w}, STD={s}, SL={sl*100:.2f}%, Trail={ts*100:.2f}%")
    
    # FASE II: TEST OUT-OF-SAMPLE (EL VEREDICTO DE LA VERDAD)
    print(f"[{symbol}] FASE II: Validación OUT-OF-SAMPLE ({len(c_test)} velas)...")
    t_pnl, t_trades, t_wr = backtest_kinematic_numba(c_test, h_test, l_test, w, s, sl, ts)
    
    print(f"[{symbol}] OOS RESULT: PnL={t_pnl*100:.2f}% | WR={t_wr*100:.2f}% | Trades={t_trades}")
    
    if t_pnl > 0.0 and t_trades > 10:
        print(f"✅ [{symbol}] EDGE VALIDADO: Supervivencia OOS Confirmada.")
    else:
        print(f"❌ [{symbol}] EDGE RECHAZADO: Falso positivo matemático (Overfitting de Bandas). OOS PnL negativo.")

if __name__ == "__main__":
    symbols = ["OPUSDT", "SUIUSDT", "UNIUSDT"]
    for sym in symbols:
        evaluate_symbol(sym)
