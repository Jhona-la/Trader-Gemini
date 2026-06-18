import os
import sys
import numpy as np
import polars as pl
from numba import njit
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# 🚀 NUMBA CORE: MICROSTRUCTURE KINEMATIC ENGINE (O(1) TICK-BY-TICK)
# ==============================================================================
@njit(fastmath=True, nogil=True)
def backtest_microstructure_numba(
    prices, volumes, is_buyer_maker, 
    ema_window, z_breakout, z_absorption, sl_pct
):
    n = len(prices)
    
    # EMA Constants
    alpha = 2.0 / (ema_window + 1.0)
    
    # State Variables (O(1) updates)
    ema_price = prices[0]
    ema_price_var = 0.0
    
    ema_cvd = 0.0
    ema_cvd_var = 0.0
    
    current_cvd = 0.0
    
    in_position = False
    direction = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    trades = 0
    wins = 0
    total_pnl = 0.0
    max_dd = 0.0
    peak_pnl = 0.0
    
    # Calentamiento (Burn-in)
    burn_in = max(1000, ema_window * 3)
    
    for i in range(1, n):
        price = prices[i]
        vol = volumes[i]
        
        # 1. Update CVD
        cvd_delta = -vol if is_buyer_maker[i] else vol
        current_cvd += cvd_delta
        
        # 2. Update O(1) Price Bollinger (EMA & Variance)
        diff_price = price - ema_price
        ema_price += alpha * diff_price
        ema_price_var = (1.0 - alpha) * (ema_price_var + alpha * diff_price * diff_price)
        price_std = np.sqrt(ema_price_var)
        
        bollinger_upper = ema_price + (2.0 * price_std)
        bollinger_lower = ema_price - (2.0 * price_std)
        
        # 3. Update O(1) CVD Z-Score
        diff_cvd = cvd_delta - ema_cvd
        ema_cvd += alpha * diff_cvd
        ema_cvd_var = (1.0 - alpha) * (ema_cvd_var + alpha * diff_cvd * diff_cvd)
        cvd_std = np.sqrt(ema_cvd_var)
        
        cvd_z_score = 0.0
        if cvd_std > 1e-9:
            cvd_z_score = (cvd_delta - ema_cvd) / cvd_std
            
        if i < burn_in:
            continue
            
        # 4. Posición Abierta (Mantenimiento y Salida)
        if in_position:
            pnl_pct = 0.0
            
            if direction == 1:
                # Update Trailing Stop
                if price > entry_price:
                    potential_ts = price * (1.0 - sl_pct)
                    if potential_ts > trailing_stop:
                        trailing_stop = potential_ts
                
                # Check Stop Loss / Trailing
                if price <= trailing_stop:
                    pnl_pct = (trailing_stop - entry_price) / entry_price
                    in_position = False
                    
                # LEADING EXIT: Si el CVD_Z se vuelve muy negativo en pleno LONG (agresores huyen)
                elif cvd_z_score < -2.0:
                    pnl_pct = (price - entry_price) / entry_price
                    in_position = False
                    
            else: # SHORT
                if price < entry_price:
                    potential_ts = price * (1.0 + sl_pct)
                    if potential_ts < trailing_stop:
                        trailing_stop = potential_ts
                        
                if price >= trailing_stop:
                    pnl_pct = (entry_price - trailing_stop) / entry_price
                    in_position = False
                    
                # LEADING EXIT
                elif cvd_z_score > 2.0:
                    pnl_pct = (entry_price - price) / entry_price
                    in_position = False
                    
            if not in_position:
                total_pnl += pnl_pct
                trades += 1
                if pnl_pct > 0:
                    wins += 1
                    
                if total_pnl > peak_pnl:
                    peak_pnl = total_pnl
                dd = peak_pnl - total_pnl
                if dd > max_dd:
                    max_dd = dd
            continue
            
        # 5. Lógica de Fusión: Evaluador de Breakouts y Absorción
        
        price_delta_pct = (price - prices[i-1]) / prices[i-1]
        
        # A) BREAKOUT VALIDADO (Confluencia Pura LONG)
        if price > bollinger_upper and cvd_z_score > z_breakout:
            in_position = True
            direction = 1
            entry_price = price
            trailing_stop = entry_price * (1.0 - sl_pct)
            continue
            
        # B) BREAKOUT VALIDADO (Confluencia Pura SHORT)
        if price < bollinger_lower and cvd_z_score < -z_breakout:
            in_position = True
            direction = -1
            entry_price = price
            trailing_stop = entry_price * (1.0 + sl_pct)
            continue
            
        # C) REVERSAL POR ABSORCIÓN (Pánico absorbido -> LONG)
        if price < bollinger_lower and cvd_z_score < -z_absorption and price_delta_pct > -0.0005:
            # Ventas masivas pero el precio casi no cae -> Absorción ballena
            in_position = True
            direction = 1
            entry_price = price
            trailing_stop = entry_price * (1.0 - sl_pct)
            continue
            
        # D) REVERSAL POR DISTRIBUCIÓN (Furia compradora absorbida -> SHORT)
        if price > bollinger_upper and cvd_z_score > z_absorption and price_delta_pct < 0.0005:
            in_position = True
            direction = -1
            entry_price = price
            trailing_stop = entry_price * (1.0 + sl_pct)
            continue
            
    wr = (wins / trades) if trades > 0 else 0.0
    return total_pnl, trades, wr, max_dd

# ==============================================================================
# ESCUDO WALK-FORWARD
# ==============================================================================
def run_grid_search(prices, volumes, is_buyer_maker):
    best_pnl = -999.0
    best_params = None
    
    ema_windows = [1000, 5000, 10000] # Ticks
    z_breakouts = [1.5, 2.0, 2.5]
    z_absorptions = [3.0, 4.0, 5.0]
    sls = [0.005, 0.01, 0.015]
    
    # Pre-compilar Numba
    backtest_microstructure_numba(prices[:2000], volumes[:2000], is_buyer_maker[:2000], 1000, 1.5, 3.0, 0.01)
    
    print("🔍 Iniciando optimización en el Train Set...")
    total_iters = len(ema_windows) * len(z_breakouts) * len(z_absorptions) * len(sls)
    count = 0
    for ema in ema_windows:
        for zb in z_breakouts:
            for za in z_absorptions:
                for sl in sls:
                    count += 1
                    pnl, trades, wr, max_dd = backtest_microstructure_numba(
                        prices, volumes, is_buyer_maker, ema, zb, za, sl
                    )
                    
                    if trades > 30 and pnl > best_pnl:
                        best_pnl = pnl
                        best_params = (ema, zb, za, sl)
                        
    return best_params, best_pnl

def run_experiment(symbol: str):
    print(f"\n==============================================")
    print(f"🛡️ ESCUDO WALK-FORWARD: {symbol} (Order Flow)")
    print(f"==============================================")
    
    parquet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history", f"{symbol}_aggTrades_7d.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"❌ Archivo no encontrado: {parquet_path}")
        return
        
    df = pl.read_parquet(parquet_path)
    print(f"📊 Cargando {len(df):,} ticks individuales...")
    
    prices = df['price'].cast(pl.Float64).to_numpy()
    volumes = df['volume'].cast(pl.Float64).to_numpy()
    is_buyer_maker = df['is_buyer_maker'].cast(pl.Boolean).to_numpy()
    
    # 70/30 Split Cronológico
    split_idx = int(len(prices) * 0.70)
    
    p_train, v_train, b_train = prices[:split_idx], volumes[:split_idx], is_buyer_maker[:split_idx]
    p_test, v_test, b_test = prices[split_idx:], volumes[split_idx:], is_buyer_maker[split_idx:]
    
    print(f"\n[FASE I] Optimizando hiperparámetros adaptativos en {len(p_train):,} ticks (70%)...")
    t0 = time.time()
    best_params, best_pnl = run_grid_search(p_train, v_train, b_train)
    t1 = time.time()
    
    if not best_params:
        print("❌ FAIL: Ningún set de parámetros logró rentabilidad estadística en Train.")
        return
        
    ema, zb, za, sl = best_params
    _, tr_trades, tr_wr, tr_dd = backtest_microstructure_numba(p_train, v_train, b_train, ema, zb, za, sl)
    
    print(f"✅ TRAIN OPTIMAL (Tardó {t1-t0:.2f}s): PnL = {best_pnl*100:.2f}% | WR = {tr_wr*100:.2f}% | MaxDD = {tr_dd*100:.2f}% | Trades = {tr_trades}")
    print(f"   Parámetros Adaptativos: EMA_Ticks={ema}, Z_Breakout={zb}, Z_Absorption={za}, SL={sl*100:.2f}%")
    
    print(f"\n[FASE II] Evaluación Ciega (OOS) en {len(p_test):,} ticks (30%)...")
    
    oos_pnl, oos_trades, oos_wr, oos_dd = backtest_microstructure_numba(p_test, v_test, b_test, ema, zb, za, sl)
    
    print(f"⚖️ OOS RESULTADO: PnL = {oos_pnl*100:.2f}% | WR = {oos_wr*100:.2f}% | MaxDD = {oos_dd*100:.2f}% | Trades = {oos_trades}")
    
    # Criterios inquebrantables
    passed = True
    if oos_wr < 0.65:
        print("❌ FAIL CRITERIO: Win Rate OOS < 65%")
        passed = False
    if oos_pnl <= 0:
        print("❌ FAIL CRITERIO: PnL OOS Negativo")
        passed = False
    if oos_dd > 0.10:
        print("❌ FAIL CRITERIO: Max Drawdown > 10%")
        passed = False
        
    if passed:
        print("\n🏆 VEREDICTO: EL EDGE MICROESTRUCTURAL ES REAL. AUTORIZADO A FORJAR EN RUST.")
    else:
        print("\n💀 VEREDICTO: EL EDGE MURIÓ EN OUT-OF-SAMPLE. RUST DENEGADO.")

if __name__ == "__main__":
    run_experiment("UNIUSDT")
