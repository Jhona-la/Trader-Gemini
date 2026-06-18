import os
import sys
import numpy as np
import polars as pl
from numba import njit
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# 🚀 NUMBA CORE: PURE QUANTUM SCALPER (O(1) TICK-BY-TICK)
# ==============================================================================
@njit(fastmath=True, nogil=True)
def backtest_pure_scalper_numba(
    prices, volumes, is_buyer_maker, bid_qtys, ask_qtys,
    ema_window, z_obi, z_cvd, z_exit, sl_pct
):
    n = len(prices)
    
    # EMA Constants
    alpha = 2.0 / (ema_window + 1.0)
    
    # State Variables (O(1) updates)
    ema_cvd = 0.0
    ema_cvd_var = 0.0
    current_cvd = 0.0
    
    ema_obi = 0.0
    ema_obi_var = 0.0
    
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
        
        # O(1) CVD Z-Score
        diff_cvd = cvd_delta - ema_cvd
        ema_cvd += alpha * diff_cvd
        ema_cvd_var = (1.0 - alpha) * (ema_cvd_var + alpha * diff_cvd * diff_cvd)
        cvd_std = np.sqrt(ema_cvd_var)
        
        cvd_z_score = 0.0
        if cvd_std > 1e-9:
            cvd_z_score = (cvd_delta - ema_cvd) / cvd_std
            
        # 2. Update OBI
        total_liquidity = bid_qtys[i] + ask_qtys[i]
        current_obi = 0.5
        if total_liquidity > 0:
            current_obi = bid_qtys[i] / total_liquidity
            
        # O(1) OBI Z-Score
        diff_obi = current_obi - ema_obi
        ema_obi += alpha * diff_obi
        ema_obi_var = (1.0 - alpha) * (ema_obi_var + alpha * diff_obi * diff_obi)
        obi_std = np.sqrt(ema_obi_var)
        
        obi_z_score = 0.0
        if obi_std > 1e-9:
            obi_z_score = (current_obi - ema_obi) / obi_std
            
        if i < burn_in:
            continue
            
        # 3. Posición Abierta (Mantenimiento y Salida Leading)
        if in_position:
            pnl_pct = 0.0
            
            if direction == 1:
                if price > entry_price:
                    potential_ts = price * (1.0 - sl_pct)
                    if potential_ts > trailing_stop:
                        trailing_stop = potential_ts
                
                # Check Stop Loss / Trailing (Hard Exit)
                if price <= trailing_stop:
                    pnl_pct = (trailing_stop - entry_price) / entry_price
                    in_position = False
                    
                # LEADING EXIT: El muro de Bid se retira O la euforia compradora se agota
                elif obi_z_score < 0.0 or cvd_z_score > z_exit:
                    pnl_pct = (price - entry_price) / entry_price
                    in_position = False
                    
            else: # SHORT
                if price < entry_price:
                    potential_ts = price * (1.0 + sl_pct)
                    if potential_ts < trailing_stop:
                        trailing_stop = potential_ts
                        
                # Hard Exit
                if price >= trailing_stop:
                    pnl_pct = (entry_price - trailing_stop) / entry_price
                    in_position = False
                    
                # LEADING EXIT: Muro de Ask se retira O pánico vendedor se agota
                elif obi_z_score > 0.0 or cvd_z_score < -z_exit:
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
            
        # 4. Lógica de Fusión Pura OBI+CVD: Fricción sin Precio
        price_delta = price - prices[i-1]
        
        # A) ABSORCIÓN DE VENTAS (Muro pasivo alcista + Ataque agresivo bajista -> LONG)
        if obi_z_score > z_obi and cvd_z_score < -z_cvd and price_delta >= 0:
            in_position = True
            direction = 1
            entry_price = price
            trailing_stop = entry_price * (1.0 - sl_pct)
            continue
            
        # B) ABSORCIÓN DE COMPRAS (Muro pasivo bajista + Ataque agresivo alcista -> SHORT)
        if obi_z_score < -z_obi and cvd_z_score > z_cvd and price_delta <= 0:
            in_position = True
            direction = -1
            entry_price = price
            trailing_stop = entry_price * (1.0 + sl_pct)
            continue
            
    wr = (wins / trades) if trades > 0 else 0.0
    return total_pnl, trades, wr, max_dd

# ==============================================================================
# ESCUDO WALK-FORWARD (SCALPER PURO)
# ==============================================================================
def run_grid_search(prices, volumes, is_buyer_maker, bid_qtys, ask_qtys):
    best_pnl = -999.0
    best_params = None
    
    ema_windows = [100, 500, 1000] # Ticks (es mucho más rápido en HFT)
    z_obis = [1.5, 2.0, 3.0]
    z_cvds = [1.5, 2.0, 3.0]
    z_exits = [1.0, 1.5, 2.0]
    sls = [0.002, 0.005]
    
    # Pre-compilar Numba
    backtest_pure_scalper_numba(prices[:2000], volumes[:2000], is_buyer_maker[:2000], bid_qtys[:2000], ask_qtys[:2000], 100, 1.5, 1.5, 1.0, 0.005)
    
    print("🔍 Iniciando optimización en el Train Set (Fricción Pura OBI+CVD)...")
    total_iters = len(ema_windows) * len(z_obis) * len(z_cvds) * len(z_exits) * len(sls)
    count = 0
    for ema in ema_windows:
        for zo in z_obis:
            for zc in z_cvds:
                for ze in z_exits:
                    for sl in sls:
                        count += 1
                        pnl, trades, wr, max_dd = backtest_pure_scalper_numba(
                            prices, volumes, is_buyer_maker, bid_qtys, ask_qtys, ema, zo, zc, ze, sl
                        )
                        
                        if trades > 50 and pnl > best_pnl:
                            best_pnl = pnl
                            best_params = (ema, zo, zc, ze, sl)
                        
    return best_params, best_pnl

def run_experiment(symbol: str):
    print(f"\n=======================================================")
    print(f"🛡️ ESCUDO WALK-FORWARD PURE SCALPER: {symbol} (OBI+CVD)")
    print(f"=======================================================")
    
    parquet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history", f"{symbol}_merged_microstructure_7d.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"❌ Archivo no encontrado: {parquet_path}")
        return
        
    df = pl.read_parquet(parquet_path)
    
    # Castear strings a tipos reales
    if df['price'].dtype == pl.String:
        df = df.with_columns([
            pl.col('price').cast(pl.Float64),
            pl.col('volume').cast(pl.Float64),
            pl.col('is_buyer_maker').str.to_lowercase().eq("true")
        ])
        
    print(f"📊 Cargando {len(df):,} ticks individuales con BBO...")
    
    prices = df['price'].cast(pl.Float64).to_numpy()
    volumes = df['volume'].cast(pl.Float64).to_numpy()
    is_buyer_maker = df['is_buyer_maker'].cast(pl.Boolean).to_numpy()
    bid_qtys = df['bid_qty'].cast(pl.Float64).to_numpy()
    ask_qtys = df['ask_qty'].cast(pl.Float64).to_numpy()
    
    # 70/30 Split Cronológico
    split_idx = int(len(prices) * 0.70)
    
    p_train, v_train, bm_train, bq_train, aq_train = prices[:split_idx], volumes[:split_idx], is_buyer_maker[:split_idx], bid_qtys[:split_idx], ask_qtys[:split_idx]
    p_test, v_test, bm_test, bq_test, aq_test = prices[split_idx:], volumes[split_idx:], is_buyer_maker[split_idx:], bid_qtys[split_idx:], ask_qtys[split_idx:]
    
    print(f"\n[FASE I] Optimizando hiperparámetros adaptativos en {len(p_train):,} ticks (70%)...")
    t0 = time.time()
    best_params, best_pnl = run_grid_search(p_train, v_train, bm_train, bq_train, aq_train)
    t1 = time.time()
    
    if not best_params:
        print("❌ FAIL: Ningún set de parámetros logró rentabilidad estadística en Train.")
        return
        
    ema, zo, zc, ze, sl = best_params
    _, tr_trades, tr_wr, tr_dd = backtest_pure_scalper_numba(p_train, v_train, bm_train, bq_train, aq_train, ema, zo, zc, ze, sl)
    
    print(f"✅ TRAIN OPTIMAL (Tardó {t1-t0:.2f}s): PnL = {best_pnl*100:.2f}% | WR = {tr_wr*100:.2f}% | MaxDD = {tr_dd*100:.2f}% | Trades = {tr_trades}")
    print(f"   Parámetros: EMA={ema}, Z_OBI={zo}, Z_CVD={zc}, Z_EXIT={ze}, SL={sl*100:.2f}%")
    
    print(f"\n[FASE II] Evaluación Ciega (OOS) en {len(p_test):,} ticks (30%)...")
    
    oos_pnl, oos_trades, oos_wr, oos_dd = backtest_pure_scalper_numba(p_test, v_test, bm_test, bq_test, aq_test, ema, zo, zc, ze, sl)
    
    print(f"⚖️ OOS RESULTADO: PnL = {oos_pnl*100:.2f}% | WR = {oos_wr*100:.2f}% | MaxDD = {oos_dd*100:.2f}% | Trades = {oos_trades}")
    
    # Criterios inquebrantables
    passed = True
    if oos_wr < 0.60:
        print("❌ FAIL CRITERIO: Win Rate OOS < 60%")
        passed = False
    if oos_pnl <= 0:
        print("❌ FAIL CRITERIO: PnL OOS Negativo")
        passed = False
    if oos_dd > 0.05:
        print("❌ FAIL CRITERIO: Max Drawdown > 5%")
        passed = False
        
    if passed:
        print("\n🏆 VEREDICTO: EL SCALPER CUÁNTICO PURO ES REAL. AUTORIZADO A FORJAR EN RUST.")
    else:
        print("\n💀 VEREDICTO: EL EDGE DE ORDER FLOW ESTÁ ARBITRADO A ESTA FRECUENCIA. RUST DENEGADO.")

if __name__ == "__main__":
    run_experiment("UNIUSDT")
