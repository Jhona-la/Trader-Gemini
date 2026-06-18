import os
import sys
import numpy as np
import polars as pl
from numba import njit
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# 🚀 NUMBA CORE: ESTRUCTURAL SWING (CAZA DE LIQUIDACIONES)
# ==============================================================================
# Importar motor C++ puro
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from core.metal.macro_engine import run_macro_cpp
except ImportError:
    print("❌ Error: Compila core/metal/macro_engine.pyx antes de ejecutar (python setup_metal.py build_ext --inplace)")
    sys.exit(1)

# ==============================================================================
# ESCUDO WALK-FORWARD (MACRO)
# ==============================================================================
def run_macro_grid_search(t, o, h, l, c, v, oi, f):
    best_pnl = -999.0
    best_params = None
    
    lookbacks = [4, 8, 12] # 1h, 2h, 3h (en velas de 15m)
    price_drops = [0.03, 0.05, 0.08] # 3%, 5%, 8% drop
    vol_mults = [2.0, 3.0, 5.0]
    oi_drops = [0.02, 0.05, 0.10] # 2%, 5%, 10% OI drop
    sls = [0.01, 0.02]
    tps = [0.05, 0.10, 0.15]
    
    # Precompilar Numba
    backtest_macro_swing_numba(t[:100], o[:100], h[:100], l[:100], c[:100], v[:100], oi[:100], f[:100], 4, 0.03, 2.0, 0.05, 0.02, 0.05)
    
    print("🔍 Iniciando optimización MACRO en el Train Set (Derivados)...")
    for lb in lookbacks:
        for pd in price_drops:
            for vm in vol_mults:
                for oid in oi_drops:
                    for sl in sls:
                        for tp in tps:
                            pnl, trades, wr, pf, max_dd = backtest_macro_swing_numba(
                                t, o, h, l, c, v, oi, f,
                                lb, pd, vm, oid, sl, tp
                            )
                            
                            # Criterio mínimo para considerar un set de parámetros
                            if trades >= 5 and pnl > best_pnl and wr >= 0.40:
                                best_pnl = pnl
                                best_params = (lb, pd, vm, oid, sl, tp)
                                
    return best_params, best_pnl

def run_experiment(symbols: list):
    print(f"\n=======================================================")
    print(f"🛡️ ESCUDO WALK-FORWARD MACRO MULTI-ASSET (2 AÑOS)")
    print(f"=======================================================")
    
    # We will accumulate trades across all symbols
    all_tr_t, all_tr_o, all_tr_h, all_tr_l, all_tr_c, all_tr_v, all_tr_oi, all_tr_f = [], [], [], [], [], [], [], []
    all_ts_t, all_ts_o, all_ts_h, all_ts_l, all_ts_c, all_ts_v, all_ts_oi, all_ts_f = [], [], [], [], [], [], [], []
    
    total_samples = 0
    
    for symbol in symbols:
        parquet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history", "macro", f"{symbol}_merged_macro.parquet")
        if not os.path.exists(parquet_path):
            print(f"⚠️ Archivo no encontrado: {parquet_path}")
            continue
            
        df = pl.read_parquet(parquet_path)
        total_samples += len(df)
        
        t = df['timestamp'].to_numpy().astype(np.int64)
        o = df['open'].to_numpy().astype(np.float64)
        h = df['high'].to_numpy().astype(np.float64)
        l = df['low'].to_numpy().astype(np.float64)
        c = df['close'].to_numpy().astype(np.float64)
        v = df['volume'].to_numpy().astype(np.float64)
        oi = df['sum_open_interest'].to_numpy().astype(np.float64)
        f = df['funding_rate'].to_numpy().astype(np.float64)
        
        # 70/30 Split Cronológico por moneda
        split_idx = int(len(t) * 0.70)
        all_tr_t.append(t[:split_idx])
        all_tr_o.append(o[:split_idx])
        all_tr_h.append(h[:split_idx])
        all_tr_l.append(l[:split_idx])
        all_tr_c.append(c[:split_idx])
        all_tr_v.append(v[:split_idx])
        all_tr_oi.append(oi[:split_idx])
        all_tr_f.append(f[:split_idx])
        
        all_ts_t.append(t[split_idx:])
        all_ts_o.append(o[split_idx:])
        all_ts_h.append(h[split_idx:])
        all_ts_l.append(l[split_idx:])
        all_ts_c.append(c[split_idx:])
        all_ts_v.append(v[split_idx:])
        all_ts_oi.append(oi[split_idx:])
        all_ts_f.append(f[split_idx:])
        
    print(f"📊 Cargadas {total_samples:,} velas de 15m para {len(symbols)} activos...")
    
    # We will search for a set of params that is robust ACROSS all assets.
    # To do this in Numba, we can wrap the evaluation.
    
    def eval_portfolio(lookbacks, price_drops, vol_mults, oi_drops, sls, tps, is_train=True):
        best_pnl = -999.0
        best_params = None
        
        for lb in lookbacks:
            for pd in price_drops:
                for vm in vol_mults:
                    for oid in oi_drops:
                        for sl in sls:
                            for tp in tps:
                                port_pnl = 0.0
                                port_trades = 0
                                port_wins = 0
                                
                                # Evaluate across all symbols
                                for i in range(len(all_tr_t)):
                                    _t = all_tr_t[i] if is_train else all_ts_t[i]
                                    _o = all_tr_o[i] if is_train else all_ts_o[i]
                                    _h = all_tr_h[i] if is_train else all_ts_h[i]
                                    _l = all_tr_l[i] if is_train else all_ts_l[i]
                                    _c = all_tr_c[i] if is_train else all_ts_c[i]
                                    _v = all_tr_v[i] if is_train else all_ts_v[i]
                                    _oi = all_tr_oi[i] if is_train else all_ts_oi[i]
                                    _f = all_tr_f[i] if is_train else all_ts_f[i]
                                    
                                    # Llamada C++
                                    pnl, trades, wr, pf, max_dd = run_macro_cpp(
                                        _t, _o, _h, _l, _c, _v, _oi, _f,
                                        lb, pd, vm, oid, sl, tp
                                    )
                                    port_pnl += pnl
                                    port_trades += trades
                                    port_wins += int(wr * trades)
                                    
                                port_wr = port_wins / port_trades if port_trades > 0 else 0.0
                                
                                if is_train:
                                    if port_trades >= 30 and port_pnl > best_pnl and port_wr >= 0.55:
                                        best_pnl = port_pnl
                                        best_params = (lb, pd, vm, oid, sl, tp)
                                else:
                                    return port_pnl, port_trades, port_wr
                                    
        return best_params, best_pnl

    print(f"\n[FASE I] Optimizando hiperparámetros de Portafolio en In-Sample (70%)...")
    t0 = time.time()
    
    lookbacks = [4, 8, 12, 16]
    price_drops = [0.03, 0.05, 0.08]
    vol_mults = [2.0, 3.0, 5.0]
    oi_drops = [0.02, 0.05, 0.10]
    sls = [0.02, 0.03]
    tps = [0.05, 0.08, 0.15]
    
    best_params, best_pnl = eval_portfolio(lookbacks, price_drops, vol_mults, oi_drops, sls, tps, is_train=True)
    t1 = time.time()
    
    if not best_params:
        print("❌ FAIL: Ningún set de parámetros logró rentabilidad estable (WR>55%, n>30) transversalmente.")
        return
        
    lb, pd, vm, oid, sl, tp = best_params
    print(f"✅ TRAIN OPTIMAL (Tardó {t1-t0:.2f}s): PnL acumulado = {best_pnl*100:.2f}%")
    print(f"   Parámetros: V_Window={lb}, Drop={pd*100}%, Vol_Mult={vm}x, OI_Drop={oid*100}%, SL={sl*100}%, TP={tp*100}%")
    
    print(f"\n[FASE II] Evaluación Ciega (OOS) de Portafolio (30%)...")
    oos_pnl, oos_trades, oos_wr = eval_portfolio([lb], [pd], [vm], [oid], [sl], [tp], is_train=False)
    
    print(f"⚖️ OOS RESULTADO MULTI-ASSET: PnL = {oos_pnl*100:.2f}% | WR = {oos_wr*100:.2f}% | Trades = {oos_trades}")
    
    passed = True
    if oos_wr < 0.55:
        print("❌ FAIL CRITERIO: Win Rate OOS < 55%")
        passed = False
    if oos_pnl <= 0:
        print("❌ FAIL CRITERIO: PnL OOS Negativo")
        passed = False
    if oos_trades < 10:
        print("❌ FAIL CRITERIO: Insuficiente N de Trades en OOS")
        passed = False
        
    if passed:
        print("\n🏆 VEREDICTO ABSOLUTO: EL SWING ESTRUCTURAL SOBREVIVE LA PRUEBA DE 2 AÑOS MULTI-ASSET Y SLIPPAGE ASIMÉTRICO.")
    else:
        print("\n💀 VEREDICTO: EL EDGE FUE OVERFITTING. COLAPSÓ BAJO EL ESCUDO MULTI-RÉGIMEN.")

if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "UNIUSDT"]
    run_experiment(symbols)
