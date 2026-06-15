#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 QUANTUM VECTOR-JIT BACKTEST ENGINE (FASE 5)
═══════════════════════════════════════════════════════════════════════════════
Ejecución masiva de Machine Learning y simulaciones HFT en milisegundos.
Evita el cuello de botella de Event-Driven y procesa matrices en memoria.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# ─── Project Root ───
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from scripts.vector_backtest.feature_engine import VectorFeatureEngine
from scripts.vector_backtest.ml_batch_infer import MLBatchInfer
from scripts.vector_backtest.numba_engine import NumbaEngine
from utils.logger import logger

def run_quantum_simulation():
    print("🚀 Iniciando Motor QUANTUM VECTOR-JIT (Santo Grial)")
    
    # Configuraciones de paridad
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'DOGE/USDT']
    days = 15
    interval = '1m'
    
    start_time = time.time()
    
    print(f"📥 Cargando datos desde cache_parquet...")
    
    all_data = {}
    cache_dir = "data/cache_parquet"
    if os.path.exists(cache_dir):
        for symbol in symbols:
            safe_sym = symbol.replace("/", "")
            matched_files = [f for f in os.listdir(cache_dir) if safe_sym in f and f.endswith(".parquet") and "vision" not in f]
            if matched_files:
                matched_files.sort(reverse=True)
                path = os.path.join(cache_dir, matched_files[0])
                all_data[symbol] = pd.read_parquet(path)
    
    total_pnl = 0.0
    total_trades = 0
    starting_cap = Config.INITIAL_CAPITAL
    
    for symbol in symbols:
        print(f"\n═══════════════════════════════════════════════════════════════")
        print(f"🧬 PROCESANDO SÍMBOLO: {symbol}")
        
        # Load raw OHLCV
        df_raw = all_data.get(symbol)
        if df_raw is None or len(df_raw) < 1000:
            print(f"⚠️ Insufficient data for {symbol}.")
            continue
            
        t0 = time.time()
        
        # 2. Vectorized Feature Engineering
        # Esto devuelve un Polars DF o Pandas DF con todo el RSI, MACD, etc.
        df_features_pl = VectorFeatureEngine.compute_all_features(df_raw, symbol)
        df_features = df_features_pl.to_pandas()
        
        t1 = time.time()
        
        # 3. Batch ML Inference (Dual Horizon)
        # Scalping
        ml_sigs_scalp, _ = MLBatchInfer.infer_all(df_features, symbol, Config, Config.MODEL_DIR, horizon="SCALPING")
        # Swing
        ml_sigs_swing, _ = MLBatchInfer.infer_all(df_features, symbol, Config, Config.MODEL_DIR, horizon="SWING")
        
        t2 = time.time()
        
        # 4. Vectorized Technical Rules (Dual Horizon)
        rsi = df_features.get('rsi_14', pd.Series(np.zeros(len(df_features)))).values
        macd = df_features.get('macd', pd.Series(np.zeros(len(df_features)))).values
        macd_signal = df_features.get('macd_signal', pd.Series(np.zeros(len(df_features)))).values
        atr_pct = df_features.get('atr_pct', pd.Series(np.zeros(len(df_features)))).values

        # Technical Signals (Scalping)
        tech_sigs_scalp = np.zeros(len(df_features), dtype=np.int8)
        tech_sigs_scalp[(rsi > 0) & (rsi < 40) & (macd > macd_signal)] = 1
        tech_sigs_scalp[(rsi > 60) & (macd < macd_signal)] = -1
        
        # Technical Signals (Swing) - Slower / More demanding
        tech_sigs_swing = np.zeros(len(df_features), dtype=np.int8)
        tech_sigs_swing[(rsi > 0) & (rsi < 30) & (macd > macd_signal)] = 1
        tech_sigs_swing[(rsi > 70) & (macd < macd_signal)] = -1
        
        # Macro Multiplier Mock (Simulando Risk On / Risk Off basado en volatilidad)
        macro_multiplier = np.ones(len(df_features), dtype=np.float32)
        macro_multiplier[atr_pct > 0.05] = 0.5  # Reduce aggressiveness in high vol
        macro_multiplier[atr_pct < 0.01] = 1.2  # Increase in low vol
        
        t3 = time.time()
        
        # 5. JIT C-Speed Simulation (Dual Horizon Integral)
        sim_config = {
            'initial_capital': starting_cap,
            'kelly_fraction': 0.19,
            'maker_fee': getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002),
            'taker_fee': getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.00075),
            'leverage_scalp': 50.0,
            'base_tp_scalp': 0.0076,
            'base_sl_scalp': 0.0162,
            'leverage_swing': 30.0,
            'base_tp_swing': 0.1732,
            'base_sl_swing': 0.0313
        }
        
        capital_curve, trades_pnl_scalp, trades_pnl_swing = NumbaEngine.run_simulation(
            df_features, 
            ml_sigs_scalp, tech_sigs_scalp, 
            ml_sigs_swing, tech_sigs_swing,
            macro_multiplier, sim_config
        )
        
        t4 = time.time()
        
        # Stats
        final_pnl_scalp = np.sum(trades_pnl_scalp)
        final_pnl_swing = np.sum(trades_pnl_swing)
        final_pnl = final_pnl_scalp + final_pnl_swing
        
        n_trades_scalp = len(trades_pnl_scalp)
        n_trades_swing = len(trades_pnl_swing)
        n_trades = n_trades_scalp + n_trades_swing
        
        wr_scalp = (np.sum(trades_pnl_scalp > 0) / n_trades_scalp * 100) if n_trades_scalp > 0 else 0
        wr_swing = (np.sum(trades_pnl_swing > 0) / n_trades_swing * 100) if n_trades_swing > 0 else 0
        
        total_pnl += final_pnl
        total_trades += n_trades
        
        print(f"⏱️  PERFORMANCE PROFILING ({len(df_features)} velas):")
        print(f"  - Feature Eng. (Polars) : {(t1-t0)*1000:.2f} ms")
        print(f"  - ML Batch (XGBoost)    : {(t2-t1)*1000:.2f} ms")
        print(f"  - Tech Signals (NumPy)  : {(t3-t2)*1000:.2f} ms")
        print(f"  - C-Speed Sim (Numba)   : {(t4-t3)*1000:.2f} ms")
        print(f"  =====================================")
        print(f"  🎯 TOTAL TIEMPO SÍMBOLO : {(t4-t0)*1000:.2f} ms")
        print(f"\n📊 RESULTADOS {symbol}:")
        print(f"  - Trades Scalp : {n_trades_scalp} (WR: {wr_scalp:.1f}%) -> PnL: ${final_pnl_scalp:.2f}")
        print(f"  - Trades Swing : {n_trades_swing} (WR: {wr_swing:.1f}%) -> PnL: ${final_pnl_swing:.2f}")
        print(f"  - Net PnL Total: ${final_pnl:.2f}")

    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f"🏆 GLOBAL QUANTUM RESULTS (15 DÍAS, {len(symbols)} SÍMBOLOS)")
    print(f"  - Tiempo Total Procesamiento : {time.time() - start_time:.2f} segundos")
    print(f"  - Total Trades (Escala HFT)  : {total_trades}")
    print(f"  - Capital Inicial            : ${starting_cap:.2f}")
    print(f"  - Net PnL Consolidado        : ${total_pnl:.2f}")
    print(f"  - ⚡ RETORNO TOTAL           : {(total_pnl/starting_cap)*100:.2f}%")
    print(f"═══════════════════════════════════════════════════════════════")

if __name__ == "__main__":
    run_quantum_simulation()
