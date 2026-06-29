#!/usr/bin/env python3
"""
🚀 MASSIVE VECTORIZED OPTIMIZATION ENGINE - MULTI-ASSET TRAILING SCALPING
Grid Search de Nanosegundos para Trader Gemini.
Ejecuta millones de simulaciones para encontrar los umbrales exactos
y el activo correcto que permitan llevar $13 USD a $26 USD en 15 días con un alto WinRate.
"""

import os
import sys
import time
import numpy as np
import polars as pl
import asyncio
import gc

# Configuración Numba/CPU para maximizar hilos
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["NUMBA_NUM_THREADS"] = "16"

# Root path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from core.vectorized_backtest import quantum_grid_search_core
from strategies.components.feature_engineering import FeatureEngineering
from utils.logger import logger

def generate_parameter_grid():
    pass

async def async_main():
    print("🚀 Iniciando Motor de Optimización Cuántica Multi-Activo...")
    
    # ---------------------------------------------------------
    # PARÁMETROS CUÁNTICOS: EXPANSIÓN A MILLONES DE COMBINACIONES
    # ---------------------------------------------------------
    ml_thresholds = np.linspace(0.60, 0.95, 8) # 8 steps
    tech_thresholds = np.linspace(0.50, 0.80, 5) # 5 steps
    vol_thresholds = np.linspace(0.4, 0.9, 4) # 4 steps
    sl_pcts = np.linspace(0.01, 0.15, 6) # 6 steps
    tp_pcts = np.linspace(0.0005, 0.008, 6) # 6 steps
    trail_acts = np.linspace(0.001, 0.005, 4) # 4 steps
    trail_dists = np.linspace(0.0002, 0.002, 4) # 4 steps
    max_holds = np.array([10, 30, 60, 100]) # 4 steps
    strategy_types = np.array([0, 1]) # 0 = Mean Reversion, 1 = Breakout
    # Total combinations = 8 * 5 * 4 * 6 * 6 * 4 * 4 * 4 * 2 = 737,280 por moneda
    # Por 26 monedas = 19.1 millones de configuraciones totales evaluadas.
    
    # Construir grilla usando np.meshgrid (Nueve dimensiones)
    mesh = np.array(np.meshgrid(
        ml_thresholds, tech_thresholds, vol_thresholds, sl_pcts, tp_pcts, trail_acts, trail_dists, max_holds, strategy_types
    )).T.reshape(-1, 9)
    
    params_grid = np.require(mesh, dtype=np.float64, requirements=['C'])
    n_combos = params_grid.shape[0]
    
    print(f"🧠 Generadas {n_combos:,} combinaciones de hiperparámetros.")
    
    global_results = []
    
    global_results = []
    
    import ccxt
    client = ccxt.binance({'enableRateLimit': True})
    
    # Use Config.Strategies.ASSETS if available, otherwise fallback
    assets_to_scan = getattr(Config.Strategies, 'ASSETS', ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    # Escanear el Top 10 para buscar la mayor eficiencia en el Interés Compuesto
    assets_to_scan = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT"
    ]
    
    print(f"🌍 Escaneando {len(assets_to_scan)} activos de forma secuencial (Protegiendo RAM)...")
    
    for symbol in assets_to_scan:
        print(f"\n==================================================")
        print(f"📥 ANALIZANDO ACTIVO: {symbol}")
        print(f"==================================================")
        
        try:
            import time
            since = int((time.time() - 30 * 24 * 3600) * 1000) # 30 days ago
            # CCXT fetch_ohlcv returns: [timestamp, open, high, low, close, volume]
            # Binance limits to 1000 candles per request, so we might need a loop or just get 1000 for quick massive test
            klines = client.fetch_ohlcv(symbol, '5m', since=since, limit=1000)
            
            if not klines or len(klines) < 500:
                print(f"❌ Datos insuficientes para {symbol}.")
                continue
                
            print(f"✅ {len(klines)} velas descargadas. Procesando Features...")
            
            # 2. Convertir a Polars para Feature Engineering
            import pandas as pd
            # Format ccxt data: [timestamp, open, high, low, close, volume]
            df_pd = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_pd[col] = df_pd[col].astype(float)
            df_pd['buy_volume'] = df_pd['volume'] * 0.5 # proxy since ccxt standard OHLCV doesn't have taker buy volume easily in 1 call
                
            df_pl = pl.from_pandas(df_pd)
            df_pl = df_pl.with_columns(pl.lit(symbol).alias('symbol'))
            
            # 3. Extraer Features
            fe = FeatureEngineering()
            df_features = fe.prepare_features(df_pl)
            
            # Extraer columnas como Numpy arrays
            close_prices = np.require(df_features['close'].to_numpy(), dtype=np.float64, requirements=['C'])
            high_prices = np.require(df_features['high'].to_numpy(), dtype=np.float64, requirements=['C'])
            low_prices = np.require(df_features['low'].to_numpy(), dtype=np.float64, requirements=['C'])
            
            # RANSAC Volatility
            if 'volatility_regime' in df_features.columns:
                vol_ratios = np.require(df_features['volatility_regime'].to_numpy() + 0.5, dtype=np.float64, requirements=['C'])
            else:
                vol_ratios = np.ones_like(close_prices)
                
            # ---------------------------------------------------------
            # FASE 2: Inferencia Batch con Modelo XGBoost Real
            # ---------------------------------------------------------
            from strategies.ml_strategy import UniversalEnsembleStrategy
            import xgboost as xgb
            
            # Instanciar estrategia solo para cargar el modelo
            strategy = UniversalEnsembleStrategy(data_provider=None, events_queue=None, symbol=symbol)
            strategy.horizon_str = "SCALPING"
            strategy._load_models()
            
            if strategy.xgb_model:
                expected_cols = strategy._feature_cols
                
                # Fill missing columns with 0
                for col in expected_cols:
                    if col not in df_features.columns:
                        if hasattr(df_features, 'with_columns'):
                            df_features = df_features.with_columns(pl.lit(0.0).alias(col))
                        else:
                            df_features[col] = 0.0
                
                if hasattr(df_features, 'select'):
                    X_df = df_features.select(expected_cols)
                else:
                    X_df = df_features[expected_cols]
                    
                X = np.require(X_df.to_numpy(), dtype=np.float32, requirements=['C'])
                
                booster = strategy.xgb_model.get_booster() if hasattr(strategy.xgb_model, 'get_booster') else strategy.xgb_model
                raw_ml_preds = booster.inplace_predict(X)
                
                if raw_ml_preds.ndim > 1 and raw_ml_preds.shape[1] > 1:
                    ml_scores = np.require(raw_ml_preds[:, 1], dtype=np.float64, requirements=['C'])
                else:
                    ml_scores = np.require(raw_ml_preds, dtype=np.float64, requirements=['C'])
                    
                from utils.math_kernel import calculate_rsi_jit
                raw_rsi = calculate_rsi_jit(close_prices, 14)
                tech_scores = np.require(raw_rsi / 100.0, dtype=np.float64, requirements=['C'])
            else:
                print(f"⚠️ HACIENDO FALLBACK AL PROXY RSI para {symbol}")
                ml_scores = np.full_like(close_prices, 0.6)
                tech_scores = np.full_like(close_prices, 0.5)

            # Compilación JIT Warmup (pequeño batch)
            _ = quantum_grid_search_core(
                close_prices[:100], high_prices[:100], low_prices[:100],
                ml_scores[:100], tech_scores[:100], vol_ratios[:100],
                params_grid[:2]
            )
            
            # 5. EJECUCIÓN MASIVA
            t0 = time.time()
            results = quantum_grid_search_core(
                close_prices, high_prices, low_prices,
                ml_scores, tech_scores, vol_ratios,
                params_grid
            )
            t1 = time.time()
            
            print(f"⏱️ Vectorizado en {t1-t0:.4f}s ({(n_combos * len(close_prices)) / (t1-t0):,.0f} velas/s)")
            
            # Evaluar Resultados y agregar a lista global
            for c in range(n_combos):
                pnl = results[c, 0]
                wr = results[c, 1]
                trades = results[c, 2]
                dd = results[c, 3]
                
                if trades > 5:
                    global_results.append({
                        'symbol': symbol,
                        'pnl': pnl, 'wr': wr, 'trades': trades, 'dd': dd,
                        'params': params_grid[c]
                    })
                    
        except Exception as e:
            print(f"❌ Error procesando {symbol}: {e}")
            
        finally:
            # Forzar liberación de memoria por cada activo pesado
            gc.collect()

    # Ordenar Ranking Global
    global_results.sort(key=lambda x: x['pnl'], reverse=True)
    
    print("\n" + "="*60)
    print("🏆 GLOBAL MULTI-ASSET LEADERBOARD - TRAILING SCALPING")
    print("="*60)
    
    if len(global_results) > 0:
        for i, res in enumerate(global_results[:20]): # Top 20 across all assets
            strat_name = "Breakout" if res['params'][8] == 1 else "Mean Reversion"
            print(f"\n--- RANK #{i+1} | {res['symbol']} ---")
            print(f"   Strategy:       {strat_name}")
            print(f"   ML Threshold:   {res['params'][0]:.2f}")
            print(f"   Tech/RSI Thr:   {res['params'][1]:.2f}")
            print(f"   Stop Loss:      {res['params'][3]*100:.2f}%")
            print(f"   Take Profit:    {res['params'][4]*100:.2f}%")
            print(f"   Trail Activa:   {res['params'][5]*100:.2f}%")
            print(f"   Trail Dist:     {res['params'][6]*100:.2f}%")
            print(f"   Max Hold Bars:  {res['params'][7]:.0f}")
            print(f"   💰 Net PnL:     {res['pnl']*100:.2f}%")
            print(f"   🎯 Win Rate:    {res['wr']*100:.2f}%")
            print(f"   📉 Max DD:      {res['dd']*100:.2f}%")
            print(f"   🔁 Trades:      {res['trades']:.0f}")
    else:
        print("\n⚠️ Ninguna configuración logró ser rentable o procesada.")

if __name__ == "__main__":
    asyncio.run(async_main())
