import os
import sys
import optuna
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier, XGBRegressor

# Ensure root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from strategies.components.feature_engineering import FeatureEngineering
from scripts.run_god_mode_backtest import run_global_backtest
from core.backtest_infra import fetch_binance_data, fetch_multi_symbol_data

# =================================================================================
# 🔮 PHASE IV: SUPREME SURROGATE OPTIMIZATION (DARK ALPHA)
# QUÉ: Entrenamos XGBoost con Hiperparámetros dictados por Optuna (Surrogado TPE)
#      y lo evaluamos *directamente* en el God Mode Backtest.
# POR QUÉ: No nos importa el F1-Score clásico. Queremos Win Rate = 100% en el
#      simulador real (con slippage, comisiones y kill switch).
# PARA QUÉ: Romper el techo de 85% de Confianza para forzar al modelo a detectar
#      Dark Alpha (Net Pressure, L2 OFI).
# CÓMO: 
#   1. Optuna sugiere params (max_depth, lr, gamma, etc.)
#   2. Entrenamos modelos XGBoost con esos params
#   3. Ejecutamos el Backtest de 1-7 días
#   4. Recompensamos fuertemente WR > 99% y PnL positivo.
# =================================================================================

def train_custom_xgboost(df_features, feature_cols, symbol, horizon, models_dir, params):
    """
    Entrena los modelos con los hiperparámetros mutados por Optuna.
    """
    closes = df_features['close'].to_numpy().astype(np.float32)
    highs = df_features['high'].to_numpy().astype(np.float32)
    lows = df_features['low'].to_numpy().astype(np.float32)
    
    lookahead = 15 if horizon == 'SCALPING' else 60
    
    N = len(closes)
    y_cls = np.zeros(N, dtype=int)
    y_reg_long = np.zeros(N, dtype=np.float32)
    y_reg_short = np.zeros(N, dtype=np.float32)
    y_next_high_pct = np.zeros(N, dtype=np.float32)
    y_next_low_pct = np.zeros(N, dtype=np.float32)
    y_time_to_peak = np.zeros(N, dtype=np.float32)
    
    for i in range(N - lookahead):
        window_highs = highs[i+1 : i+lookahead+1]
        window_lows = lows[i+1 : i+lookahead+1]
        
        y_cls[i] = int(closes[i+lookahead] > closes[i])
        max_high = np.max(window_highs)
        min_low = np.min(window_lows)
        y_reg_long[i] = max(0.0, (max_high - closes[i]) / closes[i])
        y_reg_short[i] = max(0.0, (closes[i] - min_low) / closes[i])
        y_next_high_pct[i] = (max_high - closes[i]) / closes[i]
        y_next_low_pct[i] = (min_low - closes[i]) / closes[i]
        peak_bar_long = np.argmax(window_highs)
        y_time_to_peak[i] = peak_bar_long / max(1, lookahead)
        
    X = df_features.select(feature_cols).to_numpy()[:-lookahead]
    y_cls = y_cls[:-lookahead]
    y_reg_long = y_reg_long[:-lookahead]
    y_reg_short = y_reg_short[:-lookahead]
    y_next_high_pct = y_next_high_pct[:-lookahead]
    y_next_low_pct = y_next_low_pct[:-lookahead]
    y_time_to_peak = y_time_to_peak[:-lookahead]
    
    if len(X) < 100:
        return False
        
    # Apply Optuna params
    md = params['max_depth']
    lr = params['learning_rate']
    ne = params['n_estimators']
    gamma = params['gamma']
    scale_pos_weight = params['scale_pos_weight']
    
    clf_model = XGBClassifier(n_estimators=ne, learning_rate=lr, max_depth=md, gamma=gamma,
                               scale_pos_weight=scale_pos_weight, eval_metric='logloss', n_jobs=8)
    reg_long_model = XGBRegressor(n_estimators=ne, learning_rate=lr, max_depth=md, gamma=gamma, n_jobs=8)
    reg_short_model = XGBRegressor(n_estimators=ne, learning_rate=lr, max_depth=md, gamma=gamma, n_jobs=8)
    reg_next_high = XGBRegressor(n_estimators=ne, learning_rate=lr, max_depth=md, gamma=gamma, n_jobs=8)
    reg_next_low = XGBRegressor(n_estimators=ne, learning_rate=lr, max_depth=md, gamma=gamma, n_jobs=8)
    reg_ttp = XGBRegressor(n_estimators=ne, learning_rate=lr, max_depth=md, gamma=gamma, n_jobs=8)
    
    clf_model.fit(X, y_cls)
    reg_long_model.fit(X, y_reg_long)
    reg_short_model.fit(X, y_reg_short)
    reg_next_high.fit(X, y_next_high_pct)
    reg_next_low.fit(X, y_next_low_pct)
    reg_ttp.fit(X, y_time_to_peak)
    
    safe_sym = symbol.replace('/', '')
    suffix = "_scalping" if horizon == 'SCALPING' else "_swing"
    
    os.makedirs(models_dir, exist_ok=True)
    clf_model.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb.ubj"))
    reg_long_model.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_long.ubj"))
    reg_short_model.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_short.ubj"))
    reg_next_high.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_next_high.ubj"))
    reg_next_low.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_next_low.ubj"))
    reg_ttp.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_ttp.ubj"))
    
    return True

def run_surrogate_pipeline(n_trials=50, symbols=None, days=2):
    if symbols is None:
        symbols = ["BTC/USDT"]

    print(f"\n🔮 [DARK ALPHA] Iniciando Pipeline Surrogado de Evolución")
    print(f"Símbolos: {symbols} | Días de Backtest: {days} | Trials: {n_trials}")
    
    models_dir = os.path.join(_project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n📦 [1/3] Descargando y Preparando Features (Zero-Copy)...")
    all_data = fetch_multi_symbol_data(symbols, days=days)
    
    # Pre-calcular features para no repetir en cada trial
    precalculated_features = {}
    for sym in symbols:
        df = all_data.get(sym)
        if df is not None and not df.empty:
            fe = FeatureEngineering()
            df_feat = fe.prepare_features(df.copy(), symbol=sym).drop_nulls()
            feature_cols = [c for c in df_feat.columns if c not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            precalculated_features[sym] = (df_feat, feature_cols)
    
    def objective(trial):
        # 1. Mutación de ADN (Hiperparámetros XGBoost)
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
        }
        
        # Opcional: También mutar parámetros del Risk Manager en la config
        Config.Risk.SOPHIA_CERTAINTY_FLOOR = trial.suggest_float('sophia_floor', 0.70, 0.95)
        Config.Horizons.Scalping['tp_pct'] = trial.suggest_float('scalp_tp', 0.005, 0.02)
        Config.Horizons.Scalping['sl_pct'] = trial.suggest_float('scalp_sl', 0.002, 0.01)
        
        print(f"\n🧬 [TRIAL {trial.number}] Mutando ADN → {params} | Floor: {Config.Risk.SOPHIA_CERTAINTY_FLOOR:.2f}")
        
        # 2. Entrenar modelos
        for sym, (df_feat, fcols) in precalculated_features.items():
            train_custom_xgboost(df_feat, fcols, sym, 'SCALPING', models_dir, params)
            train_custom_xgboost(df_feat, fcols, sym, 'SWING', models_dir, params)
            
        # 3. Ejecutar Backtest Reales (15 segundos)
        try:
            res = run_global_backtest(all_data, symbols, days=days, verbose=False, isolated_strategy="ml")
            metrics = res["metrics"]
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return -999.0
            
        pnl = metrics['Net PnL %']
        win_rate = metrics['Win Rate %']
        trades = metrics['Total Trades']
        
        print(f"  🏁 RESULT: Trades: {trades} | WR: {win_rate:.1f}% | PnL: {pnl:.2f}%")
        
        if trades < 3:
            return -50.0  # Castigo por inactividad
            
        # 4. Función de Recompensa Cuántica
        # Queremos maximizar PnL pero con una condición ASESINA de Win Rate > 99%
        wr_penalty = 0
        if win_rate < 99.0:
            wr_penalty = (99.0 - win_rate) * 5.0  # Fuerte castigo por perder un solo trade
            
        fitness = pnl - wr_penalty + (trades * 0.1)
        
        return fitness
        
    study = optuna.create_study(direction='maximize', study_name="Dark_Alpha_Surrogate")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n🏆 [SURROGATE OPTIMIZATION COMPLETE]")
    print(f"  Best Trial: {study.best_trial.number}")
    print(f"  Best Fitness: {study.best_trial.value:.2f}")
    print(f"  Best Params: {study.best_trial.params}")
    
    # Save the best parameters to a JSON file for the C-Engine and Python config
    best_params_path = os.path.join(_project_root, "models", "supreme_surrogate_best.json")
    with open(best_params_path, "w") as f:
        import json
        json.dump(study.best_trial.params, f, indent=4)
        
    print(f"💾 Guardado en {best_params_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--symbols", type=str, default="BTC/USDT")
    args = parser.parse_args()
    
    syms = [s.strip() for s in args.symbols.split(",")]
    run_surrogate_pipeline(n_trials=args.trials, symbols=syms, days=args.days)
