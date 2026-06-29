import os
import sys
import json
import random
import argparse
import numpy as np
import joblib

# Añadir raíz al path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from scripts.run_mirror_backtest import run_mirror, fetch_binance_data

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestRegressor

def generate_random_config():
    """Generates a mutated configuration."""
    params = {
        "scalp_tp": round(random.uniform(0.005, 0.05), 4),
        "scalp_sl": round(random.uniform(0.005, 0.03), 4),
        "swing_tp": round(random.uniform(0.02, 0.15), 4),
        "swing_sl": round(random.uniform(0.01, 0.05), 4),
        "max_risk_pct": round(random.uniform(0.01, 0.10), 3),
        "leverage": random.choice([5, 10, 15, 20])
    }
    return params

def apply_config(params):
    Config.Horizons.Scalping['tp_pct'] = params["scalp_tp"]
    Config.Horizons.Scalping['sl_pct'] = params["scalp_sl"]
    Config.Horizons.Swing['tp_pct'] = params["swing_tp"]
    Config.Horizons.Swing['sl_pct'] = params["swing_sl"]
    Config.Risk.MAX_RISK_PER_TRADE_PCT = params["max_risk_pct"]
    Config.BINANCE_LEVERAGE = params["leverage"]

def run_training_pipeline(n_trials=100, days=7):
    print(f"🔮 [FASE 3A] Generando {n_trials} Espejos de Producción (Data Gen)...")
    
    # Pre-fetch data to memory so it doesn't do I/O on every run
    symbols = ["BTC/USDT"]
    for sym in symbols:
        fetch_binance_data(sym, days=days)
        
    dataset_X = []
    dataset_y = []
    
    for i in range(n_trials):
        params = generate_random_config()
        apply_config(params)
        
        print(f"  [{i+1}/{n_trials}] Ejecutando Universo {i}...")
        res = run_mirror(symbols, days=days)
        
        # Guardar feature vector
        x_vec = [
            params["scalp_tp"], params["scalp_sl"], 
            params["swing_tp"], params["swing_sl"],
            params["max_risk_pct"], params["leverage"]
        ]
        
        # Penalizamos TD infinitos (999) como 14 días para no distorsionar el modelo
        td_target = 14.0 if res["TD"] > 14.0 else res["TD"]
        
        y_vec = [
            td_target,
            res["Sharpe"],
            res["MaxDD"],
            res["PnL"]
        ]
        
        dataset_X.append(x_vec)
        dataset_y.append(y_vec)
        
    print("\n🧠 [FASE 3B] Entrenando Oráculo Surrogado...")
    X = np.array(dataset_X)
    Y = np.array(dataset_y)
    
    # Entrenar 4 regresores independientes para predecir las 4 métricas
    models = {}
    metric_names = ["TD", "Sharpe", "MaxDD", "PnL"]
    
    for idx, name in enumerate(metric_names):
        y_target = Y[:, idx]
        if HAS_XGB:
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            
        model.fit(X, y_target)
        models[name] = model
        print(f"  ✅ Modelo para {name} entrenado (Score R2 aprox: {model.score(X, y_target):.2f})")
        
    # Guardar oráculo
    os.makedirs(os.path.join(_project_root, ".models_backtest"), exist_ok=True)
    oracle_path = os.path.join(_project_root, ".models_backtest", "surrogate_oracle.pkl")
    joblib.dump(models, oracle_path)
    print(f"\n🔮 Oráculo guardado en {oracle_path}!")
    return oracle_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    
    run_training_pipeline(args.trials, args.days)
