import os
import sys
import json
import optuna
import joblib
import numpy as np
import argparse

# Añadir raíz al path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.run_mirror_backtest import run_mirror
from config import Config

def load_oracle():
    oracle_path = os.path.join(_project_root, ".models_backtest", "surrogate_oracle.pkl")
    if not os.path.exists(oracle_path):
        print(f"❌ Error: No se encuentra el Oráculo en {oracle_path}. Entrénalo primero.")
        sys.exit(1)
    return joblib.load(oracle_path)

def objective(trial, models):
    # Generar hiperparámetros de prueba
    scalp_tp = trial.suggest_float("scalp_tp", 0.005, 0.05, step=0.005)
    scalp_sl = trial.suggest_float("scalp_sl", 0.005, 0.03, step=0.005)
    swing_tp = trial.suggest_float("swing_tp", 0.02, 0.15, step=0.01)
    swing_sl = trial.suggest_float("swing_sl", 0.01, 0.05, step=0.01)
    max_risk_pct = trial.suggest_float("max_risk_pct", 0.01, 0.10, step=0.01)
    leverage = trial.suggest_categorical("leverage", [5, 10, 15, 20])
    
    # Predecir con el Oráculo (O(1) ms en vez de N minutos)
    x_vec = np.array([[scalp_tp, scalp_sl, swing_tp, swing_sl, max_risk_pct, leverage]])
    
    td_pred = models["TD"].predict(x_vec)[0]
    sharpe_pred = models["Sharpe"].predict(x_vec)[0]
    maxdd_pred = models["MaxDD"].predict(x_vec)[0]
    pnl_pred = models["PnL"].predict(x_vec)[0]
    
    # ─── FUNCIÓN OBJETIVO ───
    # Queremos MINIMIZAR el TD (T_duplicacion).
    # Optuna por defecto minimiza. Si queremos minimizar TD, retornamos TD.
    # Penalizamos duramente configuraciones inseguras.
    
    penalty = 0.0
    if sharpe_pred < 1.0:
        penalty += 100.0
    if maxdd_pred > 0.20:
        penalty += (maxdd_pred - 0.20) * 1000.0
        
    final_score = td_pred + penalty
    
    # Guardar métricas en el trial para inspección
    trial.set_user_attr("Sharpe", float(sharpe_pred))
    trial.set_user_attr("MaxDD", float(maxdd_pred))
    trial.set_user_attr("PnL", float(pnl_pred))
    
    return float(final_score)

def run_collapse(n_trials=10000, validation_top_k=3, days=7):
    print("🧠 Cargando Oráculo Surrogado...")
    models = load_oracle()
    
    print(f"🌌 [FASE 3C] Colapsando {n_trials} Universos en la mente del Oráculo...")
    # TPE Sampler por defecto
    study = optuna.create_study(direction="minimize", study_name="UniverseCollapse")
    
    # Optimizamos pasándole los modelos
    study.optimize(lambda t: objective(t, models), n_trials=n_trials, n_jobs=-1)
    
    print("\n" + "="*50)
    print("🏆 UNIVERSOS EVALUADOS CON ÉXITO")
    print("="*50)
    
    best_trials = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], 
                         key=lambda t: t.value)[:validation_top_k]
                         
    print(f"\n🔍 [FASE 3D] Validación Final de los Top {len(best_trials)} en la Realidad Absoluta:")
    
    for rank, trial in enumerate(best_trials):
        params = trial.params
        print(f"\n🥇 Rank #{rank+1} (Predicción del Oráculo: TD={trial.value:.2f} días | Sharpe={trial.user_attrs['Sharpe']:.2f})")
        print(f"   Parámetros: {params}")
        
        # Aplicamos la configuración
        Config.Horizons.Scalping['tp_pct'] = params["scalp_tp"]
        Config.Horizons.Scalping['sl_pct'] = params["scalp_sl"]
        Config.Horizons.Swing['tp_pct'] = params["swing_tp"]
        Config.Horizons.Swing['sl_pct'] = params["swing_sl"]
        Config.Risk.MAX_RISK_PER_TRADE_PCT = params["max_risk_pct"]
        Config.BINANCE_LEVERAGE = params["leverage"]
        
        # Ejecutamos el Espejo Real
        res = run_mirror(["BTC/USDT"], days=days)
        
        # Comparamos
        td_real = res["TD"]
        td_pred = trial.value
        error_pct = abs(td_real - td_pred) / max(td_pred, 0.01) * 100
        
        print("   ✅ [REALIDAD] vs [ORÁCULO]")
        print(f"   TD (Días):      {td_real:.2f}   vs   {td_pred:.2f}  (Error: {error_pct:.1f}%)")
        print(f"   Sharpe:         {res['Sharpe']:.2f}   vs   {trial.user_attrs['Sharpe']:.2f}")
        print(f"   MaxDD:          {res['MaxDD']*100:.2f}% vs   {trial.user_attrs['MaxDD']*100:.2f}%")
        print(f"   PnL Bruto:      ${res['PnL']:.2f} vs   ${trial.user_attrs['PnL']:.2f}")
        
        if td_real <= 3.0:
            print("   🔥🔥 LA SINGULARIDAD SE HA CONFIRMADO. OBJETIVO DE 3 DÍAS LOGRADO. 🔥🔥")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    
    run_collapse(args.trials, args.top_k, args.days)
