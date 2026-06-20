#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 HYPER-EVOLVER: MASSIVE SYSTEM-WIDE OPTIMIZATION VIA GENETIC TPE (OPTUNA)
═══════════════════════════════════════════════════════════════════════════════

QUÉ: Orquestador evolutivo masivo. Usa algoritmos genéticos para mutar el
     ADN del sistema y encontrar los parámetros de supervivencia (El Santo Grial).
POR QUÉ: Resolver a fuerza bruta 1000 variables tomaría 950 años. 
PARA QUÉ: Lograr un interés compuesto exponencial en microcuentas de $13 USD
     maximizando el Win Rate y minimizando el Drawdown a < 15%.
CÓMO: Optuna (TPE) -> Subprocess -> run_god_mode_backtest.py -> Fitness Score.
"""

import optuna
import subprocess
import json
import os
import sys
import uuid
import time
from datetime import datetime

# ─── CONFIGURATION ───
DAYS = 1
SYMBOLS = "ALL"
CAPITAL = 13.0
N_TRIALS = 2
STUDY_NAME = "hyper_evolver_micro_compounding"
RESULTS_DIR = "archive/hyper_evolver"

os.makedirs(RESULTS_DIR, exist_ok=True)

def objective(trial):
    run_id = f"gen_{trial.number}_{uuid.uuid4().hex[:6]}"
    output_json = os.path.join(RESULTS_DIR, f"{run_id}.json")
    
    # ─── 1. MUTAR EL ADN (Definir el Genoma) ───
    # Optimizamos a nivel arquitectónico y táctico
    
    # A. Tácticas de Scalping y Microscalping
    m_rsi_buy = trial.suggest_int('micro_rsi_buy', 15, 35)
    m_rsi_sell = trial.suggest_int('micro_rsi_sell', 65, 85)
    m_strength = trial.suggest_float('micro_ml_strength', 0.45, 0.65)
    m_vol_ratio = trial.suggest_float('micro_vol_ratio', 0.5, 1.5)
    m_tp = trial.suggest_float('micro_tp_default', 0.002, 0.006)
    m_sl = trial.suggest_float('micro_sl_default', 0.001, 0.004)
    m_consensus = trial.suggest_float('micro_consensus_gate', 1.0, 3.0)
    
    s_rsi_buy = trial.suggest_int('scalp_rsi_buy', 25, 45)
    s_rsi_sell = trial.suggest_int('scalp_rsi_sell', 55, 75)
    s_strength = trial.suggest_float('scalp_ml_strength', 0.40, 0.60)
    
    # B. Asignación de Capital y Riesgo Dinámico
    kelly_fraction = trial.suggest_float('max_risk_per_trade', 0.05, 0.25)
    
    # Construcción del diccionario Overrides (Recursivo)
    overrides = {
        "max_risk_per_trade": kelly_fraction,
        "Horizons": {
            "Microscalping": {
                "rsi_buy": m_rsi_buy,
                "rsi_sell": m_rsi_sell,
                "strength_threshold": m_strength,
                "min_volume_ratio": m_vol_ratio,
                "tp_pct": m_tp,
                "sl_pct": m_sl,
                "consensus_gate_mult": m_consensus
            },
            "Scalping": {
                "rsi_buy": s_rsi_buy,
                "rsi_sell": s_rsi_sell,
                "strength_threshold": s_strength
            }
        }
    }
    
    overrides_str = json.dumps(overrides)
    
    # ─── 2. EJECUTAR EL SIMULADOR DE PRODUCCIÓN ───
    print(f"\n🧬 [Generación {trial.number}] Mutando ADN... Ejecutando Universo {run_id}")
    cmd = [
        sys.executable,
        "scripts/run_god_mode_backtest.py",
        "--days", str(DAYS),
        "--symbols", SYMBOLS,
        "--capital", str(CAPITAL),
        "--output", output_json,
        "--quiet",  # Suprimir el ruido del backtest interno
        "--override", overrides_str
    ]
    
    start_time = time.time()
    try:
        # Ejecutamos en un proceso aislado para evitar fugas de memoria y contaminación de estado
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=1200) # 20 mins max per backtest
        
        if process.returncode != 0:
            print(f"❌ Simulación falló (Return Code {process.returncode}). Error: {process.stderr[-200:]}")
            raise optuna.exceptions.TrialPruned()
            
    except subprocess.TimeoutExpired:
        print("⏳ Timeout de simulación. Podando mutación estancada...")
        raise optuna.exceptions.TrialPruned()
        
    execution_time = time.time() - start_time
    
    # ─── 3. LEER EL OMNI-REGISTRO Y EVALUAR APTITUD (FITNESS) ───
    if not os.path.exists(output_json):
        print("❌ El archivo JSON no se generó. Podando mutación...")
        raise optuna.exceptions.TrialPruned()
        
    try:
        with open(output_json, "r") as f:
            res = json.load(f)
            
        metrics = res["metrics"]
        
        total_return_pct = metrics["total_return_pct"]
        win_rate = metrics["win_rate"]
        max_drawdown = metrics["max_drawdown_pct"]
        final_capital = metrics["final_equity"]
        total_trades = metrics["total_trades"]
        
        if total_trades < 10:
            # Mutación cobarde (no operó lo suficiente)
            return -999.0
            
        # 🧪 THE SANTO GRIAL FITNESS FUNCTION
        # Queremos maximizar el retorno neto (interés compuesto), PERO necesitamos
        # penalizar severamente el Drawdown, especialmente si supera el 15% (Liquidación)
        
        # Base: Crecimiento de la cuenta (Compound Growth)
        fitness = total_return_pct 
        
        # Penalización exponencial por Drawdown peligroso
        if max_drawdown > 15.0:
            fitness -= (max_drawdown - 15.0) * 10  # Castigo de muerte
        elif max_drawdown > 5.0:
            fitness -= (max_drawdown - 5.0) * 2    # Castigo de advertencia
            
        # Bonus por alta efectividad (Requerido para cuenta micro)
        if win_rate > 70.0:
            fitness += 5.0
            
        print(f"🏆 [Gen {trial.number} Result] PnL: {total_return_pct:+.2f}% | WR: {win_rate:.1f}% | DD: {max_drawdown:.2f}% | Fitness: {fitness:.2f} | Time: {execution_time:.1f}s")
        return fitness
        
    except Exception as e:
        print(f"⚠️ Error al leer resultados: {e}")
        raise optuna.exceptions.TrialPruned()

def main():
    print(f"==========================================================")
    print(f" 🧬 INICIANDO HYPER-EVOLVER MASIVO (OPTUNA GENETIC ENGINE)")
    print(f"==========================================================")
    print(f"▸ Días de Simulación : {DAYS}")
    print(f"▸ Activos (Multiverso): {SYMBOLS}")
    print(f"▸ Capital Inicial    : ${CAPITAL}")
    print(f"▸ Objetivo Fitness   : Máximo Interés Compuesto + Supervivencia")
    print(f"==========================================================")
    
    # Utiliza la base de datos SQLite para permitir pausa/reanudación y persistencia
    storage_url = f"sqlite:///hyper_evolver.db"
    
    study = optuna.create_study(
        study_name=STUDY_NAME, 
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\n🛑 Evolución interrumpida manualmente.")
        
    print(f"\n==========================================================")
    print(f" 🥇 EVOLUCIÓN COMPLETADA")
    print(f"==========================================================")
    print("El ADN más apto encontrado (SANTO GRIAL) es:")
    best_params = study.best_params
    print(json.dumps(best_params, indent=4))
    
    print(f"\nMejor Fitness Score: {study.best_value:.2f}")
    print(f"Recomendación: Mapea estos valores a tu config.py para producción.")

if __name__ == "__main__":
    main()
