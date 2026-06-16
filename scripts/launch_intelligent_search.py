import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.surrogate_model import LightGBMSurrogate
from core.quantum_optimizer import QuantumOptimizer
from core.vectorized_backtest import run_backtest_fidelity

def launch_search():
    print("==================================================")
    print("🚀 INICIANDO BÚSQUEDA CUÁNTICA MULTI-FIDELIDAD")
    print("==================================================")
    
    surrogate = LightGBMSurrogate(model_path="surrogate_model.txt")
    optimizer = QuantumOptimizer(study_name="trader_gemini_quantum", db_path="sqlite:///optim_study.db")
    
    # ---------------------------------------------------------
    # FASE 1: Pre-entrenamiento (Random Search F2)
    # ---------------------------------------------------------
    print("\n[FASE 1] Recolectando datos base (Fidelidad F2) para entrenar Surrogate...")
    start_time = time.time()
    
    n_pre_train = 500
    X_train = np.zeros((n_pre_train, 5), dtype=np.float32)
    y_train = np.zeros(n_pre_train, dtype=np.float32)
    
    for i in range(n_pre_train):
        # Muestreo aleatorio
        params = {
            'rsi_period': np.random.randint(5, 50),
            'rsi_lower': np.random.uniform(10.0, 40.0),
            'rsi_upper': np.random.uniform(60.0, 90.0),
            'stop_loss': np.random.uniform(0.01, 0.10),
            'take_profit': np.random.uniform(0.02, 0.20)
        }
        
        # Ejecutar en fidelidad F2 (1000 velas)
        score = run_backtest_fidelity('F2', params)
        
        X_train[i] = [params['rsi_period'], params['rsi_lower'], params['rsi_upper'], params['stop_loss'], params['take_profit']]
        y_train[i] = score
        
    # Entrenar modelo
    surrogate.train(X_train, y_train)
    print(f"✅ Fase 1 Completada en {time.time() - start_time:.2f} segundos.")
    
    # ---------------------------------------------------------
    # FASE 2: Inferencia Virtual Masiva
    # ---------------------------------------------------------
    print("\n[FASE 2] Inferencia Virtual Masiva con Surrogate (1,000,000 muestras)...")
    start_time = time.time()
    
    X_virtual = surrogate.generate_virtual_samples(1_000_000)
    y_pred = surrogate.predict(X_virtual)
    
    # Seleccionar el top 1% (10,000 mejores)
    top_indices = np.argsort(y_pred)[-10000:]
    X_top = X_virtual[top_indices]
    y_top = y_pred[top_indices]
    
    # Seleccionar el top 10 absolutos para inyectar a Optuna
    top_10 = X_top[-10:]
    candidates_to_enqueue = []
    for row in top_10:
        candidates_to_enqueue.append({
            'rsi_period': int(row[0]),
            'rsi_lower': float(row[1]),
            'rsi_upper': float(row[2]),
            'stop_loss': float(row[3]),
            'take_profit': float(row[4])
        })
        
    optimizer.enqueue_virtual_candidates(candidates_to_enqueue)
    
    print(f"✅ Fase 2 Completada en {time.time() - start_time:.2f} segundos.")
    print(f"   (1 millón de backtests virtuales en ~segundos. Predicción Top 1: {y_top[-1]:.4f})")
    
    # ---------------------------------------------------------
    # FASE 3: Optimización Bayesiana TPE Multi-Fidelidad
    # ---------------------------------------------------------
    print("\n[FASE 3] Optimización Bayesiana (TPE + Pruner) en Optuna...")
    print("   El sistema usará todos los cores disponibles y cortará trials malos tempranamente.")
    
    start_time = time.time()
    # Para la demo y evitar problemas de SQLite Threading en Windows, usamos n_jobs=1
    # En producción real se puede usar Redis o PostgreSQL para n_jobs=-1
    optimizer.optimize(n_trials=50, n_jobs=1)
    
    print(f"\n✅ Fase 3 Completada en {time.time() - start_time:.2f} segundos.")
    
    best = optimizer.get_best_params()
    print("\n🏆 MEJOR CONFIGURACIÓN ENCONTRADA:")
    for k, v in best.items():
        print(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")

if __name__ == "__main__":
    # Asegurar que el path log de optuna no sature la consola
    import logging
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    launch_search()
