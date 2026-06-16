import optuna
import os
import sqlite3
from core.vectorized_backtest import run_backtest_fidelity

class QuantumOptimizer:
    """
    Orquestador Multi-Fidelidad de Optuna con TPE + Pruner.
    Usa SQLite para permitir reanudación en caso de apagado.
    """
    
    def __init__(self, study_name="trader_gemini_quantum", db_path="sqlite:///optim_study.db"):
        self.study_name = study_name
        self.db_path = db_path
        
        # Configurar Hyperband / ASHA Pruner
        # Este pruner elimina las malas ejecuciones antes de llegar a alta fidelidad
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=100, # Velas iniciales
            max_resource=10000, # Velas máximas
            reduction_factor=3 # Elimina 2/3 de las peores configuraciones cada ronda
        )
        
        # Sampler TPE (Tree-structured Parzen Estimator)
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
        
        # Crear o reanudar el estudio
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.db_path,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
    def objective(self, trial):
        """
        Función objetivo que mapea el ensayo de Optuna a las múltiples fidelidades.
        """
        params = {
            'rsi_period': trial.suggest_int('rsi_period', 5, 50),
            'rsi_lower': trial.suggest_float('rsi_lower', 10.0, 40.0),
            'rsi_upper': trial.suggest_float('rsi_upper', 60.0, 90.0),
            'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.10),
            'take_profit': trial.suggest_float('take_profit', 0.02, 0.20)
        }
        
        # Evaluar en múltiples pasos (ASHA / Hyperband logic)
        fidelities = ['F1', 'F2', 'F3', 'F4']
        candle_steps = [100, 1000, 5000, 10000]
        
        last_score = -1.0
        
        for step, fidelity in zip(candle_steps, fidelities):
            # Ejecutar Numba JIT para este nivel de fidelidad
            score = run_backtest_fidelity(fidelity, params)
            last_score = score
            
            # Reportar el paso al Pruner de Optuna
            trial.report(score, step)
            
            # Decidir si la prueba es tan mala que se corta aquí mismo
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        # Retornar el score final (en la máxima fidelidad alcanzada)
        return last_score

    def enqueue_virtual_candidates(self, top_candidates):
        """
        Inyecta las mejores predicciones del Surrogate Model en la cola de Optuna
        para que las valide realmente con la función objetivo.
        """
        for cand in top_candidates:
            # Cand debe ser un dict con los parámetros
            self.study.enqueue_trial(cand)

    def optimize(self, n_trials=1000, n_jobs=-1):
        """
        Inicia la optimización real de Optuna usando múltiples cores paralelos.
        n_jobs=-1 usa todos los cores del CPU.
        """
        # Para evitar bloquear el OS, limitamos n_jobs a cores_disponibles - 2
        import multiprocessing
        cores = max(1, multiprocessing.cpu_count() - 2)
        if n_jobs == -1:
            n_jobs = cores
            
        print(f"[OPTUNA] Iniciando {n_trials} trials con {n_jobs} cores paralelos...")
        self.study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        
    def get_best_params(self):
        return self.study.best_params
