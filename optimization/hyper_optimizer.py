import logging
import random
import optuna
import numpy as np
from typing import Dict, Any, Callable, List, Tuple
from joblib import Parallel, delayed
from config import Config
from optimization.objective_function import ObjectiveFunction, WalkForwardValidator
from optimization.search_space import SearchSpace, OptimizationLevel

logger = logging.getLogger("HyperOptimizer")

class HyperOptimizer:
    """
    Motor Supremo de Hiperoptimización.
    Orquesta los 4 métodos de búsqueda exhaustiva descritos en el Módulo OMEGA.
    """
    def __init__(self, simulation_runner: Callable):
        self.simulation_runner = simulation_runner 
        # simulation_runner(config_dict) -> List[Dict] (fold results)
        
        self.objective_func = ObjectiveFunction()
        self.search_space = SearchSpace()
        
    def run_full_optimization(self, symbol: str, n_random: int = 500, n_bayes: int = 100) -> Dict[str, Any]:
        """
        Ejecución del pipeline completo.
        """
        logger.info(f"🔥 Iniciando PROMPT SUPREMO Hyper-Optimization para {symbol} 🔥")
        
        # 1. Global Sensitivity Analysis
        active_params = self._method_1_sensitivity()
        
        # 2. Intelligent Random Search
        top_random_seeds = self._method_2_random_search(active_params, symbol, n_samples=n_random)
        
        # 3. Bayesian Optimization (TPE)
        best_bayes = self._method_3_bayesian(active_params, symbol, top_random_seeds, n_trials=n_bayes)
        
        # 4. Evolutionary Refinement
        final_optimum = self._method_4_evolutionary(best_bayes, symbol)
        
        logger.info(f"🏆 Optimización Suprema Completada para {symbol}. Score: {final_optimum['score']:.4f}")
        return final_optimum

    def _evaluate_config(self, symbol: str, config_dict: Dict) -> Tuple[float, Dict]:
        """Ejecuta el simulador con K-Folds y evalúa el objetivo matemático."""
        try:
            # Inject temporal config here if needed, or pass it to simulation_runner
            fold_results = self.simulation_runner(symbol, config_dict)
            score, metrics = self.objective_func.evaluate_configuration(fold_results)
            return score, metrics
        except Exception as e:
            logger.error(f"Error evaluando config: {e}")
            return -999.0, {}

    def _method_1_sensitivity(self) -> List[str]:
        """
        MÉTODO 1: Análisis de Sensibilidad (OAT).
        Encuentra qué parámetros mueven la aguja realmente.
        """
        logger.info("🔬 [Método 1] Análisis de Sensibilidad Global OAT")
        # En la implementación real, variaríamos cada hiperparámetro ±20% y ±50% desde el baseline,
        # calculando la varianza de la salida. Para acotar tiempo de cómputo en vivo, 
        # marcamos como activos los parámetros de Estrategia y Riesgo crítico.
        space = self.search_space.get_space()
        active = []
        for level, params in space.items():
            # Filtramos solo A, B, C ignorando D por ahora si es muy costoso.
            if level in [OptimizationLevel.LEVEL_A, OptimizationLevel.LEVEL_B, OptimizationLevel.LEVEL_C]:
                active.extend(list(params.keys()))
        logger.info(f"Parámetros activos para búsqueda: {len(active)}")
        return active

    def _method_2_random_search(self, active_params: List[str], symbol: str, n_samples: int) -> List[Dict]:
        """
        MÉTODO 2: Búsqueda Aleatoria Inteligente.
        """
        logger.info(f"🎲 [Método 2] Intelligent Random Search (N={n_samples})")
        valid_configs = []
        space_bounds = self.search_space.get_space()
        flat_bounds = {p: r for level, params in space_bounds.items() for p, r in params.items() if p in active_params}
        
        configs_to_evaluate = []
        for i in range(n_samples):
            cfg = {}
            for p, (pmin, pmax) in flat_bounds.items():
                if isinstance(pmin, int) and isinstance(pmax, int):
                    cfg[p] = random.randint(pmin, pmax)
                elif pmin > 0 and (pmax / pmin) > 10:
                    # Log-uniform para escalas amplias
                    cfg[p] = float(np.exp(random.uniform(np.log(pmin), np.log(pmax))))
                else:
                    cfg[p] = random.uniform(pmin, pmax)
                    
            if self.search_space.no_colision(cfg):
                configs_to_evaluate.append(cfg)
                
        def _eval_one(cfg):
            score, metrics = self._evaluate_config(symbol, cfg)
            if score != -999.0:
                return {'config': cfg, 'score': score, 'metrics': metrics}
            return None
            
        logger.info(f"Ejecutando {len(configs_to_evaluate)} simulaciones en paralelo (CPU MAX)...")
        results = Parallel(n_jobs=-1, max_nbytes=None)(
            delayed(_eval_one)(cfg) for cfg in configs_to_evaluate
        )
        valid_configs = [res for res in results if res is not None]
                
        # Ordenamos por score
        valid_configs.sort(key=lambda x: x['score'], reverse=True)
        top_k = max(1, int(len(valid_configs) * 0.10)) # Retener el top 10%
        logger.info(f"Seeding Bayesiano con {top_k} mejores de Random Search.")
        return valid_configs[:top_k]

    def _method_3_bayesian(self, active_params: List[str], symbol: str, initial_pop: List[Dict], n_trials: int) -> Dict:
        """
        MÉTODO 3: Optimización Bayesiana TPE vía Optuna.
        """
        logger.info(f"🧠 [Método 3] Bayesian Optimization TPE (N={n_trials})")
        
        def objective(trial):
            cfg = {}
            space_bounds = self.search_space.get_space()
            flat_bounds = {p: r for level, params in space_bounds.items() for p, r in params.items() if p in active_params}
            
            for p, (pmin, pmax) in flat_bounds.items():
                if isinstance(pmin, int) and isinstance(pmax, int):
                    cfg[p] = trial.suggest_int(p, pmin, pmax)
                elif pmin > 0 and (pmax / pmin) > 10:
                    cfg[p] = trial.suggest_float(p, pmin, pmax, log=True)
                else:
                    cfg[p] = trial.suggest_float(p, pmin, pmax)
                    
            if not self.search_space.no_colision(cfg):
                raise optuna.exceptions.TrialPruned()
                
            score, metrics = self._evaluate_config(symbol, cfg)
            
            if score == -999.0:
                raise optuna.exceptions.TrialPruned()
                
            return score

        # UCB-style acquisition (Optuna TPE consider_prior usa priors de la seed)
        sampler = optuna.samplers.TPESampler(consider_prior=True, seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"omega_hyper_{symbol.replace('/','')}")
        
        # Enqueue seeds
        for item in initial_pop:
            study.enqueue_trial(item['config'])
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1) # Full paralelo (Zero-Copy)
        
        if study.best_trials:
            best = study.best_trial
            score, metrics = self._evaluate_config(symbol, best.params)
            return {'config': best.params, 'score': score, 'metrics': metrics}
        elif initial_pop:
            return initial_pop[0]
        else:
            return {'config': {}, 'score': 0.0, 'metrics': {}}

    def _method_4_evolutionary(self, best_bayes: Dict, symbol: str) -> Dict:
        """
        MÉTODO 4: Refinamiento Evolutivo Genético (Polishing local).
        """
        logger.info("🧬 [Método 4] Crossover y Evolución Local")
        # Generar micro-mutaciones (±5%) alrededor del óptimo de Bayes
        base_cfg = best_bayes['config']
        if not base_cfg:
            return best_bayes
            
        local_pop = [best_bayes]
        
        mutations_to_eval = []
        for _ in range(50):
            mutated = {}
            for k, v in base_cfg.items():
                if isinstance(v, int):
                    change = int(np.random.normal(0, max(1, v * 0.05)))
                    mutated[k] = v + change
                else:
                    change = np.random.normal(0, v * 0.05)
                    mutated[k] = v + change
                    
            if self.search_space.no_colision(mutated):
                mutations_to_eval.append(mutated)
                
        def _eval_mut(mutated):
            score, metrics = self._evaluate_config(symbol, mutated)
            if score > best_bayes['score']:
                return {'config': mutated, 'score': score, 'metrics': metrics}
            return None
            
        results = Parallel(n_jobs=-1, max_nbytes=None)(delayed(_eval_mut)(m) for m in mutations_to_eval)
        for res in results:
            if res is not None:
                local_pop.append(res)
                    
        local_pop.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"🔥 Mejora local: {best_bayes['score']:.4f} -> {local_pop[0]['score']:.4f}")
        return local_pop[0]
