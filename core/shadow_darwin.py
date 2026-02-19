"""
OMEGA PROTOCOL: Shadow Darwin + Optuna Bayesian Optimizer (Phase 99/6)
======================================================================
QUÉ: Reemplaza la búsqueda por fuerza bruta del algoritmo genético con
     Tree-Parzen Estimator (TPE) de Optuna para explorar el espacio de
     parámetros del Genotype de forma inteligente.
POR QUÉ: Grid/Random search desperdicia ~70% de evaluaciones en regiones
         subóptimas. TPE aprende un modelo probabilístico del espacio y
         concentra trials en zonas prometedoras.
PARA QUÉ: Encontrar el ADN óptimo (bollinger, RSI, TP/SL, pesos) en
          ~70% menos evaluaciones → menor consumo CPU para Metal-Core.
CÓMO: Optuna `create_study` con TPE sampler + SQLite storage (persistencia
      entre sesiones). Cada trial parametriza un Genotype y lo evalúa con
      el SimulationEngine existente.
CUÁNDO: Llamado desde el loop de shadow optimization en background.
DÓNDE: core/shadow_darwin.py
QUIÉN: ShadowDarwin.run_epoch_optuna()
"""
import os
import logging
import json
from typing import List, Dict, Optional
import pandas as pd

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from core.genotype import Genotype
from core.evolution import EvolutionEngine, FitnessCalculator
from core.simulation import SimulationEngine, SimDataProvider

logger = logging.getLogger("ShadowDarwin")

# Suppress Optuna's verbose logging (it logs every trial)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ShadowDarwin:
    """
    Orquestador Evolutivo (Trinidad Omega - Phase 18/99).
    
    DUAL MODE:
    - run_epoch()          → Genetic Algorithm (legacy, population-based)
    - run_epoch_optuna()   → Bayesian TPE (Phase 99, Optuna-powered)
    """
    def __init__(self, data_provider: SimDataProvider, population_size: int = 50, 
                 use_neural: bool = False, wandb_tracker=None):
        self.data_provider = data_provider
        self.population_size = population_size
        self.use_neural = use_neural
        self.evolution_engine = EvolutionEngine()
        self.simulator = SimulationEngine(data_provider)
        self.populations: Dict[str, List[Genotype]] = {}
        
        # Phase 99: Optuna SQLite persistence
        self.optuna_db = "sqlite:///data/optuna_studies.db"
        os.makedirs("data", exist_ok=True)
        
        # Phase 99: WandB hook (optional)
        self.wandb_tracker = wandb_tracker

    # ═══════════════════════════════════════════════════════
    # LEGACY: Genetic Algorithm (preserved for compatibility)
    # ═══════════════════════════════════════════════════════
    
    def initialize_population(self, symbol: str):
        """Crea población inicial aleatoria/sembrada para un símbolo"""
        population = []
        input_dim = 25
        output_dim = 4

        adam = Genotype(symbol=symbol)
        if self.use_neural: adam.init_brain(input_dim, output_dim)
        population.append(adam)
        
        for _ in range(self.population_size - 1):
            mutant = Genotype(symbol=symbol)
            if self.use_neural: mutant.init_brain(input_dim, output_dim)
            self.evolution_engine.mutate(mutant)
            self.evolution_engine.mutate(mutant)
            population.append(mutant)
            
        self.populations[symbol] = population

    def run_epoch(self, symbol: str, generations: int = 1):
        """Ejecuta N generaciones de evolución genética para un símbolo"""
        if symbol not in self.populations:
            self.initialize_population(symbol)
            
        population = self.populations[symbol]
        
        for gen in range(generations):
            for individual in population:
                trades = self.simulator.run(individual, symbol)
                fitness = FitnessCalculator.calculate_fitness(trades)
                individual.fitness_score = fitness
                individual.generation += 1
            
            diversity = self.evolution_engine.calculate_diversity(population)
            self.evolution_engine.adjust_mutation_rate(diversity)
            
            population.sort(key=lambda x: x.fitness_score, reverse=True)
            best_fitness = population[0].fitness_score
            
            logger.info(
                f"🧬 [Epoch {gen}] {symbol} "
                f"Fitness: {best_fitness:.4f} | Div: {diversity:.4f} | "
                f"Mut: {self.evolution_engine.mutation_rate:.2f}"
            )
            
            # WandB tracking
            if self.wandb_tracker:
                self.wandb_tracker.log_generation(
                    gen_id=gen, fitness=best_fitness, 
                    diversity=diversity, 
                    params=population[0].genes,
                    symbol=symbol
                )
            
            elite_count = max(1, int(self.population_size * 0.1))
            population = self.evolution_engine.evolve_generation(population, elite_count)
            self.populations[symbol] = population
            
        self.save_hall_of_fame(population, top_n=3)
        return population[0]

    # ═══════════════════════════════════════════════════════
    # PHASE 99: OPTUNA BAYESIAN OPTIMIZER (TPE)
    # ═══════════════════════════════════════════════════════

    def run_epoch_optuna(self, symbol: str, n_trials: int = 50) -> Dict:
        """
        Ejecuta optimización Bayesiana con Optuna para un símbolo.
        
        Args:
            symbol: Par de trading (e.g., 'BTC/USDT')
            n_trials: Número de evaluaciones (cada trial = 1 simulación)
            
        Returns:
            dict con best_params + best_fitness
        """
        clean_sym = symbol.replace('/', '')
        study_name = f"omega_{clean_sym}"
        
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.optuna_db,
                direction='maximize',
                sampler=TPESampler(
                    n_startup_trials=10,    # Random exploration first
                    multivariate=True,      # Model parameter correlations
                    seed=42
                ),
                pruner=MedianPruner(
                    n_startup_trials=5,     # Don't prune too early
                    n_warmup_steps=3
                ),
                load_if_exists=True         # Resume previous study
            )
        except Exception as e:
            logger.error(f"❌ [Optuna] Failed to create study for {symbol}: {e}")
            # Fallback: in-memory study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
        
        # Run optimization
        study.optimize(
            lambda trial: self._optuna_objective(trial, symbol),
            n_trials=n_trials,
            show_progress_bar=False,
            n_jobs=1  # Sequential — SimulationEngine is not thread-safe
        )
        
        best = study.best_trial
        logger.info(
            f"🧬⚗️ [Optuna] {symbol} BEST: "
            f"Fitness={best.value:.4f} | Trial #{best.number} | "
            f"Total trials: {len(study.trials)}"
        )
        
        # WandB tracking
        if self.wandb_tracker:
            self.wandb_tracker.log_generation(
                gen_id=best.number,
                fitness=best.value,
                diversity=0.0,
                params=best.params,
                symbol=symbol,
                method="optuna_tpe"
            )
        
        # Save best as Genotype
        best_genotype = self._params_to_genotype(best.params, symbol)
        best_genotype.fitness_score = best.value
        best_genotype.save(f"data/genotypes/{clean_sym}_gene_optuna_best.json")
        
        return {
            'best_params': best.params,
            'best_fitness': best.value,
            'total_trials': len(study.trials),
            'genotype': best_genotype
        }
    
    def _optuna_objective(self, trial: optuna.Trial, symbol: str) -> float:
        """
        Objective function for Optuna with Walk-Forward Cross-Validation (DF-B4).
        Evaluates parameters on multiple time folds to ensure robustness using Mean - StdDev.
        """
        # ─── Suggest parameters (same ranges as Genotype defaults ±50%) ───
        genes = {
            # Technical Indicators
            "bollinger_period": trial.suggest_int("bollinger_period", 10, 40),
            "bollinger_std": trial.suggest_float("bollinger_std", 1.0, 3.5, step=0.1),
            "rsi_period": trial.suggest_int("rsi_period", 7, 28),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 85),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 40),
            "adx_threshold": trial.suggest_int("adx_threshold", 15, 40),
            "strength_threshold": trial.suggest_float("strength_threshold", 0.3, 0.9, step=0.05),
            
            # Risk Management
            "tp_pct": trial.suggest_float("tp_pct", 0.005, 0.04, step=0.001),
            "sl_pct": trial.suggest_float("sl_pct", 0.005, 0.05, step=0.001),
            "atr_sl_multiplier": trial.suggest_float("atr_sl_multiplier", 1.0, 4.0, step=0.25),
            "trend_ema_period": trial.suggest_int("trend_ema_period", 100, 400, step=50),
            "trailing_activation_rsi": trial.suggest_int("trailing_activation_rsi", 50, 80),
            
            # Weights (must sum to ~1.0)
            "weight_trend": trial.suggest_float("weight_trend", 0.1, 0.7, step=0.05),
            "weight_momentum": trial.suggest_float("weight_momentum", 0.1, 0.7, step=0.05),
            "weight_volatility": trial.suggest_float("weight_volatility", 0.05, 0.4, step=0.05),
            
            # Neural weights not optimized by Optuna (too many dims)
            "brain_weights": []
        }
        
        # Normalize weights to sum to 1.0
        total_w = genes["weight_trend"] + genes["weight_momentum"] + genes["weight_volatility"]
        if total_w > 0:
            genes["weight_trend"] /= total_w
            genes["weight_momentum"] /= total_w
            genes["weight_volatility"] /= total_w
        
        # Create Genotype from suggested params
        genotype = Genotype(symbol=symbol, genes=genes)
        
        # --- DF-B4: WALK-FORWARD CROSS-VALIDATION ---
        try:
            # 1. Get Data Length
            if symbol not in self.simulator.data.arrays:
                return -999.0
            
            n_samples = len(self.simulator.data.arrays[symbol])
            if n_samples < 500: # Need enough data for splits
                 # Fallback to single run
                 trades = self.simulator.run(genotype, symbol)
                 return FitnessCalculator.calculate_fitness(trades)
            
            # 2. Define Folds (4 Folds)
            k_folds = 4
            fold_size = n_samples // k_folds
            
            fold_scores = []
            
            for i in range(k_folds):
                # Define Range [Start, End]
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size
                
                # Run Simulation on Slice
                trades = self.simulator.run(genotype, symbol, start_idx=start_idx, end_idx=end_idx)
                
                # Calculate Fitness for this Fold
                # Note: Small folds might yield 0 trades, so fitness handles it.
                score = FitnessCalculator.calculate_fitness(trades)
                fold_scores.append(score)
            
            # 3. Aggregate (Robustness Metric)
            # Fitness = Mean - StdDev (Penalize regime sensitivity)
            mean_score = np.mean(fold_scores)
            std_dev = np.std(fold_scores)
            
            # Penalize high variance
            final_fitness = mean_score - (std_dev * 0.5)
            
            return float(final_fitness)

        except Exception as e:
            logger.warning(f"⚠️ [Optuna] Trial failed for {symbol}: {e}")
            return -999.0
    
    def _params_to_genotype(self, params: Dict, symbol: str) -> Genotype:
        """Converts Optuna best_params dict into a Genotype."""
        genes = dict(Genotype(symbol="_default_").genes)  # Start with defaults
        for key, value in params.items():
            if key in genes:
                genes[key] = value
        
        # Normalize weights
        total_w = genes.get("weight_trend", 0.4) + genes.get("weight_momentum", 0.4) + genes.get("weight_volatility", 0.2)
        if total_w > 0:
            genes["weight_trend"] /= total_w
            genes["weight_momentum"] /= total_w
            genes["weight_volatility"] /= total_w
            
        return Genotype(symbol=symbol, genes=genes)

    def get_study_stats(self, symbol: str) -> Optional[Dict]:
        """Returns Optuna study statistics for a symbol."""
        clean_sym = symbol.replace('/', '')
        study_name = f"omega_{clean_sym}"
        try:
            study = optuna.load_study(study_name=study_name, storage=self.optuna_db)
            return {
                'total_trials': len(study.trials),
                'best_fitness': study.best_value,
                'best_params': study.best_params,
                'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            }
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════
    # SHARED: Hall of Fame Persistence
    # ═══════════════════════════════════════════════════════

    def save_hall_of_fame(self, population: List[Genotype], top_n: int = 3):
        """Persiste los mejores N genomas únicos"""
        saved_count = 0
        for i, genotype in enumerate(population):
            if saved_count >= top_n:
                break
                
            suffix = "_alpha" if i == 0 else f"_beta_{i}"
            path = f"data/genotypes/{genotype.symbol.replace('/','')}_gene{suffix}.json"
            genotype.save(path)
            saved_count += 1
            
        logger.info(f"🏆 Hall of Fame saved for {population[0].symbol} (Top {saved_count})")
