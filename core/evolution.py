import numpy as np
import pandas as pd
import random
import secrets # Phase 26: Stochastic Purity
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("EvolutionEngine")

@dataclass
class TradeResult:
    """Estructura mínima para evaluación de fitness"""
    pnl_pct: float
    duration_seconds: float
    drawdown_pct: float = 0.0
    is_win: bool = False

class FitnessCalculator:
    """
    Motor de Evaluación Financiera (Trinidad Omega - Fase 16).
    Calcula un 'Score' único para clasificar la supervivencia de los genomas.
    """
    
    @staticmethod
    def calculate_fitness(trades: List[TradeResult], risk_free_rate: float = 0.0) -> float:
        """
        Fórmula Maestra de Fitness:
        Score = Sortino Ratio * log(1 + Total Trades) * Conflict Penalty
        """
        if not trades:
            return 0.0
            
        # 1. Extract Metrics
        returns = np.array([t.pnl_pct for t in trades])
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        
        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        # 2. Profit Factor (Safe Division)
        gross_profit = sum(t.pnl_pct for t in wins)
        gross_loss = abs(sum(t.pnl_pct for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # 3. Sharpe / Sortino
        avg_return = np.mean(returns)
        std_dev = np.std(returns)
        
        # Downside Deviation for Sortino
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.001
        
        sortino = (avg_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # 4. Drawdown Penalty (Max Drawdown of the equity curve)
        equity_curve = np.cumsum(returns)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # 5. Trinity Score Formula
        # Reward consistency (Sortino), Activity (log trades), and penalize deep drawdowns
        if total_trades < 10:
            # Penalty for lack of statistical significance
            significance_penalty = 0.5 
        else:
            significance_penalty = 1.0
            
        score = (sortino * 2.0) + (profit_factor * 1.0) - (max_drawdown * 10.0)
        
        # Normalize
        score = max(0.0, score) * significance_penalty
        
        return float(score)

    @staticmethod
    def evaluate_genotype(genotype_id: str, trades: List[Dict]) -> float:
        """Helper para convertir dicts a TradeResult y calcular"""
        trade_objects = [
            TradeResult(
                pnl_pct=t.get('pnl_pct', 0.0),
                duration_seconds=t.get('duration', 0),
                is_win=t.get('pnl', 0) > 0
            ) for t in trades
        ]
        return FitnessCalculator.calculate_fitness(trade_objects)

class EvolutionEngine:
    """
    Motor de Selección Natural (Trinidad Omega - Fase 17).
    Gestiona el ciclo de vida evolutivo: Selección -> Cruce -> Mutación.
    """
    
    def __init__(self, population_size=50, mutation_rate: float = 0.1, mutation_strength: float = 0.2):
        # 🧬 PHASE 26: Hardware Seeding
        # Use secrets (OS CSPRNG) to seed Random and Numpy
        seed_val = secrets.randbits(32)
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength # Std dev for gaussian mutation

    def select_parents_tournament(self, population: List[Any], tournament_size: int = 3) -> Any:
        """Selecciona el mejor individuo de un subconjunto aleatorio"""
        import random
        tournament = random.sample(population, min(tournament_size, len(population)))
        # Asume que los individuos tienen atributo 'fitness_score'
        return max(tournament, key=lambda ind: ind.fitness_score)

    def crossover(self, parent_a: Any, parent_b: Any, child_genotype: Any) -> Any:
        """
        Uniform Crossover: Mezcla genes de ambos padres 50/50.
        Retorna el genotipo hijo modificado.
        """
        import random
        
        genes_a = parent_a.genes
        genes_b = parent_b.genes
        child_genes = child_genotype.genes
        
        for gene_key in child_genes.keys():
            # Heredar de A o B aleatoriamente
            if random.random() < 0.5:
                child_genes[gene_key] = genes_a.get(gene_key, child_genes[gene_key])
            else:
                child_genes[gene_key] = genes_b.get(gene_key, child_genes[gene_key])
                
        child_genotype.parent_id = f"{parent_a.generation}+{parent_b.generation}"
        return child_genotype

    def mutate(self, genotype: Any) -> Any:
        """
        Gaussian Mutation: Altera levemente los genes numéricos.
        """
        import random
        
        genes = genotype.genes
        for key, value in genes.items():
            if random.random() < self.mutation_rate:
                # Mutación específica por tipo de dato
                if isinstance(value, int):
                    # Enteros (Periodos): +/- cambio pequeño
                    change = int(random.gauss(0, value * self.mutation_strength))
                    new_val = max(1, min(200, value + change))  # F11: Upper bound added
                    genes[key] = new_val
                elif isinstance(value, float):
                    # Flotantes (Porcentajes): +/- cambio proporcional
                    change = random.gauss(0, value * self.mutation_strength)
                    new_val = max(0.001, min(10.0, value + change))  # F11: Upper bound added
                    genes[key] = new_val
                elif isinstance(value, list) and key == 'brain_weights' and value:
                    # Neural Mutation: Perturb weights
                    # Convert to numpy for speed
                    weights = np.array(value)
                    mask = np.random.random(weights.shape) < self.mutation_rate
                    # Add noise only to selected weights
                    noise = np.random.normal(0, self.mutation_strength, size=weights.shape)
                    weights[mask] += noise[mask]
                    # Clip weights to prevent explosion? Maybe not strictly needed yet
                    genes[key] = weights.tolist()                    
        return genotype

    def calculate_diversity(self, population: List[Any]) -> float:
        """
        Calcula la diversidad genética promedio (Distancia Euclidiana Normalizada).
        Retorna un valor entre 0.0 (Clones) y 1.0 (Caos total).
        """
        if len(population) < 2:
            return 0.0
            
        # Extract numerical genes for all individuals
        # Normalize to 0-1 range for fair comparison
        keys = [k for k, v in population[0].genes.items() if isinstance(v, (int, float))]
        
        if not keys:
            return 0.0
            
        matrix = []
        for ind in population:
            row = [float(ind.genes[k]) for k in keys]
            matrix.append(row)
            
        data = np.array(matrix)
        
        # Min-Max Normalization per column
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        ranges = max_vals - min_vals
        
        # Avoid division by zero
        ranges[ranges == 0] = 1.0
        
        normalized = (data - min_vals) / ranges
        
        # Calculate pairwise distances (mean)
        # Simplify: Calculate distance from centroid (mean genome)
        centroid = np.mean(normalized, axis=0)
        distances = np.linalg.norm(normalized - centroid, axis=1)
        
        return float(np.mean(distances))

    def adjust_mutation_rate(self, current_diversity: float, target_diversity: float = 0.2):
        """
        Adaptación Dinámica (Simulated Annealing inverse).
        Si la diversidad es baja, calienta el sistema (Sube mutación).
        Si la diversidad es alta, enfría (Baja mutación para explotar).
        """
        if current_diversity < target_diversity:
            # Boost mutation
            self.mutation_rate = min(0.8, self.mutation_rate * 1.5)
            self.mutation_strength = min(1.0, self.mutation_strength * 1.5)
        else:
            # Cool down
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
            self.mutation_strength = max(0.05, self.mutation_strength * 0.9)

    # ---------------------------------------------------------
    # VECTORIZED ENGINE (Phase 31)
    # ---------------------------------------------------------
    
    def evolve_generation(self, population: List[Any], elite_count: int) -> List[Any]:
        """
        Master method for JIT-accelerated generation step.
        1. Explodes Genotypes to NumPy Arrays.
        2. Runs Numba Kernels (Selection, Crossover, Mutation).
        3. Reconstructs Genotypes.
        """
        if not population:
            return []
            
        n_pop = len(population)
        if n_pop < 2:
            return population # Cannot breed
            
        # 1. Configuration / Gene Mapping
        scalar_keys = [
            'bollinger_period', 'bollinger_std', 'rsi_period', 'rsi_overbought', 
            'rsi_oversold', 'adx_threshold', 'tp_pct', 'sl_pct', 
            'weight_trend', 'weight_momentum', 'weight_volatility'
        ]
        
        # Bounds (Min, Max) - could be config driven, hardcoded for speed here
        # Ints are cast to float for unifying the array
        bounds_min = np.array([5.0, 0.1, 5.0, 50.0, 10.0, 5.0, 0.001, 0.001, 0.0, 0.0, 0.0], dtype=np.float32)
        bounds_max = np.array([100.0, 5.0, 50.0, 95.0, 50.0, 60.0, 0.100, 0.100, 1.0, 1.0, 1.0], dtype=np.float32)

        # 2. Extract Data to Arrays
        # Structure: scalars_matrix[n_ind, n_genes]
        scalars_matrix = np.zeros((n_pop, len(scalar_keys)), dtype=np.float32)
        fitness_scores = np.zeros(n_pop, dtype=np.float32)
        
        # Neural weights: matrix[n_ind, n_weights]
        # Assume all have same brain size
        sample_brain = population[0].genes.get('brain_weights', [])
        n_brain = len(sample_brain)
        has_brain = n_brain > 0
        
        brains_matrix = None
        if has_brain:
            brains_matrix = np.zeros((n_pop, n_brain), dtype=np.float32)
            
        for i, ind in enumerate(population):
            fitness_scores[i] = ind.fitness_score
            for k_idx, k in enumerate(scalar_keys):
                scalars_matrix[i, k_idx] = float(ind.genes.get(k, 0))
            if has_brain:
                brains_matrix[i] = np.array(ind.genes['brain_weights'], dtype=np.float32)

        # 3. Elitism (Preserve top N objects directly)
        # Assumes population is already sorted by fitness? calling function should do it.
        # But we can sort here to be safe or assume indices match.
        # We will assume new population is separate.
        
        new_genotypes = []
        # Copy elites
        for i in range(elite_count):
            if i < n_pop:
                new_genotypes.append(population[i]) # Keep original object (ref) or deepcopy?
                                                    # Ref is risky if we mutate in place later. 
                                                    # Safe to assume elites are untouched.
        
        slots_needed = n_pop - len(new_genotypes)
        
        # 4. Vectorized Breeding
        from utils.evolution_kernels import (
            jit_tournament_selection, 
            jit_crossover_uniform, 
            jit_mutate_gaussian
        )
        
        # Generate Children Data
        child_scalars = np.zeros((slots_needed, len(scalar_keys)), dtype=np.float32)
        child_brains = None
        if has_brain:
            child_brains = np.zeros((slots_needed, n_brain), dtype=np.float32)
            
        for i in range(slots_needed):
            # A. Select Parents (Indices)
            idx_a = jit_tournament_selection(fitness_scores, 3)
            idx_b = jit_tournament_selection(fitness_scores, 3)
            
            # B. Crossover & Mutate Scalars
            parent_a_sc = scalars_matrix[idx_a]
            parent_b_sc = scalars_matrix[idx_b]
            
            child_sc = jit_crossover_uniform(parent_a_sc, parent_b_sc, 0.5)
            child_sc = jit_mutate_gaussian(child_sc, self.mutation_rate, self.mutation_strength, bounds_min, bounds_max)
            child_scalars[i] = child_sc
            
            # C. Crossover & Mutate Brains
            if has_brain:
                parent_a_br = brains_matrix[idx_a]
                parent_b_br = brains_matrix[idx_b]
                
                # Use standard bounds for weights (-1, 1) or larger
                # We can use a simpler mutator for weights without strict clipping or wide clipping
                w_min = np.full(n_brain, -5.0, dtype=np.float32)
                w_max = np.full(n_brain, 5.0, dtype=np.float32)
                
                child_br = jit_crossover_uniform(parent_a_br, parent_b_br, 0.5)
                # Mutate weights
                child_br = jit_mutate_gaussian(child_br, self.mutation_rate, self.mutation_strength, w_min, w_max)
                child_brains[i] = child_br

        # 5. Reconstruct Objects
        from core.genotype import Genotype
        symbol = population[0].symbol
        
        for i in range(slots_needed):
            new_ind = Genotype(symbol=symbol, generation=population[0].generation + 1)
            
            # Map Scalars back
            for k_idx, k in enumerate(scalar_keys):
                val = child_scalars[i, k_idx]
                # Cast back to int if needed
                if k in ['bollinger_period', 'rsi_period', 'adx_threshold']:
                    new_ind.genes[k] = int(round(val))
                else:
                    new_ind.genes[k] = float(val)
            
            # Map Brain back
            if has_brain:
                new_ind.genes['brain_weights'] = child_brains[i].tolist()
                
            new_genotypes.append(new_ind)
            
        return new_genotypes

