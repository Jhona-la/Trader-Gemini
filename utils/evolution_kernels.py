import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def jit_calculate_diversity(population_matrix: np.ndarray) -> float:
    """
    Calculates average Euclidean distance from centroid.
    Fastmath enabled for speed.
    """
    n_samples, n_features = population_matrix.shape
    if n_samples < 2:
        return 0.0
    
    # 1. Calculate Centroid
    centroid = np.empty(n_features, dtype=np.float32)
    for j in range(n_features):
        sum_val = 0.0
        for i in range(n_samples):
            sum_val += population_matrix[i, j]
        centroid[j] = sum_val / n_samples
        
    # 2. Calculate Mean Distance
    total_dist = 0.0
    for i in range(n_samples):
        dist_sq = 0.0
        for j in range(n_features):
            diff = population_matrix[i, j] - centroid[j]
            dist_sq += diff * diff
        total_dist += np.sqrt(dist_sq)
        
    return total_dist / n_samples

@njit(fastmath=True, cache=True)
def jit_tournament_selection(fitness_scores: np.ndarray, tournament_size: int) -> int:
    """
    Selects indices of the winner from a random tournament.
    Returns the INDEX of the winner.
    """
    n_pop = len(fitness_scores)
    best_idx = -1
    best_fitness = -1e9
    
    for _ in range(tournament_size):
        # Random pick
        idx = np.random.randint(0, n_pop)
        score = fitness_scores[idx]
        if score > best_fitness:
            best_fitness = score
            best_idx = idx
            
    return best_idx

@njit(fastmath=True, cache=True)
def jit_crossover_uniform(parent_a: np.ndarray, parent_b: np.ndarray, rate: float = 0.5) -> np.ndarray:
    """
    Mixes genes from two parents into a new child array.
    """
    n_genes = len(parent_a)
    child = np.empty(n_genes, dtype=np.float32)
    
    for i in range(n_genes):
        if np.random.random() < rate:
            child[i] = parent_a[i]
        else:
            child[i] = parent_b[i]
            
    return child

@njit(fastmath=True, cache=True, parallel=True)
def jit_crossover_mutation_batch(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_prob: float,
    mutation_prob: float,
    mutation_step: float,
    num_offspring: int
) -> np.ndarray:
    """
    Parallel version of crossover and mutation for population expansion.
    """
    offspring = np.zeros((num_offspring, parent1.shape[0]))
    for i in prange(num_offspring):
        # 1. Crossover
        for j in range(parent1.shape[0]):
            if np.random.random() < crossover_prob:
                offspring[i, j] = parent1[j]
            else:
                offspring[i, j] = parent2[j]
        
        # 2. Mutation
        for j in range(parent1.shape[0]):
            if np.random.random() < mutation_prob:
                offspring[i, j] += np.random.randn() * mutation_step
                
    return offspring

@njit(fastmath=True, cache=True)
def jit_mutate_gaussian(
    genome: np.ndarray, 
    rate: float, 
    strength: float, 
    bounds_min: np.ndarray, 
    bounds_max: np.ndarray
) -> np.ndarray:
    """
    Applies Gaussian mutation to a single genome array.
    Respects bounds.
    """
    n_genes = len(genome)
    mutated = np.copy(genome)
    
    for i in range(n_genes):
        if np.random.random() < rate:
            # Apply mutation
            change = np.random.normal(0.0, strength * np.abs(mutated[i])) 
            # If value is 0, add absolute noise
            if mutated[i] == 0:
                change = np.random.normal(0.0, strength)
                
            new_val = mutated[i] + change
            
            # Clip bounds
            if new_val < bounds_min[i]:
                new_val = bounds_min[i]
            elif new_val > bounds_max[i]:
                new_val = bounds_max[i]
                
            mutated[i] = new_val
            
    return mutated
