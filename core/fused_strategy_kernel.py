import numpy as np
from numba import njit, float32, float64
import math

@njit(fastmath=True, cache=True)
def fused_compute_step(
    closes: np.ndarray,
    volumes: np.ndarray,
    portfolio_state: np.ndarray, # [has_pos, pnl_norm, dur_norm]
    gene_params: np.ndarray,      # [sl_norm, tp_norm]
    brain_weights: np.ndarray,   # (25 * 4 = 100 flattened)
    l2_state: np.ndarray,        # [ofi, microprice_divergence]
    window: int = 5
) -> np.ndarray:
    """
    ULTRA-FUSED KERNEL (Phase 65).
    Combines: Indicators -> State Tensor -> Neural Mapping.
    Target Latency: <2μs end-to-end.
    """
    # 1. Indicator Pre-calculations (Last 'window' bars)
    n = len(closes)
    if n < 30: # Minimum bars for basic indicators
        return np.zeros(4, dtype=np.float32)
        
    state_tensor = np.zeros(25, dtype=np.float32)
    
    # 1A. Market Data (20 Features)
    # Returns (5)
    for i in range(window):
        idx = n - window + i
        val = (closes[idx] - closes[idx-1]) / closes[idx-1]
        state_tensor[i] = val
        
    # Volatility (5) - Simplistic: Price / Rolling Mean
    vol_sum = 0.0
    for i in range(n-20, n): vol_sum += volumes[i]
    mean_vol = vol_sum / 20.0
    if mean_vol < 1e-8: mean_vol = 1.0
    
    for i in range(window):
        idx = n - window + i
        state_tensor[5 + i] = volumes[idx] / mean_vol
        
    # RSI Placeholder mapping (Actual RSI logic is too heavy for single-tick fusion 
    # if we recalculate full history, so we use a fast-tracked last RSI value)
    # For now, we use a simplified momentum proxy to stay within latency budget
    for i in range(window):
        idx = n - window + i
        # Simple Momentum proxy
        mom = (closes[idx] / closes[idx-2] - 1.0) if idx >= 2 else 0.0
        state_tensor[10 + i] = mom
        # Placeholder for 4th feature
        state_tensor[15 + i] = 0.0
        
    # Inject L2 Data (Phase 66: Orderbook Vectorization)
    state_tensor[18] = l2_state[0] # ofi
    state_tensor[19] = l2_state[1] # microprice_divergence
        
    # 2. Add Portfolio & Gene (5 Features)
    state_tensor[20] = portfolio_state[0] # has_pos
    state_tensor[21] = portfolio_state[1] # pnl_norm
    state_tensor[22] = portfolio_state[2] # dur_norm
    state_tensor[23] = gene_params[0]      # sl_norm
    state_tensor[24] = gene_params[1]      # tp_norm
    
    # 3. Neural Inference (100 Weights -> 4 Actions)
    # This is a Dot Product: Output(4) = Weights(4, 25) * State(25)
    action_scores = np.zeros(4, dtype=np.float32)
    
    for act in range(4):
        score = 0.0
        base_idx = act * 25
        for j in range(25):
            score += state_tensor[j] * brain_weights[base_idx + j]
        action_scores[act] = score
        
    return action_scores


# =====================================================================
# NUMBA TREE COMPILER: NANO-LATENCY SKLEARN INFERENCE
# =====================================================================

@njit(fastmath=True)
def predict_rf_jit(
    X: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
    tree_offsets: np.ndarray
) -> float:
    """
    Executes a Random Forest inference at nanosecond speed.
    Bypasses the entire Python Scikit-learn object overhead.
    Returns the probability of the positive class.
    """
    n_trees = len(tree_offsets) - 1
    total_prob = 0.0
    
    for i in range(n_trees):
        node = tree_offsets[i]
        
        # Traverse tree
        while children_left[node] != -1: # -1 indicates a leaf node in sklearn
            f_idx = feature[node]
            if X[f_idx] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
                
        total_prob += value[node]
        
    return total_prob / n_trees

@njit(fastmath=True)
def predict_gb_jit(
    X: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
    tree_offsets: np.ndarray,
    init_score: float,
    learning_rate: float
) -> float:
    """
    Executes a Gradient Boosting classification inference at nanosecond speed.
    Bypasses Python Sklearn overhead.
    Returns the probability of the positive class via Sigmoid.
    """
    n_trees = len(tree_offsets) - 1
    score = init_score
    
    for i in range(n_trees):
        node = tree_offsets[i]
        
        # Traverse tree
        while children_left[node] != -1:
            f_idx = feature[node]
            if X[f_idx] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
                
        score += learning_rate * value[node]
        
    # Sigmoid to convert log-odds to probability
    if score >= 0:
        prob = 1.0 / (1.0 + np.exp(-score))
    else:
        exp_s = np.exp(score)
        prob = exp_s / (1.0 + exp_s)
        
    return prob
