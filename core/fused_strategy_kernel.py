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
