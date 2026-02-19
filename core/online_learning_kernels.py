import numpy as np
from numba import njit, float32, float64, prange

@njit(fastmath=True, cache=True)
def jit_sgd_batch(
    weights: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    learning_rate: float,
    clip_value: float,
    gamma: float,
    target_clip: float = 10.0
) -> tuple:
    """
    Performs Stochastic Gradient Descent on a batch of transitions.
    Model: Linear Approximation -> Q(s, a) ~= W . s
    Target: r + gamma * max_a' (W . s') 
    
    BUT, our weights are usually specific to the policy or simpler.
    If 'weights' is (InputDim, OutputDim=4), then we have Q-Network.
    If 'weights' is flattened, we need to know structure.
    
    ASSUMPTION for Phase 32:
    The Genotype 'brain_weights' is a flattened array representing a Dense Layer 
    Input(25) -> Output(4).
    Shape: (100,) or (25, 4).
    """
    
    # 1. Reshape Weights to (Input, Output) if 1D
    batch_size = len(states)
    input_dim = states.shape[1]
    n_weights = len(weights)
    output_dim = n_weights // input_dim
    
    # Create view or copy reshaped
    # Numba doesn't always like reshape in fastmath, but let's try
    # Better: Manual index math or assume logic matches.
    
    # 2. Iterate Batch
    total_error = 0.0
    
    # We need a mutable copy of weights to update
    new_weights = weights.copy()
    
    # To treat as matrix:
    # shape (inputs, outputs)
    
    for i in range(batch_size):
        s = states[i] # (InputDim,)
        a_idx = int(actions[i])
        r = rewards[i]
        sn = next_states[i] # (InputDim,)
        
        # A. Current Prediction Q(s, a)
        # Linear dot product of Input * Column(a)
        # Weight Indexing: weight_idx = col + row*output_dim ? 
        # Usually weights are stored (Input, Output) or (Output, Input).
        # Let's assume standard (Input, Output) flattend row-major: 
        # Index = input_idx * output_dim + output_idx 
        
        pred = 0.0
        for j in range(input_dim):
            # w_idx = j * output_dim + a_idx
            # pred += s[j] * weights[w_idx]
            # WAIT: Genotypes might store weights differently. 
            # neural_bridge says: 25 input -> 4 output.
            # Let's assume standard contiguous block per input feature? 
            # Or block per output neuron?
            # Standard NN: Output = Activation(Weights * Input + Bias)
            # Row = Output Neuron.
            # Index = output_idx * input_dim + input_idx
            
            w_idx = a_idx * input_dim + j
            if w_idx < n_weights:
                pred += s[j] * new_weights[w_idx]
                
        # B. Target calculation
        # Max Q(s')
        max_q_next = -1e9
        
        for act in range(output_dim):
            q_next_val = 0.0
            for j in range(input_dim):
                w_idx = act * input_dim + j
                if w_idx < n_weights:
                    q_next_val += sn[j] * new_weights[w_idx]
            if q_next_val > max_q_next:
                max_q_next = q_next_val
                
        target = r + (gamma * max_q_next)
        
        # [DF-B5 FIX] Target Clipping to prevent Brain Corruption on outliers
        if target > target_clip: target = target_clip
        elif target < -target_clip: target = -target_clip
        
        # C. Error
        error = target - pred
        total_error += abs(error)
        
        # D. Update Weights (only for taken action a_idx)
        # dE/dw = -error * input
        # w_new = w + lr * error * input
        
        grad_factor = learning_rate * error
        # Clip gradient magnitude?
        # Standard clip is on total delta, let's clip error term proxy
        if grad_factor > clip_value: grad_factor = clip_value
        elif grad_factor < -clip_value: grad_factor = -clip_value
            
        for j in range(input_dim):
            w_idx = a_idx * input_dim + j
            if w_idx < n_weights:
                new_weights[w_idx] += grad_factor * s[j]
                
    return new_weights, total_error / batch_size

@njit(fastmath=True, cache=True)
def jit_sgd_single(
    weights: np.ndarray,
    state: np.ndarray,
    action: float,
    reward: float,
    next_state: np.ndarray,
    learning_rate: float,
    clip_value: float,
    gamma: float,
    target_clip: float = 10.0
) -> float:
    """
    Single-step SGD update optimized for SIMD and Branch Prediction.
    [PHASE 64] Uses branchless clipping and optimized loop structures.
    """
    input_dim = state.shape[0]
    n_weights = weights.shape[0]
    output_dim = n_weights // input_dim
    a_idx = int(action)
    
    # 1. Prediction Q(s, a)
    pred = 0.0
    base_idx = a_idx * input_dim
    for j in range(input_dim):
        pred += state[j] * weights[base_idx + j]
            
    # 2. Target Max Q(s')
    max_q_next = -1e12
    for act in range(output_dim):
        q_next_val = 0.0
        act_base = act * input_dim
        for j in range(input_dim):
            q_next_val += next_state[j] * weights[act_base + j]
        
        # Use np.fmax for branchless hardware acceleration
        max_q_next = np.fmax(max_q_next, q_next_val)
            
    # Branchless reset for dead actions
    if max_q_next < -1e11: 
        max_q_next = 0.0
            
    target = reward + (gamma * max_q_next)
    
    # [DF-B5 FIX] Branchless Target Clipping
    target = np.fmin(np.fmax(target, -target_clip), target_clip)
    
    # 3. Error & Update
    error = target - pred
    grad_factor = learning_rate * error
    
    # Branchless Clipping using np.clip
    grad_factor = np.fmin(np.fmax(grad_factor, -clip_value), clip_value)
        
    for j in range(input_dim):
        weights[base_idx + j] += grad_factor * state[j]
            
    return abs(error)

@njit(fastmath=True, cache=True, parallel=True)
def jit_sgd_parallel(
    weights_batch: np.ndarray,      # (N_SYMBOLS, N_WEIGHTS)
    states_batch: np.ndarray,       # (N_SYMBOLS, INPUT_DIM)
    actions_batch: np.ndarray,      # (N_SYMBOLS,)
    rewards_batch: np.ndarray,      # (N_SYMBOLS,)
    next_states_batch: np.ndarray,  # (N_SYMBOLS, INPUT_DIM)
    learning_rate: float,
    clip_value: float,
    gamma: float,
    target_clip: float = 10.0
) -> np.ndarray:
    """
    Parallel SGD update across multiple symbols.
    Uses prange to parallelize per-symbol learning.
    """
    n_symbols = weights_batch.shape[0]
    errors = np.zeros(n_symbols, dtype=np.float32)
    
    for i in prange(n_symbols):
        errors[i] = jit_sgd_single(
            weights_batch[i],
            states_batch[i],
            actions_batch[i],
            rewards_batch[i],
            next_states_batch[i],
            learning_rate,
            clip_value,
            gamma,
            target_clip
        )
    return errors

