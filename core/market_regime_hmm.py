import numpy as np
try:
    import cython
    IS_CYTHON = cython.compiled
except ImportError:
    IS_CYTHON = False

if IS_CYTHON:
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]) and not kwargs: return args[0]
        return decorator
else:
    from numba import njit
from utils.logger import logger
from typing import Tuple

@njit(fastmath=True, cache=True)
def norm_pdf_jit(x: float, mu: float, std: float) -> float:
    """
    [NANO-SPEED] Normal Probability Density Function JIT precompiled.
    """
    if std <= 0.0:
        return 0.0
    exponent = -0.5 * ((x - mu) / std) ** 2
    # 2.5066282746310002 is sqrt(2 * pi)
    return (1.0 / (std * 2.5066282746310002)) * np.exp(exponent)

@njit(fastmath=True, cache=True)
def hmm_update_jit(obs: float, state_probabilities: np.ndarray, A: np.ndarray, 
                   means: np.ndarray, stds: np.ndarray) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    [NANO-SPEED] Precompiled Forward Algorithm Step for Hidden Markov Model.
    """
    n_states = len(state_probabilities)
    
    # 1. Compute likelihoods
    likelihoods = np.empty(n_states, dtype=np.float64)
    for i in range(n_states):
        likelihoods[i] = norm_pdf_jit(obs, means[i], stds[i])
        
    # 2. Prediction Step P(s_t | obs_{1:t-1})
    predicted_probs = np.zeros(n_states, dtype=np.float64)
    for j in range(n_states):
        val = 0.0
        for i in range(n_states):
            val += state_probabilities[i] * A[i, j]
        predicted_probs[j] = val
        
    # 3. Update Step P(s_t | obs_{1:t})
    updated_probs = predicted_probs * likelihoods
    
    # Normalization
    sum_probs = 0.0
    for i in range(n_states):
        sum_probs += updated_probs[i]
        
    new_state_probs = np.empty(n_states, dtype=np.float64)
    if sum_probs > 0.0:
        for i in range(n_states):
            new_state_probs[i] = updated_probs[i] / sum_probs
    else:
        new_state_probs = predicted_probs.copy()
        
    # 4. Argmax selection
    current_state = 0
    max_prob = new_state_probs[0]
    for i in range(1, n_states):
        if new_state_probs[i] > max_prob:
            max_prob = new_state_probs[i]
            current_state = i
            
    # 5. Transition Risk (Probability of changing state at t+1)
    prob_stay = new_state_probs[current_state] * A[current_state, current_state]
    transition_risk = 1.0 - prob_stay
    
    # Next state prediction for t+1
    next_state_probs = np.zeros(n_states, dtype=np.float64)
    for j in range(n_states):
        val = 0.0
        for i in range(n_states):
            val += new_state_probs[i] * A[i, j]
        next_state_probs[j] = val
        
    return current_state, transition_risk, new_state_probs, next_state_probs

@njit(fastmath=True, cache=True)
def hmm_calibrate_jit(historical_returns: np.ndarray, means: np.ndarray, stds: np.ndarray):
    """
    [NANO-SPEED] Precompiled statistical moments calibration of emissions.
    """
    n = len(historical_returns)
    if n < 50:
        return
        
    mu = np.mean(historical_returns)
    vol = np.std(historical_returns)
    
    # Numerical stability safeguard
    if vol < 1e-6:
        vol = 1e-6
        
    # State 0: Low Vol Sideways
    means[0] = mu * 0.1
    stds[0] = vol * 0.5
    
    # State 1: High Vol Sideways/Choppy
    means[1] = mu * 0.2
    stds[1] = vol * 2.0
    
    # State 2: Bull Trend
    means[2] = max(1e-5, mu + 0.5 * vol)
    stds[2] = vol * 1.2
    
    # State 3: Bear Trend
    means[3] = min(-1e-5, mu - 0.5 * vol)
    stds[3] = vol * 1.2


class HiddenMarkovModelDetector:
    """
    Sovereign HMM-style Market Regime Detector.
    QUÉ: Clasifica el mercado en estados ocultos basados en retornos y volatilidad.
    POR QUÉ: Los indicadores tradicionales son reactivos; HMM busca la estructura probabilística subyacente.
    PARA QUÉ: Anticipar cambios de régimen y ajustar el riesgo proactivamente.
    DÓNDE: core/market_regime_hmm.py
    """
    
    REGIMES = {
        0: 'RANGING',
        1: 'CHOPPY',
        2: 'TRENDING_BULL',
        3: 'TRENDING_BEAR'
    }
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        # Probabilidades iniciales (Equitativas)
        self.pi = np.array([0.25] * n_states, dtype=np.float64)
        # Matriz de Transición (Persistencia fuerte en el estado actual)
        self.A = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
            [0.025, 0.025, 0.9, 0.05],
            [0.025, 0.025, 0.05, 0.9]
        ], dtype=np.float64)
        # Parámetros de Emisión (Media de retornos y Desviación Estándar)
        self.means = np.array([0.0, 0.0, 0.0005, -0.0005], dtype=np.float64)
        self.stds = np.array([0.0005, 0.002, 0.0015, 0.0015], dtype=np.float64)
        
        self.last_state = 0
        self.state_probabilities = self.pi.copy()

    def update(self, returns: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Actualiza el estado actual usando el algoritmo Forward precompilado JIT.
        """
        try:
            if len(returns) == 0:
                # Retornar estado actual por fallback
                next_state_probs = self.state_probabilities @ self.A
                return self.REGIMES[self.last_state], 0.0, next_state_probs

            # Tomar el último retorno para la actualización online
            obs = float(returns[-1])
            
            # Invocar cálculo JIT
            current_state, transition_risk, new_probs, next_probs = hmm_update_jit(
                obs, self.state_probabilities, self.A, self.means, self.stds
            )
            
            self.state_probabilities = new_probs
            self.last_state = current_state
            
            return self.REGIMES[current_state], transition_risk, next_probs

        except Exception as e:
            logger.error(f"HMM Update Error: {e}")
            fallback_probs = self.state_probabilities @ self.A
            return self.REGIMES[self.last_state], 0.5, fallback_probs

    def calibrate(self, historical_returns: np.ndarray):
        """
        Calibración adaptativa basada en momentos estadísticos usando funciones JIT.
        """
        try:
            if len(historical_returns) < 50:
                return
            
            # Invocar calibración JIT
            hmm_calibrate_jit(historical_returns, self.means, self.stds)
            
            logger.info(
                f"🧠 [HMM Calibrate] Calibrated emission parameters. "
                f"Means: {self.means.round(5)}, Stds: {self.stds.round(5)}"
            )
            
        except Exception as e:
            logger.error(f"HMM Calibration Error: {e}")
