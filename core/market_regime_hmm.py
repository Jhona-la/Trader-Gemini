import numpy as np
from scipy.stats import norm
from utils.logger import logger
from typing import Tuple, List, Dict

class HiddenMarkovModelDetector:
    """
    Sovereign HMM-style Market Regime Detector.
    QUÉ: Clasifica el mercado en estados ocultos basados en retornos y volatilidad.
    POR QUÉ: Los indicadores tradicionales (ADX/RSI) son reactivos; HMM busca la estructura probabilística subyacente.
    PARA QUÉ: Anticipar cambios de régimen y ajustar el riesgo proactivamente.
    """
    
    REGIMES = {
        0: 'LOW_VOL_SIDEWAYS',
        1: 'VOLATILE_SIDEWAYS',
        2: 'TREND_BULL',
        3: 'TREND_BEAR'
    }
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        # Probabilidades iniciales (Equitativas)
        self.pi = np.array([0.25] * n_states)
        # Matriz de Transición (Persistencia fuerte en el estado actual)
        self.A = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
            [0.025, 0.025, 0.9, 0.05],
            [0.025, 0.025, 0.05, 0.9]
        ])
        # Parámetros de Emisión (Media de retornos y Desviación Estándar)
        # 0: Low Vol Side (Mean 0, Std 0.001)
        # 1: High Vol Side (Mean 0, Std 0.005)
        # 2: Bull Trend (Mean 0.002, Std 0.003)
        # 3: Bear Trend (Mean -0.002, Std 0.003)
        self.means = np.array([0.0, 0.0, 0.0005, -0.0005])
        self.stds = np.array([0.0005, 0.002, 0.0015, 0.0015])
        
        self.last_state = 0
        self.state_probabilities = self.pi.copy()

    def update(self, returns: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Actualiza el estado actual usando el algoritmo Forward simplificado (Online).
        returns: Array de retornos recientes (pct_change).
        """
        try:
            if len(returns) == 0:
                return self.REGIMES[self.last_state], 0.0, self.A[self.last_state]

            # Tomar el último retorno para la actualización online
            obs = returns[-1]
            
            # 1. Calcular verosimilitudes (Likelihoods) de la observación en cada estado
            likelihoods = np.array([norm.pdf(obs, m, s) for m, s in zip(self.means, self.stds)])
            
            # 2. Paso de Predicción (P(s_t | obs_{1:t-1}))
            predicted_probs = self.state_probabilities @ self.A
            
            # 3. Paso de Actualización (P(s_t | obs_{1:t}))
            updated_probs = predicted_probs * likelihoods
            
            # Normalizar
            sum_probs = np.sum(updated_probs)
            if sum_probs > 0:
                self.state_probabilities = updated_probs / sum_probs
            else:
                self.state_probabilities = predicted_probs # Fallback if likelihoods are zero
                
            # 4. Determinar estado más probable
            current_state = int(np.argmax(self.state_probabilities))
            self.last_state = current_state
            
            # 5. Probabilidad de transición (Probabilidad de que el estado cambie)
            # 1 - P(permanecer en el mismo estado en t+1)
            transition_matrix = self.A
            prob_stay = self.state_probabilities[current_state] * transition_matrix[current_state, current_state]
            transition_risk = 1.0 - prob_stay
            
            # Próximo vector de probabilidad (P(s_{t+1}))
            next_state_probs = self.state_probabilities @ self.A
            
            return self.REGIMES[current_state], transition_risk, next_state_probs

        except Exception as e:
            logger.error(f"HMM Update Error: {e}")
            return self.REGIMES[self.last_state], 0.5, self.pi

    def calibrate(self, historical_returns: np.ndarray):
        """
        Calibración simple basada en momentos estadísticos de datos históricos.
        QUÉ: Ajusta las medias y varianzas de los estados HMM.
        POR QUÉ: Los mercados cambian; las 'piscinas' y 'ríos' evolucionan.
        """
        try:
            if len(historical_returns) < 100:
                return
            
            # Dividir retornos en cuantiles de volatilidad para mapear estados
            vol = np.std(historical_returns)
            mu = np.mean(historical_returns)
            
            # Low Vol (0)
            self.means[0] = mu * 0.1
            self.stds[0] = vol * 0.5
            
            # High Vol (1)
            self.means[1] = mu * 0.2
            self.stds[1] = vol * 2.0
            
            # Bull (2)
            self.means[2] = max(0.0001, mu + 0.5 * vol)
            self.stds[2] = vol * 1.2
            
            # Bear (3)
            self.means[3] = min(-0.0001, mu - 0.5 * vol)
            self.stds[3] = vol * 1.2
            
            logger.info(f"🧠 [HMM] Calibrated with {len(historical_returns)} samples. Vol Reference: {vol:.5f}")
            
        except Exception as e:
            logger.error(f"HMM Calibration Error: {e}")
