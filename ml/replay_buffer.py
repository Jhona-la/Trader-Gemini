"""
🧠 Prioritized Experience Replay (PER) - Phase 9 (NEURAL-FORTRESS)
Memoria probabilística para la estrategia ML. Prioriza muestrear "Cisnes Negros" 
(trades que terminan en pérdidas catastróficas o caídas estructurales) 
para acelerar el aprendizaje del modelo PPO.
"""
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        :param capacity: Total número de transiciones almacenadas
        :param alpha: Nivel de priorización (0 = uniforme, 1 = priorización total)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, 
            log_prob: float, axioma_reason: str, error: float = None):
        """
        Añade una experiencia al buffer.
        State, action, reward, next_state, log_prob, axioma_reason.
        El error inicial (prioridad) es el máximo conocido para asegurar 
        que las nuevas experiencias sean muestreadas al menos una vez.
        """
        # Calcular error/prioridad inicial
        max_prio = max(self.priorities.max() if self.buffer else 1.0, 1.0)
        
        # Inyectar severidad si Axioma falló duro
        if "THESIS" in axioma_reason or "CRASH" in axioma_reason:
            max_prio *= 2.0  # Double priority for structural failures
        elif reward < -0.5:
             max_prio *= 1.5 # Boost priority for high negative rewards
            
        if error is not None:
             max_prio = error
             
        experience = (state, action, reward, next_state, log_prob, axioma_reason)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Muestrea un batch probabilísticamente basado en las prioridades.
        Beta controla el sesgo de importancia (Importance Sampling).
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
            
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # P(i) = p_i^alpha / sum(p_k^alpha)
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance Sampling Weights: W(i) = (N * P(i)) ^ -beta
        total_items = len(self.buffer)
        weights = (total_items * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize to [0, 1] bounds for stability
        
        return experiences, indices, np.array(weights, dtype=np.float32)
        
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Actualiza las prioridades basadas en el Temporal Difference Error post-entrenamiento.
        """
        for count, idx in enumerate(indices):
            # Agregar epsilon mínimo para que nunca tenga probabilidad 0 de ser vuelto a muestrear
            self.priorities[idx] = abs(td_errors[count]) + 1e-5
            
    def __len__(self):
        return len(self.buffer)

    def save(self, filepath: str):
        """
        Persistencia: Guarda la memoria en disco (atomicity mediante joblib provisional).
        """
        try:
            import joblib
            joblib.dump({
                'buffer': self.buffer,
                'pos': self.pos,
                'priorities': self.priorities,
                'alpha': self.alpha,
                'capacity': self.capacity
            }, filepath)
        except Exception as e:
            from utils.logger import logger
            logger.error(f"Failed to save ReplayBuffer: {e}")

    def load(self, filepath: str):
        """
        Persistencia: Carga la memoria desde el disco.
        """
        import os
        if not os.path.exists(filepath):
            return
        try:
            import joblib
            data = joblib.load(filepath)
            self.buffer = data.get('buffer', [])
            self.pos = data.get('pos', 0)
            self.priorities = data.get('priorities', self.priorities)
            self.alpha = data.get('alpha', self.alpha)
            self.capacity = data.get('capacity', self.capacity)
            from utils.logger import logger
            logger.info(f"ReplayBuffer loaded {len(self.buffer)} experiences from {filepath}.")
        except Exception as e:
            from utils.logger import logger
            logger.error(f"Failed to load ReplayBuffer: {e}")
