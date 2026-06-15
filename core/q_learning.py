import numpy as np
import pickle
import os
from utils.logger import logger
from config import Config

class QLearningAgent:
    """
    🧠 Q-Learning On-The-Fly para Ajuste Dinámico de TP/SL
    Aprende de las pérdidas (SL) y ganancias (TP) en tiempo real para optimizar hiperparámetros.
    """
    def __init__(self, actions=None, alpha=0.1, gamma=0.9, epsilon=0.1):
        # Acciones: Modificadores de TP/SL (ej. (1.0, 1.0) = no cambiar, (1.2, 0.8) = Aumentar TP, Bajar SL)
        if actions is None:
            self.actions = [
                (1.0, 1.0),   # Baseline
                (1.2, 0.9),   # Expand TP, Tighten SL (Trend following)
                (0.8, 1.1),   # Tighten TP, Expand SL (High win-rate scalping)
                (0.9, 0.9),   # Tighten Both (High volatility, quick exits)
                (1.1, 1.1)    # Expand Both (Low volatility, let it breathe)
            ]
        else:
            self.actions = actions
            
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.pending_trades = {} # Maps symbol -> (state_key, action_idx)
        
        self.model_path = os.path.join(Config.DATA_DIR, "q_learning_table.pkl")
        self.load_model()

    def _get_state_key(self, regime: str, volatility_level: int, obi_velocity_level: int) -> str:
        return f"{regime}_{volatility_level}_{obi_velocity_level}"

    def get_action(self, state_key: str) -> tuple:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
            
        # Epsilon-Greedy Exploration
        if np.random.uniform(0, 1) < self.epsilon:
            action_idx = np.random.choice(len(self.actions))
        else:
            action_idx = np.argmax(self.q_table[state_key])
            
        return action_idx, self.actions[action_idx]

    def update_q_value(self, state_key: str, action_idx: int, reward: float, next_state_key: str):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))
            
        current_q = self.q_table[state_key][action_idx]
        max_next_q = np.max(self.q_table[next_state_key])
        
        # Bellman Equation
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q
        
        # Auto-Save periodically or directly here
        self.save_model()
        logger.debug(f"🧠 [Q-LEARNING] Updated State {state_key} Action {action_idx} -> Q: {new_q:.3f}")

    def save_model(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            logger.error(f"Failed to save Q-Table: {e}")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.q_table = pickle.load(f)
                logger.info(f"🧠 [Q-LEARNING] Loaded Q-Table with {len(self.q_table)} states.")
            except Exception as e:
                logger.error(f"Failed to load Q-Table: {e}")

# Instancia Global
q_agent = QLearningAgent()
