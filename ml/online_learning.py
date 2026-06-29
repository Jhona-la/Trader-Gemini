import numpy as np
from collections import deque
from utils.logger import logger

class HotAdapterRL:
    """
    [PHASE 2 POWER] Online Reinforcement Learning (Hot Adapter)
    Updates model prediction bias tick-by-tick without full retraining.
    """
    def __init__(self, learning_rate: float = 0.05, max_memory: int = 1000):
        self.learning_rate = learning_rate
        self.experience_buffer = deque(maxlen=max_memory)
        self.bias_vector = {} # {symbol_direction: float}

    def update_weights(self, symbol: str, is_win: bool, pnl_pct: float, direction: str):
        """
        Calculates gradient step based on the immediate trade result.
        """
        key = f"{symbol}_{direction}"
        if key not in self.bias_vector:
            self.bias_vector[key] = 1.0 # Base multiplier

        # Reward formulation
        # If win, increase confidence multiplier slightly. If loss, decrease it sharply.
        reward = pnl_pct * 100 # percentage points
        
        # Asymmetric penalty: Losses hurt more than wins help (capital preservation)
        if not is_win:
            penalty = abs(reward) * 1.5 
            step = - (self.learning_rate * penalty)
        else:
            step = self.learning_rate * reward
            
        old_bias = self.bias_vector[key]
        new_bias = max(0.2, min(2.0, old_bias + step)) # Bound between 0.2x and 2.0x
        
        self.bias_vector[key] = new_bias
        
        self.experience_buffer.append({
            'symbol': symbol,
            'direction': direction,
            'pnl': pnl_pct,
            'bias_shift': new_bias - old_bias
        })
        
        logger.info(f"🧠 [HotAdapterRL] {symbol} {direction} | Trade {'WON' if is_win else 'LOST'} ({pnl_pct*100:.2f}%) -> Bias adjusted to {new_bias:.3f}x")

    def get_bias(self, symbol: str, direction: str) -> float:
        """
        Returns the real-time bias multiplier for the base XGBoost model.
        """
        key = f"{symbol}_{direction}"
        return self.bias_vector[key]
