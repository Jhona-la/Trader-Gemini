import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from utils.logger import logger

@dataclass
class TradeOutcome:
    """
    Data Transfer Object for Trade Results.
    Used to pass full context from Portfolio/Engine to Strategy/Learner.
    """
    entry_price: float
    exit_price: float
    direction: int # 1 for Long, -1 for Short
    leverage: float
    max_adverse_excursion: float # MAE
    max_favorable_excursion: float # MFE
    duration_seconds: float
    latency_ms: float = 0.0
    entry_features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0: return 0.0
        raw_pnl = (self.exit_price - self.entry_price) / self.entry_price
        return raw_pnl * self.direction * self.leverage

class RewardSystem:
    """
    Reward System for Neural Fortress.
    Calculates non-linear rewards to incentivize stability and penalize extreme risks.
    """
    def __init__(self, drawdown_penalty_factor: float = 2.0, skew_penalty_factor: float = 0.5):
        self.drawdown_penalty_factor = drawdown_penalty_factor
        self.skew_penalty_factor = skew_penalty_factor
        self.rolling_returns = []
        self.window_size = 50 

    def calculate_reward(self, outcome: TradeOutcome, current_drawdown: float) -> float:
        """
        Calculate the reward for a completed trade using TradeOutcome.
        
        Args:
            outcome (TradeOutcome): The result of the trade.
            current_drawdown (float): Current drawdown percentage (e.g., 0.02 for 2%).
        
        Returns:
            float: The calculated reward.
        """
        # 1. Base Reward: Tanh of PnL
        pnl = outcome.pnl_pct
        scaled_pnl = pnl * 10
        base_reward = np.tanh(scaled_pnl)

        # 2. Drawdown Penalty (Exponential)
        drawdown_penalty = self.drawdown_penalty_factor * (np.exp(current_drawdown * 10) - 1.0)
        
        # NEURAL-FORTRESS: Latency Penalty (Phase 9)
        # Penalize if latency > 100ms. 
        # Structure: w3 * (latency_ms / 100)
        latency_penalty = 0.0
        if outcome.latency_ms > 0:
            # 100ms = 0.1s. If latency is 200ms, penalty = 0.1 * (200/100) = 0.2
            latency_factor = 0.1 
            latency_penalty = latency_factor * (outcome.latency_ms / 100.0)

        # 3. Volatility/Skewness Penalty
        self.rolling_returns.append(pnl)
        if len(self.rolling_returns) > self.window_size:
            self.rolling_returns.pop(0)
            
        skew_penalty = 0.0
        if len(self.rolling_returns) >= 10:
            returns_array = np.array(self.rolling_returns)
            mean = np.mean(returns_array)
            std = np.std(returns_array)
            
            if std > 1e-6:
                skewness = np.mean(((returns_array - mean) / std) ** 3)
                if skewness < -0.5:
                   skew_penalty = self.skew_penalty_factor * abs(skewness)

        # 4. Total Reward (Weighted)
        # R = w1*PnL - w2*DD - w3*Latency - w4*Skew
        total_reward = base_reward - drawdown_penalty - latency_penalty - skew_penalty
        
        return total_reward
        
    def reset(self):
        self.rolling_returns = []
