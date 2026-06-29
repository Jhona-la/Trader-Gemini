"""
🤖 SOPHIA-REWARDS: Asymmetric Reward Shaping Engine (Phase 9 - NEURAL-FORTRESS)
Genera recompensas no lineales para el algoritmo PPO basadas en PnL, Drawdown y feedback de AXIOMA.
"""
from enum import Enum
from typing import Dict, Any

class TesisDecayReason(Enum):
    NONE = 0
    THESIS_DECAY = 1   # Trade failed because structural advantage was lost over time
    DEPTH_CRASH = 2    # Trade failed due to sudden lack of order book liquidity
    MOMENTUM_REVERSE = 3 # Trade failed due to immediate counter-trend momentum

class AdvancedRewardEngine:
    def __init__(self):
        # Coefficients based on Phase 9 Specs
        self.pnl_weight = 10.0
        self.drawdown_penalty_weight = 20.0
        
        # Axioma Penalties
        self.penalty_thesis_decay = -0.5
        self.penalty_depth = -0.1
        self.penalty_momentum = -0.3
        
    def calculate_reward(self, 
                         pnl_pct: float, 
                         max_drawdown_pct: float, 
                         axioma_diagnosis: TesisDecayReason = TesisDecayReason.NONE,
                         duration_seconds: float = 0.0) -> float:
        """
        Calcula el reward final del trade.
        Reward = (PnL_pct * 10) - (Max_Drawdown_pct * 20) - Penales(Axioma)
        """
        # 1. Base PnL Reward
        base_reward = pnl_pct * self.pnl_weight
        
        # 2. Risk/Volatility Penalty (Asymmetric to heavily punish pain)
        # Even winning trades are punished if they suffered massive drawdown first.
        drawdown_penalty = abs(max_drawdown_pct) * self.drawdown_penalty_weight
        
        reward = base_reward - drawdown_penalty
        
        # 3. Time Decay Penalty for Scalping
        # In HFT, being stuck in a trade for > 5 minutes is risky
        if duration_seconds > 300: # 5 minutes
            time_decay = (duration_seconds - 300) / 1000.0 # Small incremental penalty
            reward -= min(time_decay, 0.2)
        
        # 4. AXIOMA Diagnosis Integration
        if axioma_diagnosis == TesisDecayReason.THESIS_DECAY:
            reward += self.penalty_thesis_decay
        elif axioma_diagnosis == TesisDecayReason.DEPTH_CRASH:
            reward += self.penalty_depth
        elif axioma_diagnosis == TesisDecayReason.MOMENTUM_REVERSE:
            reward += self.penalty_momentum
            
        return max(-2.0, min(reward, 2.0)) # Clip terminal reward between [-2, 2] for stable learning

# Global singleton
reward_engine = AdvancedRewardEngine()
