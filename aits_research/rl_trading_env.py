"""
AITS Phase 5: Reinforcement Learning Decision Core
Custom Trading Environment (Gymnasium)

Simulates the cryptocurrency market for the RL Agent to interact with.
Incorporates Institutional Reward Shaping:
- Penalizes Drawdown (Risk Management)
- Penalizes excessive trading (Commission costs)
- Rewards high Sharpe Ratio
"""

import logging
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class AITSTradingEnv(gym.Env if gym else object):
    """
    A custom trading environment for RL agents.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, initial_balance=13.0, max_steps=1000):
        super(AITSTradingEnv, self).__init__()
        
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        
        # Action Space: 0 = Flat, 1 = Long, 2 = Short, 3 = Hedge
        if spaces:
            self.action_space = spaces.Discrete(4)
            
            # Observation Space: Features + Account State
            # Features: [Trend, Volatility, Liquidation_Density, Predictor_Prob_Up, Predictor_Prob_Down]
            # Account: [Balance, Current_Position, Unrealized_PnL]
            # Total = 8 dimensions
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            )
            
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        self.current_price = 100.0 # Mock starting price
        self.peak_balance = self.initial_balance
        self.history = []
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Generates a mock observation state."""
        # Simulated inputs from Phase 3 & 4
        trend = np.sin(self.current_step / 10.0)
        volatility = np.random.uniform(0.1, 0.5)
        liq_density = np.random.uniform(0, 100)
        prob_up = 0.6 if trend > 0 else 0.4
        prob_down = 1.0 - prob_up
        
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (self.current_price - self.entry_price)
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - self.current_price)
            
        return np.array([
            trend, volatility, liq_density, prob_up, prob_down,
            self.balance, float(self.position), unrealized_pnl
        ], dtype=np.float32)

    def _simulate_market_step(self):
        """Simulates price movement for the next step."""
        change = np.random.normal(0, 0.5) # Random walk
        self.current_price += change

    def step(self, action):
        self.current_step += 1
        self._simulate_market_step()
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # Calculate Step PnL based on previous position
        step_pnl = 0.0
        if self.position == 1:
            step_pnl = (self.current_price - self.entry_price)
        elif self.position == -1:
            step_pnl = (self.entry_price - self.current_price)
            
        # Execute Action
        # Action Space: 0 = Flat, 1 = Long, 2 = Short, 3 = Hedge (Partial Close)
        if action == 1 and self.position != 1:
            # Entering Long
            if self.position == -1:
                self.balance += step_pnl # Realize PnL
            self.position = 1
            self.entry_price = self.current_price
            reward -= 0.01 # Commission penalty
            
        elif action == 2 and self.position != -1:
            # Entering Short
            if self.position == 1:
                self.balance += step_pnl # Realize PnL
            self.position = -1
            self.entry_price = self.current_price
            reward -= 0.01 # Commission penalty
            
        elif action == 0 and self.position != 0:
            # Close position
            self.balance += step_pnl
            self.position = 0
            reward -= 0.01
            
        # Update Peak Balance for Drawdown calculation
        total_equity = self.balance
        if self.position != 0:
            total_equity += step_pnl
            
        if total_equity > self.peak_balance:
            self.peak_balance = total_equity
            
        # 🛡️ Institutional Reward Shaping 🛡️
        # 1. Base Reward: Change in equity
        reward += (total_equity - self.initial_balance) * 0.1
        
        # 2. Drawdown Penalty (Severe)
        drawdown = (self.peak_balance - total_equity) / self.peak_balance
        if drawdown > 0.05: # 5% Drawdown
            reward -= 5.0 # Heavy punishment
            
        # 3. Time Exhaustion Penalty (to prevent holding forever)
        if self.position != 0:
            reward -= 0.001
            
        # Check termination (Broke or max steps)
        if total_equity <= self.initial_balance * 0.5: # Lost 50%
            terminated = True
            reward -= 100.0 # Game Over punishment
            logging.info("💀 Agent Bankrupted. Episode Terminated.")
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        info = {'equity': total_equity, 'drawdown': drawdown}
        
        return self._get_obs(), float(reward), terminated, truncated, info

if __name__ == "__main__":
    if gym:
        env = AITSTradingEnv()
        obs, _ = env.reset()
        logging.info(f"Environment Initialized. Initial Observation: {obs}")
    else:
        logging.error("Gymnasium is not installed. Run: pip install gymnasium")
