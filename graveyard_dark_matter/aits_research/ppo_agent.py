"""
AITS Phase 5: Reinforcement Learning Decision Core
PPO Agent Implementation

This script trains a Proximal Policy Optimization (PPO) agent
using the custom AITSTradingEnv.
PPO is chosen for its stability in stochastic environments like financial markets.
"""

import logging
import os
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl_trading_env import AITSTradingEnv
except ImportError:
    PPO = None
    DummyVecEnv = None
    AITSTradingEnv = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = "rl_models"
LOGS_DIR = "rl_logs"

def train_ppo_agent():
    if not PPO:
        logging.error("stable-baselines3 is not installed. Run: pip install stable-baselines3")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    logging.info("--- Initializing AITS PPO Agent ---")
    
    # 1. Instantiate the Environment
    # DummyVecEnv wraps the environment for vectorization (required by SB3)
    env = DummyVecEnv([lambda: AITSTradingEnv(initial_balance=13.0, max_steps=1000)])
    
    # 2. Instantiate the PPO Agent
    # MlpPolicy uses a standard Multi-Layer Perceptron neural network for the policy (Actor) and value function (Critic).
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR, learning_rate=0.0003)
    
    # 3. Train the Agent
    # In production, timesteps would be in the millions (e.g., 10_000_000)
    training_steps = 10000
    logging.info(f"🚀 Starting PPO Training for {training_steps} timesteps...")
    
    model.learn(total_timesteps=training_steps, progress_bar=True)
    
    # 4. Save the Model
    model_path = os.path.join(MODELS_DIR, "ppo_aits_v1")
    model.save(model_path)
    logging.info(f"✅ Model saved successfully to {model_path}.zip")

    # 5. Evaluate the Agent (Simulation)
    logging.info("--- Evaluating Trained Agent ---")
    obs = env.reset()
    for _ in range(10): # Run a quick 10-step simulation
        # predict() returns the optimal action and the hidden state
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, rewards, dones, info = env.step(action)
        
        # Decode action for logging
        action_name = {0: "FLAT", 1: "LONG", 2: "SHORT", 3: "HEDGE"}.get(action[0], "UNKNOWN")
        logging.info(f"Action Taken: {action_name} | Reward: {rewards[0]:.4f} | Equity: ${info[0]['equity']:.2f}")
        
        if dones[0]:
            logging.info("Episode finished during evaluation.")
            break

if __name__ == "__main__":
    train_ppo_agent()
