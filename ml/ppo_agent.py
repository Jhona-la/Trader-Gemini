import os
import numpy as np
from typing import Dict, Tuple, List
from utils.logger import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        Module = object

if TORCH_AVAILABLE:
    class ActorCriticNetwork(nn.Module):
        """
        AITS Phase 5: Actor-Critic architecture for PPO.
        Optimized for <100μs inference latency on CPU.
        """
        def __init__(self, state_dim: int, hidden_dim: int = 64):
            super().__init__()
            # Shared Feature Extractor
            self.base = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            )
            
            # Actor Head: Outputs a continuous action [-1, 1] for Direction/Aggressiveness
            # Negative: Short Bias, Positive: Long Bias. Magnitude: Sizing / Confidence.
            self.actor_mean = nn.Linear(hidden_dim, 1)
            # Log std for continuous action exploration
            self.actor_log_std = nn.Parameter(torch.zeros(1))
            
            # Critic Head: Predicts Expected Value of the state
            self.critic = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            features = self.base(x)
            mean = torch.tanh(self.actor_mean(features)) # Bounds between [-1, 1]
            log_std = self.actor_log_std.expand_as(mean)
            std = torch.exp(log_std)
            value = self.critic(features)
            return mean, std, value

class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    Handles continuous action spaces for dynamic position sizing and confidence vetoing.
    """
    def __init__(self, state_dim: int = 15, lr: float = 3e-4, gamma: float = 0.99, clip_eps: float = 0.2):
        self.state_dim = state_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        
        if not TORCH_AVAILABLE:
            logger.warning("🧠 [PPOAgent] PyTorch not available. RL Core bypassed.")
            self.network = None
            return
            
        self.device = torch.device("cpu") # Force CPU for nano-latency
        self.network = ActorCriticNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # ⚡ Dynamic Quantization for Inference Speed
        # We only quantize linear layers. Parameters remain FP32 for training.
        # Since PPO trains online, quantization might interfere with dynamic backprop,
        # so we keep it in FP32. CPU FP32 for small MLPs is already <50μs.

    def get_action_and_value(self, state: np.ndarray) -> Tuple[float, float, float]:
        """
        Inference step: Given a state, return the deterministic action (mean), 
        sampled action (for exploration), log probability, and estimated value.
        State should be shape (state_dim,)
        """
        if self.network is None:
            return 0.5, 0.0, 0.0 # Default safe fallback
            
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, std, value = self.network(state_tensor)
            
            # In production inference, we often use deterministic 'mean' or sample it.
            # We sample from Normal distribution for exploration during training
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # We return clamped deterministic action for the engine execution
            # and the raw log_prob for the buffer.
            # For execution: magnitude -> sizing, sign -> directional agreement
            execution_action = torch.clamp(mean, -1.0, 1.0).item()
            return execution_action, log_prob.item(), value.item()

    def update(self, states, actions, log_probs_old, returns, advantages):
        """
        Executes PPO clipped surrogate objective update.
        Called asynchronously to avoid blocking the event loop.
        """
        if self.network is None or len(states) == 0:
            return
            
        self.network.train()
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1).to(self.device)
        log_probs_old = torch.FloatTensor(np.array(log_probs_old)).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).unsqueeze(1).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Epochs
        for _ in range(4): # 4 epochs per batch
            mean, std, value = self.network(states)
            dist = torch.distributions.Normal(mean, std)
            log_probs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ratio
            ratios = torch.exp(log_probs_new - log_probs_old)
            
            # Clipped Surrogate Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss (MSE)
            critic_loss = F.mse_loss(value, returns)
            
            # Total Loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
        logger.info(f"🧠 [PPO Agent] Update complete. Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

ppo_agent = PPOAgent()
