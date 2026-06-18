"""
AITS Phase 0: Mathematical Preparation
Demonstrates a Hidden Markov Model (HMM) for Regime Detection.
This will replace the basic heuristic `market_regime.py` classifier in AITS Layer 6.
HMMs probabilistically determine unobservable market states (Chop, Trend, Panic)
based on observable emissions (Returns, Volatility).

Dependencies: pip install hmmlearn numpy
"""

import numpy as np
import logging

try:
    from hmmlearn import hmm
except ImportError:
    hmm = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_market_data(n_samples=500):
    """Generates synthetic returns simulating 3 regimes: Calm, Trending, Panic"""
    np.random.seed(42)
    # Regime 0: Calm/Chop (Low mean, Low vol)
    regime_0 = np.random.normal(0.0001, 0.002, int(n_samples * 0.4))
    # Regime 1: Trending Bull (Positive mean, Med vol)
    regime_1 = np.random.normal(0.0050, 0.005, int(n_samples * 0.4))
    # Regime 2: Panic/Crash (Negative mean, High vol)
    regime_2 = np.random.normal(-0.0100, 0.015, int(n_samples * 0.2))
    
    # Concatenate to simulate market cycles
    returns = np.concatenate([regime_0, regime_1, regime_2, regime_0])
    return returns.reshape(-1, 1)

def run_hmm_simulation():
    if not hmm:
        logging.error("hmmlearn is not installed. Run: pip install hmmlearn")
        return

    logging.info("Generating synthetic market returns...")
    returns = generate_market_data()
    
    # Initialize a Gaussian HMM with 3 hidden states
    logging.info("Training Gaussian Hidden Markov Model (3 States)...")
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    
    # Fit the model to the returns
    model.fit(returns)
    
    # Predict the hidden state for each observation
    hidden_states = model.predict(returns)
    
    logging.info("--- HMM Training Complete ---")
    logging.info(f"Transition Matrix:\n{np.round(model.transmat_, 3)}")
    
    # Analyze the discovered states
    for i in range(model.n_components):
        state_returns = returns[hidden_states == i]
        mean_ret = np.mean(state_returns)
        vol = np.std(state_returns)
        
        # Heuristic labeling based on discovered parameters
        label = "Unknown"
        if vol > 0.01:
            label = "Panic / High Volatility"
        elif mean_ret > 0.003:
            label = "Trending / Directional"
        else:
            label = "Calm / Chop"
            
        logging.info(f"State {i} ({label}): Mean Return = {mean_ret:.4f}, Volatility = {vol:.4f}")
        
    # Live Inference Simulation
    latest_return = np.array([[-0.012]]) # Sudden drop
    state_probs = model.predict_proba(latest_return)
    predicted_state = np.argmax(state_probs)
    
    logging.info("\n--- Live Inference ---")
    logging.info(f"Latest Market Return: {latest_return[0][0]:.4f}")
    logging.info(f"Probabilities: {np.round(state_probs[0], 3)}")
    logging.info(f"AITS Regime Governor detects State: {predicted_state}")

if __name__ == "__main__":
    run_hmm_simulation()
