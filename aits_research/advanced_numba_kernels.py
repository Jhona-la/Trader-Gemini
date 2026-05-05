"""
AITS Phase 0: Advanced Mathematical and Computational Preparation
This module contains highly optimized Numba kernels for stochastic calculus,
Monte Carlo simulations, and advanced algebraic routines required for AITS.

Dependencies: pip install numba numpy
"""

import numpy as np
import time
import logging

try:
    from numba import njit, prange
except ImportError:
    njit = lambda *args, **kwargs: lambda f: f
    prange = range

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@njit(parallel=True, fastmath=True)
def simulate_geometric_brownian_motion(S0, mu, sigma, T, dt, num_paths):
    """
    Simulates asset price paths using Geometric Brownian Motion (GBM).
    Used for AITS Layer 6 (Risk Survival Governor) Monte Carlo risk assessment.
    
    S0: Initial price
    mu: Drift (expected return)
    sigma: Volatility
    T: Total time period
    dt: Time step increment
    num_paths: Number of simulation paths
    """
    num_steps = int(T / dt)
    paths = np.zeros((num_paths, num_steps + 1), dtype=np.float64)
    paths[:, 0] = S0
    
    # Pre-compute constants
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    for i in prange(num_paths):
        # Generate random normal shocks for the entire path
        Z = np.random.randn(num_steps)
        for t in range(1, num_steps + 1):
            paths[i, t] = paths[i, t-1] * np.exp(drift + diffusion * Z[t-1])
            
    return paths

@njit(fastmath=True)
def calculate_order_flow_imbalance(bid_vols, ask_vols, prev_bid_vols, prev_ask_vols, 
                                   bid_prices, ask_prices, prev_bid_prices, prev_ask_prices):
    """
    Calculates Order Flow Imbalance (OFI), a critical high-frequency institutional feature.
    AITS Layer 3 (Predictive Intelligence) uses this to predict short-term liquidity sweeps.
    """
    n = len(bid_vols)
    ofi = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        # Bid side imbalance
        if bid_prices[i] >= prev_bid_prices[i]:
            bid_imbalance = bid_vols[i]
        elif bid_prices[i] == prev_bid_prices[i]:
            bid_imbalance = bid_vols[i] - prev_bid_vols[i]
        else:
            bid_imbalance = -prev_bid_vols[i]
            
        # Ask side imbalance
        if ask_prices[i] <= prev_ask_prices[i]:
            ask_imbalance = ask_vols[i]
        elif ask_prices[i] == prev_ask_prices[i]:
            ask_imbalance = ask_vols[i] - prev_ask_vols[i]
        else:
            ask_imbalance = -prev_ask_vols[i]
            
        ofi[i] = bid_imbalance - ask_imbalance
        
    return ofi

def run_benchmarks():
    logging.info("Starting Numba GBM Benchmark (10,000 paths, 1,000 steps)...")
    
    # Warmup compilation
    _ = simulate_geometric_brownian_motion(100.0, 0.05, 0.2, 1.0, 0.001, 2)
    
    start_time = time.time()
    paths = simulate_geometric_brownian_motion(S0=50000.0, mu=0.01, sigma=0.4, T=1.0, dt=0.001, num_paths=10000)
    elapsed = time.time() - start_time
    
    logging.info(f"✅ GBM Simulation completed in {elapsed:.4f} seconds.")
    logging.info(f"Generated {paths.shape[0] * paths.shape[1]:,} data points.")
    logging.info(f"Mean Final Price: {np.mean(paths[:, -1]):.2f}")

if __name__ == "__main__":
    run_benchmarks()
