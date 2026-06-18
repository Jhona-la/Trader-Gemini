import logging
import argparse
import sys
import pandas as pd
import numpy as np
from typing import Dict

# Setup Paths
import os
sys.path.append(os.getcwd())

from core.shadow_darwin import ShadowDarwin
from core.simulation import SimDataProvider
from core.genotype import Genotype
from utils.wandb_tracker import wandb_tracker
from config import Config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Trainer")

def load_data(symbol: str, days: int = 30) -> Dict[str, pd.DataFrame]:
    """Generates dummy data for training prototype. In prod, load from DB."""
    logger.info(f"Loading {days} days of data for {symbol}...")
    
    # Generate Synthetic Geometric Brownian Motion for realistic testing
    # S_t = S_0 * exp((mu - 0.5*sigma^2)t + sigma*W_t)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days*1440, freq='1min')
    n = len(dates)
    dt = 1/1440 # 1 minute steps
    mu = 0.0001 # Drift
    sigma = 0.01 # Volatility
    
    # Price path
    s0 = 10000
    prices = np.zeros(n)
    prices[0] = s0
    shocks = np.random.normal(0, 1, n)
    
    for i in range(1, n):
        prices[i] = prices[i-1] * np.exp((mu - 0.5*sigma**2)*1 + sigma*shocks[i])
        
    df = pd.DataFrame(index=dates)
    df['open'] = prices
    df['close'] = prices * (1 + np.random.normal(0, 0.0005, n)) # Slight noise
    df['high'] = np.maximum(df['open'], df['close']) * (1 + abs(np.random.normal(0, 0.001, n)))
    df['low'] = np.minimum(df['open'], df['close']) * (1 - abs(np.random.normal(0, 0.001, n)))
    df['volume'] = np.abs(np.random.normal(100, 50, n)) * prices / 100
    
    # Validation constraints
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    df.index.name = 'timestamp' # Required by SimDataProvider
    
    logger.info(f"Data Loaded: {len(df)} candles.")
    return {symbol: df}

def main():
    parser = argparse.ArgumentParser(description="Trinidad Omega Training Ground")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Symbol to train on')
    parser.add_argument('--epochs', type=int, default=50, help='Number of generations')
    parser.add_argument('--pop_size', type=int, default=20, help='Population size')
    args = parser.parse_args()

    symbol = args.symbol
    data_map = load_data(symbol)
    provider = SimDataProvider(data_map)
    
    # Initialize WandB if configured
    wandb_tracker.init_run(
        project=Config.WANDB_PROJECT,
        entity=Config.WANDB_ENTITY,
        config={
            "pop_size": args.pop_size,
            "epochs": args.epochs,
            "use_neural": True,
            "symbol": symbol
        }
    )

    # Initialize Shadow Darwin with Neural Engine
    darwin = ShadowDarwin(provider, population_size=args.pop_size, use_neural=True, wandb_tracker=wandb_tracker)
    
    logger.info(f"🧬 Starting Neuroevolution for {symbol} ({args.epochs} Epochs)")
    
    best_fitness_history = []
    
    for epoch in range(args.epochs):
        # Run 1 Generation per loop allow hooks
        winner = darwin.run_epoch(symbol, generations=1)
        best_fitness_history.append(winner.fitness_score)
        
        # Log Progress
        if epoch % 5 == 0:
            logger.info(f"--- Epoch {epoch} Complete ---")
            logger.info(f"    Best Fitness: {winner.fitness_score:.4f}")
            logger.info(f"    Genes: TP={winner.genes['tp_pct']:.4f}, SL={winner.genes['sl_pct']:.4f}")
            
    # Final Report
    logger.info("🏆 Training Complete.")
    logger.info(f"Top Fitness: {max(best_fitness_history):.4f}")
    
    # Verify Learning Curve (Simple check)
    if best_fitness_history[-1] > best_fitness_history[0]:
        logger.info("✅ Learning Confirmed: Final Fitness > Initial Fitness.")
    else:
        logger.warning("⚠️ No Learning Detected: Check Hyperparameters.")

    # Finish WandB
    wandb_tracker.finish()

if __name__ == "__main__":
    main()
