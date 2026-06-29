import asyncio
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.getcwd())

from core.strategy_selector import StrategySelector
from data.binance_loader import BinanceData
from core.portfolio import Portfolio
from config import Config
from utils.logger import logger

async def run_diagnostic():
    logger.info("🧪 Starting Meta-Brain Diagnostic...")
    
    # 1. Setup Mock/Real components
    import queue
    events_queue = queue.Queue()
    
    # Use BTC as the primary proxy for the diagnostic
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    data_provider = BinanceData(events_queue, symbols)
    
    # Load some history for the provider
    logger.info("📡 Fetching recent history for evaluation...")
    await data_provider.update_symbol_list(symbols)
    # Give it a moment to pull data
    await asyncio.sleep(5) 
    
    portfolio = Portfolio(initial_capital=15.0)
    # Mock some performance in Portfolio to see the blend
    portfolio.strategy_performance = {
        'TECHNICAL': {'pnl': 5.2, 'wins': 10, 'losses': 2, 'trades': 12},
        'ML_XGBOOST': {'pnl': -1.2, 'wins': 4, 'losses': 6, 'trades': 10},
    }
    
    # 2. Initialize Selector
    selector = StrategySelector(portfolio=portfolio, data_provider=data_provider)
    
    # 3. RUN THE BRAIN
    selector.update_strategy_rankings()
    
    # 4. REPORT
    print("\n" + "="*50)
    print("🧠 SOVEREIGN META-BRAIN DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Timestamp: {datetime.now(timezone.utc)}")
    print("-"*50)
    
    for strat, data in selector.strategy_health.items():
        score = data['score']
        rank = data['rank']
        mult = selector.get_strategy_multiplier(strat)
        veto = "✅ ALLOWED" if selector.should_allow_trade(strat) else "❌ VETOED"
        
        icon = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
        print(f"{icon} Rank {rank}: {strat:20} | Score: {score:.2f} | Risk Mult: {mult:.1f}x | {veto}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_diagnostic())
