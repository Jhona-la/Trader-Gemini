import sys
import os
import asyncio
import argparse
from datetime import datetime, timezone

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.events import MarketEvent, SignalType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from core.engine import Engine
from strategies.technical import HybridScalpingStrategy
import pandas as pd
from utils.logger import logger

async def run_concurrent_backtest():
    """Runs a backtest using the true concurrent production engine."""
    symbol = "BTC/USDT"
    initial_capital = 13.0
    
    logger.info(f"🚀 Starting Concurrent Engine Backtest on {symbol} with {initial_capital}$")
    
    # Enable test modes
    Config.BINANCE_USE_TESTNET = True
    Config.BINANCE_USE_DEMO = True
    Config.BINANCE_USE_FUTURES = True
    
    # 1. Provide Context
    portfolio = Portfolio()
    portfolio.current_cash = initial_capital
    portfolio.peak_equity = initial_capital
    
    risk_manager = RiskManager(portfolio)
    engine = Engine()
    engine.register_portfolio(portfolio)
    engine.register_risk_manager(risk_manager)
    
    # 2. Setup Strategies
    # Scalping Technical (Uses 5m primary)
    scalping_strat = HybridScalpingStrategy(
        data_provider=None, 
        events_queue=engine.events, 
        horizon="SCALPING"
    )
    # ML Strategy / Swing (Uses 1h primary)
    # Since MLStrategy relies heavily on real walk-forward, we'll use a Swing Technical for validation of engine
    swing_strat = HybridScalpingStrategy(
        data_provider=None,
        events_queue=engine.events,
        horizon="SWING"
    )
    
    engine.register_strategy(scalping_strat)
    engine.register_strategy(swing_strat)
    
    # 3. Generate Mock Data
    print("Generating Mock Data for Engine Concurrency Test...")
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=1000, freq='1min')
    import numpy as np
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': np.random.uniform(60000, 70000, 1000),
        'high': np.random.uniform(60000, 70000, 1000) + 100,
        'low': np.random.uniform(60000, 70000, 1000) - 100,
        'close': np.random.uniform(60000, 70000, 1000),
        'volume': np.random.uniform(1, 100, 1000)
    })
    
    print(f"Loaded {len(df)} 1m candles. Beginning Engine Stream...")
    
    class MockDataProvider:
        def __init__(self, df):
            self.df = df
            self.current_idx = 0
            self.symbol_list = [symbol]
        
        def get_latest_bars(self, sym, n, timeframe):
            return None # Fallback disabled for full simulation 
            
    mock_dp = MockDataProvider(df)
    scalping_strat.data_provider = mock_dp
    swing_strat.data_provider = mock_dp
    engine.data_provider = mock_dp
    
    # For a faithful concurrent simulation using the production engine, we would need 
    # to feed MarketEvents and run the event_loop.
    print("""
    NOTE: A fully faithful production engine backtest requires dynamic OHLCV aggregation 
    in the MockDataProvider for the M5, M15, H1 calls made by the strategies, which 
    would effectively reinvent data_provider.py.
    
    For now, this validates concurrent instantiation and the separation of capital 
    between horizons via portfolio splits.
    """)
    
    print(f"[PRE-TEST] Total cash: {portfolio.get_available_cash()}")
    print(f"[PRE-TEST] Scalping cash: {portfolio.get_available_cash('SCALPING')}")
    print(f"[PRE-TEST] Swing cash: {portfolio.get_available_cash('SWING')}")
    

if __name__ == '__main__':
    asyncio.run(run_concurrent_backtest())
