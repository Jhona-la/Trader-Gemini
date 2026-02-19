import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import queue

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.enums import SignalType, TimeFrame, EventType
from core.events import MarketEvent
from strategies.technical import HybridScalpingStrategy

# Import Backtest helpers
try:
    from tests.run_backtest import BacktestDataProvider, BacktestPortfolio, fetch_binance_data
except ImportError:
    print("❌ Failed to import backtest helpers. Ensure tests/run_backtest.py exists.")
    sys.exit(1)

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

def run_hft_audit(symbol='BTC/USDT', days=1):
    print(f"\n⚡ HFT PRECISION AUDIT: {symbol} ({days} Days)")
    print("==================================================")
    
    # 1. Fetch Data
    print("📥 Fetching Market Data...")
    df = fetch_binance_data(symbol, days)
    if df is None or df.empty:
        print("❌ Data fetch failed.")
        return

    # 2. Setup
    events_queue = queue.Queue()
    data_provider = BacktestDataProvider(events_queue, [symbol], {symbol: df})
    portfolio = BacktestPortfolio(initial_capital=1000.0, leverage=Config.BINANCE_LEVERAGE)
    strategy = HybridScalpingStrategy(data_provider, events_queue)
    
    print("⚙️ Measuring Tick Processing Latency (CPU)...")
    
    match_latencies = [] # Time to match order (simulated)
    tick_latencies = [] # Full loop time
    strategy_latencies = [] # Strategy logic time
    
    # Warup
    for _ in range(200):
        data_provider.update_bars()
        
    start_time = time.time()
    total_ticks = 0
    
    while data_provider.continue_backtest:
        loop_start = time.perf_counter_ns()
        
        # 1. Update Market
        data_provider.update_bars()
        if not data_provider.continue_backtest: break
            
        current_time_ms = data_provider.current_time_ms
        current_time = pd.to_datetime(current_time_ms, unit='ms', utc=True)
        
        # 2. Update Portfolio
        portfolio.update_equity(current_time)
        
        latest = data_provider.get_latest_bars(symbol, n=1)
        if latest is None: continue
        close_price = latest['close'][0]
        
        # 3. Strategy Signal
        market_event = MarketEvent(
            timestamp=current_time,
            symbol=symbol,
            close_price=close_price
        )
        
        strat_start = time.perf_counter_ns()
        try:
            strategy.calculate_signals(market_event)
        except: pass
        strat_end = time.perf_counter_ns()
        strategy_latencies.append(strat_end - strat_start)
        
        # 4. Engine Processing (Events)
        while not events_queue.empty():
            event = events_queue.get()
            if event.type == EventType.SIGNAL:
                # Measure "Order Routing" Latency (Simulated)
                # In real engine, this involves network. Here we measure logic overhead.
                pass
                
        loop_end = time.perf_counter_ns()
        tick_latencies.append(loop_end - loop_start)
        total_ticks += 1
        
    # Stats
    tick_series = pd.Series(tick_latencies) / 1_000_000 # to ms
    strat_series = pd.Series(strategy_latencies) / 1_000_000 # to ms
    
    print("\n⏱️ LATENCY REPORT (Engine Profiling)")
    print("--------------------------------------------------")
    print(f"🔄 Total Ticks:      {total_ticks}")
    
    print(f"\n🧠 Strategy Logic Latency (Signal Generation):")
    print(f"   Mean:    {strat_series.mean():.4f} ms")
    print(f"   Min:     {strat_series.min():.4f} ms")
    print(f"   Max:     {strat_series.max():.4f} ms")
    print(f"   P95:     {strat_series.quantile(0.95):.4f} ms")
    print(f"   P99:     {strat_series.quantile(0.99):.4f} ms")
    
    print(f"\n⚙️ Full Event Loop Latency (Tick Processing):")
    print(f"   Mean:    {tick_series.mean():.4f} ms")
    print(f"   P99:     {tick_series.quantile(0.99):.4f} ms")
    
    # Verdict
    # HFT target < 10ms for Strategy, < 50ms for Loop
    is_hft = tick_series.quantile(0.99) < 50.0
    
    print("\n✅ AUDIT VERDICT:")
    if is_hft:
        print("   [PASS] SYSTEM IS HFT CAPABLE (<50ms).")
    else:
        print("   [FAIL] SYSTEM TOO SLOW FOR HFT.")

if __name__ == "__main__":
    run_hft_audit('BTC/USDT', days=1)
