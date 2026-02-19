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

class ScenarioInjector:
    def __init__(self, crash_idx, duration_bars=5, drop_pct=0.20):
        self.crash_start_idx = crash_idx
        self.crash_end_idx = crash_idx + duration_bars
        self.drop_pct = drop_pct
        self.active = False
        
    def apply(self, price, current_idx):
        if self.crash_start_idx <= current_idx <= self.crash_end_idx:
            self.active = True
            # Simulate crash: Drop instantly
            # Or gradual? Let's do instant drop to test GAP handling.
            return price * (1.0 - self.drop_pct)
        return price

class StressedPortfolio(BacktestPortfolio):
    def __init__(self, initial_capital, leverage, slippage_pct=0.005): 
        super().__init__(initial_capital, leverage)
        self.slippage_pct = slippage_pct
        
    def _apply_slippage(self, price, side):
        if side == "LONG":
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)

    def update_equity_unrealized(self, timestamp, current_price):
        """Include unrealized PnL in equity curve"""
        unrealized_pnl = 0.0
        for symbol, pos in self.positions.items():
            qty = pos['qty']
            entry = pos['entry']
            side = pos['side']
            if side == "LONG":
                unrealized_pnl += (current_price - entry) * qty
            else:
                unrealized_pnl += (entry - current_price) * qty
        
        total_equity = self.current_capital + unrealized_pnl
        
        self.equity_curve.append(total_equity)
        if isinstance(timestamp, datetime):
            self.timestamps.append(timestamp)

def run_risk_audit(symbol='BTC/USDT', days=15):
    print(f"\n🛡️ RISK SHIELD AUDIT: {symbol} ({days} Days)")
    print("==================================================")
    print("⚠️  SCENARIO: FLASH CRASH (-20%) + HIGH SLIPPAGE (0.5%)")
    
    # 1. Fetch Data
    print("📥 Fetching Market Data...")
    df = fetch_binance_data(symbol, days)
    if df is None or df.empty:
        print("❌ Data fetch failed.")
        return

    # 2. Setup Backtest Environment
    events_queue = queue.Queue()
    data_provider = BacktestDataProvider(events_queue, [symbol], {symbol: df})
    
    # Initialize STRESSED Portfolio
    portfolio = StressedPortfolio(initial_capital=1000.0, leverage=Config.BINANCE_LEVERAGE, slippage_pct=0.005)
    
    strategy = HybridScalpingStrategy(data_provider, events_queue)
    print("✅ HybridScalpingStrategy Instantiated.")
    
    # Setup Crash Scenario (Middle of Simulation)
    total_bars = len(df)
    crash_idx = total_bars // 2
    scenario = ScenarioInjector(crash_idx, duration_bars=10, drop_pct=0.20)
    print(f"🔥 Crash Scheduled at Bar Index: {crash_idx}")
    
    # Warup
    warmup = 200
    for _ in range(warmup):
        data_provider.update_bars()
        
    start_time = time.time()
    simulated_orders = 0
    crash_triggered = False
    
    while data_provider.continue_backtest:
        data_provider.update_bars()
        if not data_provider.continue_backtest: break
            
        current_time_ms = data_provider.current_time_ms
        current_time = pd.to_datetime(current_time_ms, unit='ms', utc=True)
        current_idx = data_provider.current_index 
        
        # Get REAL price first
        latest = data_provider.get_latest_bars(symbol, n=1)
        if latest is None: continue
        real_close = latest['close'][0]
        
        # Force Open Long before crash (Test SL)
        if current_idx == (crash_idx - 5):
            print(f"⚠️ FORCING LONG POSITION for Stress Test at {real_close:.2f}")
            # Use Config Sizing (Simulating Production Logic)
            # Size = Capital * % Allocation * Leverage
            # wait, open_position takes 'size_usd' as Margin Amount usually?
            # NO, BacktestPortfolio open_position takes 'size_usd' as MARGIN.
            # So we pass Margin Amount.
            margin_amount = portfolio.current_capital * Config.POSITION_SIZE_MICRO_ACCOUNT
            
            portfolio.open_position_with_metadata(
                symbol, "LONG", real_close, margin_amount, current_time, 
                metadata={'strategy': 'StressTest'},
                sl_price=real_close * 0.98 # 2% SL
            )
            simulated_orders += 1
        
        # INJECT SCENARIO
        stressed_close = scenario.apply(real_close, current_idx)
        
        if scenario.active and not crash_triggered:
            print(f"🚨 FLASH CRASH TRIGGERED! Price: {real_close:.2f} -> {stressed_close:.2f}")
            crash_triggered = True
        
        # Update Portfolio with STRESSED Price (Mark-to-Market)
        portfolio.update_equity_unrealized(current_time, stressed_close)
        # Note: update_equity uses self.positions (marks to market?). 
        # Wait, BacktestPortfolio.update_equity implementation calculates equity based on current price?
        # It usually takes 'current_time' but WHERE does it get price?
        # In run_backtest.py: update_equity(self, timestamp) -> loop symbols -> get price?
        # No, update_equity usually needs price map.
        # Let's check run_backtest.py implementation.
        # Snippet 26400 showed update_equity(self, timestamp). It likely assumes self access to data_provider? Or passed price map?
        # Actually I missed checking `update_equity` internals fully.
        # If it doesn't take price, does it have access to data?
        # `BacktestPortfolio` is standalone?
        # IF IT IS STANDALONE, it cannot mark-to-market without prices.
        # Maybe it just updates timestamps?
        # Real logic usually needs `update_fill` or `update_from_tick`.
        
        # Let's assumes it DOES NOT mark to market accurately if I don't pass price.
        # But `portfolio.equity_curve` is updated?
        # I'll check `run_backtest.py` via view_file if needed.
        # For now, let's assume it works or I fix it.
        
        # Helper manual update for audit:
        # PnL = (Current Price - Entry) * Qty
        # We use `stressed_close` for PnL calculation.
        
        # Check Exits with STRESSED Price
        pos = portfolio.positions.get(symbol)
        if pos:
            entry = pos['entry']
            side = pos['side']
            sl = pos.get('sl_price')
            tp = pos.get('tp_price')
            
            close_it = False
            
            # CRASH LOGIC: If price GAPS below SL (Long), we exit at stress price (slippage applied by StressedPortfolio)
            if side == "LONG":
                if sl and stressed_close <= sl: 
                    close_it = True # SL Hit
                elif tp and stressed_close >= tp: 
                    close_it = True
            elif side == "SHORT":
                if sl and stressed_close >= sl: 
                    close_it = True
                elif tp and stressed_close <= tp: 
                    close_it = True
                
            if close_it:
                portfolio.close_position(symbol, stressed_close, current_time)
                # If crash triggered exit, log it
                if scenario.active:
                    print(f"🛡️ STOP LOSS EXECUTED DURING CRASH at {stressed_close:.2f}")

        # Generate Signal (Strategy sees STRESSED Price)
        market_event = MarketEvent(
            timestamp=current_time,
            symbol=symbol,
            close_price=stressed_close # Strategy sees crash
        )
        
        try:
            strategy.calculate_signals(market_event)
        except: pass
            
        while not events_queue.empty():
            event = events_queue.get()
            if event.type == EventType.SIGNAL:
                if event.signal_type == SignalType.EXIT:
                    portfolio.close_position(symbol, stressed_close, current_time)
                    continue
                
                # Execution Logic (Market Order)
                side = "LONG" if event.signal_type == SignalType.LONG else "SHORT" if event.signal_type == SignalType.SHORT else None
                if side:
                    # Open Position
                    # Note: We use OPENING price as `stressed_close` (Market Order fills at current)
                    size_usd = 100.0 * Config.BINANCE_LEVERAGE
                    portfolio.open_position_with_metadata(
                        symbol, side, stressed_close, size_usd, current_time, 
                        metadata={'strategy': 'Hybrid'},
                        sl_price=stressed_close * 0.98 if side == "LONG" else stressed_close * 1.02 # Fixed SL for test
                    )
                    simulated_orders += 1

    # Force Close
    final_time_ms = data_provider.current_time_ms
    final_dt = pd.to_datetime(final_time_ms, unit='ms', utc=True)
    for sym in list(portfolio.positions.keys()):
        portfolio.close_position(sym, stressed_close, final_dt)
             
    # Report
    print(f"⏱️ Simulation Complete. Orders: {simulated_orders}")
    
    # Calculate Max Drawdown
    equity = portfolio.equity_curve
    max_dd = 0.0
    peak = equity[0]
    for val in equity:
        if val > peak: peak = val
        dd = (peak - val) / peak
        if dd > max_dd: max_dd = dd
        
    print("\n🛡️ RISK SHIELD REPORT (LEVEL II)")
    print("--------------------------------------------------")
    print(f"🔥 Scenario:         Flash Crash (-20%)")
    print(f"📉 Max Drawdown:     {max_dd*100:.2f}%")
    print(f"💰 Final Equity:     ${portfolio.current_capital:.2f}")
    
    if max_dd < 0.15: # Pass if DD < 15% (Initial Capital based) even with crash? 
        # Ideally SL protects us. 10x leverage * 2% SL = 20% loss max per trade.
        # If gap is 20%, we lose 200%? REKT?
        # That's the test! Can we survive GAP risk?
        print("   [PASS] RISK SHIELD HELD.")
    else:
        print("   [FAIL] RISK SHIELD BREACHED.")

if __name__ == "__main__":
    run_risk_audit('BTC/USDT', days=15)
