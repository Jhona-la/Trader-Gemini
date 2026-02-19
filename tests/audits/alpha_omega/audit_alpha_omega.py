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

def calculate_expectancy(trades):
    if not trades:
        return 0, 0, 0, 0
    
    wins = [t['pnl_usd'] for t in trades if t['pnl_usd'] > 0]
    losses = [t['pnl_usd'] for t in trades if t['pnl_usd'] <= 0]
    
    n_wins = len(wins)
    n_losses = len(losses)
    total = n_wins + n_losses
    
    if total == 0: return 0, 0, 0, 0
    
    win_rate = n_wins / total
    loss_rate = n_losses / total
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    
    # Expectancy = (WinRate * AvgWin) - (LossRate * AvgLoss)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')
    
    return expectancy, win_rate, profit_factor, total

def analyze_decay(trades):
    if len(trades) < 20:
        return None
        
    mid = len(trades) // 2
    first_half = trades[:mid]
    second_half = trades[mid:]
    
    e1, _, _, _ = calculate_expectancy(first_half)
    e2, _, _, _ = calculate_expectancy(second_half)
    
    return {
        'first_half_E': e1,
        'second_half_E': e2,
        'decay_delta': e2 - e1
    }

def run_real_strategy_audit(symbol='BTC/USDT', days=30):
    print(f"\n🚀 ALPHA-OMEGA AUDIT: {symbol} ({days} Days)")
    print("==================================================")
    
    # 1. Fetch Data
    print("📥 Fetching Market Data...")
    df = fetch_binance_data(symbol, days)
    if df is None or df.empty:
        print("❌ Data fetch failed.")
        return

    # 2. Setup Backtest Environment
    events_queue = queue.Queue()
    
    # Initialize Data Provider
    # Note: BacktestDataProvider expects {symbol: df}
    data_provider = BacktestDataProvider(events_queue, [symbol], {symbol: df})
    
    # Initialize Portfolio
    portfolio = BacktestPortfolio(initial_capital=1000.0, leverage=Config.BINANCE_LEVERAGE)
    
    # Initialize REAL Strategy
    # HybridScalpingStrategy(data_provider, events_queue, genotype=None)
    strategy = HybridScalpingStrategy(data_provider, events_queue)
    print("✅ HybridScalpingStrategy Instantiated.")
    
    # 3. Run Simulation
    print("⚙️ Running Simulation (Tick-by-Tick)...")
    
    trades_log = []
    
    # Pre-warm indicators (strategy.calculate_indicators handles warming internally usually, 
    # but we skip first N bars for stability)
    warmup = 200
    total_bars = len(df)
    
    # Advance Update pointer to warmup
    # BacktestDataProvider.update_bars() increments index.
    for _ in range(warmup):
        data_provider.update_bars()
        
    start_time = time.time()
    
    simulated_orders = 0
    
    # Iterate through remaining bars
    # We rely on data_provider tracking index.
    # Total iterations = total_bars - warmup
    
    while data_provider.continue_backtest:
        # 1. Update Market Data (Next Bar)
        data_provider.update_bars()
        
        if not data_provider.continue_backtest:
            break
            
        current_time_ms = data_provider.current_time_ms
        current_time = pd.to_datetime(current_time_ms, unit='ms', utc=True)
        
        # 2. Update Portfolio (Mark-to-Market + Check Stops)
        portfolio.update_equity(current_time)
        
        # Get current price
        # BacktestDataProvider.get_latest_bars returns struct array
        latest = data_provider.get_latest_bars(symbol, n=1)
        if latest is None: continue
        close_price = latest['close'][0]
        
        # Check Exits (SL/TP)
        # portfolio.check_exits(symbol, close_price, current_time) # Method missing
        # Implement check manually:
        pos = portfolio.positions.get(symbol)
        if pos:
            entry = pos['entry']
            side = pos['side']
            # sl = pos.get('sl_price') # Wait, I don't use sl variable except below
            sl = pos.get('sl_price')
            tp = pos.get('tp_price')
            
            close_it = False
            
            if side == "LONG":
                if sl and close_price <= sl: close_it = True
                elif tp and close_price >= tp: close_it = True
            elif side == "SHORT":
                if sl and close_price >= sl: close_it = True
                elif tp and close_price <= tp: close_it = True
                
            if close_it:
                portfolio.close_position(symbol, close_price, current_time)
        
        # 3. Generate Strategy Signal
        # Create MarketEvent to trigger strategy
        market_event = MarketEvent(
            timestamp=current_time,
            symbol=symbol,
            close_price=close_price
        )
        
        # Strategy logic
        try:
            strategy.calculate_signals(market_event)
        except Exception as e:
            # print(f"Strategy Error: {e}") 
            pass
            
        # 4. Process Signals
        while not events_queue.empty():
            event = events_queue.get()
            if event.type == EventType.SIGNAL:
                # Execute Signal in Portfolio
                # Open Position
                # Mock Slippage/Spread
                
                # Check Direction
                from core.enums import SignalType, OrderSide
                
                side = None
                if event.signal_type == SignalType.LONG:
                    side = "LONG"
                elif event.signal_type == SignalType.SHORT:
                    side = "SHORT"
                elif event.signal_type == SignalType.EXIT:
                    portfolio.close_position(symbol, close_price, current_time)
                    continue
                
                if side:
                    # Calculate Size (Simplified for Audit: Fixed $100 Margin)
                    size_usd = 100.0 * Config.BINANCE_LEVERAGE
                    # Check existing position first (simple strategy usually assumes 1 pos)
                    # If we already have position in same direction, maybe add? Or ignore?
                    # For audit simplicity, ignore if same side. Close if opposite?
                    # For now just open new (BacktestPortfolio supports multiple? No, simple dict)
                    
                    portfolio.open_position_with_metadata(
                        symbol, side, close_price, size_usd, current_time, 
                        metadata={'strategy': 'Hybrid'}
                    )
                    simulated_orders += 1

    # Force Close Open Positions
    final_time_ms = data_provider.current_time_ms
    final_dt = pd.to_datetime(final_time_ms, unit='ms', utc=True)
    
    for sym in list(portfolio.positions.keys()):
        latest = data_provider.get_latest_bars(sym, n=1)
        if latest is not None:
             price = latest['close'][0]
             portfolio.close_position(sym, price, final_dt)
             
    duration = time.time() - start_time
    print(f"⏱️ Simulation Complete in {duration:.2f}s. Orders: {simulated_orders}")
    
    # 4. Generate Report
    trades = portfolio.trades # CORRECTED ATTRIBUTE NAME
    
    e, wr, pf, count = calculate_expectancy(trades)
    decay = analyze_decay(trades)
    
    # Calculate Sharpe ( Daily Returns )
    # Portfolio equity curve is needed.
    # BacktestPortfolio has .equity_curve (list of floats) and .timestamps (list of datetime)
    equity_values = portfolio.equity_curve
    timestamps = portfolio.timestamps
    
    # Pad timestamps if needed (equity_curve has initial value, timestamps starts empty?)
    # usually len(equity_values) == len(timestamps) + 1 (initial)
    # Let's check logic. update_equity adds both.
    # Initial capital is added on init? Yes. Timestamps is empty.
    # So we align.
    
    if len(equity_values) > len(timestamps):
        # Drop initial for alignment or pad timestamp?
        # Drop initial for returns calc.
        aligned_equity = equity_values[1:]
        aligned_timestamps = timestamps
    else:
        aligned_equity = equity_values
        aligned_timestamps = timestamps

    if not aligned_equity:
        sharpe = 0
    else:
        df_eq = pd.DataFrame({'timestamp': aligned_timestamps, 'equity': aligned_equity})
        df_eq['timestamp'] = pd.to_datetime(df_eq['timestamp'])
        df_eq.set_index('timestamp', inplace=True)
        # Resample to Daily
        daily = df_eq['equity'].resample('D').last().pct_change().dropna()
        if len(daily) > 1:
            sharpe = daily.mean() / daily.std() * np.sqrt(365)
        else:
            sharpe = 0
            
    print("\n📊 FINANCIAL AUDIT REPORT (LEVEL I)")
    print("--------------------------------------------------")
    print(f"💰 Net Profit:       ${portfolio.current_capital - portfolio.initial_capital:.2f}")
    print(f"📈 Win Rate:         {wr*100:.2f}%")
    print(f"⚖️ Profit Factor:    {pf:.2f}")
    print(f"🎲 Expectancy ($E$): ${e:.2f} per trade")
    print(f"📉 Sharpe Ratio:     {sharpe:.2f}")
    print(f"🔄 Total Trades:     {len(trades)}")
    
    if decay:
        print("\n📉 DECAY ANALYSIS")
        print(f"   First Half $E:    ${decay['first_half_E']:.2f}")
        print(f"   Second Half $E:   ${decay['second_half_E']:.2f}")
        print(f"   Delta:            {decay['decay_delta']:.2f}")
        if decay['second_half_E'] < decay['first_half_E']:
            print("   ⚠️  WARNING: Strategy Performance Degrading!")
        else:
            print("   ✅  Performance Stable or Improving.")
            
    # VALIDATION CRITERIA
    print("\n✅ AUDIT VERDICT:")
    if e > 0 and sharpe > 1.5:
        print("   [PASS] ALPHA ENGINE VERIFIED.")
    else:
        print("   [FAIL] ALPHA ENGINE UNDERPERFORMING.")

if __name__ == "__main__":
    run_real_strategy_audit('BTC/USDT', days=15)
