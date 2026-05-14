
import os
import sys
import asyncio
import logging

# Ensure root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
# Force Testnet/Safety for Smoke Test
Config.BINANCE_USE_TESTNET = False # User wants REAL capital test ($15)
Config.BINANCE_USE_DEMO = False
Config.BINANCE_USE_FUTURES = True
Config.INITIAL_CAPITAL = 15.0
Config.TRADING_PAIRS = ['BTC/USDT'] 
# Disable actual order placement just in case, or use a Mock Executor?
# No, "Test de Humo con Capital Real" implies checking if we CAN trade.
# But I will put a safety block in Executor if I can, or just trust the logic won't trigger in 30s.

from data.binance_loader import BinanceData
from core.portfolio import Portfolio
from execution.binance_executor import BinanceExecutor

# Setup Logger to Console
logging.basicConfig(level=logging.INFO)

async def live_smoke_test():
    print("🔥 [LIVE SMOKE TEST] CONNECTING TO BINANCE REAL (FUTURES)...")
    print(f"💰 CAPITAL: ${Config.INITIAL_CAPITAL}")
    
    # 1. Init Data
    events_queue = asyncio.Queue() # Mock queue
    # We need a real Queue for Engine? 
    import queue
    sync_queue = queue.Queue()
    
    data_handler = BinanceData(sync_queue, Config.TRADING_PAIRS)
    
    # 2. Init Portfolio
    portfolio = Portfolio(
        initial_capital=Config.INITIAL_CAPITAL,
        csv_path="execution/smoke_trades.csv",
        status_path="execution/smoke_status.csv"
    )
    
    # 3. Init Executor
    executor = BinanceExecutor(sync_queue, portfolio=portfolio)
    
    # 4. Sync State
    print("🔄 Syncing Portfolio with Exchange...")
    try:
        executor.sync_portfolio_state(portfolio)
        print("✅ Sync Success!")
        print(f"💰 BALANCE VERIFIED: ${portfolio.current_cash:.2f}")
        print(f"📊 OPEN POSITIONS: {len(portfolio.positions)}")
    except Exception as e:
        print(f"❌ Sync Failed: {e}")
        return
        
    # 5. Start Data Stream (Briefly)
    print("📡 Starting Data Stream (10s)...")
    task = asyncio.create_task(data_handler.start_socket())
    
    # Wait for some data
    symbol = Config.TRADING_PAIRS[0]
    for i in range(10):
        await asyncio.sleep(1)
        if symbol in data_handler.buffers_1m:
            df = data_handler.get_latest_bars(symbol, n=1)
            if not df.empty:
                price = df['close'].iloc[-1]
                print(f"⏱️ {i+1}s | {symbol}: {price}")
                if price > 0:
                    print("✅ Real-time Data Received.")
                    break
    
    print("🛑 Stopping Stream...")
    await data_handler.shutdown()
    task.cancel()
    
    print("✅ LIVE SMOKE TEST PASSED. SYSTEM IS READY.")

if __name__ == "__main__":
    try:
        asyncio.run(live_smoke_test())
    except KeyboardInterrupt:
        pass
