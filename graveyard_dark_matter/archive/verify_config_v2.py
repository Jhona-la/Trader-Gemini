
from config import Config
from data.binance_loader import BinanceData
import queue
import os

# Set environment variables for test
os.environ['BINANCE_USE_TESTNET'] = 'True'

def test_config_access():
    print("=== Testing Config Access ===")
    try:
        lookback = Config.Strategies.ML_LOOKBACK_BARS
        print(f"✅ Config.Strategies.ML_LOOKBACK_BARS: {lookback}")
    except AttributeError as e:
        print(f"❌ Error accessing Config.Strategies.ML_LOOKBACK_BARS: {e}")
        return

    print("\n=== Testing BinanceLoader & MLStrategy Access ===")
    events_queue = queue.Queue()
    symbols = ["BTC/USDT"]
    try:
        # Initialize loader (will trigger fetch_initial_history with 105 hours)
        loader = BinanceData(events_queue, symbols)
        print("✅ BinanceLoader initialized successfully.")
        
        from strategies.ml_strategy import MLStrategyHybridUltimate as MLStrategy
        ml_strat = MLStrategy(
            data_provider=loader,
            events_queue=events_queue,
            symbol="BTC/USDT",
            lookback=Config.Strategies.ML_LOOKBACK_BARS
        )
        print(f"✅ MLStrategy lookback: {ml_strat.lookback}")
        if ml_strat.lookback == 5000:
            print("✅ MLStrategy is using the CORRECT configured lookback (5000).")
        else:
            print(f"❌ MLStrategy is using WRONG lookback: {ml_strat.lookback} (expected 5000)")
            
    except Exception as e:
        print(f"❌ Initialization check failed: {e}")

if __name__ == "__main__":
    test_config_access()
