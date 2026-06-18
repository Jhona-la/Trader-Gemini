import os
import sys
import queue
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.events import MarketEvent
from strategies.kinematic_strategy import KinematicStrategy
from config import Config

class MockDataProvider:
    def __init__(self, symbol):
        self.symbol = symbol
        from core.quantum.mmap_storage import QuantumMMAP
        mmap = QuantumMMAP(symbol)
        df = mmap.to_dataframe()
        if len(df) > 100:
            self.bars = df.iloc[-100:].values.tolist()
        else:
            self.bars = []
            
    def get_latest_bars(self, symbol, timeframe="1m", n=45):
        if len(self.bars) >= n:
            return self.bars[-n:]
        return self.bars

def run_test():
    q = queue.Queue()
    
    symbol = "OPUSDT" # We know this one has edge in the matrix
    dp = MockDataProvider(symbol)
    
    strategy = KinematicStrategy(data_provider=dp, events_queue=q, symbol=symbol, horizon="SCALPING")
    
    if not strategy.is_active:
        print("❌ Strategy not active for OPUSDT! Matrix loading failed.")
        return
        
    print(f"✅ Strategy Active. SL: {strategy.edge_config['sl_pct']}, Trailing: {strategy.edge_config['kinematic_umbral']}")
    
    event = MarketEvent(symbol=symbol, close_price=dp.bars[-1][4])
    strategy.calculate_signals(event)
    
    if q.empty():
        print("⏸️ No signal generated on the last bar. (Normal behavior for breakout)")
    
    while not q.empty():
        sig = q.get()
        print(f"📡 SIGNAL GENERATED: {sig.signal_type.name} at {sig.datetime}")
        print(f"   - SL: {sig.sl_pct}, TP: {sig.tp_pct}, Trailing: {sig.kinematic_umbral}")
        
if __name__ == "__main__":
    run_test()
