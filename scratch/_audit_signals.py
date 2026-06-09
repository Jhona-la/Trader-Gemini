# VERIFICATION v2: Complete MockDP matching BinanceData interface
import sys, os, traceback, queue, time
sys.path.insert(0, '.')
os.environ["TRADER_GEMINI_BACKTEST"] = "true"

import numpy as np

print("="*70)
print("  VERIFICATION v2: Signal Gate Fix (Complete Mock)")
print("="*70)

# 1. Synthetic structured array
n = 300
np.random.seed(42)
base = 100000.0
prices = [base]
for i in range(1, n):
    drift = 5.0 if i < 200 else -15.0
    prices.append(prices[-1] + drift + np.random.normal(0, 20))

prices = np.array(prices, dtype=np.float32)
highs = prices + np.abs(np.random.normal(10, 5, n)).astype(np.float32)
lows = prices - np.abs(np.random.normal(10, 5, n)).astype(np.float32)
opens = prices - np.random.normal(0, 3, n).astype(np.float32)
volumes = np.random.uniform(100, 1000, n).astype(np.float32)
now_ms = int(time.time() * 1000)
timestamps = np.array([now_ms - (n - i) * 60000 for i in range(n)], dtype=np.int64)

struct_dtype = [('timestamp', 'i8'), ('open', 'f4'), ('high', 'f4'), 
                ('low', 'f4'), ('close', 'f4'), ('volume', 'f4')]
data = np.empty(n, dtype=struct_dtype)
data['timestamp'] = timestamps
data['open'] = opens
data['high'] = highs
data['low'] = lows
data['close'] = prices
data['volume'] = volumes

print(f"  {n} bars: ${prices[0]:.0f} → ${prices[199]:.0f} → ${prices[-1]:.0f}")

# 2. Complete MockDP
class MockDP:
    def __init__(self, data_arr):
        self._data = data_arr
        self.symbol_list = ['BTC/USDT']
        self.is_backtest = True
    
    def get_latest_bars(self, symbol, n=1, timeframe='1m'):
        arr = self._data
        if n > len(arr): return arr
        return arr[-n:]
    
    def get_active_positions(self):
        return {}
    
    def get_order_flow_metrics(self, symbol):
        return None  # No order flow in backtest
    
    def get_derivatives_metrics(self, symbol):
        return {}
    
    def get_lead_lag(self, symbol):
        return None
    
    def get_liquidation_history(self, symbol):
        return []

# 3. Test
from strategies.technical import HybridScalpingStrategy
from core.events import MarketEvent

events_q = queue.Queue()
dp = MockDP(data)

strat = HybridScalpingStrategy(
    data_provider=dp,
    events_queue=events_q,
    horizon='SCALPING'
)

print(f"  Weights: {strat.MULTI_TIMEFRAME_WEIGHTS}")

# Scan
total_signals = 0
signal_log = []
errors_seen = set()

for i in range(100, n):
    sub = data[:i+1]
    dp._data = sub
    
    me = MarketEvent(symbol='BTC/USDT', close_price=float(sub['close'][-1]))
    
    try:
        strat.calculate_signals(me)
    except Exception as ex:
        err_key = str(type(ex).__name__)
        if err_key not in errors_seen:
            print(f"  ❌ New error at bar {i}: {ex}")
            traceback.print_exc()
            errors_seen.add(err_key)
        continue
    
    while not events_q.empty():
        s = events_q.get()
        total_signals += 1
        sig_type = getattr(s, 'signal_type', '?')
        strength = getattr(s, 'strength', 0)
        price = float(sub['close'][-1])
        if total_signals <= 15:
            print(f"  [bar {i}] {sig_type} | str={strength:.3f} | ${price:.0f}")
        signal_log.append({'bar': i, 'type': str(sig_type), 'strength': strength, 'price': price})

print(f"\n{'='*70}")
if total_signals > 0:
    longs = sum(1 for s in signal_log if 'LONG' in s['type'])
    shorts = sum(1 for s in signal_log if 'SHORT' in s['type'])
    avg_str = np.mean([s['strength'] for s in signal_log])
    print(f"  ✅ SIGNALS: {total_signals} (LONG={longs}, SHORT={shorts})")
    print(f"     Avg strength: {avg_str:.3f}")
    print(f"     Rate: {total_signals/(n-100)*100:.1f}% of bars")
else:
    print(f"  ❌ STILL ZERO SIGNALS")
    print(f"     Unique errors: {errors_seen}")
    print(f"     last_processed_times count: {len(strat.last_processed_times)}")
    # Check if dedup is killing everything
    keys = list(strat.last_processed_times.keys())[:5]
    print(f"     Sample dedup keys: {keys}")
print(f"{'='*70}")
