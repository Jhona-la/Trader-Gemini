import sys, os
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timezone
from queue import Queue
from core.events import MarketEvent
from core.backtest_infra import BacktestDataProvider, fetch_binance_data
from strategies.technical import HybridScalpingStrategy

print("Loading data...")
df = fetch_binance_data("BTC/USDT", days=7)
print(f"Data bars loaded: {len(df)}")

events_queue = Queue()
data_provider = BacktestDataProvider(events_queue, ["BTC/USDT"], {"BTC/USDT": df})
data_provider.is_backtest = True
tech = HybridScalpingStrategy(data_provider, events_queue, horizon="SCALPING")

stats = {
    "evaluated": 0,
    "momentum_pass": 0,
    "breakout_pass": 0,
    "mean_rev_pass": 0,
    "total_signals": 0,
    "strength_fail": 0
}

warmup = 100
for i in range(warmup):
    data_provider.update_bars()
    if not events_queue.empty(): events_queue.get()

while data_provider.continue_backtest:
    data_provider.update_bars()
    if events_queue.empty(): continue
    me = events_queue.get()
    
    tf_data = tech.get_multi_timeframe_data("BTC/USDT")
    if tech.PRIMARY_TF not in tf_data: continue
    
    pkg = tf_data[tech.PRIMARY_TF]
    params = tech.get_symbol_params("BTC/USDT")
    setups = tech.detect_setup(pkg, params, "BTC/USDT")
    
    stats["evaluated"] += 1
    
    if setups.get('long_momentum') or setups.get('short_momentum'): stats["momentum_pass"] += 1
    if setups.get('long_scalp_break') or setups.get('short_scalp_break'): stats["breakout_pass"] += 1
    if setups.get('long_mean_rev') or setups.get('short_mean_rev'): stats["mean_rev_pass"] += 1
    
    if any([setups.get('long_momentum'), setups.get('short_momentum'), setups.get('long_scalp_break'), setups.get('short_scalp_break'), setups.get('long_mean_rev'), setups.get('short_mean_rev')]):
        confluence = tech.calculate_multi_timeframe_confluence(tf_data, "BTC/USDT")
        setup_type = "MOMENTUM" if setups.get('long_momentum') or setups.get('short_momentum') else "UNKNOWN"
        volatility = setups['atr'] / setups['close']
        strength = tech.calculate_signal_strength(setups, confluence, volatility, "BTC/USDT", setup_type)
        if strength >= params['strength_threshold']:
            stats["total_signals"] += 1
        else:
            stats["strength_fail"] += 1

print("\n--- RESULTS ---")
for k, v in stats.items():
    print(f"{k}: {v}")
