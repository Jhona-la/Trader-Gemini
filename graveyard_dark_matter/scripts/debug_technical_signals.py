"""
DEBUG: Verifica si HybridScalpingStrategy puede generar señales.
Usa datos simulados para aislar el problema del backtest.
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TRADER_GEMINI_BACKTEST'] = 'true'

import numpy as np
import pandas as pd
from queue import Queue
from datetime import datetime, timezone
from config import Config

print('=== DIRECT TEST: HybridScalpingStrategy Signal Generation ===\n')

# Mock DataProvider
class MockDP:
    is_backtest = True
    symbol_list = ['BTC/USDT']
    
    def get_latest_bars(self, symbol, n=300, timeframe='1m'):
        np.random.seed(42)
        n_bars = max(n, 400)
        # Simulated BTC with some volatility + trend
        prices = 80000.0 + np.cumsum(np.random.randn(n_bars) * 80)
        
        # Add some mean-reversion dips to force BB touch
        for i in range(50, n_bars, 100):
            prices[i] -= 400  # Create oversold dip
            prices[i+1] -= 200
        
        dtype = np.dtype([
            ('timestamp', 'i8'), ('open', 'f8'), ('high', 'f8'),
            ('low', 'f8'), ('close', 'f8'), ('volume', 'f8'),
        ])
        bars = np.zeros(n_bars, dtype=dtype)
        for i in range(n_bars):
            vol_factor = 1.0 + abs(np.random.randn()) * 0.5
            bars[i]['timestamp'] = 1700000000000 + i * 60000
            bars[i]['close'] = prices[i]
            bars[i]['open'] = prices[i] * (1 + np.random.randn() * 0.0008)
            bars[i]['high'] = max(bars[i]['open'], bars[i]['close']) * (1 + abs(np.random.randn()) * 0.001)
            bars[i]['low'] = min(bars[i]['open'], bars[i]['close']) * (1 - abs(np.random.randn()) * 0.001)
            bars[i]['volume'] = (1000 + np.random.rand() * 5000) * vol_factor
        return bars[-n:]

# Import strategy
from strategies.technical import HybridScalpingStrategy
from core.events import MarketEvent

events_queue = Queue()
dp = MockDP()

# Init strategy
tech = HybridScalpingStrategy(dp, events_queue)
print(f'Strategy: {tech.strategy_id}')
print(f'Symbol: {tech.symbol}')
print(f'STRENGTH_THRESHOLD: {tech.STRENGTH_THRESHOLD}')
print(f'MIN_VOLUME_RATIO: {tech.MIN_VOLUME_RATIO}')
print()

# Get MTF data
print('--- Multi-Timeframe Data ---')
mtf = tech.get_multi_timeframe_data('BTC/USDT')
print(f'Timeframes available: {list(mtf.keys())}')
if not mtf:
    print('ERROR: No MTF data! Data provider returning None.')
    sys.exit(1)

# Test setup detection
primary_tf = tech.PRIMARY_TF
pkg = mtf.get(primary_tf) or list(mtf.values())[0]
print(f'Primary TF: {primary_tf} -> Using: {list(mtf.keys())[0]}')
print()

print('--- Setup Detection ---')
setups = tech.detect_setup(pkg, symbol='BTC/USDT')
for k, v in setups.items():
    if isinstance(v, float):
        print(f'  {k}: {v:.4f}')
    else:
        print(f'  {k}: {v}')
print()

# Test confluence
print('--- Confluence ---')
confluence = tech.calculate_multi_timeframe_confluence(mtf, symbol='BTC/USDT')
print(f'  confluence_score: {confluence:.4f}')
print()

# Test signal strength
print('--- Signal Strength ---')
volatility = setups['atr'] / setups['close']
strength = tech.calculate_signal_strength(setups, confluence, volatility, symbol='BTC/USDT')
print(f'  strength: {strength:.4f}')
print(f'  threshold: {tech.STRENGTH_THRESHOLD}')

if strength >= tech.STRENGTH_THRESHOLD:
    print('  -> Would GENERATE signal!')
else:
    print(f'  -> Would NOT generate (delta: {strength - tech.STRENGTH_THRESHOLD:.4f})')
print()

# Try calling calculate_signals multiple times with forced conditions
print('--- Signal Generation Test (50 iterations) ---')
signal_count = 0
for i in range(50):
    event = MarketEvent(
        symbol='BTC/USDT',
        close_price=80000.0 + np.random.randn() * 100,
        timestamp=datetime.now(timezone.utc)
    )
    tech.calculate_signals(event)
    while not events_queue.empty():
        sig = events_queue.get()
        signal_count += 1
        print(f'  [Bar {i}] SIGNAL: {sig.signal_type} | strength={getattr(sig, "strength", "N/A")}')

print()
if signal_count > 0:
    print(f'✅ SUCCESS: {signal_count} signals generated over 50 bars')
else:
    print('❌ ZERO signals generated over 50 bars')
    print()
    print('DIAGNOSTIC OUTPUT:')
    print(f'  long_mean_rev={setups.get("long_mean_rev")}')
    print(f'  short_mean_rev={setups.get("short_mean_rev")}')
    print(f'  rsi={setups["rsi"]:.2f}')
    print(f'  volume_ratio={setups["volume_ratio"]:.2f}')
    print(f'  strength={strength:.4f}')
    print(f'  threshold={tech.STRENGTH_THRESHOLD}')
    print()
    print('POSSIBLE ROOT CAUSES:')
    if not setups.get('long_mean_rev') and not setups.get('short_mean_rev'):
        print('  1. MEAN_REV setups not detecting (BB touch + RSI extreme + volume needed)')
    if strength < tech.STRENGTH_THRESHOLD:
        print(f'  2. Signal strength too weak ({strength:.2f} < {tech.STRENGTH_THRESHOLD})')
    if not mtf.get(primary_tf):
        print(f'  3. Primary TF ({primary_tf}) data missing, using fallback')
