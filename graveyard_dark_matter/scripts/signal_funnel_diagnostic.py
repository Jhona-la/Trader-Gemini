"""
🔬 SIGNAL FUNNEL DIAGNOSTIC v2 — Capture ALL errors
"""
import sys, os, traceback
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TRADER_GEMINI_BACKTEST'] = 'true'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from queue import Queue
from config import Config
from core.events import MarketEvent
from core.backtest_infra import BacktestDataProvider, fetch_binance_data
from strategies.technical import HybridScalpingStrategy
from utils.logger import logger
import logging
logger.setLevel(logging.DEBUG)

print("=" * 70)
print("🔬 SIGNAL FUNNEL DIAGNOSTIC v2")
print("=" * 70)

# Download 1 day of data
df = fetch_binance_data("BTC/USDT", days=1)
if df is None or len(df) < 500:
    print("❌ Insufficient data")
    sys.exit(1)

print(f"✅ {len(df):,} bars loaded")

symbol = "BTC/USDT"
events_queue = Queue()
historical_data = {symbol: df}
data_provider = BacktestDataProvider(events_queue, [symbol], historical_data)
data_provider.is_backtest = True

tech = HybridScalpingStrategy(data_provider, events_queue)

# Advance to bar 350 (past warmup)
for i in range(350):
    data_provider.update_bars()
    events_queue.get()  # consume MarketEvent

print(f"\n📊 Testing bar 350 with MANUAL deep trace...\n")

# Advance 1 more bar
data_provider.update_bars()
me = events_queue.get()
print(f"MarketEvent: symbol={me.symbol}, price={me.close_price}, ts={me.timestamp}")

# Now manually step through generate_signals
print("\n--- MANUAL SIGNAL GENERATION TRACE ---")
print(f"tech.symbol = {tech.symbol}")
print(f"tech.horizon = {tech.horizon}")
print(f"tech.PRIMARY_TF = {tech.PRIMARY_TF}")
print(f"tech.HORIZON_TIMEFRAMES = {tech.HORIZON_TIMEFRAMES}")

# 1. Check symbols list
symbols = []
if tech.symbol:
    symbols = [tech.symbol]
elif me and getattr(me, 'symbol', None):
    symbols = [me.symbol]
else:
    symbols = tech.data_provider.symbol_list
print(f"\nSymbols to process: {symbols}")

# 2. Check timeframe data  
for s in symbols:
    print(f"\n--- Processing {s} ---")
    try:
        tf_data = tech.get_multi_timeframe_data(s)
        print(f"  Timeframes found: {list(tf_data.keys())}")
        
        primary_tf = tech.PRIMARY_TF
        print(f"  Primary TF: {primary_tf}")
        print(f"  Primary TF in data? {primary_tf in tf_data}")
        
        if primary_tf not in tf_data:
            print(f"  ❌ PRIMARY TF '{primary_tf}' NOT in timeframe_data!")
            print(f"  Available keys: {list(tf_data.keys())}")
            # Check what data provider has
            print(f"  Data provider struct_data keys: {list(data_provider.struct_data.get(s, {}).keys())}")
            continue
        
        pkg = tf_data[primary_tf]
        data = pkg['data']
        inds = pkg['inds']
        print(f"  Data bars: {len(data)}")
        
        # 3. Check setup detection
        params = tech.get_symbol_params(s)
        print(f"  Params: adx_thresh={params['adx_threshold']}, strength_thresh={params['strength_threshold']}")
        
        setups = tech.detect_setup(pkg, params, s)
        print(f"  Setups detected: {setups}")
        
        # 4. Check confluence
        confluence = tech.calculate_multi_timeframe_confluence(tf_data, s)
        print(f"  Confluence score: {confluence}")
        
        # 5. Check signal direction
        signal_type = None
        setup_type = "UNKNOWN"
        if setups:
            if setups.get('long_mean_rev') or setups.get('short_mean_rev'):
                from core.enums import SignalType
                signal_type = SignalType.LONG if setups.get('long_mean_rev') else SignalType.SHORT
                setup_type = "MEAN_REV"
            elif setups.get('long_momentum') or setups.get('short_momentum'):
                signal_type = SignalType.LONG if setups.get('long_momentum') else SignalType.SHORT
                setup_type = "MOMENTUM"
            elif setups.get('long_scalp_break') or setups.get('short_scalp_break'):
                signal_type = SignalType.LONG if setups.get('long_scalp_break') else SignalType.SHORT
                setup_type = "SCALP_BREAKOUT"
        
        print(f"  Signal type: {signal_type}")
        print(f"  Setup type: {setup_type}")
        
        if signal_type:
            # 6. Check strength
            volatility = setups['atr'] / setups['close']
            strength = tech.calculate_signal_strength(setups, confluence, volatility, s, setup_type)
            print(f"  Strength: {strength}")
            print(f"  Strength threshold: {params['strength_threshold']}")
            print(f"  Strength passes? {strength >= params['strength_threshold']}")
            
            # 7. Check ADX
            print(f"  ADX: {setups['adx']}")
            print(f"  ADX passes? {setups['adx'] >= params['adx_threshold']}")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()

print(f"\n{'='*70}")
print(f"DONE")
print(f"{'='*70}")
