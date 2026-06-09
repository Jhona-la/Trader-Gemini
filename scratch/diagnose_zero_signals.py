import sys
import os
from queue import Queue
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from config import Config
from core.backtest_infra import BacktestDataProvider, fetch_multi_symbol_data
from strategies.technical import HybridScalpingStrategy

def main():
    print("🧪 Starting granular diagnostics for zero-signal anomaly...")
    symbols = ["BTC/USDT"]
    days = 3
    
    # 1. Download/Fetch data
    all_data = fetch_multi_symbol_data(symbols, days=days)
    if not all_data or "BTC/USDT" not in all_data:
        print("❌ Failed to fetch BTC/USDT data.")
        return
        
    df = all_data["BTC/USDT"]
    print(f"📊 Downloaded BTC/USDT data: {len(df)} rows. Index range: {df.index[0]} to {df.index[-1]}")
    
    # 2. Instantiate DataProvider
    events_queue = Queue()
    data_provider = BacktestDataProvider(events_queue, symbols, all_data)
    
    # 3. Check what timeframes are loaded and their lengths
    print("\n🔍 Checking timeframes in struct_data:")
    struct = data_provider.struct_data["BTC/USDT"]
    for tf, arr in struct.items():
        print(f"   • Timeframe {tf}: {len(arr)} bars. First timestamp: {arr['timestamp'][0]}, Last timestamp: {arr['timestamp'][-1]}")
        
    # 4. Instantiate Technical Strategy for SCALPING
    print("\n Instantiating HybridScalpingStrategy for SCALPING...")
    strategy = HybridScalpingStrategy(data_provider=data_provider, events_queue=events_queue, horizon="SCALPING")
    
    # Let's advance data provider to a point where warmup is done (e.g. 500 epochs)
    print("\n🚀 Simulating first 500 epochs and evaluating setups...")
    
    for epoch in range(1, 600):
        data_provider.update_bars()
        
        # Drain events queue to keep it clean
        while not events_queue.empty():
            events_queue.get()
            
        if epoch in [10, 100, 200, 300, 400, 500, 550, 580]:
            print(f"\n--- Epoch {epoch} (Time: {pd.to_datetime(data_provider.current_time_ms, unit='ms', utc=True)}) ---")
            
            # Retrieve Multi-Timeframe Data
            tf_data = strategy.get_multi_timeframe_data("BTC/USDT")
            print(f"   Available TFs in strategy cache: {list(tf_data.keys())}")
            
            primary_tf = strategy.PRIMARY_TF
            if primary_tf in tf_data:
                pkg = tf_data[primary_tf]
                data_len = len(pkg['data'])
                print(f"   Primary TF ({primary_tf}) data length: {data_len}")
                
                # Check indicators
                inds = pkg['inds']
                print(f"   Indicators checked - Close: {pkg['data']['close'][-1]:.2f}, RSI: {inds['rsi'][-1]:.2f}, ATR: {inds['atr'][-1]:.4f}, ADX: {inds['adx'][-1]:.2f}")
                
                # Let's perform a forensic inspect inside detect_setup
                data = pkg['data']
                idx = -2
                bbu, bbl = inds['bb_upper'][idx], inds['bb_lower'][idx]
                last_close = data['close'][idx]
                last_low = data['low'][idx]
                last_high = data['high'][idx]
                last_rsi = inds['rsi'][idx]
                last_vol_ratio = inds['volume_ratio'][idx]
                prev_rsi = inds['rsi'][idx - 1]
                
                rsi_buy, rsi_sell = strategy._get_dynamic_rsi_levels(inds)
                adx_thresh = strategy._get_dynamic_adx_threshold(inds)
                adx_extreme = inds['adx'][idx] > 35
                is_strong_trend = inds['adx'][idx] > adx_thresh
                
                rsi_oversold = last_rsi < rsi_buy
                rsi_overbought = last_rsi > rsi_sell
                
                if is_strong_trend:
                    rsi_oversold = last_rsi < min(20, rsi_buy)
                    rsi_overbought = last_rsi > max(80, rsi_sell)
                    
                is_range = not is_strong_trend or rsi_oversold or rsi_overbought
                if adx_extreme:
                    is_range = False
                    
                price_at_lower = last_low <= bbl
                price_at_upper = last_high >= bbu
                
                vol_min = 0.85 # since we are in backtest
                high_volume = last_vol_ratio > vol_min
                rsi_turning_up = last_rsi > prev_rsi
                rsi_turning_down = last_rsi < prev_rsi
                
                candle_stabilizing_long = (last_close > data['open'][idx]) or ((last_close - last_low) > (last_high - last_close))
                candle_stabilizing_short = (last_close < data['open'][idx]) or ((last_high - last_close) > (last_close - last_low))
                
                # Print all sub-components
                print("   Forensic Sub-Conditions for LONG_MEAN_REV:")
                print(f"      • price_at_lower: {price_at_lower} (last_low={last_low:.2f} <= bbl={bbl:.2f})")
                print(f"      • rsi_oversold: {rsi_oversold} (last_rsi={last_rsi:.2f} < rsi_buy={rsi_buy:.2f})")
                print(f"      • high_volume: {high_volume} (last_vol_ratio={last_vol_ratio:.2f} > vol_min={vol_min:.2f})")
                print(f"      • is_range: {is_range} (adx={inds['adx'][idx]:.2f}, threshold={adx_thresh:.2f})")
                print(f"      • rsi_turning_up: {rsi_turning_up} (last_rsi={last_rsi:.2f} > prev_rsi={prev_rsi:.2f})")
                print(f"      • candle_stabilizing_long: {candle_stabilizing_long}")
                
                # Check setups
                params = strategy.get_symbol_params("BTC/USDT")
                setups = strategy.detect_setup(pkg, params, "BTC/USDT")
                if setups:
                    print("   Final Strategy Decided Setups:")
                    print(f"      - long_mean_rev: {setups.get('long_mean_rev')}")
                    print(f"      - short_mean_rev: {setups.get('short_mean_rev')}")
                    print(f"      - long_momentum: {setups.get('long_momentum')}")
                    print(f"      - short_momentum: {setups.get('short_momentum')}")
                else:
                    print("   ❌ detect_setup returned None!")
            else:
                print(f"   ❌ Primary TF {primary_tf} NOT in strategy cache!")

if __name__ == "__main__":
    import pandas as pd
    main()
