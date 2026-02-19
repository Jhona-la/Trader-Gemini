import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from binance.client import Client

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from data.data_provider import DataProvider
from strategies.technical import HybridScalpingStrategy

class SimpleDataProvider(DataProvider):
    def __init__(self, symbol):
        self.symbol = symbol
        self.struct_data = {}
        self.load_data()

    def load_data(self):
        try:
            client = Client(Config.BINANCE_API_KEY, Config.BINANCE_SECRET_KEY)
        except:
            client = Client()
            
        print(f"  > Fetching data for {self.symbol}...")
        
        # Fetch 5m
        api_symbol = self.symbol.replace('/', '')
        klines = client.futures_klines(symbol=api_symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=1000)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        
        df['timestamp'] = df['timestamp'].astype('int64')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        struct_dtype = [
            ('timestamp', 'i8'), ('open', 'f4'), ('high', 'f4'), 
            ('low', 'f4'), ('close', 'f4'), ('volume', 'f4')
        ]
        
        arr = np.empty(len(df), dtype=struct_dtype)
        arr['timestamp'] = df['timestamp'].values
        arr['open'] = df['open'].values
        arr['high'] = df['high'].values
        arr['low'] = df['low'].values
        arr['close'] = df['close'].values
        arr['volume'] = df['volume'].values
        
        self.struct_data['5m'] = arr
        
        # Fetch 1h
        klines_1h = client.futures_klines(symbol=api_symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=200)
        df_1h = pd.DataFrame(klines_1h, columns=[
             'timestamp', 'open', 'high', 'low', 'close', 'volume', 'time', 'qav', 'num', 'tbv', 'tqv', 'ignore'
        ])
        df_1h['timestamp'] = df_1h['timestamp'].astype('int64')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_1h[col] = df_1h[col].astype(float)
            
        arr_1h = np.empty(len(df_1h), dtype=struct_dtype)
        arr_1h['timestamp'] = df_1h['timestamp'].values
        arr_1h['open'] = df_1h['open'].values
        arr_1h['high'] = df_1h['high'].values
        arr_1h['low'] = df_1h['low'].values
        arr_1h['close'] = df_1h['close'].values
        arr_1h['volume'] = df_1h['volume'].values
        self.struct_data['1h'] = arr_1h
        
        # Also need 15m for strategy? Strategy uses 5m, 15m, 1h
        klines_15m = client.futures_klines(symbol=api_symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=500)
        df_15m = pd.DataFrame(klines_15m, columns=[
             'timestamp', 'open', 'high', 'low', 'close', 'volume', 'time', 'qav', 'num', 'tbv', 'tqv', 'ignore'
        ])
        df_15m['timestamp'] = df_15m['timestamp'].astype('int64')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_15m[col] = df_15m[col].astype(float)
        
        arr_15m = np.empty(len(df_15m), dtype=struct_dtype)
        arr_15m['timestamp'] = df_15m['timestamp'].values
        arr_15m['open'] = df_15m['open'].values
        arr_15m['high'] = df_15m['high'].values
        arr_15m['low'] = df_15m['low'].values
        arr_15m['close'] = df_15m['close'].values
        arr_15m['volume'] = df_15m['volume'].values
        self.struct_data['15m'] = arr_15m

    def get_latest_bars(self, symbol, n=1, timeframe='5m'):
        if timeframe not in self.struct_data:
            return None
        arr = self.struct_data[timeframe]
        return arr[-n:]
        
    def update_bars(self):
        pass
        
    def get_active_positions(self):
        return {}
    
    @property
    def symbol_list(self):
        return [self.symbol]

def diagnose():
    print("🔍 DIAGNOSTIC MODE: Checking Strength for BTC/USDT")
    dp = SimpleDataProvider('BTC/USDT')
    strategy = HybridScalpingStrategy(dp, None)
    
    print("\n📊 INSPECTION (Last 50 bars):")
    print(f"{'Time':<20} | {'Type':<6} | {'Str':<5} | {'ADX':<5} | {'Conf':<5} | {'VolR':<5} | {'Result'}")
    print("-" * 100)
    
    # We need to manually iterate and construct 'timeframe_data' like generate_signals does
    
    # get 5m data first to iterate
    data_5m_all = dp.get_latest_bars('BTC/USDT', 1000, '5m')
    
    count = 0
    start_idx = len(data_5m_all) - 50
    
    for i in range(start_idx, len(data_5m_all)):
        # Construct timeframe data ending at i
        # We need to slice "up to i"
        # Since get_latest_bars returns the *last n* segments, we can't easily query "at index i" from DP.
        # But we can pass sliced arrays to strategy methods.
        
        current_data_5m = data_5m_all[:i+1] # Slice up to current bar
        if len(current_data_5m) < 300: continue
        
        # Mock timeframe_data
        # We need to be careful: Strategy calls get_latest_bars(n=300).
        # We should simulate that.
        
        # 5m
        subset_5m = current_data_5m[-300:]
        inds_5m = strategy.calculate_indicators(subset_5m)
        pkg_5m = {'data': subset_5m, 'inds': inds_5m}
        
        # 1h (Need to find corresponding 1h index? Or just take latest available?)
        # For simplicity, taking latest 1h data available at that time
        # This is hard to sync perfectly without re-fetching/resampling. 
        # But 'SimpleDataProvider' has static 1h arrays. 
        # Strategy calls get_latest_bars('1h', n=300).
        # We can just use the latest 1h data for now, assuming it's roughly synced or just checking logic.
        # Ideally we should filter 1h data by timestamp <= 5m timestamp
        
        curr_ts = subset_5m['timestamp'][-1]
        
        data_1h_all = dp.struct_data['1h']
        idx_1h = np.searchsorted(data_1h_all['timestamp'], curr_ts, side='right')
        subset_1h = data_1h_all[max(0, idx_1h-300):idx_1h]
        if len(subset_1h) < 50: continue
        inds_1h = strategy.calculate_indicators(subset_1h)
        pkg_1h = {'data': subset_1h, 'inds': inds_1h}
        
        # 15m
        data_15m_all = dp.struct_data['15m']
        idx_15m = np.searchsorted(data_15m_all['timestamp'], curr_ts, side='right')
        subset_15m = data_15m_all[max(0, idx_15m-200):idx_15m]
        if len(subset_15m) < 30: continue
        inds_15m = strategy.calculate_indicators(subset_15m)
        pkg_15m = {'data': subset_15m, 'inds': inds_15m}
        
        timeframe_data = {
            '5m': pkg_5m,
            '15m': pkg_15m,
            '1h': pkg_1h
        }
        
        # 1. Confluence
        confluence_score = strategy.calculate_multi_timeframe_confluence(timeframe_data)
        
        # 2. Setups
        setups = strategy.detect_scalping_setup(pkg_5m)
        if not setups: continue
        
        volatility = setups['atr'] / setups['close']
        
        # 3. Strength
        strength = strategy.calculate_signal_strength(setups, confluence_score, volatility)
        
        signal_type = ""
        if setups['long_mean_rev'] or setups['long_momentum']: signal_type = "LONG"
        elif setups['short_mean_rev'] or setups['short_momentum']: signal_type = "SHORT"
        
        if signal_type:
            ts_dt = pd.to_datetime(curr_ts, unit='ms', utc=True)
            res_str = "PASS" if strength >= 0.6 else "FAIL"
            
            # Additional check ADX < 20 for ranges
            adx_val = setups['adx']
            rsi_val = setups['rsi']
            
            # Pre-filter logic match
            keep = True
            if adx_val < 20: 
                if not (rsi_val < 15 or rsi_val > 85): keep = False
            
            if keep:
                 print(f"{str(ts_dt):<20} | {signal_type:<6} | {strength:<5.2f} | {adx_val:<5.1f} | {confluence_score:<5.2f} | {setups['volume_ratio']:<5.2f} | {res_str}")
                 count += 1
            else:
                 if strength >= 0.5:
                     print(f"{str(ts_dt):<20} | {signal_type:<6} | {strength:<5.2f} | {adx_val:<5.1f} | {confluence_score:<5.2f} | {setups['volume_ratio']:<5.2f} | FILTERED (ADX)")
    
    print(f"\n✅ Found {count} valid setups.")

if __name__ == "__main__":
    diagnose()
