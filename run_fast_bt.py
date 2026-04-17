import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.run_multi_horizon_backtest import fetch_data, run_strategy_backtest, HORIZON_PROFILES
import pandas as pd

def fast_test():
    symbol = 'BTC/USDT'
    days = 5
    print(f"Fetching {days} days of data for {symbol}...")
    df = fetch_data(symbol, days)
    if df is None or len(df) < 500:
        print("Not enough data.")
        return
        
    print("Running Technical strategy...")
    res_tech = run_strategy_backtest(df, symbol, 'Technical', initial_capital=13, leverage=10, horizon_days=1)
    
    # We can also run XGBoost to test ML
    print("Running ML_XGBoost strategy...")
    res_xgb = run_strategy_backtest(df, symbol, 'ML_XGBoost', initial_capital=13, leverage=10, horizon_days=1)
    
    print("\n[Technical] Results:")
    print(f"  Trades: {res_tech['trades']}")
    print(f"  Win Rate: {res_tech['win_rate']:.2f}%")
    print(f"  PNL: ${res_tech['pnl_usd']:.4f}")
    
    print("\n[ML_XGBoost] Results:")
    print(f"  Trades: {res_xgb['trades']}")
    print(f"  Win Rate: {res_xgb['win_rate']:.2f}%")
    print(f"  PNL: ${res_xgb['pnl_usd']:.4f}")
    
if __name__ == '__main__':
    fast_test()
