import pandas as pd
import numpy as np
import time
import itertools
from core.nano_backtester import vectorized_signals, vectorized_backtest_core

# Load Data
df = pd.read_csv('data/historical/BTC_USDT_1m.csv')
if 'close' not in df.columns:
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore']

closes = df['close'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)

# Grid setup
rsi_windows = [5, 10, 14]
rsi_os_levels = [30, 40]
rsi_ob_levels = [60, 70]
macd_fast = [3, 12]
macd_slow = [6, 26]

# Backtest params
sl_pct = 0.001
tp_pct = 0.002
leverage = 50.0
fee_rate = 0.0002
max_hold = 20

print(f"📊 Running Nano-Grid Search over {len(closes)} bars...")

best_return = -999.0
best_params = None
best_stats = {}

t0 = time.perf_counter()
iterations = 0

for w, os_lvl, ob_lvl, mf, ms in itertools.product(rsi_windows, rsi_os_levels, rsi_ob_levels, macd_fast, macd_slow):
    signals = vectorized_signals(closes, w, os_lvl, ob_lvl, mf, ms)
    pnl_arr, dur_arr = vectorized_backtest_core(
        highs, lows, closes, signals,
        sl_pct, tp_pct, leverage, fee_rate, max_hold
    )
    
    capital = 13.0
    wins = 0
    trades = 0
    for pnl in pnl_arr:
        if pnl != 0:
            trades += 1
            capital *= (1.0 + pnl)
            if pnl > 0: wins += 1
            
    ret = (capital / 13.0) - 1.0
    if ret > best_return:
        best_return = ret
        best_params = (w, os_lvl, ob_lvl, mf, ms)
        best_stats = {'trades': trades, 'wr': wins/trades if trades > 0 else 0, 'capital': capital}
    
    iterations += 1

t1 = time.perf_counter()

print("=========================================================")
print(f"🏆 MEJORES HIPERPARÁMETROS (RUST GRID SEARCH)")
print("=========================================================")
print(f"Iteraciones   : {iterations}")
print(f"Tiempo Total  : {(t1-t0)*1000:.2f} ms")
print(f"Params (w,os,ob,mf,ms): {best_params}")
print(f"Retorno Neto  : {best_return*100:.2f}%")
print(f"Capital Final : ${best_stats.get('capital', 13.0):.2f}")
print(f"Operaciones   : {best_stats.get('trades', 0)}")
print(f"Win Rate Real : {best_stats.get('wr', 0)*100:.2f}%")
print("=========================================================")
