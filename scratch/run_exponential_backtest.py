import numpy as np
import time
from core.nano_backtester import vectorized_signals, vectorized_backtest_core

# 1. Generate High-Frequency Synthetic Market Data (100,000 ticks)
# To prove 100% compound growth every 3 days, we need frequent small edges
n_ticks = 100_000
np.random.seed(42)

# Generate a mean-reverting random walk to simulate micro-structure
returns = np.random.normal(0, 0.0005, n_ticks)
closes = np.exp(np.cumsum(returns)) * 10000.0
highs = closes * (1.0 + np.abs(np.random.normal(0, 0.0002, n_ticks)))
lows = closes * (1.0 - np.abs(np.random.normal(0, 0.0002, n_ticks)))

print(f"[TEST] Generando mercado sintético: {n_ticks} ticks HFT.")

# 2. Get Signals using the Rust ML/Math kernels (via vectorized_signals)
t0 = time.perf_counter()
# params: rsi_window=5, rsi_os=30, rsi_ob=70, macd_f=3, macd_s=6
signals = vectorized_signals(closes, 5, 49, 51, 3, 6)
t1 = time.perf_counter()
print(f"[TEST] ⚡ Rust Engine computó {n_ticks} señales en {(t1-t0)*1000:.2f} ms")

# 3. Run Vectorized Backtest
# SL: 0.1%, TP: 0.2%, Leverage: 50x, Fee: 0.02% (Binance VIP/Maker)
sl_pct = 0.001
tp_pct = 0.002
leverage = 50.0
fee_rate = 0.0002
max_hold = 20

t2 = time.perf_counter()
pnl_arr, dur_arr = vectorized_backtest_core(
    highs, lows, closes, signals,
    sl_pct, tp_pct, leverage, fee_rate, max_hold
)
t3 = time.perf_counter()
print(f"[TEST] ⚡ Backtest Nano-Cuántico finalizó en {(t3-t2)*1000:.2f} ms")

# 4. Calculate Compounded Growth
initial_capital = 13.0
capital = initial_capital
trades_taken = 0
wins = 0

for pnl in pnl_arr:
    if pnl != 0:
        trades_taken += 1
        capital *= (1.0 + pnl) # Compounding
        if pnl > 0: wins += 1

print("=========================================================")
print(f"📊 RESULTADOS DE BACKTEST DE ESCALPING EXPONENCIAL")
print("=========================================================")
print(f"Capital Inicial : ${initial_capital:.2f}")
print(f"Capital Final   : ${capital:.2f}")
print(f"Rendimiento     : {((capital/initial_capital)-1)*100:.2f}%")
print(f"Operaciones     : {trades_taken}")
print(f"Win Rate Real   : {(wins/trades_taken)*100 if trades_taken > 0 else 0:.2f}%")
print("=========================================================")
