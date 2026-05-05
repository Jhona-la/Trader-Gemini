"""
FORENSIC TOOL: Measure actual M1 ATR to calibrate SL/TP for scalping.
Uses ccxt directly to fetch data without depending on internal loader.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import ccxt

exchange = ccxt.binance({'options': {'defaultType': 'future'}})
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=1440)  # ~24h

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['prev_close'] = df['close'].shift(1)
df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
df['tr_pct'] = (df['tr'] / df['close']) * 100
df['bar_return_pct'] = ((df['close'] - df['open']) / df['open']) * 100
df['bar_abs_return_pct'] = df['bar_return_pct'].abs()
df['intrabar_range_pct'] = ((df['high'] - df['low']) / df['close']) * 100

atr_14 = df['tr_pct'].rolling(14).mean().dropna()
atr_50 = df['tr_pct'].rolling(50).mean().dropna()

print(f"\n{'='*60}")
print(f"📊 BTC/USDT M1 ATR ANALYSIS (last 24h, {len(df)} bars)")
print(f"{'='*60}")

print(f"\n📐 TRUE RANGE (% of price):")
print(f"   ATR(14):         {atr_14.iloc[-1]:.4f}%")
print(f"   ATR(50):         {atr_50.iloc[-1]:.4f}%")
print(f"   Median TR:       {df['tr_pct'].median():.4f}%")
print(f"   Mean TR:         {df['tr_pct'].mean():.4f}%")
print(f"   P10 TR:          {df['tr_pct'].quantile(0.10):.4f}%")
print(f"   P25 TR:          {df['tr_pct'].quantile(0.25):.4f}%")
print(f"   P75 TR:          {df['tr_pct'].quantile(0.75):.4f}%")
print(f"   P90 TR:          {df['tr_pct'].quantile(0.90):.4f}%")

print(f"\n📈 BAR RETURNS (close-to-close %):")
print(f"   Mean Abs Return: {df['bar_abs_return_pct'].mean():.4f}%")
print(f"   Median Abs Ret:  {df['bar_abs_return_pct'].median():.4f}%")
print(f"   P75 Abs Return:  {df['bar_abs_return_pct'].quantile(0.75):.4f}%")
print(f"   P90 Abs Return:  {df['bar_abs_return_pct'].quantile(0.90):.4f}%")

print(f"\n🎯 INTRA-BAR RANGE (high-low %):")
print(f"   Mean Range:      {df['intrabar_range_pct'].mean():.4f}%")
print(f"   Median Range:    {df['intrabar_range_pct'].median():.4f}%")

# Rolling max favorable excursion over N bars
for n_bars in [3, 5, 10, 15, 30, 60, 90]:
    max_up = df['close'].rolling(n_bars).apply(lambda x: (x.max() - x[0]) / x[0] * 100, raw=True)
    max_down = df['close'].rolling(n_bars).apply(lambda x: (x[0] - x.min()) / x[0] * 100, raw=True)
    mfe = max_up.dropna()
    mae = max_down.dropna()
    
    # How often does MFE exceed thresholds?
    tp_reach_015 = (mfe >= 0.15).mean() * 100
    tp_reach_020 = (mfe >= 0.20).mean() * 100
    tp_reach_030 = (mfe >= 0.30).mean() * 100
    tp_reach_050 = (mfe >= 0.50).mean() * 100
    
    print(f"\n   {n_bars}-bar MFE/MAE ({n_bars}min window):")
    print(f"     Mean MFE: +{mfe.mean():.4f}% | P50: +{mfe.median():.4f}% | P90: +{mfe.quantile(0.90):.4f}%")
    print(f"     Mean MAE: -{mae.mean():.4f}% | P50: -{mae.median():.4f}% | P90: -{mae.quantile(0.90):.4f}%")
    print(f"     TP Hit Rate: 0.15%={tp_reach_015:.1f}% | 0.20%={tp_reach_020:.1f}% | 0.30%={tp_reach_030:.1f}% | 0.50%={tp_reach_050:.1f}%")

# CURRENT vs EMPIRICAL
print(f"\n{'='*60}")
print(f"⚡ DIAGNOSIS: CURRENT PARAMS vs MARKET REALITY")
print(f"{'='*60}")
atr = atr_14.iloc[-1]
print(f"   ATR(14) M1:       {atr:.4f}%")
print(f"   Current TP:       0.5000% → {0.50/atr:.1f}x ATR (needs {0.50/atr:.0f} perfectly directional bars)")
print(f"   Current SL:       0.2500% → {0.25/atr:.1f}x ATR (hit by {0.25/atr:.0f} adverse bars)")
print(f"")
print(f"   Round-trip fees:  ~0.0575% (maker+taker with BNB)")
print(f"   Net TP after fee: {0.50 - 0.0575:.4f}%")
print(f"   Net SL after fee: {0.25 + 0.0575:.4f}% (SL + fees = total loss)")
print(f"   Breakeven WR:     {(0.25 + 0.0575) / (0.50 - 0.0575 + 0.25 + 0.0575) * 100:.1f}%")
