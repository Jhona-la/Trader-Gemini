"""
COMPREHENSIVE HARD_SL ANALYSIS
Find patterns to reduce HARD_SL while maintaining alpha.
Analyze ALL backtests for maximum sample size.
"""
import pandas as pd
import glob

files = glob.glob('results/forensic/trade_replay_A_*.csv')
all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"=== DATASET: {len(all_df)} trades from {len(files)} backtests ===\n")

hard = all_df[all_df['exit_reason'] == 'HARD_SL'].copy()
non_hard = all_df[all_df['exit_reason'] != 'HARD_SL'].copy()

print(f"HARD_SL: {len(hard)} trades, PnL=${hard['net_pnl'].sum():+.4f}")
print(f"Non-HARD_SL: {len(non_hard)} trades, PnL=${non_hard['net_pnl'].sum():+.4f}\n")

# MFE Analysis - do HARD_SL trades EVER show profit?
print("=== MFE ANALYSIS (Did the trade ever show profit?) ===")
for threshold in [0.05, 0.10, 0.15, 0.20, 0.30]:
    low_mfe = hard[hard['mfe_pct'] * 100 < threshold]
    print(f"  MFE < {threshold:.2f}%: {len(low_mfe)}/{len(hard)} trades ({len(low_mfe)/max(len(hard),1)*100:.0f}%), PnL=${low_mfe['net_pnl'].sum():+.4f}")

# Confidence Analysis
print("\n=== CONFIDENCE ANALYSIS ===")
for threshold in [0.65, 0.70, 0.75, 0.80]:
    above = hard[hard['prediction_confidence'] >= threshold]
    below = hard[hard['prediction_confidence'] < threshold]
    print(f"  conf >= {threshold}: {len(above)} trades, PnL=${above['net_pnl'].sum():+.4f}")
    print(f"  conf <  {threshold}: {len(below)} trades, PnL=${below['net_pnl'].sum():+.4f}")

# Duration Analysis
print("\n=== DURATION ANALYSIS ===")
for threshold in [5, 10, 15, 30, 60]:
    fast = hard[hard['duration_seconds'] / 60 < threshold]
    slow = hard[hard['duration_seconds'] / 60 >= threshold]
    print(f"  dur < {threshold}min: {len(fast)} trades, PnL=${fast['net_pnl'].sum():+.4f}")

# Symbol Analysis
print("\n=== HARD_SL BY SYMBOL ===")
for sym in hard['symbol'].sort_values().unique():
    s = hard[hard['symbol'] == sym]
    total = all_df[all_df['symbol'] == sym]
    print(f"  {sym}: {len(s)}/{len(total)} trades ({len(s)/max(len(total),1)*100:.1f}% of all trades), PnL=${s['net_pnl'].sum():+.4f}")

# Direction Analysis
print("\n=== HARD_SL BY DIRECTION ===")
for d in ['LONG', 'SHORT']:
    s = hard[hard['direction'] == d]
    t = all_df[all_df['direction'] == d]
    print(f"  {d}: {len(s)}/{len(t)} trades ({len(s)/max(len(t),1)*100:.1f}%), PnL=${s['net_pnl'].sum():+.4f}")

# Winning trades MFE comparison
print("\n=== MFE: WINNERS vs HARD_SL ===")
winners = all_df[all_df['net_pnl'] > 0]
print(f"  Winners avg MFE: {winners['mfe_pct'].mean()*100:.3f}%")
print(f"  HARD_SL avg MFE: {hard['mfe_pct'].mean()*100:.3f}%")
print(f"  All trades avg MFE: {all_df['mfe_pct'].mean()*100:.3f}%")

# Early detection: trades that will become HARD_SL have low MFE early
print("\n=== EARLY DETECTION POTENTIAL ===")
low_mfe_hard = hard[hard['mfe_pct'] * 100 < 0.10]
low_mfe_winners = winners[winners['mfe_pct'] * 100 < 0.10]
print(f"  Trades with MFE < 0.10% that hit HARD_SL: {len(low_mfe_hard)}")
print(f"  Trades with MFE < 0.10% that won: {len(low_mfe_winners)}")
print(f"  Ratio: If MFE < 0.10%, {len(low_mfe_hard)/max(len(low_mfe_hard)+len(low_mfe_winners),1)*100:.0f}% chance of HARD_SL")
