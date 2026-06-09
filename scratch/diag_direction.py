import pandas as pd
df = pd.read_csv('results/forensic/trade_replay_A_7d64c47f.csv')

print("=== ALL TRADES BY DIRECTION ===")
for d in ['LONG', 'SHORT']:
    s = df[df['direction'] == d]
    wr = s['net_pnl'].gt(0).mean() * 100
    pnl = s['net_pnl'].sum()
    print(f"  {d}: {len(s)} trades, WR={wr:.1f}%, PnL=${pnl:+.4f}")

print("\n=== LOSERS BY EXIT REASON ===")
losers = df[df['net_pnl'] < 0]
for _, t in losers.iterrows():
    print(f"  {t['exit_reason']:25s} {t['symbol']:12s} {t['direction']:6s} PnL=${t['net_pnl']:+.4f} dur={t['duration_seconds']/60:.0f}min")

print("\n=== KILL SWITCH IMPACT ===")
print("24 trades were rejected by KILL_SWITCH")
print("These are potential alpha opportunities lost")

# Compare with 8213f2d7
print("\n=== COMPARISON WITH BEST RUN (8213f2d7) ===")
df2 = pd.read_csv('results/forensic/trade_replay_A_8213f2d7.csv')
print(f"  8213f2d7: {len(df2)} trades, PnL=${df2['net_pnl'].sum():+.4f}")
print(f"  7d64c47f: {len(df)} trades, PnL=${df['net_pnl'].sum():+.4f}")
print(f"  Difference: {len(df2)-len(df)} fewer trades, ${df['net_pnl'].sum()-df2['net_pnl'].sum():+.4f} worse")
