"""Analyze FLIP_EXIT trades from regression backtest."""
import pandas as pd
df = pd.read_csv('results/forensic/trade_replay_A_d6e45615.csv')

print("=== EXIT REASON BREAKDOWN ===")
for reason in df['exit_reason'].unique():
    sub = df[df['exit_reason'] == reason]
    wins = sub[sub['net_pnl'] > 0]
    print(f"  {reason}: {len(sub)} trades, WR={len(wins)/max(len(sub),1)*100:.0f}%, PnL=${sub['net_pnl'].sum():+.4f}")

print("\n=== FLIP_EXIT DETAIL ===")
flip = df[df['exit_reason'].str.contains('FLIP', na=False)]
for _, t in flip.iterrows():
    sym = str(t['symbol'])
    d = str(t['direction'])
    conf = float(t.get('prediction_confidence', 0) or 0)
    mae = float(t.get('mae_pct', 0) or 0) * 100
    mfe = float(t.get('mfe_pct', 0) or 0) * 100
    pnl = float(t['net_pnl'])
    dur = int(t.get('duration_seconds', 0) or 0)
    print(f"  {sym:12s} {d:6s} conf={conf:.3f} MAE={mae:.3f}% MFE={mfe:.3f}% PnL=${pnl:+.4f} dur={dur}s")

print("\n=== COMPARISON: FLIP vs REVERSAL vs TIMEOUT ===")
for reason_key in ['FLIP', 'REVERSAL', 'TIMEOUT']:
    sub = df[df['exit_reason'].str.contains(reason_key, na=False)]
    if len(sub) > 0:
        print(f"  {reason_key}: avg_conf={sub['prediction_confidence'].mean():.3f}, avg_MAE={sub['mae_pct'].mean()*100:.3f}%, avg_MFE={sub['mfe_pct'].mean()*100:.3f}%, avg_dur={sub['duration_seconds'].mean():.0f}s")
