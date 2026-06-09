import pandas as pd
df = pd.read_csv('results/forensic/trade_replay_A_8213f2d7.csv')
hard = df[df['exit_reason']=='HARD_SL']
print('=== 6 HARD_SL TRADES ===')
for _, t in hard.iterrows():
    sym = str(t['symbol'])
    d = str(t['direction'])
    conf = float(t.get('prediction_confidence', 0) or 0)
    mae = float(t.get('mae_pct', 0) or 0) * 100
    mfe = float(t.get('mfe_pct', 0) or 0) * 100
    pnl = float(t['net_pnl'])
    dur = float(t.get('duration_seconds', 0) or 0) / 60
    print(f"  {sym:12s} {d:6s} conf={conf:.3f} MAE={mae:.3f}% MFE={mfe:.3f}% PnL=${pnl:+.4f} dur={dur:.0f}min")

print("\nBy symbol:")
for sym in hard['symbol'].unique():
    s = hard[hard['symbol'] == sym]
    print(f"  {sym}: {len(s)} trades, PnL=${s['net_pnl'].sum():+.4f}")
