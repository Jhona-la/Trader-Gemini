"""Forensic analysis of trade directions and confidence vs outcome."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv('results/forensic/trade_replay_A_57a2ba6f.csv')

print('=== DIRECTION ANALYSIS ===')
for d in ['LONG', 'SHORT']:
    sub = df[df['direction'] == d]
    wins = sub[sub['net_pnl'] > 0]
    wr = len(wins) / max(1, len(sub)) * 100
    pnl = sub['net_pnl'].sum()
    print(f"  {d}: {len(sub)} trades, WR={wr:.0f}%, PnL=${pnl:+.4f}")

print()
print('=== ALL TRADES: CONFIDENCE vs OUTCOME ===')
print(f"{'Symbol':12s} {'Dir':6s} {'Conf':>6s} {'PnL':>10s} {'Result':6s} {'Exit':20s} {'MAE%':>8s}")
print("-" * 75)
for _, t in df.iterrows():
    outcome = 'WIN' if t['net_pnl'] > 0 else 'LOSS'
    sym = str(t['symbol'])
    dr = str(t['direction'])
    conf = float(t.get('prediction_confidence', 0) or 0)
    pnl = float(t['net_pnl'])
    exit_r = str(t['exit_reason'])[:20]
    mae = float(t.get('mae_pct', 0) or 0) * 100
    print(f"  {sym:12s} {dr:6s} {conf:6.3f} ${pnl:+9.4f} {outcome:6s} {exit_r:20s} {mae:7.3f}%")

print()
print('=== KEY INSIGHT ===')
shorts = df[df['direction'] == 'SHORT']
longs = df[df['direction'] == 'LONG']
print(f"SHORT trades: {len(shorts)}, WR={len(shorts[shorts['net_pnl']>0])/max(1,len(shorts))*100:.0f}%, total PnL=${shorts['net_pnl'].sum():+.4f}")
print(f"LONG trades:  {len(longs)}, WR={len(longs[longs['net_pnl']>0])/max(1,len(longs))*100:.0f}%, total PnL=${longs['net_pnl'].sum():+.4f}")
print(f"\nSHORT bias COST: ${shorts['net_pnl'].sum():+.4f} vs LONG profit: ${longs['net_pnl'].sum():+.4f}")
