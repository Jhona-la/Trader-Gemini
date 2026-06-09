"""Analyze remaining HARD_SL trades from post-fix backtest."""
import pandas as pd

df = pd.read_csv('results/forensic/trade_replay_A_fe300436.csv')
hard = df[df['exit_reason'] == 'HARD_SL']

print('=== 2 REMAINING HARD_SL TRADES ===')
for _, t in hard.iterrows():
    sym = t['symbol']
    d = t['direction']
    mae = float(t.get('mae_pct', 0) or 0) * 100
    mfe = float(t.get('mfe_pct', 0) or 0) * 100
    conf = float(t.get('prediction_confidence', 0) or 0)
    pnl = float(t['net_pnl'])
    dur = int(t.get('duration_seconds', 0) or 0)
    strat = t['strategy_id']
    print(f"  {sym} {d} conf={conf:.3f} MAE={mae:.3f}% MFE={mfe:.3f}% PnL=${pnl:+.4f} dur={dur}s strat={strat}")

print()
wins = df[df['net_pnl'] > 0]
losses = df[df['net_pnl'] <= 0]
print(f"Wins: {len(wins)}, avg=${wins['net_pnl'].mean():+.4f}")
print(f"Losses: {len(losses)}, avg=${losses['net_pnl'].mean():+.4f}")
print(f"Total PnL: ${df['net_pnl'].sum():+.4f}")
no_hs = df[df['exit_reason'] != 'HARD_SL']
print(f"Without HARD_SL: ${no_hs['net_pnl'].sum():+.4f} ({len(no_hs)} trades, WR={len(no_hs[no_hs['net_pnl']>0])/len(no_hs)*100:.0f}%)")
