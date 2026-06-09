"""Analyze 9 HARD_SL trades from 7-day backtest to find pattern."""
import pandas as pd

df = pd.read_csv('results/forensic/trade_replay_A_75c88d6b.csv')
hard = df[df['exit_reason'] == 'HARD_SL']
good = df[df['exit_reason'] != 'HARD_SL']

print('=== 9 HARD_SL TRADES — PATTERN ANALYSIS ===')
print(f"{'Symbol':12s} {'Dir':6s} {'Conf':>6s} {'MAE%':>8s} {'MFE%':>8s} {'PnL':>10s} {'Dur(s)':>8s} {'Strategy':>40s}")
print("-" * 100)
for _, t in hard.iterrows():
    sym = str(t['symbol'])
    d = str(t['direction'])
    conf = float(t.get('prediction_confidence', 0) or 0)
    mae = float(t.get('mae_pct', 0) or 0) * 100
    mfe = float(t.get('mfe_pct', 0) or 0) * 100
    pnl = float(t['net_pnl'])
    dur = int(t.get('duration_seconds', 0) or 0)
    strat = str(t.get('opener_strategy_id', t.get('strategy_id', '')))[-40:]
    print(f"  {sym:12s} {d:6s} {conf:6.3f} {mae:7.3f}% {mfe:7.3f}% ${pnl:+9.4f} {dur:8d} {strat:>40s}")

print()
print('=== COMPARISON: HARD_SL vs GOOD TRADES ===')
print(f"HARD_SL (n={len(hard)}):")
print(f"  Avg confidence: {hard['prediction_confidence'].mean():.3f}")
print(f"  Avg MAE: {(hard['mae_pct']*100).mean():.3f}%")
print(f"  Avg MFE: {(hard['mfe_pct']*100).mean():.3f}%")
print(f"  Avg duration: {hard['duration_seconds'].mean():.0f}s")
print(f"  Direction: LONG={len(hard[hard['direction']=='LONG'])}, SHORT={len(hard[hard['direction']=='SHORT'])}")
print(f"  Total PnL: ${hard['net_pnl'].sum():+.4f}")

print(f"\nGOOD TRADES (n={len(good)}):")
print(f"  Avg confidence: {good['prediction_confidence'].mean():.3f}")
print(f"  Avg MAE: {(good['mae_pct']*100).mean():.3f}%")
print(f"  Avg MFE: {(good['mfe_pct']*100).mean():.3f}%")  
print(f"  Avg duration: {good['duration_seconds'].mean():.0f}s")
print(f"  Direction: LONG={len(good[good['direction']=='LONG'])}, SHORT={len(good[good['direction']=='SHORT'])}")
print(f"  Total PnL: ${good['net_pnl'].sum():+.4f}")

print()
print('=== STRATEGY BREAKDOWN FOR HARD_SL ===')
for strat in hard['opener_strategy_id'].unique():
    sub = hard[hard['opener_strategy_id'] == strat]
    print(f"  {strat}: {len(sub)} trades, PnL=${sub['net_pnl'].sum():+.4f}")

print()
print('=== CONFIDENCE DISTRIBUTION ===')
for bucket in [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.0)]:
    lo, hi = bucket
    sub = df[(df['prediction_confidence'] >= lo) & (df['prediction_confidence'] < hi)]
    hs = sub[sub['exit_reason'] == 'HARD_SL']
    wins = sub[sub['net_pnl'] > 0]
    if len(sub) > 0:
        print(f"  conf [{lo:.2f}-{hi:.2f}): {len(sub)} trades, WR={len(wins)/len(sub)*100:.0f}%, HARD_SL={len(hs)}, PnL=${sub['net_pnl'].sum():+.4f}")
