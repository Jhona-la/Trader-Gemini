"""Analyze 5-symbol backtest HARD_SL trades."""
import pandas as pd
df = pd.read_csv('results/forensic/trade_replay_A_accc53f3.csv')
hard = df[df['exit_reason'] == 'HARD_SL']

print('=== 10 HARD_SL TRADES ===')
print(f"{'Symbol':12s} {'Dir':6s} {'Conf':>6s} {'MAE%':>8s} {'MFE%':>8s} {'PnL':>10s} {'Dur(s)':>8s} {'Horizon':>12s}")
print("-" * 80)
for _, t in hard.iterrows():
    sym = str(t['symbol'])
    d = str(t['direction'])
    conf = float(t.get('prediction_confidence', 0) or 0)
    mae = float(t.get('mae_pct', 0) or 0) * 100
    mfe = float(t.get('mfe_pct', 0) or 0) * 100
    pnl = float(t['net_pnl'])
    dur = int(t.get('duration_seconds', 0) or 0)
    hz = str(t.get('horizon', '?'))
    print(f"  {sym:12s} {d:6s} {conf:6.3f} {mae:7.3f}% {mfe:7.3f}% ${pnl:+9.4f} {dur:8d} {hz:>12s}")

print(f"\n=== HARD_SL BY SYMBOL ===")
for sym in hard['symbol'].unique():
    sub = hard[hard['symbol'] == sym]
    print(f"  {sym}: {len(sub)} trades, PnL=${sub['net_pnl'].sum():+.4f}")

print(f"\n=== HARD_SL BY HORIZON ===")
for hz in hard['horizon'].unique():
    sub = hard[hard['horizon'] == hz]
    print(f"  {hz}: {len(sub)} trades, PnL=${sub['net_pnl'].sum():+.4f}")

print(f"\n=== ALL TRADES BY HORIZON ===")
for hz in df['horizon'].unique():
    sub = df[df['horizon'] == hz]
    wins = sub[sub['net_pnl'] > 0]
    print(f"  {hz}: {len(sub)} trades, WR={len(wins)/max(len(sub),1)*100:.0f}%, PnL=${sub['net_pnl'].sum():+.4f}")
