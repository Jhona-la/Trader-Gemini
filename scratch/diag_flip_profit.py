"""
FLIP_EXIT PROFITABILITY ANALYSIS
QUÉ: Analizar todos los FLIP_EXIT para encontrar condiciones bajo las cuales SON rentables.
POR QUÉ: El principio dice "cada feature debe ser rentable por sí sola."
PARA QUÉ: Configurar FLIP_EXIT correctamente en vez de eliminarlo.
"""
import pandas as pd
import os, glob

# Load ALL backtest trade replays to maximize sample size
replay_files = glob.glob('results/forensic/trade_replay_A_*.csv')
all_trades = []
for f in replay_files:
    try:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        all_trades.append(df)
    except:
        pass

if not all_trades:
    print("No trade replay files found!")
    exit()

df = pd.concat(all_trades, ignore_index=True)
print(f"=== TOTAL DATASET: {len(df)} trades from {len(replay_files)} backtests ===\n")

# Separate FLIP vs non-FLIP
flip = df[df['exit_reason'].str.contains('FLIP', na=False)].copy()
non_flip = df[~df['exit_reason'].str.contains('FLIP', na=False)].copy()

print(f"FLIP_EXIT trades: {len(flip)}")
print(f"Non-FLIP trades: {len(non_flip)}\n")

if len(flip) == 0:
    print("No FLIP trades to analyze")
    exit()

# Winning vs Losing FLIP trades
flip_win = flip[flip['net_pnl'] > 0]
flip_loss = flip[flip['net_pnl'] <= 0]

print(f"=== FLIP: WINNERS ({len(flip_win)}) vs LOSERS ({len(flip_loss)}) ===")
for label, subset in [("WINNERS", flip_win), ("LOSERS", flip_loss)]:
    if len(subset) == 0:
        print(f"  {label}: none")
        continue
    print(f"\n  {label} ({len(subset)} trades):")
    print(f"    avg_confidence: {subset['prediction_confidence'].mean():.3f}")
    print(f"    avg_MAE%: {subset['mae_pct'].mean()*100:.3f}%")
    print(f"    avg_MFE%: {subset['mfe_pct'].mean()*100:.3f}%")
    print(f"    avg_duration: {subset['duration_seconds'].mean():.0f}s ({subset['duration_seconds'].mean()/60:.0f}min)")
    print(f"    avg_PnL: ${subset['net_pnl'].mean():+.4f}")
    print(f"    total_PnL: ${subset['net_pnl'].sum():+.4f}")

# Find the discriminator
print(f"\n=== DISCRIMINATOR ANALYSIS ===")
for threshold_col, name in [('prediction_confidence', 'Confidence'), ('duration_seconds', 'Duration(s)'), ('mae_pct', 'MAE%')]:
    if threshold_col not in flip.columns:
        continue
    vals = flip[threshold_col].dropna().sort_values()
    for pct in [0.25, 0.50, 0.75]:
        threshold = vals.quantile(pct)
        above = flip[flip[threshold_col] >= threshold]
        below = flip[flip[threshold_col] < threshold]
        above_wr = len(above[above['net_pnl'] > 0]) / max(len(above), 1) * 100
        below_wr = len(below[below['net_pnl'] > 0]) / max(len(below), 1) * 100
        print(f"  {name} >= {threshold:.4f}: {len(above)} trades, WR={above_wr:.0f}%, PnL=${above['net_pnl'].sum():+.4f}")
        print(f"  {name} <  {threshold:.4f}: {len(below)} trades, WR={below_wr:.0f}%, PnL=${below['net_pnl'].sum():+.4f}")
        print()

# Per-symbol FLIP analysis
print(f"\n=== FLIP BY SYMBOL ===")
for sym in flip['symbol'].unique():
    sub = flip[flip['symbol'] == sym]
    wins = sub[sub['net_pnl'] > 0]
    print(f"  {sym}: {len(sub)} trades, WR={len(wins)/max(len(sub),1)*100:.0f}%, PnL=${sub['net_pnl'].sum():+.4f}")

# Per-direction FLIP analysis
print(f"\n=== FLIP BY DIRECTION ===")
for d in flip['direction'].unique():
    sub = flip[flip['direction'] == d]
    wins = sub[sub['net_pnl'] > 0]
    print(f"  {d}: {len(sub)} trades, WR={len(wins)/max(len(sub),1)*100:.0f}%, PnL=${sub['net_pnl'].sum():+.4f}")

# Detail all FLIP trades
print(f"\n=== ALL FLIP TRADES (sorted by PnL) ===")
flip_sorted = flip.sort_values('net_pnl', ascending=False)
print(f"{'Symbol':12s} {'Dir':6s} {'Conf':>6s} {'MAE%':>8s} {'MFE%':>8s} {'PnL':>10s} {'Dur(min)':>10s}")
print("-" * 70)
for _, t in flip_sorted.iterrows():
    sym = str(t['symbol'])
    d = str(t['direction'])
    conf = float(t.get('prediction_confidence', 0) or 0)
    mae = float(t.get('mae_pct', 0) or 0) * 100
    mfe = float(t.get('mfe_pct', 0) or 0) * 100
    pnl = float(t['net_pnl'])
    dur = float(t.get('duration_seconds', 0) or 0) / 60
    print(f"  {sym:12s} {d:6s} {conf:6.3f} {mae:7.3f}% {mfe:7.3f}% ${pnl:+9.4f} {dur:9.1f}m")
