import json

d = json.load(open("results/backtests/god_mode_2d125e00_1d.json"))
th = d.get("trade_history", {})

all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            all_trades.append(t)

# BTC trades analysis
btc = [t for t in all_trades if t.get("symbol") == "BTC/USDT"]
print(f"BTC/USDT: {len(btc)} trades")
print(f"  Wins: {sum(1 for t in btc if t.get('net_pnl', t.get('pnl', 0)) > 0)}")
print(f"  Losses: {sum(1 for t in btc if t.get('net_pnl', t.get('pnl', 0)) <= 0)}")

# Check directions
dirs = [t.get("direction") for t in btc]
print(f"  LONG: {dirs.count('LONG')} SHORT: {dirs.count('SHORT')}")

# BTC winning trades analysis
print(f"\nBTC WINNING trades:")
for t in btc:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    if pnl > 0:
        print(f"  Dir={t.get('direction')} PnL=${pnl:.4f} Exit={t.get('exit_reason','?')[:40]}")

# Check if BTC losses are all from LONG during downtrend
print(f"\nBTC LOSS distribution:")
losses_by_dir = {}
for t in btc:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    if pnl <= 0:
        d = t.get("direction", "?")
        losses_by_dir[d] = losses_by_dir.get(d, 0) + 1
print(f"  {losses_by_dir}")

# SOL analysis for comparison
sol = [t for t in all_trades if t.get("symbol") == "SOL/USDT"]
print(f"\nSOL/USDT: {len(sol)} trades")
for t in sol:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    print(f"  Dir={t.get('direction'):5} PnL=${pnl:+.4f} Exit={t.get('exit_reason','?')[:40]}")
