"""Compare the latest backtest result against previous runs."""
import json, glob, os

results_dir = "results/backtests"
files = sorted(glob.glob(f"{results_dir}/god_mode_*_7d.json"), key=os.path.getmtime, reverse=True)

if len(files) < 2:
    print("Need at least 2 results to compare")
    exit()

latest = json.load(open(files[0]))
prev = json.load(open(files[1]))

print("=" * 70)
print(f"LATEST:   {os.path.basename(files[0])}")
print(f"PREVIOUS: {os.path.basename(files[1])}")
print("=" * 70)

for label, d in [("PREVIOUS", prev), ("LATEST  ", latest)]:
    m = d["metrics"]
    print(f"\n--- {label} ---")
    print(f"  Capital:    ${m.get('final_capital', 0):.2f} (Return: {m.get('total_return_pct', 0):.2f}%)")
    print(f"  Trades:     {m.get('wins', 0)}W/{m.get('losses', 0)}L  WR={m.get('win_rate', 0):.1f}%")
    print(f"  Signals:    {m.get('signals_generated', 0)} → Orders: {m.get('orders_generated', 0)} → Rejected: {m.get('orders_rejected', 0)}")
    print(f"  Max DD:     {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Fees:       ${m.get('fees_paid', 0):.4f}")
    print(f"  Kill Switch: {m.get('kill_switch_triggered', '?')}")

# Strategy attribution
print("\n" + "=" * 70)
print("STRATEGY ATTRIBUTION (LATEST)")
print("=" * 70)
sa = latest.get("forensic_strategy_attribution", latest.get("strategy_attribution", {}))
total_pnl = 0
for name, info in sorted(sa.items(), key=lambda x: x[1].get("net_pnl", 0), reverse=True):
    w = info.get("wins", 0)
    l = info.get("losses", 0)
    pnl = info.get("net_pnl", info.get("gross_pnl", 0))
    total_pnl += pnl
    wr = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f"  {name:50s} {w}W/{l}L WR={wr:5.1f}% PnL=${pnl:+.4f}")
print(f"  {'TOTAL':50s} PnL=${total_pnl:+.4f}")

# Trade directions
print("\n" + "=" * 70)
print("TRADE DIRECTIONS (LATEST)")
print("=" * 70)
th = latest.get("trade_history", {})
for cat, trades in th.items():
    if isinstance(trades, list):
        dirs = [t.get("direction", "?") for t in trades]
        print(f"  {cat}: {len(trades)} trades | LONG={dirs.count('LONG')} SHORT={dirs.count('SHORT')}")

# Top rejections aggregated
print("\n" + "=" * 70)
print("TOP REJECTIONS (LATEST, aggregated)")
print("=" * 70)
rej = latest.get("rejection_reasons", {})
agg = {}
for k, v in rej.items():
    if "SOPHIA" in k:
        agg["SOPHIA_VETO"] = agg.get("SOPHIA_VETO", 0) + v
    elif "KILL_SWITCH" in k:
        agg["KILL_SWITCH"] = agg.get("KILL_SWITCH", 0) + v
    elif "DIRECTIONAL" in k:
        agg["DIRECTIONAL_SAFETY"] = agg.get("DIRECTIONAL_SAFETY", 0) + v
    elif "INSUFFICIENT" in k:
        agg["INSUFFICIENT_MARGIN"] = agg.get("INSUFFICIENT_MARGIN", 0) + v
    elif "FLIP" in k:
        agg["FLIP_GATE"] = agg.get("FLIP_GATE", 0) + v
    else:
        key = k.split(" for ")[0] if " for " in k else k
        agg[key] = agg.get(key, 0) + v
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:15]:
    print(f"  {k}: {v}")

# Trade timeline (last 10)
print("\n" + "=" * 70)
print("LAST 10 TRADES (LATEST)")
print("=" * 70)
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            all_trades.append(t)
all_trades.sort(key=lambda x: str(x.get("exit_time", x.get("entry_time", ""))))
running = sum(t.get("net_pnl", t.get("pnl", 0)) for t in all_trades)
for t in all_trades[-10:]:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    sym = t.get("symbol", "?")
    d = t.get("direction", "?")
    ex = t.get("exit_reason", "?")[:30]
    print(f"  {sym:10} {d:5} PnL=${pnl:+.4f} Exit={ex}")
print(f"  Total PnL: ${running:+.4f}")
