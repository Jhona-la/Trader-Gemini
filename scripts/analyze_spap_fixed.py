import json

f = "results/backtests/god_mode_4a929bba_1d.json"
d = json.load(open(f))
m = d["metrics"]
print(f"=== SPAP Unified Fixed ===")
print(f"Capital: ${m['final_capital']:.2f}")
print(f"Return: {m['total_return_pct']:.2f}%")
print(f"Trades: {m.get('wins',0)}W/{m.get('losses',0)}L WR={m.get('win_rate',0):.1f}%")

th = d.get("trade_history", {})
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        all_trades.extend(trades)
all_trades.sort(key=lambda x: str(x.get("exit_time", x.get("entry_time", ""))))
print(f"\nTrade Details ({len(all_trades)} trades):")
for t in all_trades:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    dur = t.get("duration_seconds", 0)
    sym = t.get("symbol", "?")
    d = t.get("direction", "?")
    exit_r = t.get("exit_reason", "?")
    strat = t.get("strategy_id", "?")[:20]
    print(f"  {sym:12s} {d:6s} PnL=${pnl:+.4f} Dur={dur:6.0f}s Exit={exit_r}")
