import json
d = json.load(open("results/backtests/god_mode_1b777d1e_3d.json"))
th = d.get("trade_history", {})
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            all_trades.append(t)

all_trades.sort(key=lambda x: str(x.get("exit_time", x.get("entry_time", ""))))
print(f"Total trades: {len(all_trades)}")
for t in all_trades:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    dur = t.get("duration_seconds", 0)
    sym = t.get("symbol", "?")
    d2 = t.get("direction", "?")
    ex = t.get("exit_reason", "?")
    print(f"  {sym:12s} {d2:6s} PnL=${pnl:+.4f} Duration={dur:.0f}s Exit={ex}")
