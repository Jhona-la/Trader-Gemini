import json
d = json.load(open("results/backtests/god_mode_e599e579_3d.json"))
m = d["metrics"]
print(f"Capital: ${m.get('final_capital',0):.2f}")
print(f"Return: {m.get('total_return_pct',0):.2f}%")
print(f"Signals: {m.get('signals_generated',0)}")
print(f"Orders: {m.get('orders_generated',0)}")
print(f"Rejected: {m.get('orders_rejected',0)}")
print(f"Trades: {m.get('wins',0)}W/{m.get('losses',0)}L WR={m.get('win_rate',0):.1f}%")
print(f"Kill Switch: {m.get('kill_switch_triggered','?')}")
rej = d.get("rejection_reasons", {})
print("Top rejections:")
for k,v in sorted(rej.items(), key=lambda x:-x[1])[:10]:
    print(f"  {k}: {v}")

# Trade details
th = d.get("trade_history", {})
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            all_trades.append(t)
all_trades.sort(key=lambda x: str(x.get("exit_time", x.get("entry_time", ""))))
print(f"\nAll {len(all_trades)} trades:")
for t in all_trades:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    print(f"  {t.get('symbol','?'):12s} {t.get('direction','?'):6s} PnL=${pnl:+.4f} Exit={t.get('exit_reason','?')}")
