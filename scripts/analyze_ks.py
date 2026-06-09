import json

d = json.load(open("results/backtests/god_mode_0ca2143c_7d.json"))
m = d["metrics"]
print(f"Kill Switch: {m.get('kill_switch_triggered')}")
print(f"Max DD: {m.get('max_drawdown_pct')}%")
print(f"Final Capital: ${m.get('final_capital', 0):.2f}")
print(f"Total trades: {m.get('total_trades')}")
print(f"Wins: {m.get('wins')} Losses: {m.get('losses')}")

rej = d.get("rejection_reasons", {})
ks = {k: v for k, v in rej.items() if "KILL" in k}
print(f"KS rejections: {ks}")

# Check daily losses count
th = d.get("trade_history", {})
losses_list = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            pnl = t.get("net_pnl", t.get("pnl", 0))
            if pnl < 0:
                losses_list.append(pnl)

print(f"Total losing trades: {len(losses_list)}")
print(f"Consecutive losses could trigger MAX_DAILY_LOSSES={10}")
