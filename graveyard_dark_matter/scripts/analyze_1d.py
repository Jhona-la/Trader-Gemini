import json

d = json.load(open("results/backtests/god_mode_6070a7f2_1d.json"))
m = d["metrics"]

print("=" * 70)
print("BACKTEST 1D — 6 FIXES APPLIED (6070a7f2)")
print("=" * 70)
print(f"  Capital:      ${m.get('final_capital', 0):.2f}")
print(f"  Return:       {m.get('total_return_pct', 0):.2f}%")
print(f"  Signals:      {m.get('signals_generated', 0)}")
print(f"  Orders:       {m.get('orders_generated', 0)}")
print(f"  Rejected:     {m.get('orders_rejected', 0)}")
print(f"  Wins/Losses:  {m.get('wins', 0)}W / {m.get('losses', 0)}L")
print(f"  Win Rate:     {m.get('win_rate', 0):.1f}%")
print(f"  Max DD:       {m.get('max_drawdown_pct', 0):.2f}%")
print(f"  Kill Switch:  {m.get('kill_switch_triggered', '?')}")
print(f"  Fees:         ${m.get('fees_paid', 0):.4f}")

# Strategy attribution
print("\n" + "=" * 70)
print("STRATEGY ATTRIBUTION")
print("=" * 70)
sa = d.get("forensic_strategy_attribution", d.get("strategy_attribution", {}))
for name, info in sorted(sa.items(), key=lambda x: x[1].get("net_pnl", 0), reverse=True):
    w = info.get("wins", 0)
    l = info.get("losses", 0)
    pnl = info.get("net_pnl", info.get("gross_pnl", 0))
    t = info.get("trades", w+l)
    wr = w / t * 100 if t > 0 else 0
    print(f"  {name:55s} {w}W/{l}L WR={wr:5.1f}% PnL=${pnl:+.4f}")

# Symbol attribution
print("\n" + "=" * 70)
print("SYMBOL ATTRIBUTION")
print("=" * 70)
th = d.get("trade_history", {})
sym_stats = {}
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            sym = t.get("symbol", "?")
            pnl = t.get("net_pnl", t.get("pnl", 0))
            win = 1 if pnl > 0 else 0
            if sym not in sym_stats:
                sym_stats[sym] = {"wins": 0, "losses": 0, "pnl": 0, "trades": 0}
            sym_stats[sym]["trades"] += 1
            sym_stats[sym]["pnl"] += pnl
            sym_stats[sym]["wins"] += win
            sym_stats[sym]["losses"] += (1 - win)
            all_trades.append(t)

for sym, s in sorted(sym_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
    wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
    print(f"  {sym:12s} {s['wins']}W/{s['losses']}L WR={wr:5.1f}% PnL=${s['pnl']:+.4f} ({s['trades']} trades)")

# Trade timeline
print("\n" + "=" * 70)
print("ALL TRADES CHRONOLOGICALLY")
print("=" * 70)
all_trades.sort(key=lambda x: str(x.get("exit_time", x.get("entry_time", ""))))
running = 0
for t in all_trades:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    running += pnl
    sym = t.get("symbol", "?")
    d_dir = t.get("direction", "?")
    ex = t.get("exit_reason", "?")[:35]
    strat = t.get("strategy_id", "?")[:25]
    print(f"  {sym:10} {d_dir:5} PnL=${pnl:+.4f} Run=${running:+.4f} Strat={strat} Exit={ex}")
