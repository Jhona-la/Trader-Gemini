import json

# Compare pre-fix7 vs post-fix7
old = json.load(open("results/backtests/god_mode_6070a7f2_1d.json"))
new = json.load(open("results/backtests/god_mode_2d125e00_1d.json"))

print("=" * 70)
print("FIX 7 IMPACT: SL 0.20% → 0.45% MICRO / 0.40% SCALP")
print("=" * 70)

for label, d in [("PRE-FIX7 (6070a)", old), ("POST-FIX7 (2d125)", new)]:
    m = d["metrics"]
    print(f"\n--- {label} ---")
    print(f"  Capital:    ${m.get('final_capital', 0):.2f} (Return: {m.get('total_return_pct', 0):.2f}%)")
    print(f"  Trades:     {m.get('wins', 0)}W/{m.get('losses', 0)}L  WR={m.get('win_rate', 0):.1f}%")
    print(f"  Max DD:     {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Kill Switch: {m.get('kill_switch_triggered', '?')}")

# Detailed trade analysis for new
print("\n" + "=" * 70)
print("TRADE TIMELINE (POST-FIX7)")
print("=" * 70)
th = new.get("trade_history", {})
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            all_trades.append(t)
all_trades.sort(key=lambda x: str(x.get("exit_time", x.get("entry_time", ""))))

running = 0
sl_count = 0
tp_count = 0
timeout_count = 0
trail_count = 0
for t in all_trades:
    pnl = t.get("net_pnl", t.get("pnl", 0))
    running += pnl
    ex = t.get("exit_reason", "?")
    if "HARD_SL" in ex: sl_count += 1
    elif "TIMEOUT" in ex: timeout_count += 1
    elif "TP" in ex or "TAKE_PROFIT" in ex: tp_count += 1
    elif "TRAIL" in ex: trail_count += 1
    sym = t.get("symbol", "?")
    d = t.get("direction", "?")
    print(f"  {sym:10} {d:5} PnL=${pnl:+.4f} Run=${running:+.4f} Exit={ex[:40]}")

print(f"\nExit Analysis:")
print(f"  HARD_SL:  {sl_count}/{len(all_trades)} ({sl_count/len(all_trades)*100:.0f}%)")
print(f"  TIMEOUT:  {timeout_count}/{len(all_trades)} ({timeout_count/len(all_trades)*100:.0f}%)")
print(f"  TP:       {tp_count}/{len(all_trades)} ({tp_count/len(all_trades)*100:.0f}%)")
print(f"  TRAIL:    {trail_count}/{len(all_trades)} ({trail_count/len(all_trades)*100:.0f}%)")
print(f"  OTHER:    {len(all_trades)-sl_count-timeout_count-tp_count-trail_count}")

# Symbol comparison
print(f"\nSymbol Performance:")
sym_stats = {}
for t in all_trades:
    sym = t.get("symbol", "?")
    pnl = t.get("net_pnl", t.get("pnl", 0))
    if sym not in sym_stats: sym_stats[sym] = {"w": 0, "l": 0, "pnl": 0}
    sym_stats[sym]["pnl"] += pnl
    if pnl > 0: sym_stats[sym]["w"] += 1
    else: sym_stats[sym]["l"] += 1
for sym, s in sorted(sym_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
    total = s["w"] + s["l"]
    wr = s["w"]/total*100 if total > 0 else 0
    print(f"  {sym:12} {s['w']}W/{s['l']}L WR={wr:5.1f}% PnL=${s['pnl']:+.4f}")
