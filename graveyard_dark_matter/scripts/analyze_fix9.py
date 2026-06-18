import json

d = json.load(open("results/backtests/god_mode_720246d0_1d.json"))
m = d["metrics"]

print("=" * 70)
print("FIX 9 RESULTS: TP 0.60%->0.35%, SL 0.45%->0.50%")
print("=" * 70)
print(f"  Capital:      ${m.get('final_capital', 0):.2f}")
print(f"  Return:       {m.get('total_return_pct', 0):.2f}%")
print(f"  Trades:       {m.get('wins', 0)}W / {m.get('losses', 0)}L")
print(f"  Win Rate:     {m.get('win_rate', 0):.1f}%")
print(f"  Max DD:       {m.get('max_drawdown_pct', 0):.2f}%")
print(f"  Kill Switch:  {m.get('kill_switch_triggered', '?')}")

# Strategy attribution
sa = d.get("forensic_strategy_attribution", d.get("strategy_attribution", {}))
print("\nStrategy Attribution:")
for name, info in sorted(sa.items(), key=lambda x: x[1].get("net_pnl", 0), reverse=True):
    w = info.get("wins", 0)
    l = info.get("losses", 0)
    pnl = info.get("net_pnl", 0)
    t = w + l
    wr = w / t * 100 if t > 0 else 0
    print(f"  {name:55s} {w}W/{l}L WR={wr:5.1f}% PnL=${pnl:+.4f}")

# Symbol + trade timeline
th = d.get("trade_history", {})
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            all_trades.append(t)
all_trades.sort(key=lambda x: str(x.get("exit_time", x.get("entry_time", ""))))

# Exit analysis
exits = {}
for t in all_trades:
    ex = t.get("exit_reason", "?")
    key = "TP" if "TP" in ex or "TAKE_PROFIT" in ex else ("SL" if "HARD_SL" in ex else ("TIMEOUT" if "TIMEOUT" in ex else ("TRAIL" if "TRAIL" in ex else "OTHER")))
    exits[key] = exits.get(key, 0) + 1
print(f"\nExit Types: {exits}")

# Per-symbol per-direction
sym_dir = {}
for t in all_trades:
    sym = t.get("symbol", "?")
    d_dir = t.get("direction", "?")
    pnl = t.get("net_pnl", t.get("pnl", 0))
    key = f"{sym}_{d_dir}"
    if key not in sym_dir: sym_dir[key] = {"w": 0, "l": 0, "pnl": 0}
    sym_dir[key]["pnl"] += pnl
    if pnl > 0: sym_dir[key]["w"] += 1
    else: sym_dir[key]["l"] += 1

print("\nSymbol x Direction:")
for key, s in sorted(sym_dir.items(), key=lambda x: x[1]["pnl"], reverse=True):
    total = s["w"] + s["l"]
    wr = s["w"]/total*100 if total > 0 else 0
    print(f"  {key:20s} {s['w']}W/{s['l']}L WR={wr:5.1f}% PnL=${s['pnl']:+.4f}")

# Progression
print("\n" + "=" * 70)
print("RETURN PROGRESSION (ALL FIXES)")
print("=" * 70)
print("  Fix 1-3:  -15.03%  (7d)")
print("  Fix 4-5:  -13.69%  (7d)")
print("  Fix 1-6:   -6.57%  (1d)")
print("  Fix 1-7:   -1.01%  (1d)")
print("  Fix 1-8:   -0.85%  (1d)")
print(f"  Fix 1-9:   {m.get('total_return_pct', 0):+.2f}%  (1d) ← CURRENT 🎉")
