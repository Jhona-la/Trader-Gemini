import json

d = json.load(open("results/backtests/god_mode_96bf3c7f_7d.json"))

# Look for the exact kill switch activation in forensic data
m = d["metrics"]
print("KS:", m.get("kill_switch_triggered"))

# Check if activation reason is in forensic scenario 
fs = d.get("forensic_scenario", {})
print("Forensic scenario:", json.dumps(fs, indent=2)[:500] if fs else "None")

# Look at trade history timestamps to see when the kill happened
th = d.get("trade_history", {})
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        for t in trades:
            all_trades.append({
                "time": t.get("exit_time", t.get("entry_time", "")),
                "symbol": t.get("symbol", "?"),
                "dir": t.get("direction", "?"),
                "pnl": t.get("net_pnl", t.get("pnl", 0)),
                "exit_reason": t.get("exit_reason", "?"),
                "strat": t.get("strategy_id", "?")[:30],
            })

# Sort by time
all_trades.sort(key=lambda x: str(x["time"]))

print(f"\nAll {len(all_trades)} trades chronologically:")
running = 0
for t in all_trades:
    running += t["pnl"]
    print(f"  {t['time']} {t['symbol']:10} {t['dir']:5} PnL=${t['pnl']:.4f} Running=${running:.4f} Exit={t['exit_reason'][:25]}")
