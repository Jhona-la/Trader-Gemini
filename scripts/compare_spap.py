import json

# Compare Fix10 vs SPAP
files = {
    "Fix10 (pre-SPAP)": "results/backtests/god_mode_9d838e05_1d.json",
    "SPAP Basic": "results/backtests/god_mode_fa07cefb_1d.json",
}

for label, f in files.items():
    d = json.load(open(f))
    m = d["metrics"]
    print(f"\n=== {label} ===")
    print(f"Capital: ${m['final_capital']:.2f}")
    print(f"Return: {m['total_return_pct']:.2f}%")
    print(f"Trades: {m.get('wins',0)}W/{m.get('losses',0)}L WR={m.get('win_rate',0):.1f}%")
    print(f"Max DD: {m.get('max_drawdown_pct',0):.2f}%")

    th = d.get("trade_history", {})
    all_trades = []
    for cat, trades in th.items():
        if isinstance(trades, list):
            all_trades.extend(trades)
    all_trades.sort(key=lambda x: str(x.get("exit_time", x.get("entry_time", ""))))
    print(f"Trade Details:")
    for t in all_trades:
        pnl = t.get("net_pnl", t.get("pnl", 0))
        dur = t.get("duration_seconds", 0)
        print(f"  {t.get('symbol','?'):12s} {t.get('direction','?'):6s} PnL=${pnl:+.4f} Dur={dur:.0f}s Exit={t.get('exit_reason','?')}")
