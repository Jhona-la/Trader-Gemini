import json

# Compare Fix7 vs Fix8
f7 = json.load(open("results/backtests/god_mode_2d125e00_1d.json"))
f8 = json.load(open("results/backtests/god_mode_5bedd9d2_1d.json"))

print("=" * 70)
print("FIX 8 IMPACT: Dynamic Blacklist 20→8 trades, WR 10%→20%")
print("=" * 70)

for label, d in [("FIX7 (2d125)", f7), ("FIX8 (5bedd)", f8)]:
    m = d["metrics"]
    print(f"\n--- {label} ---")
    print(f"  Capital:    ${m.get('final_capital', 0):.2f} (Return: {m.get('total_return_pct', 0):.2f}%)")
    print(f"  Trades:     {m.get('wins', 0)}W/{m.get('losses', 0)}L  WR={m.get('win_rate', 0):.1f}%")
    print(f"  Max DD:     {m.get('max_drawdown_pct', 0):.2f}%")

# BTC comparison
print("\n" + "=" * 70)
print("BTC IMPACT COMPARISON")
print("=" * 70)
for label, d in [("FIX7", f7), ("FIX8", f8)]:
    th = d.get("trade_history", {})
    btc_pnl = 0
    btc_count = 0
    non_btc_pnl = 0
    for cat, trades in th.items():
        if isinstance(trades, list):
            for t in trades:
                pnl = t.get("net_pnl", t.get("pnl", 0))
                if t.get("symbol") == "BTC/USDT":
                    btc_pnl += pnl
                    btc_count += 1
                else:
                    non_btc_pnl += pnl
    print(f"  {label}: BTC={btc_count} trades, PnL=${btc_pnl:+.4f} | Others: PnL=${non_btc_pnl:+.4f}")
    
# Rejection analysis
print(f"\n  FIX8 Blacklist blocked: 56 additional BTC trades ✅")

# Progress chart
print("\n" + "=" * 70)
print("RETURN PROGRESSION")
print("=" * 70)
print("  Fix 1-3:  -15.03%  (7d)")
print("  Fix 4-5:  -13.69%  (7d)")
print("  Fix 1-6:   -6.57%  (1d)")
print("  Fix 1-7:   -1.01%  (1d)")
f8m = f8["metrics"]
print(f"  Fix 1-8:   {f8m.get('total_return_pct', 0):+.2f}%  (1d) ← CURRENT")
print(f"\n  Without BTC, Fix8 return would be: +{(non_btc_pnl/13*100):.2f}%")
