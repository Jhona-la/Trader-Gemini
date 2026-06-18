import json

d = json.load(open("results/backtests/god_mode_96bf3c7f_7d.json"))

# Check for the forensic data
fv10 = d.get("forensic_v10", {})
print("FORENSIC V10 KEYS:", list(fv10.keys())[:20])

# Check strategy attribution in detail  
sa = d.get("forensic_strategy_attribution", d.get("strategy_attribution", {}))
print("\nDETAILED STRATEGY ATTR:")
for name, info in sa.items():
    print(f"  {name}:")
    for k, v in info.items():
        print(f"    {k}: {v}")

# Check if KS activation reason is stored
m = d.get("metrics", {})
print(f"\nKS Triggered: {m.get('kill_switch_triggered')}")
print(f"KS Reason: {m.get('kill_switch_reason', 'N/A')}")

# Check equity curve sample to see when DD occurred
eq = d.get("equity_curve_sample", [])
if eq:
    print(f"\nEquity curve points: {len(eq)}")
    for p in eq[:5]:
        print(f"  {p}")
    print("  ...")
    for p in eq[-5:]:
        print(f"  {p}")
