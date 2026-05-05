import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TRADER_GEMINI_BACKTEST"] = "true"

from config import Config

scalp = getattr(Config.Strategies, 'SCALPING_PARAMS', {})
swing = getattr(Config.Strategies, 'SWING_PARAMS', {})

print("=" * 60)
print("P2 FORENSIC: SL/TP PARAMETER PARITY AUDIT")
print("=" * 60)

print(f"\n[Config.Strategies.SCALPING_PARAMS]")
print(f"  tp_pct = {scalp.get('tp_pct', 'MISSING')} = {scalp.get('tp_pct', 0)*100:.2f}%")
print(f"  sl_pct = {scalp.get('sl_pct', 'MISSING')} = {scalp.get('sl_pct', 0)*100:.2f}%")

print(f"\n[Config.Strategies.SWING_PARAMS]")
print(f"  tp_pct = {swing.get('tp_pct', 'MISSING')} = {swing.get('tp_pct', 0)*100:.2f}%")
print(f"  sl_pct = {swing.get('sl_pct', 'MISSING')} = {swing.get('sl_pct', 0)*100:.2f}%")

mk = getattr(Config, "BINANCE_MAKER_FEE_BNB", 0.0002)
tk = getattr(Config, "BINANCE_TAKER_FEE_BNB", 0.000375)
fb = mk + tk
sl = scalp.get('sl_pct', 0.006)
dz = max(fb * 1.5, sl * 0.5)

print(f"\n[Dead Zone Calculation]")
print(f"  Fee buffer (maker+taker): {fb*100:.4f}%")
print(f"  Fee buffer * 1.5: {fb*1.5*100:.4f}%")
print(f"  SCALP SL * 0.5: {sl*0.5*100:.4f}%")
print(f"  DEAD ZONE = max(...) = {dz*100:.4f}%")
print(f"  FLIP threshold: -0.15%")

# Simulate the losing trade scenarios
print(f"\n{'='*60}")
print(f"SCENARIO ANALYSIS")
print(f"{'='*60}")

for pnl_pct in [-0.0107, -0.0166, -0.005, -0.002, -0.001, 0.0032]:
    ft = -0.0015
    in_dz = pnl_pct >= ft and pnl_pct < dz
    below_flip = pnl_pct < ft
    result = "FLIP BLOCKED (dead zone)" if in_dz else ("FLIP ALLOWED (heavy loss)" if below_flip else "FLIP ALLOWED (profitable)")
    # Would HARD SL have fired?
    hard_sl = abs(pnl_pct) >= sl
    print(f"  PnL={pnl_pct*100:+.2f}% -> {result} | Hard SL fires? {hard_sl} (SL={sl*100:.2f}%)")

print(f"\n{'='*60}")
print(f"KEY INSIGHT:")
print(f"  Trade at -1.07%: FLIP ALLOWED because -1.07% < flip_threshold (-0.15%)")
print(f"  BUT Hard SL (0.60%) should have fired FIRST at -0.60%!")
print(f"  This means check_stops() is NOT evaluating fast enough,")
print(f"  OR the backtest is skipping ticks where SL would trigger.")
print(f"{'='*60}")
