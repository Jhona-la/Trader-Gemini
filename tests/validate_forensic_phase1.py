"""FORENSIC VALIDATION: Test all fixes applied in Phase 1"""
import sys
sys.path.insert(0, '.')

from config import Config
from risk.risk_manager import FeeCalculator

print("=" * 60)
print("FORENSIC FIX VALIDATION - PHASE 1")
print("=" * 60)

# Test 1: MAKER fee exists
print(f"\n[TEST 1] Fee Configuration")
print(f"  MAKER FEE: {Config.BINANCE_MAKER_FEE_BNB}")
print(f"  TAKER FEE: {Config.BINANCE_TAKER_FEE_BNB}")
assert hasattr(Config, 'BINANCE_MAKER_FEE_BNB'), "MAKER FEE NOT FOUND"
print("  ✅ PASS")

# Test 2: FeeCalc order_type
print(f"\n[TEST 2] FeeCalculator order_type distinction")
fee_limit = FeeCalculator.calculate_round_trip_fee(52.0, order_type='LIMIT')
fee_market = FeeCalculator.calculate_round_trip_fee(52.0, order_type='MARKET')
print(f"  Fee LIMIT (notional $52): ${fee_limit:.4f} ({fee_limit/52*100:.3f}%)")
print(f"  Fee MARKET (notional $52): ${fee_market:.4f} ({fee_market/52*100:.3f}%)")
savings = (fee_market - fee_limit) / fee_market * 100
print(f"  MAKER savings: {savings:.1f}%")
assert fee_limit < fee_market, f"LIMIT fee ({fee_limit}) should be < MARKET fee ({fee_market})"
print("  ✅ PASS")

# Test 3: REGIME_MAP leverage
print(f"\n[TEST 3] REGIME_MAP leverage viability for $13 account")
regime_map = Config.Sniper.REGIME_MAP
all_ok = True
for regime, params in regime_map.items():
    lev = params['leverage']
    notional = 13 * 0.40 * lev
    status = "OK" if notional >= 5 else "FAIL"
    if status == "FAIL" and regime != 'ZOMBIE':
        all_ok = False
    print(f"  {regime:15s}: leverage={lev:2d}x, notional=${notional:.1f} ({status})")
if all_ok:
    print("  ✅ PASS - All active regimes produce viable notional")
else:
    print("  ❌ FAIL - Some regimes produce sub-minimum notional")

# Test 4: Sizing parameters
print(f"\n[TEST 4] Sizing & Operational Parameters")
print(f"  MAX_CONCURRENT_POSITIONS: {Config.MAX_CONCURRENT_POSITIONS}")
print(f"  COOLDOWN_PERIOD_SECONDS: {Config.COOLDOWN_PERIOD_SECONDS}s")
print(f"  Sniper MIN_LEV: {Config.Sniper.MIN_LEVERAGE}")
print(f"  Sniper MAX_LEV: {Config.Sniper.MAX_LEVERAGE}")
print(f"  Sniper DEFAULT_LEV: {Config.Sniper.DEFAULT_LEVERAGE}")
assert Config.MAX_CONCURRENT_POSITIONS >= 3, f"MAX_CONCURRENT should be >= 3, got {Config.MAX_CONCURRENT_POSITIONS}"
assert Config.COOLDOWN_PERIOD_SECONDS <= 15, f"COOLDOWN should be <= 15s, got {Config.COOLDOWN_PERIOD_SECONDS}"
print("  ✅ PASS")

# Test 5: Mathematical Viability
print(f"\n[TEST 5] Mathematical Viability (EV per trade)")
capital = 13.0
margin_pct = 0.40
leverage = 8
margin = capital * margin_pct
notional = margin * leverage
tp_pct = 0.004  # 0.4%
sl_pct = 0.0015  # 0.15%
fee_rt = FeeCalculator.calculate_round_trip_fee(notional, order_type='LIMIT')
profit_gross = notional * tp_pct
loss_gross = notional * sl_pct
profit_net = profit_gross - fee_rt
loss_net = loss_gross + fee_rt
win_rate = 0.60

ev_per_trade = (win_rate * profit_net) - ((1 - win_rate) * loss_net)
print(f"  Capital: ${capital:.2f}")
print(f"  Margin: ${margin:.2f} ({margin_pct*100:.0f}%)")
print(f"  Notional: ${notional:.2f} (leverage {leverage}x)")
print(f"  Fee RT (MAKER): ${fee_rt:.4f}")
print(f"  Profit (gross): ${profit_gross:.4f}")
print(f"  Profit (net): ${profit_net:.4f}")
print(f"  Loss (gross): ${loss_gross:.4f}")
print(f"  Loss (net): ${loss_net:.4f}")
print(f"  Win Rate: {win_rate*100:.0f}%")
print(f"  EV per trade: ${ev_per_trade:.4f}")

trades_per_day = 10
daily_ev = ev_per_trade * trades_per_day
days_to_double = capital / daily_ev if daily_ev > 0 else float('inf')
print(f"  Trades/day: {trades_per_day}")
print(f"  Daily EV: ${daily_ev:.4f} ({daily_ev/capital*100:.2f}%/day)")
print(f"  Days to double: {days_to_double:.1f}")
if days_to_double <= 15:
    print("  ✅ PASS - Doubling in <=15 days is feasible")
else:
    print(f"  ⚠️ WARNING - Needs {days_to_double:.0f} days at current params")

print(f"\n{'=' * 60}")
print("ALL PHASE 1 TESTS COMPLETE")
print(f"{'=' * 60}")
