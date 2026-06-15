"""Comparative Forensic Analysis: V155 (Before) vs V160 (After)"""

print("=" * 70)
print("  COMPARATIVE FORENSIC: V155 (BEFORE) vs V160 (AFTER)")
print("=" * 70)

# V155 (Before) - 30 trades
v155_pnls = [
    -0.00, 0.01, 0.01, -0.01, 0.01, -0.01, -0.01, -0.02,
    0.01, 0.01, -0.01, 0.01, -0.01, -0.01, -0.01, 0.01,
    -0.02, -0.01, -0.00, 0.01, -0.01,
    0.15,  # SWING SOL
    -0.03, -0.04, -0.03,
    0.01, 0.02, 0.01, 0.01,
    0.06,  # SWING BNB
]

# V160 (After) - 34 trades (extracted from log)
v160_pnls = [
    0.00,   # ETH LONG scalp (EXIT ORACLE) +$0.00
    -0.00,  # SOL SHORT micro
    0.01, 0.01, -0.01, 0.01, -0.01, -0.01, -0.02,
    0.01, 0.01, -0.01, 0.01, -0.01, -0.01, -0.01, 0.01,
    -0.02, -0.01, -0.00, 0.01, -0.01,
    0.15,   # SWING SOL exit
    0.05,   # ETH LONG scalp (EXIT ORACLE) +$0.05 ← NEW
    -0.03, -0.04, -0.03,
    0.01, 0.02, 0.01, 0.01,
    0.06,   # SWING BNB exit
    -0.15,  # ETH LONG loss (large) ← NEW
    0.02,   # SOL LONG scalp (EXIT ORACLE) +$0.02 ← NEW
]

def analyze(label, pnls):
    total = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    flat = [p for p in pnls if p == 0]
    wr = len(wins)/len(pnls)*100 if pnls else 0
    pf = sum(wins)/abs(sum(losses)) if losses and sum(losses) != 0 else 99.9
    
    print(f"\n  {label}")
    print(f"  {'─'*50}")
    print(f"  Trades:         {len(pnls)}")
    print(f"  Wins:           {len(wins)} | Losses: {len(losses)} | Flat: {len(flat)}")
    print(f"  Win Rate:       {wr:.1f}%")
    print(f"  Total PnL:      USD {total:.4f}")
    print(f"  Avg Win:        USD {sum(wins)/len(wins):.4f}" if wins else "  Avg Win:        N/A")
    print(f"  Avg Loss:       USD {sum(losses)/len(losses):.4f}" if losses else "  Avg Loss:       N/A")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"  Best Trade:     USD {max(pnls):.4f}")
    print(f"  Worst Trade:    USD {min(pnls):.4f}")
    return wr, total, pf

wr1, pnl1, pf1 = analyze("V155 (BEFORE — tp=0.35%, sl=0.50%, gate=3.0x)", v155_pnls)
wr2, pnl2, pf2 = analyze("V160 (AFTER  — tp=0.55%, sl=0.35%, gate=2.0x)", v160_pnls)

print(f"\n{'=' * 70}")
print(f"  COMPARISON DELTA")
print(f"{'=' * 70}")
print(f"  Win Rate:       {wr1:.1f}% → {wr2:.1f}% ({wr2-wr1:+.1f}pp)")
print(f"  PnL:            USD {pnl1:.4f} → USD {pnl2:.4f} ({pnl2-pnl1:+.4f})")
print(f"  Profit Factor:  {pf1:.2f} → {pf2:.2f} ({pf2-pf1:+.2f})")
print(f"  Trade Count:    {len(v155_pnls)} → {len(v160_pnls)} ({len(v160_pnls)-len(v155_pnls):+d})")

# NEW features in V160
print(f"\n{'=' * 70}")
print(f"  NEW BEHAVIORS IN V160")
print(f"{'=' * 70}")
print("""
  ✅ LONG trades now appearing (3 new LONGs: ETH, ETH, SOL)
     - ETH LONG scalp: +$0.05 (EXIT ORACLE via BB touch)
     - SOL LONG scalp: +$0.02 (EXIT ORACLE via BB touch)
     - ETH LONG: -$0.15 (LARGEST LOSS — new risk)
  
  ✅ Relaxed volatility gate (2.0x) allowed more trading
     - 4 more trades executed vs V155 (34 vs 30)
  
  ⚠️ SWING UNCHANGED: Still 100% WR, +$0.21
     - SOL SWING: +$0.15 (identical)
     - BNB SWING: +$0.06 (identical)
  
  ⚠️ NEW RISK: ETH LONG -$0.15 is the largest single loss
     - This suggests LONG entries need additional filtering
     - The MARKET PANIC veto was too relaxed for this entry

  🔍 ROOT CAUSE of remaining poor scalping WR:
     The old positions opened BEFORE our config changes still used
     tp_pct=0.003 (visible in DEBUG logs). The new config only 
     affects positions opened AFTER the config load. The backtest
     processes the same historical data, so early positions inherit
     the old parameters from the position dict, NOT from Config.
     
     → The tp_pct is SET AT ENTRY TIME, not checked dynamically.
     → We need to find WHERE tp_pct is assigned to positions.
""")

# Final balance comparison
print(f"  V155 Final Balance: USD {13.0 + pnl1:.2f} ({pnl1/13*100:+.2f}%)")
print(f"  V160 Final Balance: USD {13.0 + pnl2:.2f} ({pnl2/13*100:+.2f}%)")
