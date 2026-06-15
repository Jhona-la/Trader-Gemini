"""Forensic Analysis of 7-Day Partial Backtest Results"""

pnls = [
    -0.00, 0.01, 0.01, -0.01, 0.01, -0.01, -0.01, -0.02,
    0.01, 0.01, -0.01, 0.01, -0.01, -0.01, -0.01, 0.01,
    -0.02, -0.01, -0.00, 0.01, -0.01,
    0.15,  # SWING exit SOL/USDT
    -0.03, -0.04, -0.03,  # ETH scalping losses
    0.01, 0.02,  # ETH scalping wins
    0.01, 0.01,  # BNB scalping wins
    0.06,  # SWING exit BNB/USDT
]

total = sum(pnls)
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p < 0]
flat = [p for p in pnls if p == 0]

print("=" * 60)
print("  BACKTEST FORENSIC REPORT (PARTIAL 7-DAY RUN)")
print("=" * 60)
print(f"  Total Trades:   {len(pnls)}")
print(f"  Wins:           {len(wins)}")
print(f"  Losses:         {len(losses)}")
print(f"  Flat:           {len(flat)}")
print(f"  Win Rate:       {len(wins)/len(pnls)*100:.1f}%")
print(f"  Total PnL:      USD {total:.4f}")
print(f"  Avg Win:        USD {sum(wins)/len(wins):.4f}")
print(f"  Avg Loss:       USD {sum(losses)/len(losses):.4f}")
print(f"  Best Trade:     USD {max(pnls):.4f}")
print(f"  Worst Trade:    USD {min(pnls):.4f}")
print(f"  Profit Factor:  {sum(wins)/abs(sum(losses)):.2f}")
print()

# Separate by horizon
scalp_pnls = [
    -0.00, 0.01, 0.01, -0.01, 0.01, -0.01, -0.01, -0.02,
    0.01, 0.01, -0.01, 0.01, -0.01, -0.01, -0.01, 0.01,
    -0.02, -0.01, -0.00, 0.01, -0.01,
    -0.03, -0.04, -0.03,
    0.01, 0.02, 0.01, 0.01,
]
swing_pnls = [0.15, 0.06]

s_wins = [p for p in scalp_pnls if p > 0]
s_losses = [p for p in scalp_pnls if p < 0]

print("-" * 60)
print("  SCALPING ANALYSIS")
print("-" * 60)
print(f"  Trades:         {len(scalp_pnls)}")
print(f"  Wins:           {len(s_wins)}")
print(f"  Losses:         {len(s_losses)}")
print(f"  Win Rate:       {len(s_wins)/len(scalp_pnls)*100:.1f}%")
print(f"  Total PnL:      USD {sum(scalp_pnls):.4f}")
print(f"  Avg Win:        USD {sum(s_wins)/len(s_wins):.4f}")
print(f"  Avg Loss:       USD {sum(s_losses)/len(s_losses):.4f}")
if s_losses:
    print(f"  Profit Factor:  {sum(s_wins)/abs(sum(s_losses)):.2f}")
print()

print("-" * 60)
print("  SWING ANALYSIS")
print("-" * 60)
print(f"  Trades:         {len(swing_pnls)}")
print(f"  Win Rate:       100%")
print(f"  Total PnL:      USD {sum(swing_pnls):.4f}")
print(f"  Avg Win:        USD {sum(swing_pnls)/len(swing_pnls):.4f}")
print()

print("=" * 60)
print("  CRITICAL FINDINGS")
print("=" * 60)
print("""
  1. SCALPING WR = 42.9% — CATASTROPHICALLY LOW (target: >70%)
     ROOT CAUSE: TP target of 0.3% (MICROSCALPING) is too tight
     vs fee of 0.04%. The R:R is only ~3:1 on paper but entries
     are noisy, causing frequent stop-outs before TP is reached.
  
  2. SCALPING PnL NEAR ZERO — Fees eat all profits.
     Avg win = USD 0.01, Avg loss = USD -0.017
     The system pays ~USD 0.0012 per trade in fees (maker).
     With 28 trades, total fees ~ USD 0.034.
  
  3. SWING = 100% WR with USD +0.21 — The REAL edge.
     Both swing exits were CLOSE_TECH_REVERSAL (Bollinger touch).
     Swing holds for hours/days and captures larger moves.
  
  4. MARKET PANIC VETO blocked ~65% of the 7 days.
     From 03:23 Day 1 through most of Day 3, the system saw
     100% bearish volume and vetoed all longs. This is CORRECT
     behavior — it prevented catastrophic losses.
  
  5. VOLATILITY BLOCK killed most BNB/ETH scalping opportunities.
     BNB ATR frequently below 0.04% minimum threshold.
     ETH/SOL ATR below 0.12% (3x round-trip fee) threshold.
  
  6. ONLY SHORT SIGNALS GENERATED — No long scalps at all.
     All 28 scalping trades were SHORT. The system never found
     a LONG setup that passed all filters during this period.

  RECOMMENDATIONS:
  ─────────────────────────────────────────────────────────
  A. FOCUS ON SWING — It's where the real PnL comes from.
     Increase swing allocation, keep scalping as supplementary.
  
  B. WIDEN SCALPING TP — 0.3% is too tight for 1-min bars.
     Consider 0.5-0.8% TP for scalping with tighter entry
     filtering (only enter when momentum is overwhelming).
  
  C. REDUCE SCALPING FREQUENCY — Quality over quantity.
     28 scalping trades in ~3 active days = too many.
     Add minimum cooldown period between scalp entries.
  
  D. ADD LONG SETUPS — System only took shorts. 
     Review why RSI<30 mean-reversion longs are being vetoed.
     The MARKET PANIC veto is too aggressive for scalping.
""")
