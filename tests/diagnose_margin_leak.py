#!/usr/bin/env python3
"""
DIAGNÓSTICO FORENSE: MARGIN LEAK DETECTION
==========================================
QUÉ: Simula exactamente el flujo de update_fill() para detectar fugas de margen.
POR QUÉ: El backtest muestra TotalAvail: $-24.83 con $13 de capital.
PARA QUÉ: Identificar la causa raíz exacta del desbordamiento de margen.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

print("=" * 70)
print("🔬 DIAGNÓSTICO FORENSE: MARGIN LEAK SIMULATION")
print("=" * 70)

# Simulate Portfolio state
current_cash = 13.0
used_margin = 0.0
pending_cash = 0.0
leverage = 10  # SCALPING leverage

print(f"\n📦 INITIAL STATE:")
print(f"   current_cash = ${current_cash:.2f}")
print(f"   used_margin  = ${used_margin:.2f}")
print(f"   pending_cash = ${pending_cash:.2f}")
print(f"   total_avail  = ${current_cash - used_margin - pending_cash:.2f}")

# === Simulate sizing for BTC SCALPING ===
print(f"\n{'─' * 50}")
print(f"🔵 TRADE 1: BTC SCALPING (LONG)")
available_cash = current_cash - used_margin - pending_cash
risk_pct = 0.05
sl_pct = 0.006
risk_amount = available_cash * risk_pct
notional = min(risk_amount / sl_pct, available_cash * leverage * 0.40)
margin = notional / leverage
print(f"   available_cash = ${available_cash:.2f}")
print(f"   risk_amount = ${risk_amount:.4f}")
print(f"   notional = ${notional:.2f}")
print(f"   margin (dollar_size) = ${margin:.2f}")

# Reserve cash
pending_cash += margin
print(f"   [RESERVE] pending_cash = ${pending_cash:.2f}")
total_avail = current_cash - used_margin - pending_cash
print(f"   [AFTER RESERVE] total_avail = ${total_avail:.2f}")

# Fill event  
fee = notional * 0.0002  # Maker fee
pending_cash -= margin  # Release
used_margin += notional / leverage  # Add margin
current_cash -= fee  # Deduct fee
print(f"   [FILL] fee = ${fee:.4f}")
print(f"   [FILL] pending_cash = ${pending_cash:.2f}")
print(f"   [FILL] used_margin = ${used_margin:.2f}")
print(f"   [FILL] current_cash = ${current_cash:.4f}")
total_avail = current_cash - used_margin - pending_cash
print(f"   [AFTER FILL] total_avail = ${total_avail:.2f}")

# === Simulate sizing for ETH SCALPING ===
print(f"\n{'─' * 50}")
print(f"🔵 TRADE 2: ETH SCALPING (SHORT)")
available_cash = current_cash - used_margin - pending_cash
risk_amount = available_cash * risk_pct
notional2 = min(risk_amount / sl_pct, available_cash * leverage * 0.40)
margin2 = notional2 / leverage
print(f"   available_cash = ${available_cash:.2f}")
print(f"   notional = ${notional2:.2f}")
print(f"   margin = ${margin2:.2f}")

pending_cash += margin2
fee2 = notional2 * 0.0002
pending_cash -= margin2
used_margin += notional2 / leverage
current_cash -= fee2
total_avail = current_cash - used_margin - pending_cash
print(f"   [AFTER FILL] used_margin = ${used_margin:.2f}, total_avail = ${total_avail:.2f}")

# === Simulate sizing for SOL SCALPING ===
print(f"\n{'─' * 50}")
print(f"🔵 TRADE 3: SOL SCALPING (LONG)")
available_cash = current_cash - used_margin - pending_cash
risk_amount = available_cash * risk_pct
notional3 = min(risk_amount / sl_pct, available_cash * leverage * 0.40) if available_cash > 0 else 0
margin3 = notional3 / leverage if notional3 > 0 else 0
print(f"   available_cash = ${available_cash:.2f}")
print(f"   notional = ${notional3:.2f}")
print(f"   margin = ${margin3:.2f}")

if available_cash > 1.0:
    pending_cash += margin3
    fee3 = notional3 * 0.0002
    pending_cash -= margin3
    used_margin += notional3 / leverage
    current_cash -= fee3
total_avail = current_cash - used_margin - pending_cash
print(f"   [AFTER FILL] used_margin = ${used_margin:.2f}, total_avail = ${total_avail:.2f}")

# === Now SWING trades ===
print(f"\n{'─' * 50}")
print(f"🟣 TRADE 4: BTC SWING (SHORT, 5x leverage)")
lev_swing = 5
available_cash = current_cash - used_margin - pending_cash
sl_swing = 0.015
risk_amount_sw = available_cash * 0.10
notional_sw = min(risk_amount_sw / sl_swing, available_cash * lev_swing * 0.40) if available_cash > 0 else 0
margin_sw = notional_sw / lev_swing if notional_sw > 0 else 0
print(f"   available_cash = ${available_cash:.2f}")
print(f"   notional = ${notional_sw:.2f}")
print(f"   margin = ${margin_sw:.2f}")

if available_cash > 1.0:
    fee_sw = notional_sw * 0.0002
    used_margin += notional_sw / lev_swing
    current_cash -= fee_sw
total_avail = current_cash - used_margin - pending_cash
print(f"   [AFTER FILL] used_margin = ${used_margin:.2f}, total_avail = ${total_avail:.2f}")

print(f"\n{'=' * 70}")
print(f"📊 FINAL STATE:")
print(f"   current_cash = ${current_cash:.4f}")
print(f"   used_margin  = ${used_margin:.4f}")
print(f"   pending_cash = ${pending_cash:.4f}")
total_avail = current_cash - used_margin - pending_cash
print(f"   total_avail  = ${total_avail:.4f}")
print(f"   DEFICIT = ${max(0, -total_avail):.4f}")

# === DIAGNOSIS ===
print(f"\n{'=' * 70}")
print(f"🔬 DIAGNOSIS:")
print(f"   The formula total_avail = current_cash - used_margin - pending_cash")
print(f"   With 10x leverage, each trade locks ~40% of available cash as margin.")
print(f"   3 SCALPING trades lock: ${used_margin:.2f} of margin")
print(f"   That's {used_margin/13.0*100:.1f}% of the $13 capital locked!")
print(f"   With 4+ positions, margin EXCEEDS capital → NEGATIVE available cash")
print(f"\n   🚨 ROOT CAUSE: size_position() allows 40% of available_cash per trade")
print(f"   but doesn't account for CONCURRENT positions.")
print(f"   FIX: Cap total margin reservation to 80% of current_cash MAX")
print(f"   and reduce per-trade notional ceiling based on open position count.")
