#!/usr/bin/env python3
"""
🔬 FORENSIC ZOMBIE DIAGNOSTIC — ROOT CAUSE ANALYSIS
═══════════════════════════════════════════════════════
QUÉ: Script forense para determinar POR QUÉ TIME_STOP_ZOMBIE se activa.
POR QUÉ: Necesitamos saber quién abre el trade, quién debería cerrarlo,
         y por qué ningún otro exit (TURBO_BE, TRAILING, HARD_SL) lo cierra.
PARA QUÉ: Encontrar la causa raíz estructural del ZOMBIE.
CÓMO: Traza el lifecycle completo de un trade zombie:
      1. Entry signal → qué estrategia lo abrió
      2. TP/SL asignados → son alcanzables?
      3. Peak PnL alcanzado → qué % del TP llegó?
      4. Por qué TURBO_BE no se activó → threshold vs peak
      5. Por qué TRAILING no se activó → progress vs stages
      6. Por qué HARD_SL no se activó → price vs SL level
      7. Conclusión: cuál es la brecha
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRADER_GEMINI_BACKTEST'] = 'true'

from config import Config
Config.Observability.EMAIL_ENABLED = False
Config.Observability.TELEGRAM_ENABLED = False

from core.events import MarketEvent, SignalEvent
from core.enums import EventType, SignalType
from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from queue import Queue

# ══════════════════════════════════════════════════════════════
# SCENARIO SIMULATOR: Simulate a trade lifecycle to track exits
# ══════════════════════════════════════════════════════════════

def simulate_zombie_scenario():
    """
    Simulates what happens to a trade in different market conditions
    to understand WHY zombies occur.
    """
    print("=" * 70)
    print("🔬 ZOMBIE ROOT CAUSE DIAGNOSTIC")
    print("=" * 70)
    
    # ── Setup minimal environment ──
    events_queue = Queue()
    portfolio = Portfolio(
        initial_capital=13.0,
        csv_path="scripts/zombie_diag_trades.csv",
        status_path="scripts/zombie_diag_status.csv",
        auto_save=False,
    )
    
    risk_manager = RiskManager(
        max_concurrent_positions=Config.MAX_CONCURRENT_POSITIONS,
        portfolio=portfolio,
    )
    risk_manager.win_count = 0
    risk_manager.loss_count = 0
    risk_manager._trade_cache = []
    
    # ── Golden Baseline Parameters ──
    scalping_tp = Config.Horizons.Scalping.get('tp_pct', 0.006)
    scalping_sl = Config.Horizons.Scalping.get('sl_pct', 0.0075)
    
    print(f"\n📊 GOLDEN BASELINE:")
    print(f"   TP: {scalping_tp*100:.2f}% | SL: {scalping_sl*100:.2f}%")
    print(f"   Leverage: {Config.BINANCE_LEVERAGE}x")
    
    # ── Calculate thresholds ──
    _maker_fee = getattr(Config, "BINANCE_MAKER_FEE_BNB", 0.0002)
    _taker_fee = getattr(Config, "BINANCE_TAKER_FEE_BNB", 0.000375)
    fee_buffer = _maker_fee + _taker_fee
    tp_target_pct = scalping_tp * 100  # e.g., 0.60%
    
    turbo_threshold = max(0.30, tp_target_pct * 0.50)
    trail_stage1 = tp_target_pct * 0.25
    trail_stage2 = tp_target_pct * 0.50
    trail_stage3 = tp_target_pct * 0.75
    
    print(f"\n🎯 EXIT THRESHOLDS (SCALPING):")
    print(f"   TAKE PROFIT:     +{tp_target_pct:.2f}%")
    print(f"   TURBO-BE:        +{turbo_threshold:.2f}% peak PnL (then crash back)")
    print(f"   TRAIL Stage 1:   +{trail_stage1:.2f}% peak PnL (BE + fees)")
    print(f"   TRAIL Stage 2:   +{trail_stage2:.2f}% peak PnL (protect 70%)")
    print(f"   TRAIL Stage 3:   +{trail_stage3:.2f}% peak PnL (protect 85%)")
    print(f"   HARD SL:         -{scalping_sl*100:.2f}%")
    print(f"   ZOMBIE TTL:      2700s (45min) default | 300s (ZOMBIE regime)")
    print(f"   Fee buffer:      {fee_buffer*100:.3f}%")
    
    # ── Simulate 4 market scenarios ──
    entry_price = 94000.0
    scenarios = [
        {
            "name": "🟢 SCENARIO A: Strong Trend (TP Hit)",
            "price_path": [94000 + i * 6 for i in range(100)],  # +0.64% in 100 bars
            "expected_exit": "TAKE_PROFIT",
        },
        {
            "name": "🟡 SCENARIO B: Partial Move + Retrace (TURBO_BE territory)",
            "price_path": [94000 + max(0, 3 * i - 0.05 * i**2) for i in range(100)],
            "expected_exit": "TURBO_BE",
        },
        {
            "name": "🔴 SCENARIO C: Flat Market (ZOMBIE territory)",
            "price_path": [94000 + 20 * np.sin(i * 0.1) for i in range(3000)],
            "expected_exit": "TIME_STOP_ZOMBIE",
        },
        {
            "name": "⚫ SCENARIO D: Slow Bleed (SL territory)",
            "price_path": [94000 - i * 0.7 for i in range(1200)],
            "expected_exit": "HARD_SL",
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  {scenario['name']}")
        print(f"  Expected: {scenario['expected_exit']}")
        print(f"{'='*60}")
        
        prices = scenario["price_path"]
        entry = prices[0]
        hwm = entry
        lwm = entry
        
        peak_pnl = 0.0
        hit_turbo = False
        hit_trail = None
        hit_sl = False
        hit_tp = False
        zombie_time = None
        
        for i, price in enumerate(prices):
            hwm = max(hwm, price)
            lwm = min(lwm, price)
            
            pnl_pct = ((price - entry) / entry) * 100
            current_peak = ((hwm - entry) / entry) * 100
            price_range_pct = ((hwm - lwm) / entry) * 100
            seconds_held = i * 60  # 1 bar = 1 minute
            
            peak_pnl = max(peak_pnl, current_peak)
            progress = current_peak / tp_target_pct if tp_target_pct > 0 else 0
            
            # Check TP
            if pnl_pct >= tp_target_pct and not hit_tp:
                hit_tp = True
                print(f"   ✅ TAKE_PROFIT at bar {i} ({seconds_held}s): +{pnl_pct:.3f}%")
                break
            
            # Check TURBO_BE
            if peak_pnl >= turbo_threshold and not hit_turbo:
                turbo_be_price = entry * (1 + fee_buffer + 0.0008)
                if price < turbo_be_price:
                    hit_turbo = True
                    print(f"   ⚡ TURBO_BE at bar {i} ({seconds_held}s): peak was +{peak_pnl:.3f}%, now at {pnl_pct:.3f}%")
                    break
                elif i > 0 and not hit_turbo:
                    # It COULD fire but price hasn't crashed back yet
                    pass
            
            # Check trailing
            if progress >= 0.25 and hit_trail is None:
                trail_be = entry * (1 + fee_buffer + 0.0005)
                if price < trail_be:
                    hit_trail = "TRAIL_STAGE_1_BE"
                    print(f"   🛡️ {hit_trail} at bar {i} ({seconds_held}s): progress {progress:.1%}")
                    break
            if progress >= 0.50 and hit_trail is None:
                trail_price = hwm - ((hwm - entry) * 0.30)
                if price < trail_price:
                    hit_trail = "TRAIL_STAGE_2_STD"
                    print(f"   🛡️ {hit_trail} at bar {i} ({seconds_held}s): progress {progress:.1%}")
                    break
            if progress >= 0.75 and hit_trail is None:
                trail_price = hwm - ((hwm - entry) * 0.15)
                if price < trail_price:
                    hit_trail = "TRAIL_STAGE_3_TIGHT"
                    print(f"   🛡️ {hit_trail} at bar {i} ({seconds_held}s): progress {progress:.1%}")
                    break
            
            # Check HARD_SL
            if price < entry * (1 - scalping_sl):
                hit_sl = True
                print(f"   🛑 HARD_SL at bar {i} ({seconds_held}s): {pnl_pct:.3f}%")
                break
            
            # Check ZOMBIE
            is_zombie = (seconds_held > 2700) and (price_range_pct < 0.15) and (pnl_pct < 0.02)
            is_bleed = (seconds_held > 3600) and (pnl_pct < -0.05)
            if (is_zombie or is_bleed) and zombie_time is None:
                zombie_time = seconds_held
                zombie_reason = "FLAT" if is_zombie else "SLOW_BLEED"
                print(f"   🧟 TIME_STOP_ZOMBIE ({zombie_reason}) at bar {i} ({seconds_held}s): pnl={pnl_pct:.3f}%, range={price_range_pct:.3f}%")
                break
        
        if not any([hit_tp, hit_turbo, hit_trail, hit_sl, zombie_time]):
            print(f"   ❓ NO EXIT TRIGGERED after {len(prices)} bars ({len(prices)}min)")
            print(f"      Final PnL: {((prices[-1]-entry)/entry)*100:.3f}%")
            print(f"      Peak PnL: +{peak_pnl:.3f}%")
            print(f"      Progress: {peak_pnl/tp_target_pct:.1%} of TP")
    
    # ══════════════════════════════════════════════════════════════
    # CRITICAL ANALYSIS: WHY DOES ZOMBIE WIN?
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"🔬 ROOT CAUSE ANALYSIS: WHY ZOMBIE WINS")
    print(f"{'='*70}")
    
    print(f"""
📋 ZOMBIE SE ACTIVA CUANDO SE CUMPLEN TODAS ESTAS CONDICIONES:
   1. seconds_held > 2700 (45 min sin TP)
   2. price_range < 0.15% (mercado plano)
   3. unrealized_pnl < 0.02% (sin ganancia real)

🔍 PARA QUE ZOMBIE NO SE ACTIVE, UNO DE ESTOS DEBE OCURRIR ANTES:
   A. TAKE_PROFIT: precio sube +{scalping_tp*100:.2f}% → necesita ${entry_price*scalping_tp:.2f} de movimiento
   B. TURBO_BE: peak sube +{turbo_threshold:.2f}% Y LUEGO cae → necesita subida + retroceso
   C. TRAILING: peak sube al menos +{trail_stage1:.2f}% y luego cae → similar a TURBO_BE
   D. HARD_SL: precio cae -{scalping_sl*100:.2f}% → necesita ${entry_price*scalping_sl:.2f} de caída
   E. MOMENTUM_EXIT: momentum se invierte → depende de _check_momentum_exit()
   F. FLIP_EXIT: nueva señal opuesta → depende de que una estrategia emita señal contraria

🧟 EL ZOMBIE GANA CUANDO:
   - El mercado no se mueve suficiente para TP (+{scalping_tp*100:.2f}%)
   - El mercado no cae suficiente para SL (-{scalping_sl*100:.2f}%)
   - No hay señal opuesta (FLIP_EXIT) porque las estrategias ML/Sniper 
     NO emiten EXIT signals independientes
   - El trade queda atrapado en un rango estrecho (< 0.15%) por > 45 min

💡 CAUSA RAÍZ ESTRUCTURAL:
   Las estrategias (ML, Sniper) solo emiten ENTRY signals.
   Una vez abierto el trade, NO hay mecanismo de "invalidación de señal".
   Si la señal de entrada se invalida (RSI vuelve a neutral, ML confidence baja),
   NADIE emite un EXIT. El trade solo puede cerrarse por:
   - Precio llegue a TP/SL (movimiento del mercado)
   - FLIP_EXIT (señal opuesta - raro en mercados planos)
   - TIME_STOP_ZOMBIE (timeout de último recurso)

🎯 SOLUCIONES RECOMENDADAS:
   1. SIGNAL_INVALIDATION_EXIT: Si la confianza ML cae por debajo de un
      umbral después de abrir, emitir EXIT proactivo (no esperar timeout)
   2. REDUCE ZOMBIE TTL en régimen detectado plano: 2700s → 900s (15 min)
   3. Implementar EXIT activo en ML Strategy cuando conditions se invalidan
   4. MFE-DECAY EXIT: Si MFE pico fue < 25% del TP después de X minutos,
      cerrar proactivamente (el trade no tiene momentum)
""")
    
    # Cleanup temp files
    for f in ["scripts/zombie_diag_trades.csv", "scripts/zombie_diag_status.csv"]:
        try: os.remove(f)
        except: pass

if __name__ == "__main__":
    simulate_zombie_scenario()
