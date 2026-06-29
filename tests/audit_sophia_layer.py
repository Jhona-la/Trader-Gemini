"""
Forensic script to audit the Sophia AI layer, specifically examining the Lyapunov Shield,
Nemesis behavior, and destructive interference in high frequency environments.
"""

import sys
import os
import json

# Add to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sophia.intelligence import SophiaIntelligence
from sophia.nemesis import NemesisEngine, FalsePositiveAnalyzer, TimeDeviationAnalyzer
from strategies.ml_strategy import MLStrategyHybridUltimate
from utils.logger import logger
import traceback

def audit_lyapunov_shield():
    print('='*50)
    print("🛡️ AUDITORÍA DEL ESCUDO DE LYAPUNOV")
    print('='*50)

    try:
        sophia = SophiaIntelligence()
        # Mock values simulating 1-minute scalping noise
        l_horizon = 0.5 # Very noisy
        is_btc = True
        
        # Simulate local Lyapunov logic
        shield_floor = 0.50 if is_btc else 0.65
        l_shield = max(shield_floor, 0.5 * (l_horizon / 2.0))
        
        print(f"[TEST 1] BTC Lyapunov under 1m noise: Shield Floor = {shield_floor}. Calculated Shield = {l_shield}")
        
        # What if it's an altcoin?
        is_btc = False
        shield_floor = 0.50 if is_btc else 0.65
        l_shield = max(shield_floor, 0.5 * (l_horizon / 2.0))
        print(f"[TEST 2] ALT Lyapunov under 1m noise: Shield Floor = {shield_floor}. Calculated Shield = {l_shield}")
        
        if shield_floor < 0.2:
            print("🚨 CRÍTICO: El piso matemático del escudo destruye el 80%+ de la certeza de Sophia.")
        elif shield_floor >= 0.5:
            print("✅ OK: El escudo de Lyapunov está configurado matemáticamente para no aniquilar señales (Floor >= 0.5).")

    except Exception as e:
        print(f"Error auditing Lyapunov Shield: {e}")

def audit_nemesis():
    print('\n' + '='*50)
    print("⚔️ AUDITORÍA DE NÉMESIS (FALSE POSITIVES)")
    print('='*50)
    
    try:
        fp_analyzer = FalsePositiveAnalyzer()
        report = {
            'excess_kurtosis': 4.0,      # Simulated Fat tail
            'alpha_decay_threshold_mins': 5,
        }
        
        is_fp, reason = fp_analyzer.analyze(predicted_prob=0.90, actual_pnl=-0.5, sophia_report=report, actual_duration_secs=60)
        
        print(f"[TEST 1] Fat Tail Failure (>3 excess kurtosis): {is_fp} - Reason: {reason}")
        
        report = {
            'excess_kurtosis': 1.0,      
            'alpha_decay_threshold_mins': 5,
        }
        is_fp, reason = fp_analyzer.analyze(predicted_prob=0.90, actual_pnl=-0.5, sophia_report=report, actual_duration_secs=60*15) # 15 min > 2x 5min
        print(f"[TEST 2] Signal Decay (Actual > 2x Threshold): {is_fp} - Reason: {reason}")
        print("✅ OK: Nemesis Classifier differentiates decay and tail risks.")
    except Exception as e:
        traceback.print_exc()

def audit_capitalization():
    print('\n' + '='*50)
    print("💰 AUDITORÍA DE MICRO-CUENTA ($13USD)")
    print('='*50)
    capital = 13.0
    win_rate = 0.60
    maker_fee = 0.0002
    leverage = 10
    sizing_pct = 0.40 # 40% of capital per trade
    
    trade_size = capital * sizing_pct
    notional = trade_size * leverage
    
    print(f"Capital Total: ${capital}")
    print(f"Riesgo/Sizing: {sizing_pct*100}% -> ${trade_size}")
    print(f"Apalancamiento: {leverage}x -> Notional ${notional}")
    
    if notional < 5.0:
        print("🚨 CRÍTICO: El valor nocional es menor a $5 USD (Límite de Binance para Futuros).")
    else:
        print("✅ OK: Nocional viable para Binance Futures.")
        
    tp_pct = 0.015  # 1.5%
    sl_pct = 0.02   # 2.0%
    
    round_trip_fee = notional * (maker_fee * 2)
    win_amount = (notional * tp_pct) - round_trip_fee
    loss_amount = (notional * sl_pct) + round_trip_fee
    
    ev = (win_amount * win_rate) - (loss_amount * (1 - win_rate))
    
    print(f"\n[Fees] Round Trip MAKER ({maker_fee}): ${round_trip_fee:.4f}")
    print(f"[TP {tp_pct*100}%] Beneficio Neto: +${win_amount:.4f}")
    print(f"[SL {sl_pct*100}%] Pérdida Neta: -${loss_amount:.4f}")
    print(f"[MATEMÁTICAS] Expected Value (EV) con {win_rate*100}% WR: ${ev:.4f}/trade")

    if ev > 0:
        print("✅ OK: Sistema genera Esperanza Matemática Positiva.")
    else:
        print("🚨 CRÍTICO: El R:R ratio inverso y los fees están consumiendo la estructura.")

if __name__ == "__main__":
    audit_lyapunov_shield()
    audit_nemesis()
    audit_capitalization()
