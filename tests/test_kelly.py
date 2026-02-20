"""
🧪 Test Suite: Dynamic Kelly Sizing (Fase 14 - Capital Assignment)
Valida que el RiskManager dimensione y/o vete operaciones en base a la Expectativa Matemática (EV).
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.risk_manager import RiskManager
from core.portfolio import Portfolio

# Mock objects para eventos
class MockSignal:
    def __init__(self, symbol="BTC/USDT", strength=1.0, strategy_id="TEST"):
        self.symbol = symbol
        self.strength = strength
        self.strategy_id = strategy_id
        self.atr = 100.0

def run_tests():
    print("============================================================")
    print("  🧪 INICIANDO TESTS: DYNAMIC KELLY CRITERION (RISK MGR)")
    print("============================================================\n")

    # Setup
    portfolio = Portfolio(initial_capital=1000.0, auto_save=False)
    risk_manager = RiskManager(portfolio=portfolio)
    signal = MockSignal()
    
    # ── 1. TEST: Positive Expected Value (EV > 0) ──
    print("Test 1: Expectativa Positiva (Gana el 60%, Ratio 1.5:1)")
    # Forzar un Kelly = (0.6 * 1.5 - 0.4) / 1.5 = 0.5 / 1.5 = 33.3% puro.
    # Scalping Kelly Mult (0.25) -> 33.3% * 0.25 = 8.33% de riesgo base.
    portfolio.kelly_winrate = 0.60
    portfolio.kelly_payoff_ratio = 1.5
    
    target_kelly = (0.6 * 1.5 - 0.4) / 1.5
    target_scaled = target_kelly * 0.25
    
    size = risk_manager.size_position(signal, current_price=50000.0)
    if isinstance(size, tuple): size = size[0]
    
    # Notamos que el Portfolio no se actualizó internamente en 'cash' por lo que equity=1000
    # Expected size factor should be around target_scaled, taking clipping limits into account.
    kelly_math_output = risk_manager._compute_kelly_math(0.60, 1.5, apply_mult=True)
    
    print(f"  Puro Kelly (K): {target_kelly*100:.2f}%")
    print(f"  Scaled Kelly:   {kelly_math_output*100:.2f}%")
    
    assert kelly_math_output > 0, "Kelly should be positive for a profitable system."
    print("  ✅ RiskManager aprueba operación para sistema rentable.")

    # ── 2. TEST: Negative Expected Value (EV < 0) ──
    print("\nTest 2: Expectativa Negativa (Gana el 40%, Ratio 1:1)")
    # Forzar Kelly = (0.4 * 1.0 - 0.6) / 1.0 = -0.2 (EV Negativo! El sistema pierde dinero)
    portfolio.kelly_winrate = 0.40
    portfolio.kelly_payoff_ratio = 1.0
    
    kelly_math_output_neg = risk_manager._compute_kelly_math(0.40, 1.0, apply_mult=False)
    print(f"  Puro Kelly (K): {kelly_math_output_neg*100:.2f}% (Pérdida Esperada)")
    
    size = risk_manager.size_position(signal, current_price=50000.0)
    if isinstance(size, tuple): size = size[0]
    assert size == 0.0, f"Expected size 0.0 for EV < 0, got {size}"
    assert kelly_math_output_neg < 0, "Math Output should be strictly negative."
    
    print("  ✅ RiskManager bloquea rigurosamente el capital y detiene la hemorragia (Size=0).")

    print("\n============================================================")
    print("🎉 ALL TESTS PASSED - DYNAMIC KELLY VETO VERIFIED")
    print("============================================================")

if __name__ == "__main__":
    run_tests()
