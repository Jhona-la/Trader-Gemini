"""
🧪 Test Suite: PROTOCOLO CRITERIO-AXIOMA
Verifica la integridad matemática, la regresión contable, 
la clasificación de causas de error (Oráculo) y la penalización de genes.
"""

import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.axioma_math import PrecisionAuditor
from sophia.axioma import AxiomDiagnoser, FallaBase, AxiomDiagnosis
from core.portfolio import Portfolio
from core.events import FillEvent
from core.enums import OrderSide, OrderType
from sophia.nemesis import NemesisEngine
from decimal import Decimal
from datetime import datetime
import json

def run_tests():
    print("============================================================")
    print("  🧪 INICIANDO TESTS: PROTOCOLO CRITERIO-AXIOMA")
    print("============================================================\n")
    
    # ── 1. TEST: PrecisionAuditor (Calc-Checker) ──
    print("Test 1: PrecisionAuditor (Floating Point Interrogation)")
    
    # Caso 1: Exact calculation
    entry = 50000.0
    exit = 50500.0
    qty = 0.5
    engine_pnl = (exit - entry) * qty # 250.0
    assert PrecisionAuditor.verify_pnl(entry, exit, qty, engine_pnl) == True
    print("  ✅ verify_pnl: Valid exact float64 calculation")
    
    # Caso 2: Precision Loss (Simulation of truncating error)
    bad_engine_pnl = 250.0000002 # > 1e-7 strict epsilon
    assert PrecisionAuditor.verify_pnl(entry, exit, qty, bad_engine_pnl) == False
    print("  ✅ verify_pnl: Caught truncating precision loss > 1e-7")
    
    # Caso 3: Fraction Verification
    assert PrecisionAuditor.verify_fraction(1.0, 3.0, 1/3) == True
    assert PrecisionAuditor.verify_fraction(1.0, 3.0, 0.333333) == False # missing decimals
    print("  ✅ verify_fraction: Caught Kelly/Size truncation")

    # ── 2. TEST: Accounting Equation (Portfolio Conservation) ──
    print("\nTest 2: Accounting Conservation (Eq = Init + PnL - Fees)")
    
    p = Portfolio(initial_capital=10000.0, auto_save=False)
    p.realized_pnl = 500.0
    p.total_fees_paid = 10.0
    
    # If cash matches theoretical, it should not poison PnL
    p.current_cash = 10490.0 # 10000 + 500 - 10
    p.verify_accounting_equation()
    assert not math.isnan(p.realized_pnl)
    print("  ✅ verify_accounting: Accepted valid equation state")
    
    # Corrupting cash state by 1 cent
    p.current_cash = 10490.01
    p.verify_accounting_equation()
    assert math.isnan(p.realized_pnl)
    print("  ✅ verify_accounting: CAUGHT Accounting Corruption! Triggered Poison PnL")
    
    # ── 3. TEST: AxiomDiagnoser (Root Cause Engine) ──
    print("\nTest 3: AxiomDiagnoser (Oráculo y Forense)")
    
    # Caso 1: Premio
    diag_win = AxiomDiagnoser.diagnose(
        pnl=20.0, direction="LONG", trigger_price=10.0, fill_price=10.0, sophia_report={}, duration_mins=5.0
    )
    assert diag_win.tipo_falla == FallaBase.NO_FALLA
    assert "Premio" in diag_win.razon or "Winner" in diag_win.razon
    print("  ✅ AxiomDiagnoser: Identified WINNER (NO_FALLA)")
    
    # Caso 2: Error de Cálculo (Slippage Devastador)
    diag_calc = AxiomDiagnoser.diagnose(
        pnl=-50.0, direction="LONG", trigger_price=100.0, fill_price=105.0, sophia_report={}, duration_mins=5.0
    )
    # slippage = 5.0 / 100.0 = 5% (> 0.5% threshold)
    assert diag_calc.tipo_falla == FallaBase.CALCULO
    assert diag_calc.is_fatal == True
    print("  ✅ AxiomDiagnoser: Identified CALCULO (Fatal Slippage 5%)")
    
    # Caso 3: Error de Profundidad (Spike stop loss)
    diag_depth = AxiomDiagnoser.diagnose(
        pnl=-10.0, direction="SHORT", trigger_price=100.0, fill_price=100.1, sophia_report={}, duration_mins=0.5
    )
    assert diag_depth.tipo_falla == FallaBase.PROFUNDIDAD
    print("  ✅ AxiomDiagnoser: Identified PROFUNDIDAD (Sub-minute loss)")
    
    # Caso 4: Error de Tesis (Alpha decay normal)
    diag_thesis = AxiomDiagnoser.diagnose(
        pnl=-20.0, direction="LONG", trigger_price=100.0, fill_price=100.0, 
        sophia_report={'decision_entropy': 0.8}, duration_mins=15.0
    )
    assert diag_thesis.tipo_falla == FallaBase.TESIS_DECAY
    assert diag_thesis.residual_pct == -1.0
    print("  ✅ AxiomDiagnoser: Identified TESIS_DECAY (Duration > 1m, low slip)")

    # ── 4. TEST: GenePenalizer integration ──
    print("\nTest 4: GenePenalizer + Manifest Integration")
    engine = NemesisEngine()
    
    # Simulate a TESIS_DECAY autopsy (bad entropy, slow loss)
    report = engine.full_autopsy(
        trade_id="TEST_AXIOM",
        symbol="BTC/USDT",
        direction="LONG",
        predicted_prob=0.90,
        predicted_exit_mins=10.0,
        predicted_tp_mins=10.0,
        predicted_sl_mins=5.0,
        actual_pnl=-50.0,
        actual_duration_mins=30.0,
        brier_score=0.9, # Terrible prediction
        sophia_report={'decision_entropy': 0.6},
        top_features=[{'feature': 'RSI', 'contribution': 0.5}],
        trigger_price=50000.0,
        fill_price=50000.0,
        genotype_id="GENE_123",
        persist_manifest=False
    )
    
    assert report.falla_base == "TESIS_DECAY"
    # Given Brier = 0.9 and Factor = 1.0 (for Thesis Decay), Penalty should be 0.90
    assert report.gene_penalty == 0.90
    print(f"  ✅ NemesisEngine: Correct penalty assigned to TESIS_DECAY -> -{report.gene_penalty} pts")
    
    assert "Oráculo Axioma" in report.manifest
    assert "TESIS_DECAY" in report.to_log_line()
    print("  ✅ ManifestWriter: Wrote Axioma adjustment and log line.")
    
    print("\n============================================================")
    print("🎉 ALL 4 TESTS PASSED - INTEGRIDAD CRITERIO-AXIOMA VERIFICADA")
    print("============================================================")

if __name__ == "__main__":
    run_tests()
