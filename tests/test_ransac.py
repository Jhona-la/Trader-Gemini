"""
🧪 Test Suite: RANSAC Volatility (Fase 10 - Quantitative Mastery)
Valida el cálculo de Desviación Estándar Robusta, ignorando outliers extremos (Spikes).
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.math_kernel import calculate_ransac_volatility

def run_tests():
    print("============================================================")
    print("  🧪 INICIANDO TESTS: RANSAC VOLATILITY (MATH KERNEL)")
    print("============================================================\n")

    # Generar serie de precio base "limpia" (Mean = 100, Std = 1.0)
    np.random.seed(42)
    clean_prices = np.random.normal(100.0, 1.0, 50)
    
    # ── 1. TEST: Dataset Limpio ──
    # RANSAC debería devolver valores muy similares a np.std y np.mean
    print("Test 1: Clean Dataset")
    raw_std = np.std(clean_prices)
    raw_mean = np.mean(clean_prices)
    
    ransac_std, ransac_mean = calculate_ransac_volatility(clean_prices)
    
    print(f"  Raw Std:    {raw_std:.4f} | RANSAC Std:    {ransac_std:.4f}")
    print(f"  Raw Mean:   {raw_mean:.4f} | RANSAC Mean:   {ransac_mean:.4f}")
    
    # Tolerancia pequeña entre raw y ransac para series limpias
    assert abs(raw_std - ransac_std) < 0.2, "RANSAC std deviates too much on clean data"
    assert abs(raw_mean - ransac_mean) < 0.2, "RANSAC mean deviates too much on clean data"
    print("  ✅ RANSAC Volatilidad converge correctamente en datos simétricos limpios")

    # ── 2. TEST: Dataset Corrupto (SPIKES) ──
    print("\nTest 2: Corrupted Dataset (Flash Crash Spikes)")
    corrupted_prices = clean_prices.copy()
    # Inyectar 3 flash crashes devastadores
    corrupted_prices[10] = 50.0
    corrupted_prices[20] = 150.0
    corrupted_prices[30] = 10.0
    
    raw_std_c = np.std(corrupted_prices)
    raw_mean_c = np.mean(corrupted_prices)
    
    ransac_std_c, ransac_mean_c = calculate_ransac_volatility(corrupted_prices)
    
    print(f"  Raw Std:    {raw_std_c:.4f} (Destruida) | RANSAC Std:    {ransac_std_c:.4f} (Robusta)")
    print(f"  Raw Mean:   {raw_mean_c:.4f} (Sesgada) | RANSAC Mean:   {ransac_mean_c:.4f} (Robusta)")
    
    # Raw std debería explotar (ej: >10)
    assert raw_std_c > 10.0, "Raw std didn't explode enough for this test"
    
    # RANSAC std debería mantenerse cercana a la original (~1.0)
    assert ransac_std_c < 2.0, f"RANSAC failed to reject spikes, got std {ransac_std_c}"
    
    print("  ✅ RANSAC bloqueó exitosamente los Spikes y aisló la verdadera volatilidad base")

    print("\n============================================================")
    print("🎉 ALL TESTS PASSED - RANSAC VOLATILITY KERNEL VERIFIED")
    print("============================================================")

if __name__ == "__main__":
    run_tests()
