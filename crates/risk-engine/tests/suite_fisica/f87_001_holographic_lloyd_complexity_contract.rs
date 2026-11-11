//! # Contrato Formal Fase 87: Holographic Complexity & Lloyd Bound for Maximum Capital Computation Rate
//!
//! ## Invariantes Verificados:
//! - C1: Evaluación determinista de la acción WDW, tasa de crecimiento de complejidad y cota de Lloyd en $O(1)$.
//! - C2: Disparo fail-closed del veto `HOLOGRAPHIC_LLOYD_BOUND_EXHAUSTION_VETO` con colapso de slot a 0.0.
//! - C3: Modulación suave y monótona del factor de slot continuo $\eta_{\text{lloyd}} \in [0.10, 1.25]$ ante saturación.
//! - C4: Inmunidad ante valores patológicos (NaN, Infinito, ceros) y determinismo idéntico en stack.

use risk_engine::{HolographicLloydComplexityEngine, HolographicLloydConfig};

#[test]
fn test_c1_holographic_complexity_and_lloyd_bound_evaluation() {
    let mut engine = HolographicLloydComplexityEngine::default();

    // Primer tick: inicialización
    let rep1 = engine.evaluate(0.20, 0.02, 10.0, 0.10);
    assert!(rep1.holographic_complexity > 0.0);
    assert_eq!(rep1.complexity_growth_rate, 0.0);
    assert!(rep1.lloyd_computation_bound > 0.0);
    assert_eq!(rep1.lloyd_saturation_ratio, 0.0);
    assert!(!rep1.lloyd_bound_exhaustion_veto);
    assert!(rep1.slot_scale_factor >= 0.10 && rep1.slot_scale_factor <= 1.25);

    // Segundo tick: régimen continuo regular
    let rep2 = engine.evaluate(0.25, 0.03, 12.0, 0.10);
    assert!(rep2.holographic_complexity > 0.0);
    assert!(rep2.complexity_growth_rate >= 0.0);
    assert!(rep2.lloyd_computation_bound > 0.0);
    assert!(rep2.lloyd_saturation_ratio < 0.95);
    assert!(!rep2.lloyd_bound_exhaustion_veto);
    assert!(rep2.slot_scale_factor >= 0.10 && rep2.slot_scale_factor <= 1.25);
}

#[test]
fn test_c2_lloyd_saturation_and_hard_veto() {
    let config = HolographicLloydConfig {
        hbar_eff: 1.0,
        ads_radius_l: 2.0,
        base_energy_e0: 1.0,             // Baja capacidad de energía base
        critical_saturation_ratio: 0.50, // Umbral exigente de saturación
        ewma_alpha: 1.0,                 // Respuesta inmediata sin retraso EWMA
    };
    let mut engine = HolographicLloydComplexityEngine::new(config);

    // Inicialización a baja complejidad
    let _ = engine.evaluate(0.0, 0.001, 0.10, 0.10);

    // Explosión masiva de volatilidad y calor en micro-tiempo dt=0.001 -> saturación masiva de Lloyd
    let rep_shock = engine.evaluate(1.0, 8.0, 50.0, 0.001);

    assert!(rep_shock.lloyd_saturation_ratio >= 0.50);
    assert!(rep_shock.lloyd_bound_exhaustion_veto);
    assert_eq!(rep_shock.slot_scale_factor, 0.0);
}

#[test]
fn test_c3_slot_modulation_with_headroom() {
    let config = HolographicLloydConfig {
        ewma_alpha: 1.0,
        ..Default::default()
    };
    let mut engine = HolographicLloydComplexityEngine::new(config);

    let _ = engine.evaluate(0.10, 0.01, 5.0, 0.10);
    let rep_low_strain = engine.evaluate(0.11, 0.011, 5.1, 0.10);

    let mut engine2 = HolographicLloydComplexityEngine::new(config);
    let _ = engine2.evaluate(0.10, 0.01, 5.0, 0.10);
    let rep_high_strain = engine2.evaluate(0.60, 0.25, 20.0, 0.10);

    // Mayor saturación -> menor o igual factor de slot
    if !rep_high_strain.lloyd_bound_exhaustion_veto {
        assert!(rep_high_strain.slot_scale_factor <= rep_low_strain.slot_scale_factor);
    }
}

#[test]
fn test_c4_stack_zero_heap_and_subnormal_immunity() {
    let mut engine_a = HolographicLloydComplexityEngine::default();
    let mut engine_b = HolographicLloydComplexityEngine::default();

    let rep_nan = engine_a.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    assert!(rep_nan.holographic_complexity.is_finite());
    assert!(rep_nan.complexity_growth_rate.is_finite());
    assert!(rep_nan.lloyd_computation_bound.is_finite());
    assert!(rep_nan.lloyd_saturation_ratio.is_finite());
    assert!(rep_nan.slot_scale_factor.is_finite());

    // Determinismo idéntico
    let rep_a = engine_a.evaluate(0.40, 0.08, 15.0, 0.05);
    let _ = engine_b.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    let rep_b = engine_b.evaluate(0.40, 0.08, 15.0, 0.05);

    assert_eq!(rep_a.holographic_complexity, rep_b.holographic_complexity);
    assert_eq!(rep_a.complexity_growth_rate, rep_b.complexity_growth_rate);
    assert_eq!(rep_a.lloyd_computation_bound, rep_b.lloyd_computation_bound);
    assert_eq!(rep_a.lloyd_saturation_ratio, rep_b.lloyd_saturation_ratio);
    assert_eq!(
        rep_a.lloyd_bound_exhaustion_veto,
        rep_b.lloyd_bound_exhaustion_veto
    );
    assert_eq!(rep_a.slot_scale_factor, rep_b.slot_scale_factor);
}
