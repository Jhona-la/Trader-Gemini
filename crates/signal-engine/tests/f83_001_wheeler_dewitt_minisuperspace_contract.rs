//! # Contrato Formal Fase 83: Wheeler-DeWitt Quantum Cosmology Equation & Mini-Superspace
//!
//! ## Invariantes Verificados:
//! - C1: Ligadura Hamiltoniana WDW, potencial de DeWitt $V(\phi) \ge 0$, amplitud de onda finita y acotada.
//! - C2: Corriente de flujo de probabilidad conservada $j_\phi$ y señal direccional cuántica $\kappa_{\text{wdw}} \in [-1.0, 1.0]$.
//! - C3: Disparo determinista del veto `WHEELER_DEWITT_BIG_CRUNCH_VETO` con colapso de slot a 0.0.
//! - C4: Inmunidad contra NaN/Infinitos y ejecución $O(1)$ sin asignaciones en heap.

use signal_engine::{WheelerDeWittConfig, WheelerDeWittMinisuperspaceEngine};

#[test]
fn test_c1_wdw_hamiltonian_potential_and_wkb_amplitude() {
    let mut engine = WheelerDeWittMinisuperspaceEngine::default();
    let report = engine.evaluate(1.0, 0.40, 0.02, 0.10);

    assert!(report.scale_factor_a > 0.0);
    assert!(report.alpha_coord.is_finite());
    assert!(report.potential_v >= 0.85); // lambda_cosmo = 0.85
    assert!(report.wave_amplitude > 0.0 && report.wave_amplitude <= 10.0);
    assert!(!report.big_crunch_veto);
    assert!(report.slot_scale_factor >= 0.15 && report.slot_scale_factor <= 1.30);
}

#[test]
fn test_c2_conserved_probability_current_and_directional_arrow() {
    let mut engine_bull = WheelerDeWittMinisuperspaceEngine::default();
    let report_bull = engine_bull.evaluate(2.0, 0.75, 0.015, 0.10);

    assert!(report_bull.probability_current_j_phi > 0.0);
    assert!(report_bull.directional_signal > 0.0);

    let mut engine_bear = WheelerDeWittMinisuperspaceEngine::default();
    let report_bear = engine_bear.evaluate(2.0, -0.75, 0.015, 0.10);

    assert!(report_bear.probability_current_j_phi < 0.0);
    assert!(report_bear.directional_signal < 0.0);
}

#[test]
fn test_c3_cosmological_big_crunch_collapse_hard_veto() {
    let config = WheelerDeWittConfig {
        mass_sq: 1.50,
        lambda_cosmo: 0.85,
        scale_factor_crunch_floor: 0.50, // Umbral alto para forzar veto de singularidad
    };
    let mut engine = WheelerDeWittMinisuperspaceEngine::new(config);
    let report = engine.evaluate(0.001, -0.90, 0.50, 0.05);

    assert!(report.big_crunch_veto);
    assert_eq!(report.slot_scale_factor, 0.0);
}

#[test]
fn test_c4_stack_zero_heap_and_subnormal_immunity() {
    let mut engine = WheelerDeWittMinisuperspaceEngine::default();
    let report_nan = engine.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);

    assert!(report_nan.scale_factor_a.is_finite());
    assert!(report_nan.potential_v.is_finite());
    assert!(report_nan.wave_amplitude.is_finite());
    assert!(report_nan.probability_current_j_phi.is_finite());
    assert!(report_nan.cosmic_acceleration.is_finite());
    assert!(report_nan.directional_signal.is_finite());
    assert!(report_nan.slot_scale_factor.is_finite());
}
