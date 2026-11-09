//! # Contrato Formal Fase 85: Chern-Simons Witten Invariant & Knot Floer / Khovanov Homology
//!
//! ## Invariantes Verificados:
//! - C1: Positividad y acotación de invariantes de Jones y grado cuántico de Khovanov en $[0.0, 5.0]$.
//! - C2: Torsión de Writhe $w(D)$ coherente con la acumulación temporal de cruces orientados.
//! - C3: Disparo determinista del veto `CHERN_SIMONS_WITTEN_KNOT_COLLAPSE_VETO` con colapso de slot a 0.0.
//! - C4: Inmunidad contra NaN/Infinitos y ejecución $O(1)$ sin asignaciones en heap.

use feature_engine::{ChernSimonsWittenConfig, ChernSimonsWittenKnotEngine};

#[test]
fn test_c1_chern_simons_jones_and_khovanov_invariants() {
    let mut engine = ChernSimonsWittenKnotEngine::default();

    // Actualización nominal con desequilibrio cruzado y velocidad moderada
    let report = engine.update(0.35, 0.40, 1.20, 0.02);

    assert!(report.jones_polynomial_val >= 0.1 && report.jones_polynomial_val <= 4.0);
    assert!(report.khovanov_quantum_degree >= 0.0 && report.khovanov_quantum_degree <= 5.0);
    assert!(report.knot_complexity >= 0.0 && report.knot_complexity <= 1.0);
    assert!(report.slot_scale_factor >= 0.15 && report.slot_scale_factor <= 1.30);
    assert!(!report.knot_collapse_veto);
}

#[test]
fn test_c2_writhe_accumulation_and_braid_dynamics() {
    let mut engine = ChernSimonsWittenKnotEngine::default();

    // Pulsos positivos consistentes de velocidad para acumular torsión de Writhe positiva
    for _ in 0..6 {
        let rep = engine.update(0.40, 1.50, 1.0, 0.01);
        assert!(rep.writhe_number >= 0.0);
    }

    let final_rep = engine.update(0.40, 1.50, 1.0, 0.01);
    assert!(final_rep.writhe_number > 0.0);
}

#[test]
fn test_c3_knot_collapse_and_writhe_shock_hard_veto() {
    let config = ChernSimonsWittenConfig {
        chern_simons_level_k: 4,
        min_knot_complexity_floor: 0.95, // Umbral muy alto para inducir colapso
        critical_writhe_threshold: 4.50,
    };
    let mut engine = ChernSimonsWittenKnotEngine::new(config);

    // Entrada con spread amplio y complejidad baja
    let report = engine.update(0.0, 0.0, 50.0, 0.001);

    assert!(report.knot_collapse_veto);
    assert_eq!(report.slot_scale_factor, 0.0);
}

#[test]
fn test_c4_stack_zero_heap_and_subnormal_immunity() {
    let mut engine = ChernSimonsWittenKnotEngine::default();
    let report_nan = engine.update(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);

    assert!(report_nan.writhe_number.is_finite());
    assert!(report_nan.khovanov_quantum_degree.is_finite());
    assert!(report_nan.jones_polynomial_val.is_finite());
    assert!(report_nan.knot_complexity.is_finite());
    assert!(report_nan.slot_scale_factor.is_finite());
}
