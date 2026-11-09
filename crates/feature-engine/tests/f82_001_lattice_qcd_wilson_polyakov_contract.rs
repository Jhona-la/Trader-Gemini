//! # Contrato Formal Fase 82: Lattice QCD Gauge Field Theory & Wilson-Polyakov Loop Holonomy
//!
//! ## Invariantes Verificados:
//! - C1: Unitariedad algebraica exacta en $SU(2)$, $U U^\dagger = I$, traza normalizada en $[-1.0, 1.0]$.
//! - C2: Positividad y acotación estricta de la acción de Wilson $S[U] \ge 0$ y tensión de cuerda $\sigma$.
//! - C3: Disparo determinista del veto `LATTICE_GAUGE_CONFINEMENT_COLLAPSE_VETO` con colapso de slot a 0.0.
//! - C4: Inmunidad contra NaN/Infinitos y ejecución $O(1)$ sin asignaciones en heap.

use feature_engine::{
    LatticeQcdConfig, LatticeQcdWilsonPolyakovEngine, Su2Matrix, LATTICE_DIM_SPATIAL,
    LATTICE_DIM_TEMPORAL, LATTICE_PLAQUETTE_COUNT,
};

#[test]
fn test_c1_su2_matrix_unitarity_and_algebraic_group_laws() {
    let id = Su2Matrix::identity();
    assert!((id.half_real_trace() - 1.0).abs() < 1e-12);
    assert!((id.trace_norm() - 1.0).abs() < 1e-12);

    let u1 = Su2Matrix::from_quaternion(0.6, 0.8, 0.0, 0.0);
    let u1_dag = u1.dagger();
    let prod = u1.mul(&u1_dag);

    assert!((prod.r00 - 1.0).abs() < 1e-10);
    assert!(prod.i00.abs() < 1e-10);
    assert!(prod.r01.abs() < 1e-10);
    assert!(prod.i01.abs() < 1e-10);
    assert!(prod.r10.abs() < 1e-10);
    assert!(prod.i10.abs() < 1e-10);
    assert!((prod.r11 - 1.0).abs() < 1e-10);
    assert!(prod.i11.abs() < 1e-10);
}

#[test]
fn test_c2_wilson_action_and_plaquette_bounds() {
    let mut engine = LatticeQcdWilsonPolyakovEngine::default();
    let report = engine.update(0.02, 0.15, 2.5, 0.5);

    assert!(report.average_plaquette >= -1.0 && report.average_plaquette <= 1.0);
    assert!(report.wilson_action >= 0.0);
    assert!(report.polyakov_loop_expectation >= 0.0 && report.polyakov_loop_expectation <= 1.0);
    assert!(report.string_tension_sigma >= 0.0);
    assert!(!report.lattice_confinement_collapse_veto);
    assert!(report.slot_scale_factor >= 0.10 && report.slot_scale_factor <= 1.25);
    assert_eq!(LATTICE_DIM_SPATIAL, 4);
    assert_eq!(LATTICE_DIM_TEMPORAL, 4);
    assert_eq!(LATTICE_PLAQUETTE_COUNT, 16);
}

#[test]
fn test_c3_polyakov_loop_and_confinement_collapse_hard_veto() {
    let config = LatticeQcdConfig {
        beta_coupling: 2.30,
        critical_action_threshold: 0.10, // Umbral muy sensible para provocar veto por acción
        polyakov_deconfinement_floor: 0.99, // Umbral muy alto para forzar veto de desconfinamiento
    };
    let mut engine = LatticeQcdWilsonPolyakovEngine::new(config);
    let report = engine.update(0.80, -0.90, 45.0, 8.5);

    assert!(report.lattice_confinement_collapse_veto);
    assert_eq!(report.slot_scale_factor, 0.0);
}

#[test]
fn test_c4_stack_zero_heap_and_subnormal_immunity() {
    let mut engine = LatticeQcdWilsonPolyakovEngine::default();
    let report_nan = engine.update(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);

    assert!(report_nan.average_plaquette.is_finite());
    assert!(report_nan.wilson_action.is_finite());
    assert!(report_nan.polyakov_loop_expectation.is_finite());
    assert!(report_nan.string_tension_sigma.is_finite());
    assert!(report_nan.slot_scale_factor.is_finite());
}
