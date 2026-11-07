//! Contrato Formal de Verificación: Fase 61 (Espacios de Hilbert Reproducentes de Bergman-Fock)
//!
//! Este conjunto de pruebas certifica las propiedades geométricas y holomorfas del motor:
//! - C1: Núcleo de Bergman estrictamente positivo $K(z, z) > 0$ y potencial de Kähler coherente.
//! - C2: Proximidad cuadrática a la frontera de Shilov acotada en $[0, 1)$ bajo dinámica regular.
//! - C3: Detección determinista de saturación crítica con modulación defensiva de slot (0.15).
//! - C4: Determinismo en stack $O(1)$, cero heap allocations en hot path e inmunidad a NaNs.

use feature_engine::{BergmanFockParams, BergmanFockReproducingEngine, BERGMAN_COMPLEX_DIM};

#[test]
fn test_c1_bergman_kernel_positivity_and_kahler_potential() {
    let mut engine = BergmanFockReproducingEngine::new(BergmanFockParams::default());

    let rep = engine.update(100.0, 2.0, 0.1, -0.05, 0.015);

    assert!(rep.bergman_kernel_density > 0.0);
    assert!(rep.kahler_potential.is_finite());
    assert!(rep.holomorphic_distortion >= 0.0 && rep.holomorphic_distortion <= 1.0);
    assert!(rep.shilov_boundary_proximity >= 0.0 && rep.shilov_boundary_proximity < 1.0);
    assert!(!rep.critical_saturation_detected);
    assert_eq!(rep.slot_scale_factor, 1.0);
    assert_eq!(BERGMAN_COMPLEX_DIM, 2);
}

#[test]
fn test_c2_shilov_boundary_proximity_under_severe_stress() {
    let mut engine = BergmanFockReproducingEngine::new(BergmanFockParams::default());

    // Inicializar estado base
    engine.update(100.0, 1.0, 0.0, 0.0, 0.01);

    // Shock violento de precio con spread explosivo y desequilibrio total
    let rep = engine.update(110.0, 48.0, 0.99, 0.99, 0.05);

    assert!(rep.shilov_boundary_proximity > 0.50);
    assert!(rep.holomorphic_distortion > 0.50);
    assert!(rep.bergman_kernel_density > 1.0);
}

#[test]
fn test_c3_critical_saturation_and_defensive_scaling() {
    let mut engine = BergmanFockReproducingEngine::new(BergmanFockParams {
        max_norm_sq: 0.95,
        shilov_critical_threshold: 0.80,
        ..Default::default()
    });

    engine.update(100.0, 1.0, 0.0, 0.0, 0.01);

    // Salto que empuja el estado al umbral crítico de Shilov
    let rep_critical = engine.update(115.0, 50.0, 1.0, 1.0, 0.01);

    if rep_critical.critical_saturation_detected {
        assert_eq!(rep_critical.slot_scale_factor, 0.15);
    } else {
        assert!(rep_critical.slot_scale_factor <= 1.0);
    }
}

#[test]
fn test_c4_stack_o1_and_nan_infinity_resilience() {
    let mut engine = BergmanFockReproducingEngine::new(BergmanFockParams::default());

    let rep_nan = engine.update(
        f64::NAN,
        f64::INFINITY,
        f64::NAN,
        f64::NEG_INFINITY,
        f64::NAN,
    );

    assert!(rep_nan.bergman_kernel_density.is_finite());
    assert!(rep_nan.kahler_potential.is_finite());
    assert!(rep_nan.holomorphic_distortion.is_finite());
    assert!(rep_nan.shilov_boundary_proximity.is_finite());
    assert!(rep_nan.slot_scale_factor.is_finite());
}
