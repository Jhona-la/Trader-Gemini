//! # Contrato Formal F73-001: Teoría Conforme de Campos (CFT) de Liouville y Gravedad Cuántica 2D
//!
//! Valida la invariancia de escala conforme, carga central $c_L \ge 25$, dimensión de Hausdorff $d_H \ge 2$,
//! detección de singularidad de curvatura de Liouville y resiliencia `rb_p1` sin pánicos.

use feature_engine::liouville_quantum_gravity::{
    LiouvilleCft2dQuantumGravityEngine, LiouvilleCftParams,
};

#[test]
fn test_c1_liouville_central_charge_and_hausdorff_dimension() {
    let params = LiouvilleCftParams::default();
    let mut engine = LiouvilleCft2dQuantumGravityEngine::new(params);

    let rep = engine.update(0.001, 0.05, 0.015, 1.2);

    // Carga central de Liouville c_L = 1 + 6(b + 1/b)^2 >= 25 para b in (0, 1)
    assert!(
        rep.central_charge_cl >= 25.0,
        "La carga central de Liouville debe ser estrictamente >= 25.0 (obtenido: {})",
        rep.central_charge_cl
    );

    // Dimensión fractal de Hausdorff d_H >= 2.0
    assert!(
        rep.hausdorff_dimension >= 2.0,
        "La dimensión fractal de Hausdorff debe ser >= 2.0 (obtenido: {})",
        rep.hausdorff_dimension
    );

    // Factor conforme e^{2b\phi} >= 1.0
    assert!(
        rep.conformal_factor >= 1.0,
        "El factor conforme métrico de Liouville debe ser >= 1.0"
    );
}

#[test]
fn test_c2_calm_smooth_metric_and_constructive_slot_scaling() {
    let params = LiouvilleCftParams::default();
    let mut engine = LiouvilleCft2dQuantumGravityEngine::new(params);

    // Microestructura en calma: spread muy ajustado (0.5 bps), baja volatilidad (0.005)
    let rep = engine.update(0.0001, 0.01, 0.005, 0.5);

    assert!(
        !rep.curvature_singularity_detected,
        "No debe haber singularidad de curvatura en calma"
    );
    assert!(
        rep.slot_scale_factor >= 0.95,
        "El factor de escala de slot debe ser constructivo en métrica suave"
    );
    assert!(
        rep.liouville_scalar_curvature < params.singularity_curvature_threshold,
        "La curvatura escalar debe mantenerse por debajo del umbral crítico"
    );
}

#[test]
fn test_c3_violent_curvature_singularity_and_defensive_damping() {
    let params = LiouvilleCftParams::default();
    let mut engine = LiouvilleCft2dQuantumGravityEngine::new(params);

    // Choque extremo: spread disparado a 30.0 bps y volatilidad térmica extrema (0.40)
    let rep = engine.update(0.08, -0.90, 0.40, 30.0);

    assert!(
        rep.curvature_singularity_detected,
        "Debe detectarse singularidad métrica de Liouville ante shock violento"
    );
    assert_eq!(
        rep.slot_scale_factor, params.min_slot_scale,
        "El modulador de slot debe colapsar rígidamente al mínimo defensivo (0.15)"
    );
}

#[test]
fn test_c4_zero_panic_rb_p1_resilience() {
    let params = LiouvilleCftParams::default();
    let mut engine = LiouvilleCft2dQuantumGravityEngine::new(params);

    let pathological_inputs = [
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::INFINITY),
        (1e-308, -1e-308, 0.0, -100.0),
        (-0.9999, 1.0, 500.0, 1000.0),
    ];

    for (ret, flow, vol, spread) in pathological_inputs {
        let rep = engine.update(ret, flow, vol, spread);
        assert!(rep.central_charge_cl.is_finite());
        assert!(rep.hausdorff_dimension.is_finite());
        assert!(rep.conformal_factor.is_finite());
        assert!(rep.slot_scale_factor.is_finite());
        assert!(rep.liouville_scalar_curvature.is_finite());
    }
}
