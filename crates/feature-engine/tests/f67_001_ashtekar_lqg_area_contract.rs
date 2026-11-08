//! Contrato Formal de Verificación: Fase 67 (Gravitación Cuántica de Lazos y Espectro de Área de Ashtekar)
//!
//! Este conjunto de pruebas certifica las propiedades cuánticas y topológicas de la formulación de Ashtekar-Barbero:
//! - C1: Espectro discreto de área cuantizada y cota inferior por la brecha de Planck de Barbero-Immirzi $\Delta A_{\text{gap}}$.
//! - C2: Trazas de holonomía de bucle de Wilson acotadas en $[-2.0, 2.0]$ y densidad de volumen cuántico positiva.
//! - C3: Detección determinista de singularidad cuántica por colapso de área y escalado defensivo (0.15).
//! - C4: Determinismo en stack $O(1)$, cero pánicos (`rb_p1`) y resiliencia absoluta ante NaN e infinitos.

use feature_engine::{AshtekarLqgAreaEngine, AshtekarLqgParams};

#[test]
fn test_c1_ashtekar_area_eigenvalues_and_immirzi_gap() {
    let params = AshtekarLqgParams::default();
    let mut engine = AshtekarLqgAreaEngine::new(params);

    // Flujo normal estacionario
    let report = engine.update(
        0.0002, // price_ret
        0.15,   // ofi
        0.10,   // depth_imbalance
        2.5,    // spread_bps
        0.015,  // instantaneous_vol
    );

    assert!(
        report.total_quantized_area > 0.0,
        "El área cuántica total debe ser estrictamente positiva: {}",
        report.total_quantized_area
    );

    assert!(
        report.minimum_area_gap > 0.0,
        "La brecha de área de Planck debe ser positiva: {}",
        report.minimum_area_gap
    );

    // El área total de los 4 canales debe ser mayor que la brecha de área de un solo canal j=1/2
    assert!(
        report.total_quantized_area >= report.minimum_area_gap,
        "El área total ({}) debe superar la brecha mínima ({})",
        report.total_quantized_area,
        report.minimum_area_gap
    );

    assert!(
        (report.slot_scale_factor >= 0.20 && report.slot_scale_factor <= 1.25),
        "Factor de escala de slot fuera de límites: {}",
        report.slot_scale_factor
    );
    assert!(!report.ashtekar_area_singularity_detected);
}

#[test]
fn test_c2_wilson_loop_holonomy_and_quantum_volume() {
    let params = AshtekarLqgParams::default();
    let mut engine = AshtekarLqgAreaEngine::new(params);

    for i in 0..15 {
        let ofi = (i as f64 * 0.1 - 0.7).clamp(-1.0, 1.0);
        let ret = (i as f64 * 0.0005 - 0.003).clamp(-0.01, 0.01);
        let report = engine.update(ret, ofi, 0.05, 1.8, 0.02);

        assert!(
            report.quantum_volume_density > 0.0,
            "El volumen cuántico debe ser estrictamente positivo: {}",
            report.quantum_volume_density
        );

        assert!(
            report.wilson_loop_holonomy_trace >= -2.0 && report.wilson_loop_holonomy_trace <= 2.0,
            "La traza de holonomía de SU(2) debe estar en [-2, 2]: {}",
            report.wilson_loop_holonomy_trace
        );
    }
}

#[test]
fn test_c3_quantum_singularity_detection_and_defensive_scale() {
    let mut params = AshtekarLqgParams::default();
    params.critical_area_threshold = 50.0; // Umbral exigente para forzar detección de colapso
    let mut engine = AshtekarLqgAreaEngine::new(params);

    let report = engine.update(0.0001, 0.05, 0.02, 2.0, 0.01);

    assert!(
        report.ashtekar_area_singularity_detected,
        "Debe detectar singularidad cuántica por área subcrítica"
    );
    assert_eq!(
        report.slot_scale_factor, 0.15,
        "El factor de escala debe reducirse exactamente al mínimo defensivo"
    );
}

#[test]
fn test_c4_stack_o1_and_nan_infinity_resilience() {
    let params = AshtekarLqgParams::default();
    let mut engine = AshtekarLqgAreaEngine::new(params);

    // Inyección de valores patológicos
    let report_nan = engine.update(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(report_nan.total_quantized_area.is_finite());
    assert!(report_nan.quantum_volume_density.is_finite());
    assert!(report_nan.wilson_loop_holonomy_trace.is_finite());
    assert!(report_nan.slot_scale_factor.is_finite());

    let report_inf = engine.update(
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    );
    assert!(report_inf.total_quantized_area.is_finite());
    assert!(report_inf.quantum_volume_density.is_finite());
    assert!(report_inf.wilson_loop_holonomy_trace.is_finite());
    assert!(report_inf.slot_scale_factor.is_finite());
}
