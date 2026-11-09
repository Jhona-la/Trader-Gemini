//! Contrato Formal de Verificación: Fase 77 (§77) — Fase Geométrica de Berry y Holonomía de Fubini-Study
//!
//! Verifica el cómputo de la distancia métrica cuántica de Fubini-Study, curvatura de Berry,
//! ángulo sólido subtendido y modulación de convicción direccional continua.

use signal_engine::berry_phase_holonomy::{BerryPhaseConfig, BerryPhaseEngine};

#[test]
fn test_c1_berry_phase_zero_holonomy_linear_flow() {
    let config = BerryPhaseConfig::default();
    let mut engine = BerryPhaseEngine::new(config);

    // Movimiento lineal colineal en el espacio de parámetros (sin encerrar área)
    let mut report = engine.evaluate(1.0, 1.0, 0.0);
    for i in 1..=10 {
        let scale = 1.0 + (i as f64) * 0.1;
        report = engine.evaluate(scale, scale, 0.0);
    }

    assert!(report.fubini_study_distance.is_finite());
    assert!(report.berry_phase.abs() < 1e-4);
    assert!(!report.adiabatic_breakdown);
    assert!((report.holographic_conviction_modulator - 1.0).abs() < 0.05);
}

#[test]
fn test_c2_berry_phase_non_trivial_cyclic_holonomy() {
    let config = BerryPhaseConfig {
        cycle_window_len: 16,
        adiabatic_breakdown_threshold: 0.95,
        conviction_sensitivity: 0.35,
    };
    let mut engine = BerryPhaseEngine::new(config);

    // Ciclo cerrado alrededor del origen: (1, 0, 0) -> (0, 1, 0) -> (0, 0, 1) -> (1, 0, 0)
    engine.evaluate(1.0, 0.01, 0.01);
    engine.evaluate(0.01, 1.0, 0.01);
    let report_loop = engine.evaluate(0.01, 0.01, 1.0);

    assert!(report_loop.berry_phase.abs() > 0.01);
    assert!(report_loop.berry_curvature_magnitude > 0.0);
    assert!(report_loop.holographic_conviction_modulator >= 0.45);
    assert!(report_loop.holographic_conviction_modulator <= 1.40);
}

#[test]
fn test_c3_berry_phase_adiabatic_breakdown_detection() {
    let config = BerryPhaseConfig {
        cycle_window_len: 16,
        adiabatic_breakdown_threshold: 0.80,
        conviction_sensitivity: 0.35,
    };
    let mut engine = BerryPhaseEngine::new(config);

    // Estado inicial ortogonal
    engine.evaluate(1.0, 0.0, 0.0);

    // Salto cuántico abrupto a estado ortogonal completo
    let report_jump = engine.evaluate(0.0, 10.0, 0.0);

    assert!(report_jump.fubini_study_distance >= 0.80);
    assert!(report_jump.adiabatic_breakdown);
    assert!(report_jump.holographic_conviction_modulator <= 0.70);
}

#[test]
fn test_c4_berry_phase_boundedness_and_stack_safety() {
    let config = BerryPhaseConfig::default();
    let mut engine = BerryPhaseEngine::new(config);

    // Entradas singulares / extremas
    let r1 = engine.evaluate(0.0, 0.0, 0.0);
    assert!(r1.fubini_study_distance.is_finite());
    assert!(r1.holographic_conviction_modulator >= 0.45 && r1.holographic_conviction_modulator <= 1.40);

    let r2 = engine.evaluate(1e12, -1e12, 1e12);
    assert!(r2.fubini_study_distance.is_finite());
    assert!(r2.holographic_conviction_modulator >= 0.45 && r2.holographic_conviction_modulator <= 1.40);
}
