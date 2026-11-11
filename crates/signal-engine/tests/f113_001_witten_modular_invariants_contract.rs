//! Contrato Formal de Certificación: Fase 113 (§113) — Invariantes Modulares de Witten & Chern-Simons SU(N)_k
//!
//! Pruebas formales de validación matemática:
//! 1. `test_c1_modular_s_matrix_unitarity_and_verlinde`: Verificación analítica de unitariedad y traza modular nominal.
//! 2. `test_c2_directional_alpha_generation`: Respuesta asimétrica de alpha modular ante flujos agresores long vs short.
//! 3. `test_c3_modular_trace_collapse_veto`: Colapso de traza modular $\operatorname{Tr}(\mathcal{S}) < 0.15$ detonando veto fail-closed.
//! 4. `test_c4_verlinde_unitarity_deviation_veto`: Violación de completitud en álgebra de fusión de Verlinde detonando veto.
//! 5. `test_c5_nan_infinity_immunity_and_zero_panic_guarantee`: Inmunidad total contra NaN/$\pm\infty$ y garantía zero-panic.

use signal_engine::{WittenModularConfig, WittenModularInvariantsEngine};

#[test]
fn test_c1_modular_s_matrix_unitarity_and_verlinde() {
    let mut engine = WittenModularInvariantsEngine::new(WittenModularConfig::default());

    let mut report = engine.update(0.0001, 0.05, 0.02, 0.02);
    for _ in 0..10 {
        report = engine.update(0.0001, 0.02, 0.01, 0.015);
    }

    assert!(
        report.total_quantum_dimension > 1.5,
        "La dimensión cuántica total para SU(2)_3 debe ser > 1.5"
    );
    assert!(
        report.modular_s_trace >= 0.20,
        "La traza modular debe ser estrictamente positiva"
    );
    assert!(
        report.verlinde_unitarity_deviation < 0.50,
        "La desviación de Verlinde debe ser pequeña en nominal"
    );
    assert!(
        !report.witten_modular_collapse_veto,
        "No debe haber veto en régimen balanceado"
    );
    assert!(report.slot_scaling_factor >= 0.50 && report.slot_scaling_factor <= 1.25);
}

#[test]
fn test_c2_directional_alpha_generation() {
    let mut engine_bull = WittenModularInvariantsEngine::new(WittenModularConfig::default());
    let mut engine_bear = WittenModularInvariantsEngine::new(WittenModularConfig::default());

    let mut report_bull = engine_bull.update(0.01, 0.80, 0.60, 0.02);
    let mut report_bear = engine_bear.update(-0.01, -0.80, -0.60, 0.02);

    for _ in 0..5 {
        report_bull = engine_bull.update(0.01, 0.85, 0.70, 0.02);
        report_bear = engine_bear.update(-0.01, -0.85, -0.70, 0.02);
    }

    assert!(
        report_bull.directional_alpha > 0.0,
        "Flujo agresor bid debe inducir alpha modular positivo"
    );
    assert!(
        report_bear.directional_alpha < 0.0,
        "Flujo agresor ask debe inducir alpha modular negativo"
    );
}

#[test]
fn test_c3_modular_trace_collapse_veto() {
    let mut engine = WittenModularInvariantsEngine::new(WittenModularConfig {
        min_modular_trace_threshold: 0.99, // Umbral artificialmente inalcanzable para forzar veto de traza
        ..Default::default()
    });

    let report = engine.update(0.001, 0.05, 0.02, 0.02);

    assert!(
        report.witten_modular_collapse_veto,
        "Traza modular insuficiente debe activar veto fail-closed"
    );
    assert_eq!(
        report.slot_scaling_factor, 0.10,
        "Slot debe colapsar a 0.10 bajo veto"
    );
}

#[test]
fn test_c4_verlinde_unitarity_deviation_veto() {
    let mut engine = WittenModularInvariantsEngine::new(WittenModularConfig {
        max_verlinde_unitarity_deviation: 0.001, // Umbral ultra sensible para detonar violación de unitariedad
        ..Default::default()
    });

    // Inyectamos choques con alta turbulencia para generar desalineación en fusión
    let mut report = engine.update(0.05, 0.90, -0.90, 5.0);
    for _ in 0..10 {
        report = engine.update(-0.05, -0.90, 0.90, 8.0);
    }

    assert!(
        report.witten_modular_collapse_veto,
        "Desviación de unitariedad de Verlinde debe forzar veto"
    );
    assert_eq!(report.slot_scaling_factor, 0.10);
}

#[test]
fn test_c5_nan_infinity_immunity_and_zero_panic_guarantee() {
    let mut engine = WittenModularInvariantsEngine::new(WittenModularConfig::default());

    let report_nan = engine.update(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(report_nan.modular_s_trace.is_finite());
    assert!(report_nan.dehn_twist_phase.is_finite());
    assert!(report_nan.directional_alpha.is_finite());
    assert!(report_nan.slot_scaling_factor >= 0.10 && report_nan.slot_scaling_factor <= 1.25);

    let report_inf = engine.update(
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    );
    assert!(report_inf.modular_s_trace.is_finite());
    assert!(report_inf.directional_alpha.is_finite());
    assert!(report_inf.slot_scaling_factor >= 0.10);
}
