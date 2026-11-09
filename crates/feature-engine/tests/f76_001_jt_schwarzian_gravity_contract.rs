//! Contrato Formal de Verificación: Fase 76 (§76) — Gravedad JT 2D y Dinámica Cuántica de Schwarzian
//!
//! Verifica matemáticamente la evaluación de la acción de Schwarzian, densidad de estados holográfica
//! y detección de singularidades en la reparametrización temporal de microestructura.

use feature_engine::jt_schwarzian_gravity::{
    JtSchwarzianConfig, JtSchwarzianEngine, JtSchwarzianReport,
};

#[test]
fn test_c1_jt_schwarzian_steady_flow_equilibrium() {
    let config = JtSchwarzianConfig::default();
    let mut engine = JtSchwarzianEngine::new(config);

    // Flujo estacionario constante
    let mut last_report = JtSchwarzianReport {
        schwarzian_derivative: 0.0,
        schwarzian_energy: 0.0,
        density_of_states_rho: 0.0,
        lyapunov_exponent: 0.0,
        schwarzian_singularity_detected: false,
        slot_scale_factor: 1.0,
    };

    for _ in 0..10 {
        last_report = engine.evaluate(1.0, 0.1);
    }

    assert!(last_report.schwarzian_derivative.abs() < 1e-4);
    assert!(last_report.schwarzian_energy < 1e-3);
    assert!(!last_report.schwarzian_singularity_detected);
    assert!((last_report.slot_scale_factor - 1.0).abs() < 0.05);
    assert!(last_report.lyapunov_exponent > 0.0);
}

#[test]
fn test_c2_jt_schwarzian_adiabatic_acceleration() {
    let config = JtSchwarzianConfig {
        schwarzian_coupling_c: 1.5,
        inverse_temperature_beta: 0.8,
        critical_schwarzian_threshold: 100.0,
        slot_sensitivity: 0.02,
    };
    let mut engine = JtSchwarzianEngine::new(config);

    // Tasa con aceleración suave continua
    let mut rate = 1.0;
    let mut last_report = engine.evaluate(rate, 0.1);
    for i in 1..=5 {
        rate += (i as f64) * 0.1;
        last_report = engine.evaluate(rate, 0.1);
    }

    assert!(last_report.schwarzian_derivative.is_finite());
    assert!(last_report.schwarzian_energy > 0.0);
    assert!(last_report.density_of_states_rho > 0.0);
    assert!(!last_report.schwarzian_singularity_detected);
    assert!(last_report.slot_scale_factor >= 0.15 && last_report.slot_scale_factor <= 1.25);
}

#[test]
fn test_c3_jt_schwarzian_chaotic_divergence_singularity_detection() {
    let config = JtSchwarzianConfig {
        schwarzian_coupling_c: 1.0,
        inverse_temperature_beta: 1.0,
        critical_schwarzian_threshold: 20.0,
        slot_sensitivity: 0.05,
    };
    let mut engine = JtSchwarzianEngine::new(config);

    // Inicializar con ritmo estable
    engine.evaluate(1.0, 0.01);
    engine.evaluate(1.0, 0.01);

    // Shock abrupto con aceleración hiperbólica violenta en delta t mínimo
    let report_shock = engine.evaluate(50.0, 0.005);

    assert!(report_shock.schwarzian_derivative.abs() >= 20.0);
    assert!(report_shock.schwarzian_singularity_detected);
    assert_eq!(report_shock.slot_scale_factor, 0.15);
    assert!(report_shock.schwarzian_energy >= 20.0);
}

#[test]
fn test_c4_jt_schwarzian_stack_safety_and_boundedness() {
    let config = JtSchwarzianConfig::default();
    let mut engine = JtSchwarzianEngine::new(config);

    // Tasa infinitesimal y dt casi nulo
    let report = engine.evaluate(1e-8, 1e-9);

    assert!(report.schwarzian_derivative.is_finite());
    assert!(report.schwarzian_energy.is_finite());
    assert!(report.density_of_states_rho.is_finite());
    assert!(report.lyapunov_exponent.is_finite());
    assert!(report.slot_scale_factor >= 0.15 && report.slot_scale_factor <= 1.25);
}
