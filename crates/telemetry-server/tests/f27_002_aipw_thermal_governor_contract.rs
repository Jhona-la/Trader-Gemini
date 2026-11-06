//! # Contrato de Verificación Formal F27-002: Inferencia Causal AIPW & Gobernador Térmico
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Batería de 6 contratos formales de inferencia causal doblemente robusta y gobernanza de silicio.
//! - **POR QUÉ:** La estimación del impacto causal de enrutamiento Maker vs Taker debe ser matemáticamente
//!   insesgada ante variables de confusión, y el gobernador térmico debe proteger la CPU contra degradación de silicio.
//! - **PARA QUÉ:** Demostrar que el sistema toma decisiones de trading causales óptimas y opera de forma
//!   autónoma y segura en laptops de pocos recursos.
//! - **CÓMO:** Verificación de teoremas de doble robustez, intervalos de confianza semiparamétricos y máquinas de estado.

#![allow(clippy::all)]

use telemetry_server::aipw_causal_governor::{
    AipwCausalGovernorEngine, AipwSample, SiliconThermalState,
};

#[test]
fn c1_aipw_doubly_robust_when_propensity_correct() {
    let mut engine = AipwCausalGovernorEngine::default();

    // Verdadero ATE = +2.5 bps de ventaja Maker.
    // Simulamos un escenario donde el modelo de resultado m(X) es incorrecto (tiene sesgo constante de +10.0),
    // pero el modelo de propensión e(X) es exactamente correcto.
    let true_ate = 2.5;

    for i in 0..50 {
        let is_maker = (i % 2 == 0) as u8;
        let e_true = 0.50; // Propensión balanceada verdadera
        let outcome = if is_maker == 1 { true_ate } else { 0.0 };

        // Modelo de resultado con sesgo aditivo severo (+10.0)
        let bad_m1 = 12.5;
        let bad_m0 = 10.0;

        engine.record_causal_event(is_maker, outcome, e_true, bad_m1, bad_m0);
    }

    let report = engine.analyze(1000, 1000);
    assert!(
        (report.average_treatment_effect - true_ate).abs() < 1e-6,
        "AIPW debe ser exacto cuando el modelo de propensión es correcto: ATE estimado={}, esperado={}",
        report.average_treatment_effect,
        true_ate
    );
}

#[test]
fn c2_aipw_doubly_robust_when_outcome_correct() {
    let mut engine = AipwCausalGovernorEngine::default();

    // Verdadero ATE = +3.0 bps.
    // Simulamos un escenario donde el modelo de resultado m(X) es correcto,
    // pero la propensión estimada está fuertemente sesgada (0.80 en lugar de 0.50).
    let true_ate = 3.0;

    for i in 0..50 {
        let is_maker = (i % 2 == 0) as u8;
        let outcome = if is_maker == 1 { 5.0 } else { 2.0 }; // 5 - 2 = 3.0

        let correct_m1 = 5.0;
        let correct_m0 = 2.0;
        let bad_propensity = 0.80; // Sesgado

        engine.record_causal_event(is_maker, outcome, bad_propensity, correct_m1, correct_m0);
    }

    let report = engine.analyze(1000, 1000);
    assert!(
        (report.average_treatment_effect - true_ate).abs() < 1e-6,
        "AIPW debe ser exacto cuando el modelo de resultado es correcto: ATE={}, esperado={}",
        report.average_treatment_effect,
        true_ate
    );
}

#[test]
fn c3_aipw_standard_error_and_ci_bounds() {
    let mut engine = AipwCausalGovernorEngine::default();

    // Inyectar datos con dispersión
    for i in 0..30 {
        let a = (i % 2 == 0) as u8;
        let y = if a == 1 { 4.0 + (i as f64 * 0.1) } else { 1.0 };
        engine.record_causal_event(a, y, 0.5, 4.5, 1.0);
    }

    let report = engine.analyze(1000, 1000);
    assert!(report.standard_error >= 0.0);
    assert!(
        report.ci_lower_95 <= report.average_treatment_effect,
        "Límite inferior del CI debe ser <= ATE"
    );
    assert!(
        report.ci_upper_95 >= report.average_treatment_effect,
        "Límite superior del CI debe ser >= ATE"
    );
}

#[test]
fn c4_silicon_thermal_governor_state_transitions() {
    let engine = AipwCausalGovernorEngine::new(75.0, 85.0);

    // 1. Nominal
    engine.update_temperature(65.0);
    let (state_nom, _) = engine.evaluate_thermal_state(1000, 1000);
    assert_eq!(state_nom, SiliconThermalState::Nominal);

    // 2. Warning
    engine.update_temperature(78.5);
    let (state_warn, _) = engine.evaluate_thermal_state(1000, 1000);
    assert_eq!(state_warn, SiliconThermalState::ThrottlingWarning);

    // 3. Freno Crítico
    engine.update_temperature(89.0);
    let (state_crit, _) = engine.evaluate_thermal_state(1000, 1000);
    assert_eq!(state_crit, SiliconThermalState::CriticalThermalBrake);
}

#[test]
fn c5_aipw_propensity_score_clamping_safety() {
    let sample_low = AipwSample {
        treatment: 1,
        outcome: 10.0,
        propensity_score: 0.000001, // Extremadamente bajo
        outcome_model_1: 5.0,
        outcome_model_0: 0.0,
    };

    let score = AipwCausalGovernorEngine::compute_aipw_score(&sample_low);
    assert!(
        !score.is_nan() && !score.is_infinite(),
        "El score no debe ser NaN ni infinito ante propensity score extremo"
    );
}

#[test]
fn c6_aperf_mperf_ratio_clamping_and_efficiency() {
    let engine = AipwCausalGovernorEngine::default();

    // 1. Ratio nominal normal
    let (_, r_nom) = engine.evaluate_thermal_state(1200, 1000);
    assert!((r_nom - 1.2).abs() < 1e-6);

    // 2. MPERF = 0 (defensa ante división por cero)
    let (_, r_zero) = engine.evaluate_thermal_state(1000, 0);
    assert_eq!(r_zero, 1.0);

    // 3. Ratio saturado superior
    let (_, r_high) = engine.evaluate_thermal_state(5000, 1000);
    assert_eq!(r_high, 2.0, "Debe quedar acotado en 2.0");
}
