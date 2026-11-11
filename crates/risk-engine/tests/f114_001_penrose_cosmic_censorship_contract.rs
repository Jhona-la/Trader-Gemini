//! Contrato Formal de Certificación: Fase 114 (§114) — Censura Cósmica de Penrose & Superficies Atrapadas de Raychaudhuri
//!
//! Pruebas formales de validación matemática:
//! 1. `test_c1_nominal_horizon_shielding`: Horizonte aparente estable, expansión positiva y veto inactivo.
//! 2. `test_c2_trapped_surface_closed_collapse_veto`: Detección de superficie atrapada $(\theta_+ < -0.20 \land \theta_- < 0)$ con veto fail-closed.
//! 3. `test_c3_naked_singularity_risk_veto`: Riesgo de singularidad desnuda $(a/M)^2 > 0.85$ detonando veto.
//! 4. `test_c4_raychaudhuri_derivative_singularity_veto`: Colapso de enfoque geodésico $d\theta/d\lambda < -4.0$ con veto.
//! 5. `test_c5_nan_infinity_immunity_and_zero_panic_guarantee`: Inmunidad total contra NaN/$\pm\infty$ y garantía zero-panic.

use risk_engine::{PenroseCosmicCensorshipConfig, PenroseCosmicCensorshipEngine};

#[test]
fn test_c1_nominal_horizon_shielding() {
    let mut engine = PenroseCosmicCensorshipEngine::new(PenroseCosmicCensorshipConfig::default());

    let mut report = engine.update(1.5, 0.05, 0.02, 0.005);
    for _ in 0..10 {
        report = engine.update(1.2, 0.02, 0.015, 0.005);
    }

    assert!(
        report.expansion_theta_plus > 0.0,
        "La expansión saliente debe ser positiva en mercado saludable"
    );
    assert!(
        !report.trapped_surface_detected,
        "No debe haber superficie atrapada en condiciones regulares"
    );
    assert!(
        report.naked_singularity_risk < 0.50,
        "El riesgo de singularidad desnuda debe ser bajo"
    );
    assert!(
        !report.penrose_cosmic_censorship_veto,
        "Veto de Penrose debe estar inactivo"
    );
    assert!(
        !report.raychaudhuri_singularity_veto,
        "Veto de Raychaudhuri debe estar inactivo"
    );
    assert!(report.slot_scaling_factor >= 0.50 && report.slot_scaling_factor <= 1.25);
}

#[test]
fn test_c2_trapped_surface_closed_collapse_veto() {
    let mut engine = PenroseCosmicCensorshipEngine::new(PenroseCosmicCensorshipConfig::default());

    // Volatilidad extrema, spread severo y drawdown forzando colapso de ambas congruencias nulas
    let mut report = engine.update(200.0, 0.95, 20.0, 0.25);
    for _ in 0..10 {
        report = engine.update(250.0, 0.98, 25.0, 0.30);
    }

    assert!(
        report.trapped_surface_detected,
        "Ambas congruencias deben colapsar hacia adentro"
    );
    assert!(
        report.penrose_cosmic_censorship_veto,
        "Superficie atrapada severa debe activar veto fail-closed"
    );
    assert_eq!(
        report.slot_scaling_factor, 0.10,
        "El slot debe colapsar a 0.10 ante veto"
    );
}

#[test]
fn test_c3_naked_singularity_risk_veto() {
    let mut engine = PenroseCosmicCensorshipEngine::new(PenroseCosmicCensorshipConfig {
        max_naked_singularity_risk: 0.30,
        ..Default::default()
    });

    // Desbalance extremo de profundidad con spread amplio inflando el ratio de spin sobre masa (a/M)^2
    let mut report = engine.update(100.0, 1.0, 0.01, 0.50);
    for _ in 0..10 {
        report = engine.update(150.0, 1.0, 0.01, 0.60);
    }

    assert!(
        report.naked_singularity_risk > 0.30,
        "El riesgo de singularidad desnuda debe superar el umbral sensible"
    );
    assert!(
        report.penrose_cosmic_censorship_veto,
        "Singularidad desnuda debe detonar veto fail-closed"
    );
    assert_eq!(report.slot_scaling_factor, 0.10);
}

#[test]
fn test_c4_raychaudhuri_derivative_singularity_veto() {
    let mut engine = PenroseCosmicCensorshipEngine::new(PenroseCosmicCensorshipConfig {
        min_raychaudhuri_derivative: -0.50,
        ..Default::default()
    });

    // Curvatura de Ricci y cizallamiento inflados para empujar d\theta/d\lambda a valores negativos
    let mut report = engine.update(50.0, 0.80, 5.0, 0.10);
    for _ in 0..10 {
        report = engine.update(60.0, 0.85, 6.0, 0.15);
    }

    assert!(
        report.raychaudhuri_derivative < -0.50,
        "La derivada de Raychaudhuri debe ser fuertemente negativa"
    );
    assert!(
        report.raychaudhuri_singularity_veto,
        "Enfoque de Raychaudhuri debe activar veto fail-closed"
    );
    assert_eq!(report.slot_scaling_factor, 0.10);
}

#[test]
fn test_c5_nan_infinity_immunity_and_zero_panic_guarantee() {
    let mut engine = PenroseCosmicCensorshipEngine::new(PenroseCosmicCensorshipConfig::default());

    let report_nan = engine.update(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(report_nan.expansion_theta_plus.is_finite());
    assert!(report_nan.expansion_theta_minus.is_finite());
    assert!(report_nan.shear_intensity_sigma2.is_finite());
    assert!(report_nan.raychaudhuri_derivative.is_finite());
    assert!(report_nan.naked_singularity_risk.is_finite());
    assert!(report_nan.slot_scaling_factor >= 0.10 && report_nan.slot_scaling_factor <= 1.25);

    let report_inf = engine.update(
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    );
    assert!(report_inf.expansion_theta_plus.is_finite());
    assert!(report_inf.raychaudhuri_derivative.is_finite());
    assert!(report_inf.slot_scaling_factor >= 0.10);
}
