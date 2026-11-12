//! # Contrato Formal de Certificación: Fase 102 (§102) — Gravedad Masiva dRGT & Apantallamiento de Vainshtein
//!
//! ## Modo Profesor: 7 Pilares de Verificación Cuántica
//! - QUÉ: Validación formal del motor `DrgtMassiveGravityEngine` y sus propiedades de gravedad masiva no lineal.
//! - POR QUÉ: Asegurar que el radio de Vainshtein, la cota de Higuchi y el potencial dRGT regulen el riesgo sistémico sin pánicos.
//! - PARA QUÉ: Blindar la cota de ruina Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ y regular el dimensionamiento continuo de capital.
//! - CÓMO: 5 contratos exhaustivos testeando inicialización, apantallamiento de Vainshtein, estabilidad de Higuchi, colapso de screening y resiliencia `rb_p1`.
//! - CUÁNDO: En cada ejecución del pipeline de integración continua de `risk-engine`.
//! - DÓNDE: `crates/risk-engine/tests/suite_fisica/f102_001_drgt_massive_gravity_contract.rs`.
//! - QUIÉN: Rol 5 (QA Engineer) y Rol 3 (Risk Manager Cuántico).

use risk_engine::drgt_massive_gravity::{DrgtMassiveGravityConfig, DrgtMassiveGravityEngine};

#[test]
fn test_c1_drgt_initialization_and_purity() {
    let mut engine = DrgtMassiveGravityEngine::default();
    let report = engine.evaluate(0.01, 0.15, 0.0, 0.05);

    assert!(
        report.effective_graviton_mass > 0.0,
        "La masa de gravitón debe ser positiva"
    );
    assert!(
        report.vainshtein_radius > 0.0,
        "El radio de Vainshtein debe ser positivo"
    );
    assert!(
        !report.drgt_higuchi_ghost_veto,
        "No debe haber veto en condiciones normales"
    );
    assert!(report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.25);
    assert!(
        report.higuchi_bound_ratio > 1.0,
        "Debe satisfacer la cota de Higuchi holgadamente"
    );
}

#[test]
fn test_c2_vainshtein_screening_and_radius_scaling() {
    let mut engine = DrgtMassiveGravityEngine::default();

    // 1. Perturbación suave muy dentro de la esfera de Vainshtein (r << r_V)
    let rep_screened = engine.evaluate(0.005, 0.10, 0.0, 0.02);
    assert!(
        rep_screened.vainshtein_screening_factor > 0.80,
        "Sub-Vainshtein debe tener apantallamiento muy alto (> 0.80), fue {}",
        rep_screened.vainshtein_screening_factor
    );

    // 2. Perturbación severa con estrés que rompe la escala local
    let rep_stressed = engine.evaluate(0.40, 0.80, 4.0, 0.90);
    assert!(
        rep_stressed.vainshtein_radius > rep_screened.vainshtein_radius,
        "Radio de Vainshtein debe crecer monótonamente con la masa del shock"
    );
}

#[test]
fn test_c3_higuchi_stability_bound() {
    let config = DrgtMassiveGravityConfig {
        graviton_mass_mg: 0.10, // Masa pequeña para hacer vulnerable a Higuchi
        beta_2: 1.0,
        beta_3: 0.5,
        higuchi_margin: 1.05,
        g_risk: 1.0,
    };
    let mut engine = DrgtMassiveGravityEngine::new(config);

    // En reposo, H es pequeño (~0.05), m_g^2 = 0.01, 2*H^2 = 0.005 -> ratio ~ 2.0 > margin
    let rep_stable = engine.evaluate(0.01, 0.05, 0.0, 0.0);
    assert!(rep_stable.higuchi_bound_ratio >= 1.0);

    // Inyectar expansión hiperbólica violenta de volatilidad: Hubble H explota a > 1.0
    // Entonces 2 * H^2 > 2.0 >> m_g^2 (0.01), ratio << 1.05 -> violación taquiónica de Higuchi
    let rep_ghost = engine.evaluate(0.10, 3.0, 8.0, 0.50);

    assert!(
        rep_ghost.higuchi_bound_ratio < config.higuchi_margin,
        "Ratio de Higuchi ({}) debe quebrar el margen ({}) bajo aceleración explosiva",
        rep_ghost.higuchi_bound_ratio,
        config.higuchi_margin
    );
    assert!(
        rep_ghost.drgt_higuchi_ghost_veto,
        "Debe detonar DRGT_HIGUCHI_GHOST_VETO"
    );
    assert!(
        (rep_ghost.slot_scaling_factor - 0.10).abs() < 1e-6,
        "Slot factor debe colapsar a 0.10"
    );
}

#[test]
fn test_c4_vainshtein_breach_veto() {
    let config = DrgtMassiveGravityConfig {
        graviton_mass_mg: 1.0, // Masa grande -> r_V pequeño
        beta_2: 1.0,
        beta_3: 0.5,
        higuchi_margin: 1.0,
        g_risk: 1.0,
    };
    let mut engine = DrgtMassiveGravityEngine::new(config);

    // Inyectar shock masivo con gran radio de perturbación r
    let rep = engine.evaluate(0.95, 0.50, 9.0, 0.99);

    assert!(
        rep.perturbation_radius > rep.vainshtein_radius,
        "La perturbación debe superar el radio de Vainshtein"
    );
    assert!(
        rep.drgt_higuchi_ghost_veto,
        "Debe detonar veto por desbordamiento de Vainshtein"
    );
    assert!((rep.slot_scaling_factor - 0.10).abs() < 1e-6);
}

#[test]
fn test_c5_rb_p1_resilience_and_extreme_inputs() {
    let mut engine = DrgtMassiveGravityEngine::default();

    // Inyectar combinaciones de NaN, Inf y valores negativos
    let rep = engine.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -5.0);

    assert!(
        rep.effective_graviton_mass.is_finite(),
        "Masa gravitón debe ser finita"
    );
    assert!(
        rep.hubble_volatility_expansion.is_finite(),
        "Hubble H debe ser finito"
    );
    assert!(
        rep.vainshtein_radius.is_finite(),
        "Radio Vainshtein debe ser finito"
    );
    assert!(
        rep.vainshtein_screening_factor.is_finite(),
        "Screening debe ser finito"
    );
    assert!(
        rep.drgt_potential_energy.is_finite(),
        "Energía potencial dRGT debe ser finita"
    );
    assert!(
        rep.higuchi_bound_ratio.is_finite(),
        "Ratio Higuchi debe ser finito"
    );
    assert!(rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25);
}
