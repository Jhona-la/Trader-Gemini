//! # Contrato de Certificación Formal: Fase 30 (§30) - Puerta Cuántica de Ignición y Certificación OOS
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Suite de contratos formales de certificación para la puerta de ignición y verificación OOS
//!   `VilleGrossmanIgnitionGate`.
//! - **POR QUÉ:** La transición de backtesting a producción con capital real requiere garantías
//!   matemáticas incondicionales contra ruina ($\mathbb{P} \le 10^{-6}$) y preservación estricta del
//!   suelo de riqueza móvil de Grossman-Zhou ($W_t \ge S_t$), impidiendo que ninguna orden se emita
//!   sin evidencia empírica OOS demostrada.
//! - **PARA QUÉ:** Asegurar que el crecimiento compuesto cada 72 horas no se vea truncado por
//!   eventos de cola cisne negro o violación de límites de drawdown.
//! - **CÓMO:** Verificación de 5 hipótesis formales (c1 a c5) cubriendo dinámica de cushion,
//!   desaceleración de riesgo asintótica, convergencia del E-proceso de Ville y certificación OOS.
//! - **CUÁNDO:** Previo al lanzamiento del bot en producción y en cada ciclo de auditoría forense.
//! - **DÓNDE:** `crates/risk-engine/tests/f30_001_ville_grossman_ignition_contract.rs`.
//! - **QUIÉN:** Consejo Colegiado (Roles 1: Arquitecto, 2: Quant Dev, 3: Risk Manager, 5: QA Engineer).

use risk_engine::ville_grossman_ignition_gate::{
    IgnitionViolationBits, VilleGrossmanIgnitionGate, VilleGrossmanParams,
};

#[test]
fn c1_ignition_gate_initialization_nominal_cushion() {
    let params = VilleGrossmanParams::default();
    let gate = VilleGrossmanIgnitionGate::new(100.0, params);

    assert_eq!(gate.initial_wealth(), 100.0);
    assert_eq!(gate.current_wealth(), 100.0);
    assert_eq!(gate.peak_wealth(), 100.0);

    let (cushion, cushion_ratio, risk_mult) = gate.compute_grossman_zhou_cushion();

    // Con peak = 100.0 y alpha = 0.985, piso = 98.50 -> cushion = 1.50
    assert!(
        (cushion - 1.50).abs() < 1e-4,
        "Cushion inicial debe ser 1.50"
    );
    assert!(
        (cushion_ratio - 0.015).abs() < 1e-4,
        "Cushion ratio inicial debe ser 1.5%"
    );
    assert_eq!(
        risk_mult, 1.0,
        "Multiplicador de riesgo inicial debe ser 1.0 (máxima capacidad)"
    );
}

#[test]
fn c2_grossman_zhou_floor_breach_and_deceleration() {
    let params = VilleGrossmanParams::default();
    let mut gate = VilleGrossmanIgnitionGate::new(100.0, params);

    // Simular que el portfolio alcanza un nuevo máximo de 120.0
    gate.update_wealth(120.0);
    assert_eq!(gate.peak_wealth(), 120.0);

    // Piso móvil = 0.985 * 120.0 = 118.20
    // Si la riqueza cae a 119.10 (a mitad de camino entre 120 y 118.20)
    gate.update_wealth(119.10);
    let (_cushion, _ratio, risk_half) = gate.compute_grossman_zhou_cushion();
    assert!(
        risk_half > 0.40 && risk_half < 0.60,
        "A mitad del colchón el multiplicador debe desacelerar suavemente (~0.50), obtenido {:.2}",
        risk_half
    );

    // Si la riqueza cae al piso exacto (118.20) o por debajo (118.00)
    gate.update_wealth(118.00);
    let (cushion_breached, _ratio_b, risk_breached) = gate.compute_grossman_zhou_cushion();
    assert_eq!(
        cushion_breached, 0.0,
        "Cushion debe ser 0.0 tras quebrar el suelo"
    );
    assert_eq!(
        risk_breached, 0.0,
        "Multiplicador de riesgo debe colapsar a 0.0"
    );

    let report = gate.evaluate_ignition_readiness();
    assert!(
        !report.certified_for_live_ignition,
        "La ignición debe ser terminantemente rechazada al violar el piso"
    );
    assert!(
        report.violation_mask & IgnitionViolationBits::GROSSMAN_FLOOR_BREACHED != 0,
        "Debe encender el bit de piso quebrado"
    );
}

#[test]
fn c3_ville_edge_evidence_is_not_ruin_probability() {
    let params = VilleGrossmanParams::default();
    let mut gate = VilleGrossmanIgnitionGate::new(100.0, params);

    for _ in 0..25 {
        gate.record_oos_trade(15.0);
    }

    let report = gate.evaluate_ignition_readiness();

    assert!(report.ville_e_value > 1.0);
    assert!(report.ruin_probability_bound.is_nan());
    assert!(report.violation_mask & (1 << 6) != 0);
    assert!(!report.certified_for_live_ignition);
}

#[test]
fn c4_full_oos_edge_does_not_open_unidentified_ruin() {
    let params = VilleGrossmanParams {
        max_ruin_probability: 0.10,
        grossman_alpha_floor: 0.985,
        max_allowed_drawdown_pct: 0.015,
        min_oos_sharpe: 2.0,
        min_oos_samples: 25,
    };
    let mut gate = VilleGrossmanIgnitionGate::new(100.0, params);

    for i in 0..40 {
        let pnl = if i % 5 == 0 { -2.0 } else { 25.0 };
        gate.record_oos_trade(pnl);
    }

    let report = gate.evaluate_ignition_readiness();

    assert!(report.oos_sample_count >= 25);
    assert!(report.oos_sharpe_estimate >= 2.0);
    assert!(report.current_drawdown_pct <= 0.015);
    assert!(report.ruin_probability_bound.is_nan());
    assert!(report.violation_mask & (1 << 6) != 0);
    assert!(!report.certified_for_live_ignition);
}

#[test]
fn c5_nan_and_infinite_wealth_safety_resilience() {
    let params = VilleGrossmanParams::default();
    let mut gate = VilleGrossmanIgnitionGate::new(100.0, params);

    gate.update_wealth(f64::NAN);
    gate.update_wealth(f64::INFINITY);
    gate.record_oos_trade(f64::NAN);

    assert!(!gate.current_wealth().is_finite());
    let report = gate.evaluate_ignition_readiness();
    assert!(!report.certified_for_live_ignition);
    assert!(report.violation_mask & IgnitionViolationBits::WEALTH_NON_FINITE != 0);
}

#[test]
fn c6_invalid_wealth_revokes_prior_live_certification() {
    for wealth in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let params = VilleGrossmanParams {
            max_ruin_probability: 0.10,
            ..VilleGrossmanParams::default()
        };
        let mut gate = VilleGrossmanIgnitionGate::new(100.0, params);
        for i in 0..40 {
            let pnl = if i % 5 == 0 { -2.0 } else { 25.0 };
            gate.record_oos_trade(pnl);
        }
        gate.update_wealth(wealth);

        if wealth.is_nan() {
            assert!(gate.current_wealth().is_nan());
        } else {
            assert_eq!(gate.current_wealth(), wealth);
        }
        let report = gate.evaluate_ignition_readiness();
        assert!(!report.certified_for_live_ignition);
        assert!(report.violation_mask & IgnitionViolationBits::WEALTH_NON_FINITE != 0);
    }
}

#[test]
fn c7_initial_wealth_invalid_has_no_fallback() {
    let gate = VilleGrossmanIgnitionGate::new(0.0, VilleGrossmanParams::default());

    assert_eq!(gate.initial_wealth(), 0.0);
    let report = gate.evaluate_ignition_readiness();
    assert!(!report.certified_for_live_ignition);
    assert!(report.violation_mask & IgnitionViolationBits::WEALTH_NON_FINITE != 0);
}
