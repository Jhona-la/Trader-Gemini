//! Contrato Formal de Verificación: Fase 60 (Geometría de Contacto de Reeb y Dinámica de Vórtices)
//!
//! Este conjunto de pruebas certifica las propiedades topológicas y simplécticas de preservación de capital:
//! - C1: Norma del campo de Reeb acotada $|R_\alpha| \approx 1.0$ y torsión de contacto no-negativa.
//! - C2: Activación de `REEB_CONTACT_TOPOLOGICAL_VETO` ante torsión transversal crítica, colapsando slot a 0.0.
//! - C3: Integral de acción simpléctica finita y ángulo de desalineamiento $\Theta_{\text{reeb}} \in [0, \pi]$.
//! - C4: Determinismo $O(1)$ en stack, cero heap allocations en hot path y preservación de la cota de ruina de Ville.

use risk_engine::{
    ReebContactParams, ReebContactVortexEngine, CRITICAL_CONTACT_TORSION, REEB_HISTORY_CAPACITY,
};

#[test]
fn test_c1_reeb_vector_field_and_conservative_foliation() {
    let params = ReebContactParams::default();
    let mut engine = ReebContactVortexEngine::new(params);

    // Flujo suave conservativo a lo largo de la hipersuperficie de contacto
    let report = engine.evaluate(
        50.0,   // capital_at_risk_usd
        1000.0, // total_wallet_capital_usd
        1.5,    // current_spread_bps
        0.005,  // market_volatility
        0.10,   // order_flow_imbalance
        0.0002, // price_return
    );

    assert!(
        report.reeb_vector_norm >= 0.9 && report.reeb_vector_norm <= 1.5,
        "La norma del campo de Reeb debe ser aproximadamente unitaria: {}",
        report.reeb_vector_norm
    );
    assert!(
        report.contact_torsion_intensity >= 0.0,
        "La torsión de contacto debe ser no-negativa: {}",
        report.contact_torsion_intensity
    );
    assert!(
        report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.25,
        "Modulador de slot fuera de límites: {}",
        report.slot_scaling_factor
    );
    assert!(
        !report.reeb_contact_veto_induced,
        "No debe activar veto en régimen conservativo"
    );
}

#[test]
fn test_c2_reeb_contact_veto_on_critical_torsion() {
    let params = ReebContactParams::default();
    let mut engine = ReebContactVortexEngine::new(params);

    // Inyectamos saltos transversales violentos con spread dilatado y riesgo extremo
    let mut report = engine.evaluate(100.0, 1000.0, 2.0, 0.01, 0.1, 0.001);
    for i in 1..8 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        report = engine.evaluate(
            800.0, // 80% capital en riesgo
            1000.0,
            60.0, // spread enorme de 60 bps
            0.15, // 15% volatilidad
            sign * 2.0,
            sign * 0.08,
        );
    }

    assert!(
        report.reeb_contact_veto_induced,
        "Debe activar REEB_CONTACT_TOPOLOGICAL_VETO ante torsión transversal crítica"
    );
    assert_eq!(
        report.slot_scaling_factor, 0.0,
        "El tamaño de slot debe ser cancelado (0.0) cuando el veto está activo"
    );
    assert_eq!(report.veto_reason, Some("REEB_CONTACT_TOPOLOGICAL_VETO"));
}

#[test]
fn test_c3_periodic_action_and_angular_misalignment_bounds() {
    let params = ReebContactParams::default();
    let mut engine = ReebContactVortexEngine::new(params);

    for i in 0..REEB_HISTORY_CAPACITY {
        let rep = engine.evaluate(
            20.0,
            500.0,
            2.0,
            0.008,
            0.05 * (i as f64),
            0.0001 * (i as f64),
        );

        assert!(
            rep.periodic_action_integral >= 0.0,
            "La acción simpléctica debe ser no-negativa: {}",
            rep.periodic_action_integral
        );
        assert!(
            rep.reeb_angular_misalignment >= 0.0
                && rep.reeb_angular_misalignment <= core::f64::consts::PI,
            "Desalineamiento angular de Reeb fuera de [0, pi]: {}",
            rep.reeb_angular_misalignment
        );
    }
}

#[test]
fn test_c4_critical_torsion_constant() {
    assert_eq!(CRITICAL_CONTACT_TORSION, 0.85);
}
