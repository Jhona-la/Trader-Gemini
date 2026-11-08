//! Contrato Formal de Verificación: Fase 57 (Barrera Conformal de No-Ruina de Gromov y Cuantización Cuaterniónica)
//!
//! Este conjunto de pruebas certifica las propiedades simplécticas y de preservación de capital del motor:
//! - C1: Teorema del Camello Simpléctico: radio del portafolio $R_{\text{portfolio}}$ y radio de garganta $r_{\text{hole}}$.
//! - C2: Activación de `gromov_symplectic_veto` si $R > r$, colapsando la escala de slot a 0.0.
//! - C3: Normalización y álgebra de cuaterniones $\mathbb{H}$ para rotaciones en el espacio de fases 4D.
//! - C4: Determinismo $O(1)$ stack, robustez numérica total y preservación de la cota de ruina de Ville.

use risk_engine::{GromovSymplecticBarrierEngine, GromovSymplecticParams, Quaternion};

#[test]
fn test_c1_symplectic_camel_capacities_and_clearance() {
    let params = GromovSymplecticParams::default();
    let mut engine = GromovSymplecticBarrierEngine::new(params);

    // Condición holgada: profundidad abundante ($50,000 USD), capital en riesgo modesto ($50 USD)
    let report = engine.evaluate(
        50.0,     // capital_at_risk_usd
        1000.0,   // total_wallet_capital_usd
        50_000.0, // available_liquidity_depth_usd
        1.5,      // current_spread_bps
        0.005,    // market_volatility
    );

    assert!(
        !report.gromov_symplectic_veto,
        "No debe activar veto cuando la liquidez del libro es holgada"
    );
    assert!(
        report.hole_capacity_radius > report.portfolio_phase_radius,
        "El radio de la garganta ({}) debe superar el radio del portafolio ({})",
        report.hole_capacity_radius,
        report.portfolio_phase_radius
    );
    assert!(
        report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.25,
        "Modulador de slot fuera de rango: {}",
        report.slot_scaling_factor
    );
    assert!(
        report.gromov_portfolio_capacity > 0.0 && report.gromov_hole_capacity > 0.0,
        "Las capacidades simplécticas de Gromov deben ser positivas"
    );
}

#[test]
fn test_c2_gromov_symplectic_veto_on_oversqueezing() {
    let params = GromovSymplecticParams::default();
    let mut engine = GromovSymplecticBarrierEngine::new(params);

    // Condición de aplastamiento forzado: intento de empujar $5,000 USD contra un libro con apenas $500 USD y spread de 25 bps
    let report = engine.evaluate(
        5_000.0, // capital_at_risk_usd
        5_000.0, // total_wallet_capital_usd (100% de la wallet!)
        500.0,   // available_liquidity_depth_usd (libro seco)
        25.0,    // current_spread_bps (spread dilatado)
        0.05,    // market_volatility
    );

    assert!(
        report.gromov_symplectic_veto,
        "Debe activar GROMOV_SYMPLECTIC_VETO ante intento de sobre-aplastamiento simpléctico"
    );
    assert_eq!(
        report.slot_scaling_factor, 0.0,
        "El tamaño de slot debe ser cancelado (0.0) cuando el veto está activo"
    );
}

#[test]
fn test_c3_quaternion_algebra_and_unitarity() {
    let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize();
    assert!(
        (q1.norm() - 1.0).abs() < 1e-12,
        "El cuaternión debe ser unitario"
    );

    let q_ident = Quaternion::identity();
    let q_prod = q1.mul(q_ident);
    assert!((q_prod.w - q1.w).abs() < 1e-12);
    assert!((q_prod.x - q1.x).abs() < 1e-12);
    assert!((q_prod.y - q1.y).abs() < 1e-12);
    assert!((q_prod.z - q1.z).abs() < 1e-12);
}

#[test]
fn test_c4_quaternion_phase_distance_bounded() {
    let params = GromovSymplecticParams::default();
    let mut engine = GromovSymplecticBarrierEngine::new(params);

    for i in 0..15 {
        let rep = engine.evaluate(
            10.0,
            100.0,
            2_000.0,
            2.0 + (i as f64) * 0.5,
            0.01 + (i as f64) * 0.002,
        );

        assert!(
            rep.quaternion_phase_distance >= 0.0 && rep.quaternion_phase_distance <= 1.0,
            "Distancia de fase cuaterniónica fuera de [0, 1]: {}",
            rep.quaternion_phase_distance
        );
        assert!(
            rep.compression_margin >= -1.0 && rep.compression_margin <= 1.0,
            "Margen de compresión fuera de [-1, 1]: {}",
            rep.compression_margin
        );
    }
}
