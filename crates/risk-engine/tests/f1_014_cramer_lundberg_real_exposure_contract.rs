//! Contract test for F1-014 / F1-015 (Ola Ω152):
//!
//! Cramér-Lundberg unit scaling must divide R_nocional by realistic exposure fraction
//! f_exposure = notional / capital, NEVER by max_exchange_leverage (e.g. 125x).
//!
//! When edge is strong (R_raw = 65.0) and f ~ 2.5, R_capital is 26.0,
//! resulting in m = ln(1/eps)/R ~ 0.115 and effective limit = min(0.25, 0.25/0.115) = 0.25.
//! The edge preserves the risk budget (does NOT falsely veto).

use risk_engine::correlation_guard::veto_por_riesgo_cramer_lundberg;
use risk_engine::cramer_lundberg::EstimadorSiniestros;

#[test]
fn test_f1_014_cramer_lundberg_real_exposure_scaling_contract() {
    let r_raw = 65.0; // Positive measured edge from 30+ net profitable closes
    let max_exchange_leverage = 125.0; // BTCUSDT spec max leverage
    let riesgo_candidata = 0.05; // 5% risk at stop
    let stop_loss = 0.02; // 2% stop distance
    let allocated_capital = 13.0; // $13 USD base capital
    let min_notional = 5.0; // Binance min notional
    let epsilon = 0.05; // 5% ruin probability tolerance
    let tope_streak = 0.25; // 25% max capital budget

    // 1. Old buggy scaling: divided by max_exchange_leverage (125.0)
    let r_buggy = r_raw / max_exchange_leverage; // 0.52
    let m_buggy = EstimadorSiniestros::margen_de_cota(r_buggy, epsilon).unwrap(); // ~5.76
    let cota_buggy = tope_streak / m_buggy; // ~0.0434
    let tope_efectivo_buggy = tope_streak.min(cota_buggy); // 0.0434
    assert!(
        tope_efectivo_buggy < riesgo_candidata,
        "Old bug collapsed ruin limit below candidate risk 0.05 ({} < {})",
        tope_efectivo_buggy,
        riesgo_candidata
    );

    // Old bug falsely vetoed candidate even with ZERO open positions:
    let riesgos_single = [riesgo_candidata];
    let vetoed_buggy = veto_por_riesgo_cramer_lundberg(
        &riesgos_single,
        None,
        tope_streak,
        Some(r_buggy),
        epsilon,
    );
    assert!(vetoed_buggy, "Old bug falsely triggered correlation veto on single candidate");

    // 2. Corrected scaling (F1-014): divide by real exposure fraction f_exposure:
    let f_exposure = (riesgo_candidata / stop_loss)
        .clamp(min_notional / allocated_capital, (0.25 * max_exchange_leverage.min(20.0)).max(1.0));
    assert!((f_exposure - 2.5).abs() < 1e-9, "f_exposure must be 2.5, got {}", f_exposure);

    let r_corrected = r_raw / f_exposure; // 26.0
    let m_corrected = EstimadorSiniestros::margen_de_cota(r_corrected, epsilon).unwrap(); // ~0.115
    let cota_corrected = tope_streak / m_corrected; // ~2.17
    let tope_efectivo_corrected = tope_streak.min(cota_corrected); // 0.25

    assert_eq!(
        tope_efectivo_corrected, tope_streak,
        "With strong measured edge, Cramér-Lundberg must not restrict healthy trade"
    );

    let vetoed_corrected = veto_por_riesgo_cramer_lundberg(
        &riesgos_single,
        None,
        tope_streak,
        Some(r_corrected),
        epsilon,
    );
    assert!(!vetoed_corrected, "Healthy trade with zero correlated positions must NOT be vetoed");

    // 3. For weak edge (R_raw = 0.5), Cramér-Lundberg correctly tightens limit:
    let r_weak = 0.5 / f_exposure; // 0.20
    let m_weak = EstimadorSiniestros::margen_de_cota(r_weak, epsilon).unwrap(); // ~14.98
    let cota_weak = tope_streak / m_weak; // ~0.0167
    let tope_efectivo_weak = tope_streak.min(cota_weak); // 0.0167
    assert!(
        tope_efectivo_weak < riesgo_candidata,
        "Weak edge correctly tightens limit below candidate risk"
    );
    let vetoed_weak = veto_por_riesgo_cramer_lundberg(
        &riesgos_single,
        None,
        tope_streak,
        Some(r_weak),
        epsilon,
    );
    assert!(vetoed_weak, "Weak edge trade should be protected/restricted by Lundberg");
}
