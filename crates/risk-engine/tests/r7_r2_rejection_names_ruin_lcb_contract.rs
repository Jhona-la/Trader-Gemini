// Contrato Formal Ola Ω64: Resolución R7-R2 — Nombres Canónicos de Rechazo, Paridad Analítica LCB en Viabilidad y Conservación de Micro-Capital ($13.00 USD)

use risk_engine::{
    REJECT_NAMES, REJECT_SLOTS,
    REJ_CONFIDENCE, REJ_CORRELATION, REJ_DRAWDOWN, REJ_EV, REJ_EXPOSURE_ZERO,
    REJ_FEE_IMPACT, REJ_FLAT_COIN, REJ_INVALID_INPUT, REJ_MARGIN_INSUFFICIENT,
    REJ_MIN_NOTIONAL, REJ_ORCHESTRATOR, REJ_OTROS, REJ_SIN_EVIDENCIA, REJ_SPEC,
    REJ_TARGET_GEOMETRY, REJ_TP_SL_FLOOR, REJ_VIABILIDAD,
};

#[test]
fn test_all_17_rejection_slots_map_to_named_constants_with_exact_parity() {
    assert_eq!(REJECT_SLOTS, 17, "Deben existir exactamente 17 ranuras de rechazo");
    assert_eq!(REJ_FLAT_COIN, 0);
    assert_eq!(REJ_EXPOSURE_ZERO, 1);
    assert_eq!(REJ_CORRELATION, 2);
    assert_eq!(REJ_SPEC, 3);
    assert_eq!(REJ_EV, 4);
    assert_eq!(REJ_FEE_IMPACT, 5);
    assert_eq!(REJ_MIN_NOTIONAL, 6);
    assert_eq!(REJ_MARGIN_INSUFFICIENT, 7);
    assert_eq!(REJ_ORCHESTRATOR, 8);
    assert_eq!(REJ_OTROS, 9);
    assert_eq!(REJ_DRAWDOWN, 10);
    assert_eq!(REJ_TP_SL_FLOOR, 11);
    assert_eq!(REJ_CONFIDENCE, 12);
    assert_eq!(REJ_VIABILIDAD, 13);
    assert_eq!(REJ_SIN_EVIDENCIA, 14);
    assert_eq!(REJ_INVALID_INPUT, 15);
    assert_eq!(REJ_TARGET_GEOMETRY, 16);

    assert_eq!(REJECT_NAMES[REJ_FLAT_COIN], "flat/coin");
    assert_eq!(REJECT_NAMES[REJ_EXPOSURE_ZERO], "exposure0");
    assert_eq!(REJECT_NAMES[REJ_CORRELATION], "correlacion");
    assert_eq!(REJECT_NAMES[REJ_SPEC], "spec");
    assert_eq!(REJECT_NAMES[REJ_EV], "EV");
    assert_eq!(REJECT_NAMES[REJ_FEE_IMPACT], "fee_impact");
    assert_eq!(REJECT_NAMES[REJ_MIN_NOTIONAL], "min_notional");
    assert_eq!(REJECT_NAMES[REJ_MARGIN_INSUFFICIENT], "margen_insuf");
    assert_eq!(REJECT_NAMES[REJ_ORCHESTRATOR], "orchestrator");
    assert_eq!(REJECT_NAMES[REJ_OTROS], "otros");
    assert_eq!(REJECT_NAMES[REJ_DRAWDOWN], "drawdown");
    assert_eq!(REJECT_NAMES[REJ_TP_SL_FLOOR], "suelo_tp_sl");
    assert_eq!(REJECT_NAMES[REJ_CONFIDENCE], "confianza");
    assert_eq!(REJECT_NAMES[REJ_VIABILIDAD], "viabilidad");
    assert_eq!(REJECT_NAMES[REJ_SIN_EVIDENCIA], "sin_evidencia");
    assert_eq!(REJECT_NAMES[REJ_INVALID_INPUT], "entrada_invalida");
    assert_eq!(REJECT_NAMES[REJ_TARGET_GEOMETRY], "geometria_invalida");
}

#[test]
fn test_zero_raw_magic_number_rejections_in_risk_engine_source() {
    let src = include_str!("../src/lib.rs");
    // Verificar que no queden llamadas a return rej(0) .. return rej(16) con números literales
    for i in 0..17 {
        let pattern_space = format!("return rej({});", i);
        let pattern_padded = format!("return rej( {} );", i);
        assert!(
            !src.contains(&pattern_space) && !src.contains(&pattern_padded),
            "src/lib.rs no debe contener literales mágicos para rej({}): encontrado en código fuente",
            i
        );
    }
}

#[test]
fn test_conservative_ruin_loss_q_protects_viability_gate_from_naive_single_win() {
    // 1 trade ganador con wr=1.0: un cálculo ingenuo daría q = 0.0 (o 0.01)
    // Con conservative_loss_q vía Jeffreys Beta LCB, la cota inferior conservadora
    // produce un q robusto en [0.20, 0.80] protegiendo al capital de sobre-exposición
    let q_single_win = risk_engine::ruin::conservative_loss_q(1.0, 1.0);
    assert!(
        q_single_win >= 0.20,
        "1 solo trade con wr=1.0 no debe producir q ingenuamente nulo: q={q_single_win}"
    );
    assert!(
        q_single_win <= 0.80,
        "q conservador debe estar acotado: q={q_single_win}"
    );

    // Cero trades: retorna CONSERVATIVE_Q exactamente
    let q_zero = risk_engine::ruin::conservative_loss_q(0.0, 0.0);
    assert_eq!(q_zero, risk_engine::ruin::CONSERVATIVE_Q);
}

#[test]
fn test_micro_capital_13_usd_invariants_never_vetoed_by_viability() {
    // Invariantes sagrados Binance Futures para $13.00 USD:
    let capital = 13.00;
    let min_notional = 5.10;
    let sl_pct = 0.0055; // 55 bps

    // Con q conservador 0.60:
    let q = risk_engine::ruin::CONSERVATIVE_Q;
    let tope_riesgo = risk_engine::ruin::clamp_ruin(1.0, q);
    assert!(tope_riesgo >= 0.10, "tope_riesgo debe ser al menos 10%");

    let viable = risk_engine::capital_regime::orden_viable(
        min_notional,
        sl_pct,
        capital,
        tope_riesgo,
    );
    assert!(
        viable,
        "La orden mínima estándar de Binance ($5.10 nocional, 55 bps SL) DEBE ser viable con $13.00 USD"
    );
}
