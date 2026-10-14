//! CONTRATO FORMAL: ADAPTABILIDAD DINÁMICA DE PARÁMETROS DEL CONSEJO (Ola Ω101).
//!
//! # Por qué existe
//! Erradica la petrificación heurística del Consejo de Deliberación (`metacortex-engine`).
//! Anteriormente, `CouncilParams` solo podía construirse mediante constantes congeladas en `Default::default()`,
//! impidiendo la inyección formal de configuraciones validadas generadas por evolución adaptativa o
//! por el orquestador lock-free del Arena.
//!
//! # Solución (Ola Ω101)
//! 1. `CouncilParams::new` valida cotas estrictas ([0, 1] y k > 0 finito) antes de la instanciación.
//! 2. `ConsejoDeliberacion::with_params` construye el tribunal supremo de deliberación directamente
//!    con la política evaluada, respondiendo con fidelidad determinista y diagnóstico en caso de error.

use metacortex_engine::consejo_seniors::{
    ConsejoDeliberacion, CouncilParams, MarketSnapshotPayload, SeniorRole, TradingHorizon,
};

fn test_snapshot() -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        horizon: TradingHorizon::Continuous,
        book_imbalance: 0.65,
        hurst_exponent: 0.60,
        ml_prob: 0.65,
        fused_score: 0.40,
        persistence: 0.55,
        atr_pct: 0.001,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.05,
        causal_veto_threshold: 0.75,
        current_drawdown_pct: 0.02,
        estimated_slippage_bps: 1.0,
        dominant_tau_ms: 30_000.0,
        whale_burst_z: 0.0,
        liquidation_severity: 0.0,
        open_interest_norm: 0.0,
        spoof_score: 0.0,
        crowd_ls_ratio: 1.0,
        crowd_taker_ratio: 1.0,
        ml_model_base: 0.5,
        ..Default::default()
    }
}

#[test]
fn ola101_council_params_new_valida_cotas_correctamente() {
    // 1. Instanciación válida
    let valid = CouncilParams::new(0.30, 0.75, 0.25, 0.70, 10.0, 0.80);
    assert!(valid.is_ok(), "Parámetros válidos deben construirse sin error");
    let p = valid.unwrap();
    assert_eq!(p.approval_threshold, 0.30);
    assert_eq!(p.supermajority_override, 0.75);
    assert_eq!(p.override_signal_floor, 0.25);
    assert_eq!(p.override_penalty, 0.70);
    assert_eq!(p.shrinkage_k, 10.0);
    assert_eq!(p.cascade_severity_breaker, 0.80);

    // 2. Validación de cotas inválidas
    assert!(CouncilParams::new(-0.01, 0.80, 0.28, 0.75, 8.0, 0.85).is_err());
    assert!(CouncilParams::new(1.01, 0.80, 0.28, 0.75, 8.0, 0.85).is_err());
    assert!(CouncilParams::new(0.35, f64::NAN, 0.28, 0.75, 8.0, 0.85).is_err());
    assert!(CouncilParams::new(0.35, 0.80, -0.10, 0.75, 8.0, 0.85).is_err());
    assert!(CouncilParams::new(0.35, 0.80, 0.28, 1.50, 8.0, 0.85).is_err());
    assert!(CouncilParams::new(0.35, 0.80, 0.28, 0.75, 0.0, 0.85).is_err());
    assert!(CouncilParams::new(0.35, 0.80, 0.28, 0.75, -5.0, 0.85).is_err());
    assert!(CouncilParams::new(0.35, 0.80, 0.28, 0.75, f64::INFINITY, 0.85).is_err());
    assert!(CouncilParams::new(0.35, 0.80, 0.28, 0.75, 8.0, 1.20).is_err());
}

#[test]
fn ola101_consejo_deliberacion_with_params_adopta_politica_dinamica() {
    let custom_params = CouncilParams::new(0.20, 0.70, 0.20, 0.60, 6.0, 0.60)
        .expect("Parámetros válidos");

    let consejo = ConsejoDeliberacion::with_params(custom_params)
        .expect("Consejo debe construirse con parámetros válidos");

    assert_eq!(consejo.params.approval_threshold, 0.20);
    assert_eq!(consejo.params.cascade_severity_breaker, 0.60);

    // Con breaker en 0.60, una severidad de 0.65 debe disparar veto de EnteMercado
    let mut p = test_snapshot();
    p.liquidation_severity = 0.65;
    let res = consejo.deliberar(&p, 0.60);
    assert!(!res.approved, "Breaker adaptativo a 0.60 debe vetar severidad 0.65");
    assert_eq!(res.vetoed_by, Some(SeniorRole::EnteMercado));
}

#[test]
fn ola101_consejo_deliberacion_with_params_falla_ante_parametros_corruptos() {
    let corrupt_params = CouncilParams {
        approval_threshold: 2.50, // Corrupto > 1.0
        supermajority_override: 0.80,
        override_signal_floor: 0.28,
        override_penalty: 0.75,
        shrinkage_k: 8.0,
        cascade_severity_breaker: 0.85,
        utility_veto_floor: 0.02,
        utility_veto_wr: 0.35,
        fast_slippage_floor: 35.0,
        slow_slippage_span: 65.0,
        max_drawdown_limit: 0.6315968501359614,
        streak_decay_rate: 0.50,
        causal_veto_threshold: 0.75,
        causal_min_book_imbalance: 0.25,
        causal_aligned_max_risk: 0.88,
        hodge_curl_attenuation: 0.70,
        hodge_laminar_floor: 0.30,
        ym_ref_current: 0.05,
        ym_max_weight: 0.20,
        ml_dead_zone: 0.05,
        ml_full_edge: 0.20,
        meta_dd_penalty_mult: 5.0,
        meta_max_dd_penalty: 0.50,
        meta_majority_factor: 0.70,
        meta_solitary_factor: 0.40,
        whale_z_threshold: 4.0,
        whale_min_floor: 0.60,
        liq_decay_mult: 1.2,
        liq_min_floor: 0.40,
        crowd_imbalance_decay: 0.25,
        crowd_min_floor: 0.70,
        devil_advocate_weight: 3.0,
        discrepancy_veto_threshold: 0.20,
        tracker_min_trials: 5,
        tracker_weight_floor: 0.50,
        tracker_weight_cap: 2.0,
    };

    match ConsejoDeliberacion::with_params(corrupt_params) {
        Err(err) => assert!(err.contains("Invalid council parameter approval_threshold")),
        Ok(_) => panic!("with_params debe fallar ante parámetros corruptos"),
    }
}
