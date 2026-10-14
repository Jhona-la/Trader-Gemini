//! CONTRATO FORMAL DE ADAPTACIÓN DINÁMICA DE TOLERANCIAS DE SLIPPAGE DE EJECUCIÓN (Ola Ω113)
//!
//! Verifica que:
//! 1. Paridad canónica exacta con los defaults históricos (35.0 bps floor, 65.0 bps span).
//! 2. Desacoplamiento dinámico en `SeniorEjecucion::evaluate` mediante `payload.fast_slippage_floor` y `payload.slow_slippage_span`.
//! 3. Propagación transparente y lock-free desde `CouncilParams::with_slippage_bounds` en `deliberar_traced`.
//! 4. Validación de cotas estricta ([0.0, 500.0] floor, [0.0, 1000.0] span) e integridad contra NaN/Infinitos.

use metacortex_engine::consejo_seniors::{
    evaluate_execution_slippage, evaluate_execution_slippage_with_bounds, ConsejoDeliberacion,
    CouncilParams, MarketSnapshotPayload, SeniorAgent, SeniorEjecucion, SeniorRole, TradingHorizon,
};

fn base_execution_payload() -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        horizon: TradingHorizon::Continuous,
        book_imbalance: 0.10,
        hurst_exponent: 0.55,
        ml_prob: 0.55,
        fused_score: 0.20,
        persistence: 0.30,
        atr_pct: 0.0010, // 10 bps ATR -> atr_mult = 1.0
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.10,
        causal_veto_threshold: 0.75,
        current_drawdown_pct: 0.0,
        estimated_slippage_bps: 40.0, // 40 bps
        dominant_tau_ms: 30_000.0,   // Fast scalping tau = 30s (s = 0.0)
        whale_burst_z: 0.0,
        liquidation_severity: 0.0,
        open_interest_norm: 0.0,
        spoof_score: 0.0,
        crowd_ls_ratio: 1.0,
        crowd_taker_ratio: 1.0,
        ml_model_base: 0.5,
        navier_reynolds_number: 0.0,
        navier_laminar_share: 1.0, // Flujo laminar puro -> effective_slippage = 40.0 bps
        ..Default::default()
    }
}

#[test]
fn test_execution_slippage_canonical_defaults_exact_parity() {
    let mut payload = base_execution_payload();
    payload.estimated_slippage_bps = 30.0;

    // Con defaults (floor = 35.0 bps, atr_mult = 1.0): 30 bps <= 35 bps -> No veto
    let op_default = SeniorEjecucion.evaluate(&payload, 0.55);
    assert_eq!(op_default.role, SeniorRole::Ejecucion);
    assert!(!op_default.is_veto, "30 bps debe ser admitido bajo el floor canónico de 35 bps");

    let (eff, max_slip, veto) = evaluate_execution_slippage(30.0, 1.0, 0.0, 0.0010);
    assert_eq!(veto, op_default.is_veto);
    assert!((eff - 30.0).abs() < 1e-12);
    assert!((max_slip - 35.0).abs() < 1e-12);

    // Con slippage = 40.0 bps: 40 bps > 35 bps -> Veto canónico
    payload.estimated_slippage_bps = 40.0;
    let op_veto = SeniorEjecucion.evaluate(&payload, 0.55);
    assert!(op_veto.is_veto, "40 bps debe ser vetado bajo el floor canónico de 35 bps");
}

#[test]
fn test_execution_slippage_dynamic_payload_override() {
    let mut payload = base_execution_payload();
    payload.estimated_slippage_bps = 40.0; // En defaults produce veto

    // 1. Verificamos que con defaults porta veto
    let op_base = SeniorEjecucion.evaluate(&payload, 0.55);
    assert!(op_base.is_veto);

    // 2. Modulamos dinámicamente el floor en el payload para un activo con mayor spread natural (ej. 50.0 bps)
    payload.fast_slippage_floor = 50.0;
    let op_relaxed = SeniorEjecucion.evaluate(&payload, 0.55);
    assert!(
        !op_relaxed.is_veto,
        "Con floor expandido a 50 bps, una orden con 40 bps debe ser admitida sin veto"
    );

    // 3. Modulamos a un régimen estricto de micro-scalping de cero tolerancia (ej. floor = 20.0 bps)
    payload.estimated_slippage_bps = 25.0;
    payload.fast_slippage_floor = 20.0;
    let op_strict = SeniorEjecucion.evaluate(&payload, 0.55);
    assert!(
        op_strict.is_veto,
        "Con floor estricto de 20 bps, una orden con 25 bps debe ser vetada"
    );
}

#[test]
fn test_execution_slippage_council_params_propagation() {
    let mut payload = base_execution_payload();
    payload.estimated_slippage_bps = 40.0; // Con floor canónico 35.0 se veta

    // Con consejo default: deliberación detecta veto de SeniorEjecucion
    let consejo_default = ConsejoDeliberacion::new();
    let trace_default = consejo_default.deliberar_traced(&payload, 0.60, None);
    let ejecucion_op_default = trace_default
        .opinions
        .iter()
        .find(|op| op.role == SeniorRole::Ejecucion)
        .expect("SeniorEjecucion debe emitir opinión");
    assert!(ejecucion_op_default.is_veto, "Debe ser vetado en consejo default");

    // Con consejo adaptado vía builder `with_slippage_bounds`:
    let params_custom = CouncilParams::default()
        .with_slippage_bounds(60.0, 100.0)
        .expect("Parámetros válidos");
    let consejo_custom = ConsejoDeliberacion::with_params(params_custom).expect("Consejo válido");

    let trace_custom = consejo_custom.deliberar_traced(&payload, 0.60, None);
    let ejecucion_op_custom = trace_custom
        .opinions
        .iter()
        .find(|op| op.role == SeniorRole::Ejecucion)
        .expect("SeniorEjecucion debe emitir opinión");
    assert!(
        !ejecucion_op_custom.is_veto,
        "Con floor=60 bps propagado desde CouncilParams, el slippage de 40 bps no debe ser vetado"
    );
}

#[test]
fn test_execution_slippage_bounds_validation_and_fail_closed() {
    // 1. Validación en CouncilParams
    let valid_params = CouncilParams::default().with_slippage_bounds(100.0, 200.0);
    assert!(valid_params.is_ok());

    let invalid_floor = CouncilParams::default().with_slippage_bounds(-5.0, 100.0);
    assert!(invalid_floor.is_err());

    let overflow_floor = CouncilParams::default().with_slippage_bounds(600.0, 100.0);
    assert!(overflow_floor.is_err());

    let overflow_span = CouncilParams::default().with_slippage_bounds(35.0, 1500.0);
    assert!(overflow_span.is_err());

    // 2. Validación en MarketSnapshotPayload
    let mut payload = base_execution_payload();
    assert!(payload.validate().is_ok());

    payload.fast_slippage_floor = -10.0;
    assert!(payload.validate().is_err());

    payload.fast_slippage_floor = 600.0;
    assert!(payload.validate().is_err());

    payload.fast_slippage_floor = f64::NAN;
    assert!(payload.validate().is_err());

    // 3. Robustez numérica de evaluate_execution_slippage_with_bounds
    let (eff_nan, max_nan, _veto_nan) =
        evaluate_execution_slippage_with_bounds(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(eff_nan.is_finite());
    assert!(max_nan.is_finite());
    assert_eq!(eff_nan, 0.0);
}
