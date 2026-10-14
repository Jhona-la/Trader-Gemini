//! CONTRATO FORMAL: DESACOPLAMIENTO DINÁMICO DE TOLERANCIAS DE DRAWDOWN Y RACHA EN SENIOR RIESGO (Ola Ω114).
//!
//! # Foco Cuántico / Causa Raíz
//! En `SeniorRiesgo::evaluate` (`crates/metacortex-engine/src/consejo_seniors.rs:815-852`), las tolerancias
//! de drawdown extremo y penalización por racha de pérdidas estaban atadas a constantes estáticas:
//! 1. `d_star = 1.0 - (0.05_f64).powf(1.0 / 3.0) ≈ 0.6316` (falsación rígida de ½ Kelly).
//! 2. `max_drawdown = d_star * (0.95 - 0.10 * s)` (piso de 60.0% en fast scalping y 53.68% en swing).
//!    Para una cuenta de micro-capital ($13 USD), permitir un drawdown de más del 50% antes de vetar
//!    es destructivo y anula la aceleración geométrica del capital.
//! 3. `streak_factor = 1.0 / (1.0 + 0.5 * streak as f64)` (penalización estática de 0.5 por racha).
//!
//! # Solución Matemática y Continua (Ola Ω114)
//! - Implementada la función pura `evaluate_risk_drawdown_and_conviction` con sanitización fail-closed.
//! - Agregados campos dinámicos `max_drawdown_limit` y `streak_decay_rate` a `MarketSnapshotPayload` y `CouncilParams`.
//! - Builder `with_risk_bounds` en `CouncilParams` con validación de cotas.
//! - Propagación lock-free `Cow` en `deliberar_traced`.
//! - Sincronización atómica en `QuantumConfig` (`council_max_drawdown_limit`, `council_streak_decay_rate`).
//! - Resolución omnisciente por activo en `council_snapshot` (`c{id}:max_drawdown_limit`, `c{id}:streak_decay_rate`).

use metacortex_engine::consejo_seniors::{CouncilParams, SeniorAgent, SeniorRiesgo};
use metacortex_engine::{
    evaluate_risk_drawdown_and_conviction, ConsejoDeliberacion, MarketSnapshotPayload, SeniorRole,
    DEFAULT_MAX_DRAWDOWN_LIMIT, DEFAULT_STREAK_DECAY_RATE,
};

fn base_risk_payload() -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        intended_direction: 1.0,
        loss_streak: 0,
        current_drawdown_pct: 0.05,
        dominant_tau_ms: 30_000.0, // Fast scalping (s = 0.0)
        ..Default::default()
    }
}

#[test]
fn test_risk_opinion_canonical_defaults_exact_parity() {
    let mut payload = base_risk_payload();
    payload.dominant_tau_ms = 30_000.0; // s = 0.0
    payload.current_drawdown_pct = 0.10;
    payload.loss_streak = 2;

    let op = SeniorRiesgo.evaluate(&payload, 0.55);
    assert_eq!(op.role, SeniorRole::Riesgo);
    assert!(!op.is_veto);

    // Paridad exacta con la función pura
    let (conf_pure, is_veto_pure, max_dd_pure) = evaluate_risk_drawdown_and_conviction(
        payload.current_drawdown_pct,
        payload.loss_streak,
        payload.spectral_s(),
        payload.intended_direction,
        DEFAULT_MAX_DRAWDOWN_LIMIT,
        DEFAULT_STREAK_DECAY_RATE,
    );
    assert_eq!(op.confidence, conf_pure);
    assert_eq!(op.is_veto, is_veto_pure);

    // En fast scalping (s = 0.0): max_dd = d* * 0.95
    let expected_max_dd_fast = DEFAULT_MAX_DRAWDOWN_LIMIT * 0.95;
    assert!((max_dd_pure - expected_max_dd_fast).abs() < 1e-12);

    // En slow swing (s = 1.0, tau = 12h = 43_200_000 ms): max_dd = d* * 0.85
    let mut payload_slow = payload.clone();
    payload_slow.dominant_tau_ms = 43_200_000.0;
    let (_c_slow, _v_slow, max_dd_slow) = evaluate_risk_drawdown_and_conviction(
        payload_slow.current_drawdown_pct,
        payload_slow.loss_streak,
        payload_slow.spectral_s(),
        payload_slow.intended_direction,
        DEFAULT_MAX_DRAWDOWN_LIMIT,
        DEFAULT_STREAK_DECAY_RATE,
    );
    let expected_max_dd_slow = DEFAULT_MAX_DRAWDOWN_LIMIT * 0.85;
    assert!((max_dd_slow - expected_max_dd_slow).abs() < 1e-12);

    // Convicción con racha = 2: streak_factor = 1 / (1 + 0.5 * 2) = 0.5
    // conviction = 0.5 * (1.0 - 0.10) = 0.45
    assert!((conf_pure - 0.45).abs() < 1e-12);
}

#[test]
fn test_risk_opinion_dynamic_payload_override() {
    let mut payload = base_risk_payload();
    payload.dominant_tau_ms = 30_000.0;
    payload.current_drawdown_pct = 0.28; // 28% drawdown
    payload.loss_streak = 0;

    // 1. Con defaults (d* ≈ 0.6316 -> max_dd = 0.60): 28% DD no es veto
    let op_default = SeniorRiesgo.evaluate(&payload, 0.55);
    assert!(!op_default.is_veto, "Con límite canónico ~60%, 28% DD no debe vetar");

    // 2. Con override adaptativo de micro-capital (d* = 0.25 -> max_dd = 0.25 * 0.95 = 0.2375)
    payload.max_drawdown_limit = 0.25;
    let op_micro = SeniorRiesgo.evaluate(&payload, 0.55);
    assert!(
        op_micro.is_veto,
        "Con límite adaptativo de micro-capital 25%, 28% DD DEBE disparar veto de protección"
    );

    // 3. Modulación adaptativa de racha de pérdidas
    payload.current_drawdown_pct = 0.05;
    payload.loss_streak = 3;
    payload.streak_decay_rate = 1.0; // Penalización severa por racha

    let op_fast_decay = SeniorRiesgo.evaluate(&payload, 0.55);
    // streak_factor = 1 / (1 + 1.0 * 3) = 0.25 -> convicción = 0.25 * 0.95 = 0.2375
    assert!((op_fast_decay.confidence - 0.2375).abs() < 1e-12);

    // Con decay suave (0.10): streak_factor = 1 / (1 + 0.1 * 3) = 1 / 1.3 ≈ 0.7692
    payload.streak_decay_rate = 0.10;
    let op_slow_decay = SeniorRiesgo.evaluate(&payload, 0.55);
    assert!(op_slow_decay.confidence > op_fast_decay.confidence);
}

#[test]
fn test_risk_opinion_council_params_propagation() {
    let consejo = ConsejoDeliberacion::with_params(
        CouncilParams::default()
            .with_risk_bounds(0.20, 0.80)
            .expect("Parámetros válidos"),
    )
    .expect("Consejo inicializado");

    let mut payload = base_risk_payload();
    payload.current_drawdown_pct = 0.22; // 22% DD

    let trace = consejo.deliberar_traced(&payload, 0.55, None);
    assert!(
        !trace.consensus.approved,
        "Drawdown de 22% debe vetarse con max_drawdown_limit=0.20 propagado desde CouncilParams"
    );
    assert_eq!(trace.consensus.vetoed_by, Some(SeniorRole::Riesgo));

    let riesgo_op = trace
        .opinions
        .iter()
        .find(|o| o.role == SeniorRole::Riesgo)
        .expect("Opinión de Riesgo presente");
    assert!(riesgo_op.is_veto);
}

#[test]
fn test_risk_opinion_bounds_validation_and_fail_closed() {
    // 1. Validación en CouncilParams
    assert!(CouncilParams::default().with_risk_bounds(0.50, 1.0).is_ok());
    assert!(CouncilParams::default().with_risk_bounds(-0.1, 1.0).is_err());
    assert!(CouncilParams::default().with_risk_bounds(0.0, 1.0).is_err());
    assert!(CouncilParams::default().with_risk_bounds(1.5, 1.0).is_err());
    assert!(CouncilParams::default().with_risk_bounds(0.5, -1.0).is_err());
    assert!(CouncilParams::default().with_risk_bounds(0.5, 15.0).is_err());

    // 2. Validación en MarketSnapshotPayload
    let mut payload = base_risk_payload();
    assert!(payload.validate().is_ok());

    payload.max_drawdown_limit = -0.05;
    assert!(payload.validate().is_err());

    payload.max_drawdown_limit = 0.0;
    assert!(payload.validate().is_err());

    payload.max_drawdown_limit = 1.20;
    assert!(payload.validate().is_err());

    payload.max_drawdown_limit = f64::NAN;
    assert!(payload.validate().is_err());

    payload.max_drawdown_limit = 0.50;
    payload.streak_decay_rate = -0.5;
    assert!(payload.validate().is_err());

    payload.streak_decay_rate = 12.0;
    assert!(payload.validate().is_err());

    // 3. Robustez numérica de evaluate_risk_drawdown_and_conviction ante NaN/Infinitos
    let (conf_nan, veto_nan, max_dd_nan) =
        evaluate_risk_drawdown_and_conviction(f64::NAN, 0, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(conf_nan.is_finite());
    assert!(max_dd_nan.is_finite());
    assert!(!veto_nan);
    assert_eq!(conf_nan, 0.0); // intended_direction NaN -> signum 0.0 -> conf 0.0
}

#[test]
fn test_risk_opinion_spectral_continuity_and_monotonicity() {
    let d_star = 0.50;
    let streak_decay = 0.50;

    // Verificar que max_drawdown disminuye monótonamente en s ∈ [0, 1]
    let mut prev_max_dd = f64::INFINITY;
    for step in 0..=100 {
        let s = step as f64 / 100.0;
        let (_conf, _veto, max_dd) =
            evaluate_risk_drawdown_and_conviction(0.10, 0, s, 1.0, d_star, streak_decay);
        assert!(max_dd <= prev_max_dd + 1e-12, "max_drawdown debe ser monótono no-creciente en s");
        prev_max_dd = max_dd;
    }
}
