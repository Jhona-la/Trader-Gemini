//! TRIAJE B (GLM 106): los 4 tests open_ de veto guards DRENADOS — las
//! superficies auxiliares ahora son fail-closed con caps honrados. Las
//! doctrinas: pico desconocido ≠ seguro; capital desconocido = veto; un
//! cap explícito es la palabra del dueño (jamás elevado por pisos).
//! Los gemelos VIVOS (lib.rs:279 peak-NaN del veto inline) quedan
//! registrados en TRIAJE_ROJOS_PERPETUOS como ola futura con oráculo.
//! FMT-214 (línea 41) sigue siendo la regresión cerrada.
use risk_engine::{correlation_guard::CorrelationGuardEngine, guard};

#[test]
fn auxiliary_drawdown_guard_fails_closed_on_unknown_peak() {
    // Peak NaN = drawdown no medible ⇒ NO asumir seguro (antes: true).
    assert!(
        !guard::check_drawdown_limit(13.0, f64::NAN, 0.05, 13.0, 1.0, 1.0, 5.0),
        "pico desconocido no fabrica permiso"
    );
    // Peak 0 contra capital positivo = 100% de drawdown consumado.
    assert!(!guard::check_drawdown_limit(13.0, 0.0, 0.05, 13.0, 1.0, 1.0, 5.0));
}

#[test]
fn correlation_helper_fails_closed_for_unknown_capital() {
    // Capital NaN = presupuesto de ruina desconocido ⇒ veto (antes: caía
    // silenciosamente al bootstrap 13.0 y dejaba pasar).
    assert!(
        CorrelationGuardEngine::is_continuous_correlation_vetoed(1, f64::NAN, 5.0, 10),
        "capital desconocido no dimensiona un clúster"
    );
}

#[test]
fn cluster_policy_honors_an_explicit_cap_of_one() {
    // Cap 1: la PRIMERA posición misma-apuesta ya cumple el umbral de veto
    // (antes: el piso .max(2) autorizaba una de más).
    assert!(
        CorrelationGuardEngine::is_continuous_correlation_vetoed(1, 1000.0, 5.0, 1),
        "un cap explícito de 1 se honra exacto"
    );
    // Y en régimen micro la interpolación D-641 no puede ELEVAR el cap.
    assert!(
        CorrelationGuardEngine::is_continuous_correlation_vetoed(1, 13.0, 5.0, 1),
        "ni el régimen micro sube un cap más estricto"
    );
}

#[test]
fn streak_policy_honors_an_explicit_cap_of_one() {
    // Cap 1: UNA pérdida ya bloquea (antes: .max(2) autorizaba la segunda).
    assert!(
        !guard::check_streak_drawdown_limit(1, 1),
        "una pérdida con cap 1 bloquea"
    );
    // Semántica general preservada: 0 pérdidas siempre pasa; con cap 2,
    // la segunda pérdida bloquea.
    assert!(guard::check_streak_drawdown_limit(0, 1));
    assert!(guard::check_streak_drawdown_limit(1, 2));
    assert!(!guard::check_streak_drawdown_limit(2, 2));
}

#[test]
fn fixed_minimum_notional_branch_preserves_integer_leverage() {
    use quantum_arena::{
        symbol_registry::{update_registry, SymbolSpec},
        GlobalArena,
    };
    use risk_engine::RiskEngine;
    use signal_engine::{SignalIntent, SignalType};
    use std::sync::atomic::Ordering::Relaxed;
    update_registry(vec![SymbolSpec {
        symbol: "BTCUSDT".into(),
        step_size: 0.001,
        tick_size: 0.01,
        min_qty: 0.001,
        min_notional: 5.0,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }]);
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    let out = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );
    assert_eq!(
        out.leverage.fract(),
        0.0,
        "FMT-214: every downstream leverage reassignment must remain integer"
    );
    println!(
        "FIXED FMT-214 integer contract: leverage={} margin={} exchange_floor={}",
        out.leverage,
        out.volume_usd,
        out.leverage.floor()
    );
}
