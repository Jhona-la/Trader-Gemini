//! Contrato F1-118 (Ola Ω155): Geometría TP/SL y admisión de órdenes Fail-Closed ante ATR no positivo o no finito.
//!
//! TEORÍA Y DOCTRINA (Doctrina 3 / F1-118):
//! Un activo sin volatilidad medida (ATR <= 0 o NaN/Inf) no puede generar órdenes con geometría
//! «válida» ficticia de 50 pb. Si un tape corrupto o un activo recién añadido sin klines
//! llega a la evaluación de riesgo, `compute_tp_sl` debe marcar `below_tradeable_floor = true`
//! y `try_compute_tp_sl` debe devolver `None`.
//!
//! Asimismo, `evaluate_quantum_order` debe rechazar fail-closed con `REJ_INVALID_INPUT`
//! si `current_atr` o `atr_pct` es <= 0.0 o no finito, evitando que se dimensione apalancamiento
//! o stops sobre geometrías inexistentes.

use quantum_arena::GlobalArena;
use risk_engine::{
    compute_tp_sl, try_compute_tp_sl, ultimo_rechazo, RiskEngine, TpSlInputs, REJ_INVALID_INPUT,
};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

fn base_inputs(atr_ratio: f64) -> TpSlInputs {
    TpSlInputs {
        tau_ms: 60_000.0,
        atr_ratio,
        hurst: 0.5,
        roundtrip_fee: 0.0010,
        sl_atr_multiplier: 1.0,
        sigma_forecast: None,
    }
}

fn fixture() -> (Arc<GlobalArena>, SignalIntent) {
    std::env::set_var("TG_GENOME_ENV", "backtest");
    static REGISTRY: std::sync::Once = std::sync::Once::new();
    REGISTRY.call_once(|| {
        quantum_arena::symbol_registry::update_registry(vec![
            quantum_arena::symbol_registry::SymbolSpec {
                symbol: "BTCUSDT".into(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
        ]);
    });
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.coins[0].current_price.store(50_000.0, Relaxed);
    arena.coins[0].current_atr.store(250.0, Relaxed); // 0.5% ATR
    arena.coins[0].hurst_exponent.store(0.55, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.0, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.75, Relaxed);
    arena.coins[0].metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);

    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.85,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };

    (arena, intent)
}

#[test]
fn f1_118_compute_tp_sl_marks_below_tradeable_floor_on_invalid_atr() {
    let invalid_atrs = [0.0, -0.005, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

    for &bad_atr in &invalid_atrs {
        let inputs = base_inputs(bad_atr);
        let res = compute_tp_sl(inputs);
        assert!(
            res.below_tradeable_floor,
            "F1-118 VIOLADO: compute_tp_sl no marcó below_tradeable_floor para atr_ratio = {}",
            bad_atr
        );

        let try_res = try_compute_tp_sl(inputs);
        assert!(
            try_res.is_none(),
            "F1-118 VIOLADO: try_compute_tp_sl no devolvió None para atr_ratio = {}",
            bad_atr
        );
    }
}

#[test]
fn f1_118_try_compute_tp_sl_succeeds_on_valid_atr() {
    let valid_inputs = base_inputs(0.005);
    let try_res = try_compute_tp_sl(valid_inputs);
    assert!(
        try_res.is_some(),
        "try_compute_tp_sl debe devolver Some con ATR válido de 0.005"
    );
    let out = try_res.unwrap();
    assert!(out.tp_pct > 0.0);
    assert!(out.sl_pct > 0.0);
    assert!(!out.below_tradeable_floor);
}

#[test]
fn f1_118_evaluate_quantum_order_fails_closed_on_invalid_atr() {
    let (arena, intent) = fixture();
    let mut engine = RiskEngine::new(100.0);

    // Baseline: con precio y ATR válidos la orden debe procesarse (no ser rechazada por REJ_INVALID_INPUT)
    let baseline = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_ne!(
        baseline.signal,
        SignalType::Flat,
        "La orden base con ATR válido debe admitirse"
    );

    // 1. ATR cero
    arena.coins[0].current_atr.store(0.0, Relaxed);
    let out_zero_atr = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_zero_atr.signal,
        SignalType::Flat,
        "ATR cero debe rechazarse fail-closed a Flat"
    );
    assert_eq!(ultimo_rechazo(), Some(REJ_INVALID_INPUT));

    // 2. ATR negativo
    arena.coins[0].current_atr.store(-50.0, Relaxed);
    let out_neg_atr = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_neg_atr.signal,
        SignalType::Flat,
        "ATR negativo debe rechazarse fail-closed a Flat"
    );
    assert_eq!(ultimo_rechazo(), Some(REJ_INVALID_INPUT));

    // 3. ATR NaN
    arena.coins[0].current_atr.store(f64::NAN, Relaxed);
    let out_nan_atr = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_nan_atr.signal,
        SignalType::Flat,
        "ATR NaN debe rechazarse fail-closed a Flat"
    );
    assert_eq!(ultimo_rechazo(), Some(REJ_INVALID_INPUT));

    // 4. ATR Infinito
    arena.coins[0].current_atr.store(f64::INFINITY, Relaxed);
    let out_inf_atr = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_inf_atr.signal,
        SignalType::Flat,
        "ATR Infinito debe rechazarse fail-closed a Flat"
    );
    assert_eq!(ultimo_rechazo(), Some(REJ_INVALID_INPUT));
}
