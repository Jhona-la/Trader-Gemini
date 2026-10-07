//! C07: recovered RUIN-T02/03/04 consumer contracts; no new execution receipt yet.
//! Origin: codex-review-2026-10-07, HEAD a054495c70c54d83190337355aa35376cfae3c15.
//! Original SHA-256: 46c75f82e2465bb38c54668470c6479040c479e8426bac01c87ca6a285d21752.
//! Recovery R4 (2026-10-07): only provenance header changed; test bodies retained.
//! RED/GREEN on the integration tree is pending; the original "rerun" label is not evidence.
//! No engine process, model, tape, exchange or claim of economic performance.
use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, ValidatedOrder, REJECT_COUNTERS_DIR, REJ_INVALID_INPUT};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex, Once};

// Admission diagnostics are process-global. Keep this binary's observations
// serial; other test executables do not share its counters or registry.
static SERIAL: Mutex<()> = Mutex::new(());

fn fixture(capital: f64, side: SignalType) -> (Arc<GlobalArena>, SignalIntent) {
    static REGISTRY: Once = Once::new();
    REGISTRY.call_once(|| {
        quantum_arena::symbol_registry::update_registry(vec![
            quantum_arena::symbol_registry::SymbolSpec {
                symbol: "BTCUSDT".into(), step_size: 0.001, tick_size: 0.01,
                min_qty: 0.001, min_notional: 5.0, max_leverage: 20,
                maker_fee: 0.0002, taker_fee: 0.0004, is_shadow: false,
            },
        ]);
    });
    let arena = GlobalArena::build_in_own_stack(capital);
    let coin = &arena.coins[0];
    coin.current_price.store(100.0, Relaxed);
    coin.current_atr.store(1.0, Relaxed);
    coin.hurst_exponent.store(0.5, Relaxed);
    coin.metrics.trade_count.store(100, Relaxed);
    coin.metrics.profit_factor.store(2.0, Relaxed);
    coin.metrics.win_rate.store(0.75, Relaxed);
    coin.metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.min_confidence_btc.store(0.62, Relaxed);
    let dir = if side == SignalType::Long { 1.0 } else { -1.0 };
    let intent = SignalIntent {
        signal: side, confidence: 0.9, win_probability: 0.9,
        expected_duration_ms: 60_000,
        tp_price_target: 100.0 + dir * 2.0,
        sl_price_target: 100.0 - dir,
        ..SignalIntent::default()
    };
    (arena, intent)
}

fn assert_healthy(capital: f64, side: SignalType) {
    let (arena, intent) = fixture(capital, side);
    let order = RiskEngine::new(capital).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(order.signal, side, "healthy C={capital}: {}", risk_engine::reject_report());
    assert!(order.volume_usd.is_finite() && order.volume_usd > 0.0);
    assert!(order.integer_leverage().is_some());
    assert!(order.volume_usd * order.leverage >= 5.0);
    assert_eq!(order.tp_target, intent.tp_price_target);
    assert_eq!(order.sl_target, intent.sl_price_target);
}

fn assert_rejected(order: ValidatedOrder, capital: f64, side: SignalType) {
    assert_eq!(order.signal, SignalType::Flat, "C={capital}, side={side:?}");
    assert_eq!(order.volume_usd.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn c07_t02_healthy_controls_admit_all_capitals_and_directions() {
    let _serial = SERIAL.lock().unwrap();
    for capital in [13.0, 25.0, 100.0] {
        for side in [SignalType::Long, SignalType::Short] {
            assert_healthy(capital, side);
        }
    }
}

#[test]
fn c07_t02_nonfinite_micro_product_cannot_reopen_through_lerp() {
    let _serial = SERIAL.lock().unwrap();
    let mut admitted = Vec::new();
    let mut wrong_rejection = Vec::new();
    let mut changed_risk = Vec::new();
    for capital in [13.0, 25.0, 100.0] {
        for side in [SignalType::Long, SignalType::Short] {
            assert_healthy(capital, side);
            let (arena, intent) = fixture(capital, side);
            arena.config.min_confidence_btc.store(f64::NEG_INFINITY, Relaxed);
            let direction = if side == SignalType::Long { 0 } else { 1 };
            let before = REJECT_COUNTERS_DIR[direction][REJ_INVALID_INPUT].load(Relaxed);
            let risk_before = arena.riesgo_por_operacion.load(Relaxed).to_bits();
            let order = RiskEngine::new(capital).evaluate_quantum_order(0, &intent, &arena);
            if order.signal != SignalType::Flat || order.volume_usd.to_bits() != 0.0_f64.to_bits() {
                admitted.push((capital, side, order.signal, order.volume_usd));
            }
            if REJECT_COUNTERS_DIR[direction][REJ_INVALID_INPUT].load(Relaxed) != before + 1 {
                wrong_rejection.push((capital, side));
            }
            if arena.riesgo_por_operacion.load(Relaxed).to_bits() != risk_before {
                changed_risk.push((capital, side));
            }
        }
    }
    // Evaluate all six cases before failing: a micro-full exposure0 rejection
    // must not hide the helper-only reopenings at partial/zero micro weight.
    assert!(admitted.is_empty(), "nonfinite product admitted: {admitted:?}");
    assert!(wrong_rejection.is_empty(), "missing invalid-input diagnostic: {wrong_rejection:?}");
    assert!(changed_risk.is_empty(), "rejected input changed risk EWMA: {changed_risk:?}");
}

#[test]
fn c07_t03_nan_raw_kelly_cannot_be_rescued_by_minimum_notional() {
    let _serial = SERIAL.lock().unwrap();
    for capital in [13.0, 25.0, 100.0] {
        for side in [SignalType::Long, SignalType::Short] {
            assert_healthy(capital, side);
            let (arena, intent) = fixture(capital, side);
            arena.coins[0].metrics.kelly_fraction.store(f64::NAN, Relaxed);
            let risk_before = arena.riesgo_por_operacion.load(Relaxed).to_bits();
            let order = RiskEngine::new(capital).evaluate_quantum_order(0, &intent, &arena);
            assert_rejected(order, capital, side);
            assert_eq!(arena.riesgo_por_operacion.load(Relaxed).to_bits(), risk_before);
        }
    }
}

#[test]
fn c07_t04_overflowed_envelope_minimum_remains_nonoperable() {
    let env = risk_engine::RiskEnvelope::new();
    // Finite inputs; only their intermediate product overflows.
    assert!((f64::MAX * 2.0 / 13.0).is_infinite());
    assert_eq!(env.max_leverage(13.0, 2.0, f64::MAX, 0.85, 10.0), (0.0, false));
    // Preserve the explicit, finite minimum-probe policy.
    let (ratio, allowed) = env.max_leverage(13.0, 0.005, 5.0, 0.85, 10.0);
    assert!(allowed);
    assert!((ratio - 5.0 / 13.0).abs() < 1e-12);
}
