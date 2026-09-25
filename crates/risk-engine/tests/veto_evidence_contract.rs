//! Admission contracts; no exchange, market data, or live engine.
use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn fixture() -> (std::sync::Arc<GlobalArena>, SignalIntent) {
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
        ])
    });
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
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
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    (arena, intent)
}

#[test]
fn finite_baseline_accepts_both_directions() {
    let (arena, mut intent) = fixture();
    for side in [SignalType::Long, SignalType::Short] {
        intent.signal = side;
        let out = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena);
        assert_eq!(out.signal, side, "{}", risk_engine::reject_report());
        assert!(out.volume_usd > 0.0);
    }
}

#[test]
fn target_on_wrong_side_or_at_entry_cannot_pass() {
    let (arena, mut intent) = fixture();
    for side in [SignalType::Long, SignalType::Short] {
        intent.signal = side;
        for (tp, sl) in if side == SignalType::Long {
            [(99.0, 99.0), (101.0, 101.0), (100.0, 99.0), (101.0, 100.0)]
        } else {
            [(101.0, 101.0), (99.0, 99.0), (100.0, 101.0), (99.0, 100.0)]
        } {
            intent.tp_price_target = tp;
            intent.sl_price_target = sl;
            assert_eq!(
                RiskEngine::new(100.0)
                    .evaluate_quantum_order(0, &intent, &arena)
                    .signal,
                SignalType::Flat,
                "side={side:?},tp={tp},sl={sl}"
            );
        }
    }
}

#[test]
fn actual_explicit_payoff_must_pass_the_ev_gate() {
    let (arena, mut intent) = fixture();
    // p=.9, gain=.0001, loss=.01 => gross EV=-.00091, before costs.
    for side in [SignalType::Long, SignalType::Short] {
        intent.signal = side;
        let d = if side == SignalType::Long { 1.0 } else { -1.0 };
        intent.tp_price_target = 100.0 + d * 0.01;
        intent.sl_price_target = 100.0 - d;
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat,
            "negative actual EV accepted, side={side:?}"
        );
    }
}

#[test]
fn valid_explicit_targets_are_preserved() {
    let (arena, mut intent) = fixture();
    for side in [SignalType::Long, SignalType::Short] {
        intent.signal = side;
        let d = if side == SignalType::Long { 1.0 } else { -1.0 };
        intent.tp_price_target = 100.0 + d * 2.0;
        intent.sl_price_target = 100.0 - d;
        let out = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena);
        assert_eq!(out.signal, side);
        assert_eq!(out.tp_target, intent.tp_price_target);
        assert_eq!(out.sl_target, intent.sl_price_target);
    }
}

#[test]
fn malformed_targets_are_not_interpreted_as_absent() {
    let (arena, mut intent) = fixture();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        for tp in [true, false] {
            intent.tp_price_target = if tp { bad } else { 0.0 };
            intent.sl_price_target = if tp { 0.0 } else { bad };
            assert_eq!(
                RiskEngine::new(100.0)
                    .evaluate_quantum_order(0, &intent, &arena)
                    .signal,
                SignalType::Flat,
                "target={bad},tp={tp}"
            );
        }
    }
}

#[test]
fn invalid_confidence_is_rejected_not_clamped_to_certainty() {
    let (arena, mut intent) = fixture();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        intent.confidence = bad;
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat,
            "confidence={bad}"
        );
    }
}

#[test]
fn malformed_kelly_interval_rejects_without_panicking() {
    let (arena, intent) = fixture();
    for (lo, hi) in [
        (1.1, 0.25),
        (0.3, 0.2),
        (f64::NAN, 0.25),
        (0.01, f64::INFINITY),
    ] {
        arena.config.kelly_clamp_min.store(lo, Relaxed);
        arena.config.kelly_clamp_max.store(hi, Relaxed);
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat,
            "Kelly [{lo},{hi}]"
        );
    }
}

#[test]
fn unknown_peak_cannot_disable_drawdown_protection() {
    let (arena, intent) = fixture();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert_eq!(
            RiskEngine::new(bad)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat,
            "peak={bad}"
        );
    }
}

#[test]
fn single_explicit_leg_uses_derived_other_leg() {
    let (arena, mut intent) = fixture();
    intent.tp_price_target = 102.0;
    let out = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out.signal, SignalType::Long);
    assert_eq!(out.tp_target, 102.0);
    assert!(out.sl_target > 0.0 && out.sl_target < 100.0);
    intent.tp_price_target = 0.0;
    intent.sl_price_target = 99.0;
    let out = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out.signal, SignalType::Long);
    assert_eq!(out.sl_target, 99.0);
    assert!(out.tp_target > 100.0);
}

#[test]
fn invalid_price_probability_and_fee_cannot_authorize_an_order() {
    let (arena, mut intent) = fixture();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        arena.coins[0].current_price.store(bad, Relaxed);
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat
        );
    }
    arena.coins[0].current_price.store(100.0, Relaxed);
    for bad in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        intent.win_probability = bad;
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat
        );
    }
    intent.win_probability = 0.0;
    for bad in [f64::NAN, f64::INFINITY] {
        arena.config.live_taker_fee.store(bad, Relaxed);
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat
        );
    }
}

#[test]
fn invalid_ev_threshold_cannot_disable_fee_hurdle() {
    let (arena, intent) = fixture();
    arena.config.ev_fee_multiplier.store(f64::NAN, Relaxed);
    assert_eq!(
        RiskEngine::new(100.0)
            .evaluate_quantum_order(0, &intent, &arena)
            .signal,
        SignalType::Flat
    );
}

#[test]
fn diagnostics_distinguish_input_geometry_and_economic_vetoes() {
    use risk_engine::{REJECT_COUNTERS_DIR, REJ_INVALID_INPUT, REJ_TARGET_GEOMETRY};
    let (arena, mut intent) = fixture();
    for (index, confidence, tp, sl) in [
        (REJ_INVALID_INPUT, f64::INFINITY, 0.0, 0.0),
        (REJ_TARGET_GEOMETRY, 0.9, 99.0, 99.0),
        (4, 0.9, 100.01, 99.0),
    ] {
        let before = REJECT_COUNTERS_DIR[0][index].load(Relaxed);
        intent.confidence = confidence;
        intent.tp_price_target = tp;
        intent.sl_price_target = sl;
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat
        );
        assert!(REJECT_COUNTERS_DIR[0][index].load(Relaxed) > before);
    }
}
