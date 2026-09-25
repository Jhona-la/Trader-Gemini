//! Final-leverage admission; offline synthetic state, no active genome writes.
use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

fn fixture(capital: f64) -> (Arc<GlobalArena>, SignalIntent) {
    static REGISTRY: Once = Once::new();
    REGISTRY.call_once(|| {
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
        }])
    });
    let a = GlobalArena::build_in_own_stack(capital);
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(1.0, Relaxed);
    a.coins[0].hurst_exponent.store(0.5, Relaxed);
    a.config.global_leverage.store(20.0, Relaxed);
    a.config.global_max_drawdown.store(0.2, Relaxed);
    a.config.kelly_clamp_min.store(0.01, Relaxed);
    a.config.kelly_clamp_max.store(0.25, Relaxed);
    a.config.latency_penalty_ms.store(0.0, Relaxed);
    a.config.live_taker_fee.store(0.0004, Relaxed);
    a.config.base_slippage_floor.store(0.00001, Relaxed);
    (
        a,
        SignalIntent {
            signal: SignalType::Long,
            confidence: 0.9,
            expected_duration_ms: 60_000,
            ..Default::default()
        },
    )
}

#[test]
fn micro_order_remains_feasible_with_integer_leverage() {
    let (a, intent) = fixture(13.0);
    let out = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &a);
    assert_eq!(
        out.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );
    assert_eq!(out.leverage.fract(), 0.0, "{out:?}");
    assert!(out.volume_usd * out.leverage >= 5.1);
    assert!(out.volume_usd <= 2.6);
}

#[test]
fn minimum_notional_rescue_never_overrides_genome_ceiling() {
    let (a, intent) = fixture(13.0);
    for cap in [1.0, 1.5, 2.0, 3.0, 4.0, 4.9, 5.0] {
        a.config.global_leverage.store(cap, Relaxed);
        let out = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &a);
        assert!(
            out.signal == SignalType::Flat || out.leverage <= cap.floor(),
            "cap={cap}, {out:?}"
        );
    }
}

#[test]
fn invalid_genomic_leverage_is_not_replaced_with_twenty() {
    let (a, intent) = fixture(13.0);
    for cap in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0, 0.9] {
        a.config.global_leverage.store(cap, Relaxed);
        assert_eq!(
            RiskEngine::new(13.0)
                .evaluate_quantum_order(0, &intent, &a)
                .signal,
            SignalType::Flat,
            "cap={cap}"
        );
    }
}

#[test]
fn final_fee_budget_applies_even_without_minimum_notional_rescue() {
    let (a, intent) = fixture(100.0);
    a.coins[0].metrics.trade_count.store(100, Relaxed);
    a.coins[0].metrics.kelly_fraction.store(0.15, Relaxed);
    a.coins[0].metrics.win_rate.store(0.75, Relaxed);
    a.coins[0].metrics.profit_factor.store(2.0, Relaxed);
    // Roundtrip fee=.00082; no leverage >=1 can satisfy this budget.
    a.config.max_fee_pct.store(0.0001, Relaxed);
    assert_eq!(
        RiskEngine::new(100.0)
            .evaluate_quantum_order(0, &intent, &a)
            .signal,
        SignalType::Flat
    );
}

#[test]
fn invalid_fee_budget_cannot_disable_the_veto() {
    let (a, intent) = fixture(100.0);
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        a.config.max_fee_pct.store(bad, Relaxed);
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &a)
                .signal,
            SignalType::Flat,
            "fee_budget={bad}"
        );
    }
}
