//! LG-5: el gate EV consume payoff neto contextual, no frecuencia de signos.
//! Estado offline; los pagos de cada fixture son retornos netos por nocional.
use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::{
    ultimo_rechazo, RiskEngine, ValidatedOrder, REJ_EV, REJ_INVALID_INPUT, REJ_SIN_EVIDENCIA,
};
use signal_engine::{NetPayoffEstimate, NetPayoffSource, SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

const FEE: f64 = 0.00082;

fn measured(values: &[f64]) -> NetPayoffEstimate {
    assert!(!values.is_empty());
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    NetPayoffEstimate {
        mean_net_return_pct: mean,
        standard_deviation: variance.sqrt(),
        sample_count: n,
        coin_id: 0,
        side: SignalType::Long,
        sample_start_ms: 1_000,
        sample_end_ms: n as u64 * 1_000,
        as_of_ms: n as u64 * 1_000,
        source: NetPayoffSource::LocalSimulation,
    }
}

fn fixture(capital: f64, trades: usize) -> (Arc<GlobalArena>, SignalIntent) {
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
    let arena = GlobalArena::build_in_own_stack(capital);
    // Explicit synthetic decision time; the largest cohort ends at 600_000.
    // evaluate() must never derive or advance this clock from net_payoff.
    arena.coins[0]
        .last_tick_timestamp_ms
        .store(600_000, Relaxed);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    let metrics = &arena.coins[0].metrics;
    metrics.trade_count.store(trades, Relaxed);
    metrics.win_rate.store(0.75, Relaxed);
    metrics.gross_wins.store(200.0, Relaxed);
    metrics.gross_losses.store(100.0, Relaxed);
    metrics.profit_factor.store(2.0, Relaxed);
    metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.global_leverage.store(20.0, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.min_confidence_btc.store(0.70, Relaxed);
    arena.config.ev_fee_multiplier.store(1.05, Relaxed);
    arena.config.max_fee_pct.store(0.01, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.55,
        expected_duration_ms: 60_000,
        tp_price_target: 102.25,
        sl_price_target: 99.0,
        ..Default::default()
    };
    (arena, intent)
}

fn evaluate(arena: &GlobalArena, intent: &SignalIntent) -> ValidatedOrder {
    RiskEngine::new(arena.unified_capital.load(Relaxed)).evaluate_quantum_order(0, intent, arena)
}

fn assert_rejected(arena: &GlobalArena, intent: &SignalIntent, reason: usize) {
    let order = evaluate(arena, intent);
    assert_eq!(order.signal, SignalType::Flat, "{order:?}");
    assert_eq!(
        ultimo_rechazo(),
        Some(reason),
        "{}",
        risk_engine::reject_report()
    );
}

fn assert_nonminimum_order(arena: &GlobalArena, intent: &SignalIntent) {
    assert_nonminimum_size(evaluate(arena, intent), intent.signal);
}

fn assert_nonminimum_size(order: ValidatedOrder, expected_side: SignalType) {
    assert_eq!(
        order.signal,
        expected_side,
        "{}",
        risk_engine::reject_report()
    );
    // El positivo debe atravesar la admisión con margen y nocional holgados
    // sobre la sonda mínima y sin alcanzar el techo de margen del fixture.
    // La matriz calcula L=1.608... con p=.55, PF_LCB=1.56256... y WR=.75;
    // el contrato del exchange lo cuantiza a 1. Ese entero es válido y no
    // convierte el margen admitido de 71.889... en una sonda mínima.
    assert!(order.volume_usd > 5.1, "margin at minimum: {order:?}");
    assert!(
        order.volume_usd < 200.0,
        "margin at capital ceiling: {order:?}"
    );
    assert!(
        order.leverage.is_finite()
            && order.leverage.fract() == 0.0
            && order.leverage >= 1.0
            && order.leverage < 20.0,
        "leverage outside admitted range: {order:?}"
    );
    assert!(
        order.volume_usd * order.leverage > 10.2,
        "notional at minimum: {order:?}"
    );
}

#[test]
fn lg5_many_small_net_wins_do_not_invent_positive_ev() {
    let (arena, mut intent) = fixture(1_000.0, 100);
    // Por 20: 3 TP brutos de 2.25R, 8 BE netos +.02R y 9 SL brutos -1R.
    // R=1% del nocional; TP/SL pagan FEE una vez. P(neto>0)=.55,
    // pero el retorno neto medio es -.001537 = -.1537R.
    let returns = (0..600)
        .map(|i| match i % 20 {
            0..=2 => 0.0225 - FEE,
            3..=10 => 0.0002,
            _ => -0.01 - FEE,
        })
        .collect::<Vec<_>>();
    let estimate = measured(&returns);
    assert!((estimate.mean_net_return_pct + 0.001537).abs() < 1e-12);
    assert!(
        0.55 * 0.0225 - 0.45 * 0.01 > 1.05 * FEE,
        "legacy gate would see positive EV"
    );
    intent.net_payoff = Some(estimate);
    assert_rejected(&arena, &intent, REJ_EV);
    intent.net_payoff = Some(measured(&[0.004; 100]));
    assert_nonminimum_order(&arena, &intent);
}

#[test]
fn lg5_net_return_pays_the_base_cost_once() {
    let (arena, mut intent) = fixture(1_000.0, 100);
    // .0002 neto supera el margen extra .05*FEE=.000041, pero no
    // 1.05*FEE=.000861: el segundo umbral duplicaría el coste base.
    assert!(0.0002 > 0.05 * FEE && 0.0002 < 1.05 * FEE);
    intent.net_payoff = Some(measured(&[0.0002; 100]));
    assert_nonminimum_order(&arena, &intent);
    intent.net_payoff = Some(measured(&[0.00002; 100]));
    assert_rejected(&arena, &intent, REJ_EV);
}

#[test]
fn lg5_unknown_zero_and_negative_are_distinct_without_probability_fallback() {
    for p in [0.0, 0.55, 1.0] {
        let (arena, mut intent) = fixture(1_000.0, 100);
        intent.win_probability = p;
        assert_rejected(&arena, &intent, REJ_SIN_EVIDENCIA);
        for mean in [0.0, -0.001] {
            intent.net_payoff = Some(measured(&[mean; 100]));
            assert_rejected(&arena, &intent, REJ_EV);
        }
        intent.net_payoff = Some(measured(&[0.004; 100]));
        assert_nonminimum_order(&arena, &intent);
    }
}

#[test]
fn lg5_invalid_or_other_context_evidence_rejects_even_a_probe() {
    for trades in [0, 100] {
        let (arena, mut intent) = fixture(1_000.0, trades);
        let valid = measured(&[0.004; 100]);
        let invalid = [
            NetPayoffEstimate {
                coin_id: 1,
                ..valid
            },
            NetPayoffEstimate {
                side: SignalType::Short,
                ..valid
            },
            NetPayoffEstimate {
                sample_count: 0,
                ..valid
            },
            NetPayoffEstimate {
                sample_start_ms: valid.sample_end_ms + 1,
                ..valid
            },
            NetPayoffEstimate {
                sample_end_ms: valid.as_of_ms + 1,
                ..valid
            },
            NetPayoffEstimate {
                standard_deviation: -1.0,
                ..valid
            },
            NetPayoffEstimate {
                standard_deviation: f64::NAN,
                ..valid
            },
            NetPayoffEstimate {
                standard_deviation: f64::INFINITY,
                ..valid
            },
            NetPayoffEstimate {
                standard_deviation: f64::NEG_INFINITY,
                ..valid
            },
            NetPayoffEstimate {
                mean_net_return_pct: f64::NAN,
                ..valid
            },
            NetPayoffEstimate {
                mean_net_return_pct: f64::INFINITY,
                ..valid
            },
            NetPayoffEstimate {
                mean_net_return_pct: f64::NEG_INFINITY,
                ..valid
            },
        ];
        for evidence in invalid {
            intent.net_payoff = Some(evidence);
            assert_rejected(&arena, &intent, REJ_INVALID_INPUT);
        }
    }
}

#[test]
fn lg5_probe_keeps_its_budget_and_cost_covering_geometry() {
    let (arena, mut intent) = fixture(13.0, 0);
    let baseline = evaluate(&arena, &intent);
    assert_eq!(
        baseline.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );
    assert!(baseline.volume_usd * baseline.leverage >= 5.1);
    // Evidencia válida negativa no cambia la política explícita de sonda;
    // no se inventa retorno ni se amplía su presupuesto.
    intent.net_payoff = Some(measured(&[-0.001; 100]));
    let measured_probe = evaluate(&arena, &intent);
    assert_eq!(
        measured_probe.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );
    assert_eq!(measured_probe.volume_usd, baseline.volume_usd);
    assert_eq!(measured_probe.leverage, baseline.leverage);
    intent.tp_price_target = 100.01; // TP de 1bp, inferior al coste de ida/vuelta.
    assert_rejected(&arena, &intent, REJ_EV);
}

#[test]
fn lg5_valid_evidence_respects_side_and_declared_local_source() {
    for side in [SignalType::Long, SignalType::Short] {
        for source in [
            NetPayoffSource::LocalSimulation,
            NetPayoffSource::ExchangeLocalEstimate,
        ] {
            let (arena, mut intent) = fixture(1_000.0, 100);
            let direction = if side == SignalType::Long { 1.0 } else { -1.0 };
            intent.signal = side;
            intent.tp_price_target = 100.0 + direction * 2.25;
            intent.sl_price_target = 100.0 - direction;
            intent.net_payoff = Some(NetPayoffEstimate {
                side,
                source,
                ..measured(&[0.004; 100])
            });
            assert_nonminimum_order(&arena, &intent);
        }
    }
}

#[test]
fn lg5_legacy_clock_admits_evidence_available_at_exact_equality() {
    let (arena, mut intent) = fixture(1_000.0, 100);
    // Both fixture schedules independently declare time 100_000; no clock
    // is copied from the payload or advanced by the evaluation helper.
    arena.coins[0]
        .last_tick_timestamp_ms
        .store(100_000, Relaxed);
    intent.net_payoff = Some(measured(&[0.004; 100]));
    assert_nonminimum_order(&arena, &intent);
}

#[test]
fn lg5_future_availability_rejects_for_each_clock_even_a_probe() {
    for trades in [0, 100] {
        for explicit_clock in [false, true] {
            let (arena, mut intent) = fixture(1_000.0, trades);
            intent.net_payoff = Some(measured(&[0.004; 100]));
            // The evidence is internally valid but only available at 100_000.
            // Explicit-time admission must not borrow the later depth clock.
            let order = if explicit_clock {
                RiskEngine::new(1_000.0).evaluate_quantum_order_at(0, &intent, &arena, 99_999)
            } else {
                arena.coins[0].last_tick_timestamp_ms.store(99_999, Relaxed);
                evaluate(&arena, &intent)
            };
            assert_eq!(order.signal, SignalType::Flat, "{order:?}");
            assert_eq!(ultimo_rechazo(), Some(REJ_INVALID_INPUT));
        }
    }
}

#[test]
fn lg5_explicit_trade_clock_can_be_newer_than_depth() {
    let (arena, mut intent) = fixture(1_000.0, 100);
    arena.coins[0].last_tick_timestamp_ms.store(10, Relaxed);
    intent.net_payoff = Some(NetPayoffEstimate {
        sample_start_ms: 1,
        sample_end_ms: 20,
        as_of_ms: 20,
        ..measured(&[0.004; 100])
    });
    // Boundary fixture for a decision triggered by a trade at t=20 while
    // the latest depth event remains t=10. The event provides the clock.
    let order = RiskEngine::new(1_000.0).evaluate_quantum_order_at(0, &intent, &arena, 20);
    assert_nonminimum_size(order, SignalType::Long);
    assert_eq!(arena.coins[0].last_tick_timestamp_ms.load(Relaxed), 10);
    assert_rejected(&arena, &intent, REJ_INVALID_INPUT);
}
