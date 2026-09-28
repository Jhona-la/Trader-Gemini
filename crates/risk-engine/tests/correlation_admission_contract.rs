//! Domain and live-admission contracts, no orders, exchange or engine process.
use quantum_arena::{state::CompactTick, GlobalArena};
use risk_engine::correlation_guard::{
    rho_efectivo_para_agregacion, rho_promedio, CorrelationGuardEngine,
};
use risk_engine::{RiskEngine, REJECT_COUNTERS};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

fn matrix(n: usize, rho: f64) -> Vec<Vec<f64>> {
    let mut c = vec![vec![rho; n]; n];
    for i in 0..n {
        c[i][i] = 1.0;
    }
    c
}

#[test]
fn incomplete_or_invalid_matrices_never_produce_a_group_correlation() {
    let mut partial = matrix(3, 0.2);
    partial[0][2] = f64::NAN;
    partial[2][0] = f64::NAN;
    let mut asymmetric = matrix(3, 0.2);
    asymmetric[2][0] = 0.4;
    let mut invalid_diagonal = matrix(3, 0.2);
    invalid_diagonal[1][1] = 2.0;
    for (name, c) in [
        ("partial", partial),
        ("asymmetric", asymmetric),
        ("diagonal", invalid_diagonal),
        ("range", matrix(3, 1.1)),
        ("negative variance", matrix(5, -0.4)),
    ] {
        assert!(rho_promedio(&c).is_none(), "{name}");
        assert!(rho_efectivo_para_agregacion(&c).is_none(), "{name}");
    }
}

#[test]
fn complete_valid_negative_and_positive_dependence_is_not_erased() {
    for (n, rho) in [(2, -1.0), (5, -0.25), (5, -0.2), (5, 0.0), (5, 0.8)] {
        assert!((rho_promedio(&matrix(n, rho)).unwrap() - rho).abs() < 1e-12);
    }
}

#[test]
fn invalid_correlation_does_not_look_like_a_hedge() {
    for bad in [-2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            CorrelationGuardEngine::es_la_misma_apuesta(Some(bad), 0.5),
            "{bad}"
        );
    }
    assert!(!CorrelationGuardEngine::es_la_misma_apuesta(
        Some(-0.9),
        0.5
    ));
    assert!(!CorrelationGuardEngine::es_la_misma_apuesta(Some(0.0), 0.5));
}

#[test]
fn impossible_equicorrelation_never_grants_a_variance_discount() {
    let cap = risk_engine::ruin::clamp_ruin(1.0, 0.5);
    for rho in [-0.2501, -0.4, -1.0, -2.0] {
        assert!(
            CorrelationGuardEngine::veto_por_exposicion_estructural(4, cap / 2.0, 0.5, Some(rho)),
            "rho={rho}"
        );
    }
}

#[test]
fn feasible_negative_correlation_remains_a_valid_algebraic_input() {
    let cap = risk_engine::ruin::clamp_ruin(1.0, 0.5);
    assert!(!CorrelationGuardEngine::veto_por_exposicion_estructural(
        4,
        cap / 2.0,
        0.5,
        Some(-0.2)
    ));
    assert!(CorrelationGuardEngine::veto_por_exposicion_estructural(
        4,
        cap / 2.0,
        0.5,
        Some(0.0)
    ));
}

fn fixture() -> (Arc<GlobalArena>, SignalIntent) {
    static REGISTRY: Once = Once::new();
    REGISTRY.call_once(|| {
        quantum_arena::symbol_registry::update_registry(
            ["BTCUSDT", "ETHUSDT"]
                .iter()
                .map(|symbol| quantum_arena::symbol_registry::SymbolSpec {
                    symbol: (*symbol).into(),
                    step_size: 0.001,
                    tick_size: 0.01,
                    min_qty: 0.001,
                    min_notional: 5.0,
                    max_leverage: 20,
                    maker_fee: 0.0002,
                    taker_fee: 0.0004,
                    is_shadow: false,
                })
                .collect(),
        )
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
    arena
        .config
        .global_correlation_threshold
        .store(0.5, Relaxed);
    let cap = risk_engine::ruin::clamp_ruin(1.0, 0.25);
    arena.riesgo_por_operacion.store(0.75 * cap, Relaxed);
    (
        arena,
        SignalIntent {
            signal: SignalType::Long,
            confidence: 0.9,
            expected_duration_ms: 60_000,
            ..SignalIntent::default()
        },
    )
}

fn open(arena: &GlobalArena, coin: usize, slot: usize, is_long: bool) {
    arena.coins[coin].positions.slots()[slot].open(
        is_long,
        100.0,
        0.01,
        0.1,
        1000,
        if is_long { 102.0 } else { 98.0 },
        if is_long { 99.0 } else { 101.0 },
    );
    assert!(arena.coins[coin].positions.slots()[slot].is_open());
}

fn pair_ticks(arena: &GlobalArena, anticorrelated: bool) {
    for i in 0..200 {
        let log_change = 0.01 * (i as f64 * 0.7).sin();
        for coin in 0..2 {
            let sign = if coin == 1 && anticorrelated {
                -1.0
            } else {
                1.0
            };
            let price = 100.0 * (sign * log_change).exp();
            arena.coins[coin].tick_ring.push(CompactTick {
                timestamp: 1000 + i * 10,
                bid_price: price,
                ask_price: price,
                bid_qty: 1.0,
                ask_qty: 1.0,
            });
        }
    }
}

#[test]
fn no_positions_does_not_add_a_blanket_correlation_veto() {
    let (arena, intent) = fixture();
    assert_eq!(
        RiskEngine::new(100.0)
            .evaluate_quantum_order(0, &intent, &arena)
            .signal,
        SignalType::Long
    );
}

#[test]
fn missing_evidence_is_not_imputed_to_independence_in_live_admission() {
    for slot in 0..3 {
        for side in [true, false] {
            let (arena, intent) = fixture();
            open(&arena, 1, slot, side);
            let before = REJECT_COUNTERS[2].load(Relaxed);
            let order = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena);
            assert_eq!(order.signal, SignalType::Flat, "slot={slot},long={side}");
            assert!(REJECT_COUNTERS[2].load(Relaxed) > before);
        }
    }
}

#[test]
fn all_spectral_slots_of_the_same_asset_are_counted() {
    for slot in 0..3 {
        let (arena, intent) = fixture();
        open(&arena, 0, slot, true);
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat,
            "slot={slot}"
        );
    }
}

#[test]
fn all_spectral_slots_of_other_assets_are_counted() {
    for slot in 0..3 {
        let (arena, intent) = fixture();
        pair_ticks(&arena, false);
        open(&arena, 1, slot, true);
        assert_eq!(
            RiskEngine::new(100.0)
                .evaluate_quantum_order(0, &intent, &arena)
                .signal,
            SignalType::Flat,
            "slot={slot}"
        );
    }
}

#[test]
fn opposite_side_on_anticorrelated_asset_is_concentrated_pnl() {
    let (arena, intent) = fixture();
    pair_ticks(&arena, true);
    open(&arena, 1, 2, false);
    assert_eq!(
        RiskEngine::new(100.0)
            .evaluate_quantum_order(0, &intent, &arena)
            .signal,
        SignalType::Flat
    );
}

#[test]
fn measured_hedge_is_not_mistaken_for_same_bet() {
    let (arena, intent) = fixture();
    pair_ticks(&arena, true);
    open(&arena, 1, 2, true);
    assert_eq!(
        RiskEngine::new(100.0)
            .evaluate_quantum_order(0, &intent, &arena)
            .signal,
        SignalType::Long
    );
}

#[test]
fn dependency_evidence_retains_unknown_counts_separately() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    for slot in 0..3 {
        open(&arena, 1, slot, slot != 1);
    }
    open(&arena, 0, 0, true);
    let e = dependency_exposure(&arena, 0, true, 0.5).unwrap();
    assert_eq!(e.open_positions, 4);
    assert_eq!(e.same_bet_positions, 4);
    assert_eq!(e.unknown_positions, 3);
}

#[test]
fn dependency_transform_covers_all_signs_and_slots() {
    use risk_engine::correlation_guard::dependency_exposure;
    for anticorrelated in [false, true] {
        for candidate_long in [false, true] {
            for position_long in [false, true] {
                for slot in 0..3 {
                    let (arena, _) = fixture();
                    pair_ticks(&arena, anticorrelated);
                    open(&arena, 1, slot, position_long);
                    let e = dependency_exposure(&arena, 0, candidate_long, 0.5).unwrap();
                    let concentrated = (candidate_long == position_long) != anticorrelated;
                    assert_eq!(e.open_positions, 1);
                    assert_eq!(e.unknown_positions, 0);
                    assert_eq!(e.same_bet_positions, usize::from(concentrated));
                }
            }
        }
    }
}

#[test]
fn same_asset_opposite_side_is_signed_without_invented_market_samples() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    open(&arena, 0, 0, false);
    let e = dependency_exposure(&arena, 0, true, 0.5).unwrap();
    assert_eq!(e.open_positions, 1);
    assert_eq!(e.same_bet_positions, 0);
    assert_eq!(e.unknown_positions, 0);
}

#[test]
fn invalid_candidate_has_no_dependency_snapshot() {
    use risk_engine::correlation_guard::dependency_exposure;
    let (arena, _) = fixture();
    assert!(dependency_exposure(&arena, arena.coins.len(), true, 0.5).is_none());
}
