use execution_engine::order_registry::{OrderRegistry, OrderStatus, TradeUpdate};
use execution_engine::order_types::{Fill, OrderAck};
use execution_engine::reconciliation::{
    reconcile_arena_checked, PositionRiskEntry, ReconciliationIssueKind, ReconciliationReport,
};
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard};

static UNIVERSE_LOCK: Mutex<()> = Mutex::new(());

struct UniverseGuard {
    previous: Vec<String>,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for UniverseGuard {
    fn drop(&mut self) {
        quantum_arena::symbols::update_dynamic_universe(std::mem::take(&mut self.previous));
    }
}

fn universe() -> UniverseGuard {
    let lock = UNIVERSE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let previous = quantum_arena::symbols::get_active_universe();
    quantum_arena::symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "ETHUSDT".into()]);
    UniverseGuard {
        previous,
        _lock: lock,
    }
}

fn position(symbol: &str, qty: f64, price: f64) -> PositionRiskEntry {
    PositionRiskEntry {
        symbol: symbol.into(),
        position_amt: qty,
        entry_price: price,
        position_side: "BOTH".into(),
        leverage: 5.0,
        update_time: 100,
        ..Default::default()
    }
}

fn report(rows: Vec<PositionRiskEntry>) -> ReconciliationReport {
    ReconciliationReport {
        open_positions: rows,
        ..Default::default()
    }
}

fn fees(symbol: &str, taker: f64) -> SymbolFees {
    SymbolFees {
        symbol: symbol.into(),
        maker: 0.9,
        taker,
    }
}

fn publish(arena: &GlobalArena, rows: Vec<SymbolFees>) {
    arena
        .config
        .observed_fees_required
        .store(true, Ordering::Release);
    arena
        .config
        .live_fee_schedule
        .store(Some(Arc::new(FeeSchedule::new(1, rows).unwrap())));
}

#[test]
fn registry_adoption_preserves_symbol_specific_zero_and_signed_estimates() {
    let registry = OrderRegistry::new();
    let report = report(vec![
        position("BTCUSDT", 2.0, 100.0),
        position("ETHUSDT", -3.0, 200.0),
    ]);
    assert_eq!(
        report.apply_to_registry_with_rates(&registry, |symbol| match symbol {
            "BTCUSDT" => Some(0.0),
            "ETHUSDT" => Some(-0.0002),
            _ => None,
        }),
        2
    );
    let btc = registry.get("adopted_BTCUSDT_100").unwrap();
    let eth = registry.get("adopted_ETHUSDT_100").unwrap();
    assert_eq!(btc.total_commission, 0.0);
    assert_eq!(eth.total_commission, 600.0 * -0.0002);
    assert_eq!(eth.order_type, "ADOPTED_ESTIMATE");
    assert_eq!(eth.order_id, 0);
    assert!(
        eth.fees_by_trade.is_empty(),
        "an estimate is not an identified fill"
    );
    assert_eq!(
        report.apply_to_registry_with_rates(&registry, |_| Some(-0.0001)),
        2
    );
    assert_eq!(
        registry
            .get("adopted_ETHUSDT_100")
            .unwrap()
            .total_commission,
        600.0 * -0.0001
    );
}

#[test]
fn registry_adoption_skips_missing_and_nonfinite_rates_without_a_default() {
    for rate in [
        None,
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
    ] {
        let registry = OrderRegistry::new();
        let report = report(vec![position("BTCUSDT", 1.0, 100.0)]);
        assert_eq!(report.apply_to_registry_with_rates(&registry, |_| rate), 0);
        assert!(registry.get("adopted_BTCUSDT_100").is_none());
    }
}

#[test]
fn registry_adoption_rejects_invalid_projection_and_imputed_fee_overflow() {
    for (qty, price, rate) in [
        (0.0, 100.0, 0.001),
        (1.0, f64::NAN, 0.001),
        (f64::MAX, 2.0, 0.0),
        (2.0, 100.0, f64::MAX),
    ] {
        let registry = OrderRegistry::new();
        assert_eq!(
            report(vec![position("BTCUSDT", qty, price)])
                .apply_to_registry_with_rates(&registry, |_| Some(rate)),
            0
        );
        assert!(registry.get("adopted_BTCUSDT_100").is_none());
    }
}

#[test]
fn legacy_scalar_wrapper_retains_its_explicit_scenario_fallback() {
    let registry = OrderRegistry::new();
    let report = report(vec![position("BTCUSDT", 1.0, 100.0)]);
    assert_eq!(report.apply_to_registry(&registry, 0.0), 1);
    assert_eq!(
        registry
            .get("adopted_BTCUSDT_100")
            .unwrap()
            .total_commission,
        100.0 * 0.0005
    );
}

#[test]
fn arena_adopts_each_symbol_with_its_current_signed_rate() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    publish(&arena, vec![fees("BTCUSDT", 0.0), fees("ETHUSDT", -0.0002)]);
    let result = reconcile_arena_checked(
        &[
            position("BTCUSDT", 2.0, 100.0),
            position("ETHUSDT", -3.0, 200.0),
        ],
        &arena,
        1_000,
    );
    assert_eq!(result.adjustments, 2);
    assert!(result.unresolved.is_empty());
    assert_eq!(
        arena.coins[0]
            .positions
            .slot_2
            .entry_fee
            .load(Ordering::Relaxed),
        0.0
    );
    assert_eq!(
        arena.coins[1]
            .positions
            .slot_2
            .entry_fee
            .load(Ordering::Relaxed),
        600.0 * -0.0002
    );
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 160.0);
}

#[test]
fn arena_missing_or_nonfinite_fee_evidence_remains_unresolved() {
    let _universe = universe();
    for required in [true, false] {
        let arena = GlobalArena::build_in_own_stack(100.0);
        arena
            .config
            .observed_fees_required
            .store(required, Ordering::Release);
        if !required {
            arena
                .config
                .live_taker_fee
                .store(f64::NAN, Ordering::Release);
        }
        let result = reconcile_arena_checked(
            &[
                position("BTCUSDT", 2.0, 100.0),
                position("ETHUSDT", 0.0, 0.0),
            ],
            &arena,
            1_000,
        );
        assert_eq!(result.adjustments, 0);
        assert!(result
            .unresolved
            .iter()
            .any(|i| i.symbol == "BTCUSDT"
                && i.kind == ReconciliationIssueKind::MissingRemoteEvidence));
        assert!(!arena.coins[0].positions.slot_2.is_open());
        assert_eq!(arena.used_margin.load(Ordering::Relaxed), 0.0);
    }
}

#[test]
fn arena_explicit_scenario_uses_config_rate_including_zero_and_rebates() {
    let _universe = universe();
    for rate in [0.0, -0.0017, 0.0017] {
        let arena = GlobalArena::build_in_own_stack(100.0);
        arena.config.live_taker_fee.store(rate, Ordering::Release);
        let result = reconcile_arena_checked(
            &[
                position("BTCUSDT", 2.0, 100.0),
                position("ETHUSDT", 0.0, 0.0),
            ],
            &arena,
            1_000,
        );
        assert_eq!(result.adjustments, 1);
        assert!(result.unresolved.is_empty());
        assert_eq!(
            arena.coins[0]
                .positions
                .slot_2
                .entry_fee
                .load(Ordering::Relaxed),
            200.0 * rate
        );
    }
}

#[test]
fn arena_fee_overflow_does_not_open_a_position_or_reserve_margin() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    publish(
        &arena,
        vec![fees("BTCUSDT", f64::MAX), fees("ETHUSDT", 0.0)],
    );
    let result = reconcile_arena_checked(
        &[
            position("BTCUSDT", 2.0, 100.0),
            position("ETHUSDT", 0.0, 0.0),
        ],
        &arena,
        1_000,
    );
    assert_eq!(result.adjustments, 0);
    assert!(
        result
            .unresolved
            .iter()
            .any(|i| i.symbol == "BTCUSDT"
                && i.kind == ReconciliationIssueKind::InvalidRemoteEvidence)
    );
    assert!(!arena.coins[0].positions.slot_2.is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 0.0);
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 100.0);
}

fn native_ack(id: &str, values: &[(u64, f64)]) -> OrderAck {
    OrderAck {
        client_order_id: id.into(),
        symbol: "BTCUSDT".into(),
        side: "BUY".into(),
        order_type: "MARKET".into(),
        order_id: 77,
        orig_qty: 1.0,
        executed_qty: 1.0,
        avg_price: 100.0,
        price: 100.0,
        cum_quote: 100.0,
        status: "FILLED".into(),
        update_time: 100,
        fills: values
            .iter()
            .map(|&(trade_id, commission)| Fill {
                price: 100.0,
                qty: 1.0,
                commission,
                commission_asset: "USDT".into(),
                trade_id,
            })
            .collect(),
    }
}

fn native_update(id: &str, trade_id: u64, commission: f64) -> TradeUpdate {
    TradeUpdate {
        client_order_id: id.into(),
        symbol: "BTCUSDT".into(),
        side: "BUY".into(),
        position_side: "BOTH".into(),
        order_type: "MARKET".into(),
        execution_type: "TRADE".into(),
        order_id: 77,
        status: OrderStatus::Filled,
        orig_qty: 1.0,
        cumulative_filled_qty: 1.0,
        last_filled_qty: 1.0,
        last_filled_price: 100.0,
        avg_price: 100.0,
        commission,
        commission_asset: "USDT".into(),
        trade_id,
        trade_time_ms: 100,
    }
}

#[test]
fn identified_rest_ws_union_deduplicates_overlap_and_preserves_disjoint_rebates() {
    for ws_first in [false, true] {
        let registry = OrderRegistry::new();
        let ack = native_ack("native", &[(11, -0.25), (12, 0.1)]);
        let overlap = native_update("native", 11, -0.25);
        if ws_first {
            registry.apply_trade_update(&overlap, 100);
        }
        registry.apply_ack(&ack, 100);
        registry.apply_trade_update(&overlap, 100);
        registry.apply_trade_update(&native_update("native", 13, -0.05), 101);
        registry.apply_ack(&ack, 102);
        let order = registry.get("native").unwrap();
        assert_eq!(order.fees_by_trade.len(), 3);
        assert_eq!(order.fees_by_trade.get(&11), Some(&-0.25));
        assert_eq!(order.fees_by_trade.get(&12), Some(&0.1));
        assert_eq!(order.fees_by_trade.get(&13), Some(&-0.05));
        assert!((order.total_commission - (-0.2)).abs() < 1e-12);
    }
}

#[test]
fn identified_nonfinite_observations_do_not_replace_prior_signed_evidence() {
    let registry = OrderRegistry::new();
    registry.apply_ack(&native_ack("native", &[(11, -0.25)]), 100);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        registry.apply_ack(&native_ack("native", &[(11, invalid)]), 101);
        registry.apply_trade_update(&native_update("native", 11, invalid), 102);
        assert_eq!(registry.get("native").unwrap().total_commission, -0.25);
    }
}

#[test]
fn estimated_adoption_cannot_overwrite_an_existing_native_order() {
    for case in 0..3 {
        let registry = OrderRegistry::new();
        let mut native = native_ack("adopted_BTCUSDT_100", &[(11, -0.25)]);
        if case == 0 {
            native.order_type = "ADOPTED_ESTIMATE".into();
            native.fills.clear();
        } else if case == 1 {
            native.order_type = "ADOPTED_ESTIMATE".into();
            native.order_id = 0;
        } else {
            native.order_id = 0;
            native.fills.clear();
        }
        registry.apply_ack(&native, 100);
        let before = registry.get("adopted_BTCUSDT_100").unwrap();
        assert_eq!(
            report(vec![position("BTCUSDT", 1.0, 100.0)])
                .apply_to_registry_with_rates(&registry, |_| Some(0.001)),
            0
        );
        let after = registry.get("adopted_BTCUSDT_100").unwrap();
        assert_eq!(after.order_id, before.order_id);
        assert_eq!(after.order_type, before.order_type);
        assert_eq!(after.updated_ms, before.updated_ms);
        assert_eq!(after.fees_by_trade, before.fees_by_trade);
        assert_eq!(after.total_commission, before.total_commission);
    }
}
