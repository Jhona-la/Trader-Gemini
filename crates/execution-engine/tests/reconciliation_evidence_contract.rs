use execution_engine::reconciliation::{reconcile_arena, PositionRiskEntry};
use execution_engine::reconciliation::{reconcile_arena_checked, ReconciliationIssueKind};
use quantum_arena::{position::PositionHorizon, symbol_registry, symbols, GlobalArena};
use std::sync::{atomic::Ordering, Mutex};

static UNIVERSE: Mutex<()> = Mutex::new(());

#[test]
fn checked_report_explains_hedge_and_is_order_invariant() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 1.0);
    let rows = [row(1.0, "LONG"), row(-1.0, "SHORT")];
    let first = reconcile_arena_checked(&rows, &a, 3);
    let second = reconcile_arena_checked(&[rows[1].clone(), rows[0].clone()], &a, 3);
    assert_eq!(first.unresolved, second.unresolved);
    assert_eq!(
        first.unresolved[0].kind,
        ReconciliationIssueKind::MultipleRemoteLegs
    );
    assert_eq!(first.adjustments, 0);
}
#[test]
fn invalid_metadata_and_unmapped_exposure_are_reported_without_adoption() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    for invalid in [0.0, f64::NAN, f64::INFINITY] {
        let mut p = row(1.0, "LONG");
        p.entry_price = invalid;
        assert_eq!(
            reconcile_arena_checked(&[p], &a, 3).unresolved[0].kind,
            ReconciliationIssueKind::InvalidRemoteEvidence
        );
        let mut p = row(1.0, "LONG");
        p.leverage = invalid;
        assert_eq!(
            reconcile_arena_checked(&[p], &a, 3).unresolved[0].kind,
            ReconciliationIssueKind::InvalidRemoteEvidence
        );
    }
    let mut overflow = row(f64::MAX, "LONG");
    overflow.entry_price = f64::MAX;
    assert_eq!(
        reconcile_arena_checked(&[overflow], &a, 3).unresolved[0].kind,
        ReconciliationIssueKind::InvalidRemoteEvidence
    );
    let mut other = row(1.0, "LONG");
    other.symbol = "UNMAPPEDUSDT".into();
    let r = reconcile_arena_checked(&[row(0.0, "BOTH"), other], &a, 3);
    assert_eq!(
        r.unresolved[0].kind,
        ReconciliationIssueKind::UnmappedInstrument
    );
    assert!(!a.coins[0].positions.position.is_open());
}
#[test]
fn mixed_position_modes_and_oversized_universe_do_not_mutate() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 1.0);
    let r = reconcile_arena_checked(&[row(1.0, "LONG"), row(0.0, "BOTH")], &a, 3);
    assert_eq!(
        r.unresolved[0].kind,
        ReconciliationIssueKind::InvalidRemoteEvidence
    );
    symbols::update_dynamic_universe((0..=a.coins.len()).map(|i| format!("X{i}USDT")).collect());
    let r = reconcile_arena_checked(&[], &a, 3);
    assert_eq!(
        r.unresolved[0].kind,
        ReconciliationIssueKind::InvalidUniverse
    );
    assert!(a.coins[0].positions.position.is_open());
}
fn arena() -> std::sync::Arc<GlobalArena> {
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("XIXUSDT")]);
    symbols::update_dynamic_universe(vec!["XIXUSDT".into()]);
    GlobalArena::build_in_own_stack(100.0)
}
fn row(qty: f64, side: &str) -> PositionRiskEntry {
    PositionRiskEntry {
        symbol: "XIXUSDT".into(),
        position_amt: qty,
        entry_price: 100.0,
        leverage: 10.0,
        position_side: side.into(),
        update_time: 2,
        ..Default::default()
    }
}
fn open(arena: &GlobalArena, index: usize, qty: f64) {
    assert!(arena.coins[0].positions.get_slot(index).open_with_horizon(
        true,
        100.0,
        qty,
        qty * 10.0,
        1,
        110.0,
        90.0,
        PositionHorizon::Continuous
    ));
    arena.used_margin.fetch_add(qty * 10.0, Ordering::Relaxed);
}

#[test]
fn balanced_hedge_does_not_clear_known_exposure() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 1.0);
    assert_eq!(
        reconcile_arena(&[row(1.0, "LONG"), row(-1.0, "SHORT")], &a, 3),
        0
    );
    assert!(a.coins[0].positions.position.is_open());
    assert_eq!(a.used_margin.load(Ordering::Relaxed), 10.0);
}
#[test]
fn unequal_hedge_does_not_replace_gross_with_net() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 2.0);
    assert_eq!(
        reconcile_arena(&[row(2.0, "LONG"), row(-1.0, "SHORT")], &a, 3),
        0
    );
    assert_eq!(
        a.coins[0]
            .positions
            .position
            .quantity
            .load(Ordering::Relaxed),
        2.0
    );
}
#[test]
fn reversal_cannot_confirm_a_stale_direction() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 1.0);
    assert_eq!(reconcile_arena(&[row(-1.0, "BOTH")], &a, 3), 0);
    let p = &a.coins[0].positions.position;
    assert!(!p.exchange_confirmed.load(Ordering::Relaxed));
    assert_eq!(p.entry_price.load(Ordering::Relaxed), 100.0);
}
#[test]
fn absent_row_is_not_explicit_flat_evidence() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 1.0);
    assert_eq!(reconcile_arena(&[], &a, 3), 0);
    assert!(a.coins[0].positions.position.is_open());
}
#[test]
fn invalid_quantity_does_not_become_flat() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 1.0);
    assert_eq!(reconcile_arena(&[row(f64::NAN, "BOTH")], &a, 3), 0);
    assert!(a.coins[0].positions.position.is_open());
}
#[test]
fn duplicate_leg_rows_are_not_added_together() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 1.0);
    assert_eq!(
        reconcile_arena(&[row(1.0, "LONG"), row(1.0, "LONG")], &a, 3),
        0
    );
    assert_eq!(
        a.coins[0]
            .positions
            .position
            .quantity
            .load(Ordering::Relaxed),
        1.0
    );
}
#[test]
fn multiple_local_allocations_are_not_each_given_total_remote_quantity() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 0, 0.4);
    open(&a, 2, 0.6);
    assert_eq!(reconcile_arena(&[row(1.2, "LONG")], &a, 3), 0);
    assert_eq!(
        a.coins[0].positions.scalp.quantity.load(Ordering::Relaxed),
        0.4
    );
    assert_eq!(
        a.coins[0]
            .positions
            .position
            .quantity
            .load(Ordering::Relaxed),
        0.6
    );
}
#[test]
fn nonzero_small_exchange_quantity_is_not_flat() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 5e-9);
    reconcile_arena(&[row(5e-9, "LONG")], &a, 3);
    assert!(a.coins[0].positions.position.is_open());
}
#[test]
fn small_drift_does_not_close_a_nonzero_position() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 2e-6);
    reconcile_arena(&[row(1e-7, "LONG")], &a, 3);
    let p = &a.coins[0].positions.position;
    assert!(p.is_open());
    assert_eq!(p.quantity.load(Ordering::Relaxed), 1e-7);
}
#[test]
fn explicit_flat_and_single_position_adoption_still_work() {
    let _guard = UNIVERSE.lock().unwrap_or_else(|p| p.into_inner());
    let a = arena();
    open(&a, 2, 1.0);
    assert_eq!(reconcile_arena(&[row(0.0, "BOTH")], &a, 3), 1);
    assert!(!a.coins[0].positions.position.is_open());
    assert_eq!(reconcile_arena(&[row(-2.0, "SHORT")], &a, 4), 1);
    assert!(!a.coins[0]
        .positions
        .position
        .is_long
        .load(Ordering::Relaxed));
}
