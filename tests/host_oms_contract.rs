//! Actual host authorization, reconciliation and dispatch functions with synthetic observations.
//! No REST requests, live credentials, training, tapes or model loading.
use arc_swap::ArcSwap;
use execution_engine::{
    entry_dispatch::{EntryRequest, EntryRoute, EntryTransport},
    executor::OrderExecutor,
    order_types::OrderAck,
    reconciliation::PositionRiskEntry,
};
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees, position::PositionHorizon};
use quantum_engine::host_oms::{OmsSession, SessionToken, SnapshotStatus};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::Notify;

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

fn executor(testnet: bool, paper: bool) -> Arc<OrderExecutor> {
    let mut executor = OrderExecutor::new(String::new(), "offline-contract".into(), testnet);
    executor.set_paper_trading(paper);
    Arc::new(executor)
}

fn session(testnet: bool, paper: bool) -> (Arc<OmsSession>, Arc<AtomicBool>) {
    // Deliberately start the external flag true: the production constructor must reset it.
    let ready = Arc::new(AtomicBool::new(true));
    let executors = Arc::new(ArcSwap::from(executor(testnet, paper)));
    (
        Arc::new(OmsSession::new(executors, Arc::clone(&ready))),
        ready,
    )
}

fn flat_positions() -> Vec<PositionRiskEntry> {
    quantum_arena::symbols::get_active_universe_ref()
        .iter()
        .map(|symbol| PositionRiskEntry {
            symbol: symbol.to_uppercase(),
            position_side: "BOTH".into(),
            ..Default::default()
        })
        .collect()
}

fn fees(token: &SessionToken) -> Result<FeeSchedule, String> {
    let rows = quantum_arena::symbols::get_active_universe_ref()
        .iter()
        .enumerate()
        .map(|(i, symbol)| SymbolFees {
            symbol: symbol.to_string(),
            maker: if i == 0 { -0.0001 } else { 0.0 },
            taker: 0.0003 + i as f64 * 0.000001,
        })
        .collect();
    FeeSchedule::new(token.generation(), rows).map_err(|e| format!("{e:?}"))
}

fn request() -> EntryRequest<'static> {
    EntryRequest {
        symbol: "BTCUSDT",
        is_long: true,
        quantity: 0.01,
        step_size: 0.001,
        tick_size: 0.1,
        leverage: 1,
        client_order_id: "oms_contract_entry",
        route: EntryRoute::Market,
    }
}

#[test]
fn startup_requires_remote_evidence_for_testnet_and_mainnet_but_not_explicit_paper() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    for testnet in [false, true] {
        let (session, ready) = session(testnet, false);
        session.initialize_arena(&arena);
        assert!(arena.config.observed_fees_required.load(Ordering::Acquire));
        assert!(!ready.load(Ordering::Acquire), "testnet={testnet}");
        assert!(
            session
                .check_entry(&session.snapshot_token(), &arena)
                .is_err()
        );
    }
    let (session, ready) = session(false, true);
    session.initialize_arena(&arena);
    assert!(!arena.config.observed_fees_required.load(Ordering::Acquire));
    assert!(ready.load(Ordering::Acquire));
    assert!(
        session
            .check_entry(&session.snapshot_token(), &arena)
            .is_ok()
    );
    let paper_token = session.snapshot_token();
    session
        .replace_executor(executor(true, false), &arena)
        .unwrap();
    assert!(!ready.load(Ordering::Acquire));
    assert!(arena.config.observed_fees_required.load(Ordering::Acquire));
    assert!(arena.config.live_fee_schedule.load_full().is_none());
    assert!(session.check_entry(&paper_token, &arena).is_err());
}

#[test]
fn observed_fees_are_per_symbol_and_fee_failure_or_wrong_generation_revokes_the_book() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    session.initialize_arena(&arena);
    let token = session.snapshot_token();
    let schedule = fees(&token).unwrap();
    let expected: Vec<_> = schedule
        .rows()
        .iter()
        .map(|r| (r.symbol.clone(), r.maker, r.taker))
        .collect();
    assert!(expected.len() > 1);
    let success = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        Ok(schedule),
        &arena,
        1_000,
    );
    assert_eq!(success.status, SnapshotStatus::Ready);
    let observed = arena.config.live_fee_schedule.load_full().unwrap();
    assert_eq!(observed.generation(), token.generation());
    for (row, (symbol, maker, taker)) in observed.rows().iter().zip(expected.iter()) {
        assert_eq!(&row.symbol, symbol);
        assert_eq!(row.maker, *maker);
        assert_eq!(row.taker, *taker);
    }
    assert_ne!(observed.rows()[0].taker, observed.rows()[1].taker);
    assert_eq!(observed.rows()[0].maker, -0.0001);
    assert_eq!(observed.rows()[1].maker, 0.0);
    let failure = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        Err("one symbol's fee query failed".into()),
        &arena,
        2_000,
    );
    assert_eq!(failure.status, SnapshotStatus::Blocked);
    assert!(!ready.load(Ordering::Acquire));
    assert!(arena.config.live_fee_schedule.load_full().is_none());
    let wrong_generation = FeeSchedule::new(
        token.generation() + 1,
        fees(&token).unwrap().rows().to_vec(),
    )
    .unwrap();
    let wrong = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        Ok(wrong_generation),
        &arena,
        3_000,
    );
    assert_eq!(wrong.status, SnapshotStatus::Blocked);
    assert!(arena.config.live_fee_schedule.load_full().is_none());
}

#[test]
fn both_queries_and_complete_arena_reconciliation_are_required_and_errors_revoke() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    for (positions, orders) in [
        (Err("positions unavailable".into()), Ok(Vec::new())),
        (Ok(flat_positions()), Err("orders unavailable".into())),
    ] {
        let outcome =
            session.apply_snapshot(&token, positions, orders, fees(&token), &arena, 1_000);
        assert_eq!(outcome.status, SnapshotStatus::Blocked);
        assert!(!ready.load(Ordering::Acquire));
    }
    let success = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        2_000,
    );
    assert_eq!(success.status, SnapshotStatus::Ready);
    assert!(session.check_entry(&token, &arena).is_ok());

    let missing = session.apply_snapshot(
        &token,
        Ok(Vec::new()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        3_000,
    );
    assert_eq!(missing.status, SnapshotStatus::Blocked);
    assert!(missing.unresolved > 0);
    assert!(!ready.load(Ordering::Acquire));

    session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        4_000,
    );
    assert!(ready.load(Ordering::Acquire));
    let error = session.apply_snapshot(
        &token,
        Err("later outage".into()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        5_000,
    );
    assert_eq!(error.status, SnapshotStatus::Blocked);
    assert!(session.check_entry(&token, &arena).is_err());
}

#[test]
fn a_regular_orphan_order_blocks_readiness_without_fabricating_cancellation() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    let orphan = OrderAck {
        symbol: "BTCUSDT".into(),
        client_order_id: "remote_resting_without_local_owner".into(),
        status: "NEW".into(),
        order_type: "LIMIT".into(),
        orig_qty: 0.01,
        price: 50_000.0,
        ..Default::default()
    };
    let outcome = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(vec![orphan]),
        fees(&token),
        &arena,
        1_000,
    );
    assert_eq!(outcome.status, SnapshotStatus::Blocked);
    assert_eq!(outcome.orphan_orders, 1);
    assert!(!ready.load(Ordering::Acquire));

    let registry = token.executor().registry();
    registry.register_intent(
        "known_pending",
        "BTCUSDT",
        "BUY",
        "BOTH",
        "LIMIT",
        0.01,
        1_000,
    );
    let known = OrderAck {
        symbol: "BTCUSDT".into(),
        client_order_id: "known_pending".into(),
        status: "NEW".into(),
        order_type: "LIMIT".into(),
        orig_qty: 0.01,
        ..Default::default()
    };
    let slot = arena.coins[0].positions.get_slot(2);
    assert!(slot.open_with_horizon(
        true,
        100.0,
        0.01,
        1.0,
        1_000,
        0.0,
        0.0,
        PositionHorizon::Continuous
    ));
    arena.used_margin.store(1.0, Ordering::Relaxed);
    let known_outcome = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(vec![known]),
        fees(&token),
        &arena,
        2_000,
    );
    assert_eq!(known_outcome.status, SnapshotStatus::Blocked);
    assert_eq!(known_outcome.pending_regular_orders, 1);
    assert!(slot.is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 1.0);
    assert_eq!(
        registry.active_orders().len(),
        1,
        "flat position does not prove a resting order canceled"
    );
    let unknown_terminal = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        3_000,
    );
    assert_eq!(unknown_terminal.status, SnapshotStatus::Blocked);
    assert_eq!(unknown_terminal.unresolved_orders, 1);
    assert_eq!(registry.active_orders().len(), 1);
    assert!(slot.is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 1.0);
}

#[test]
fn a_reordered_fee_book_is_rejected_before_publication() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    let mut rows = fees(&token).unwrap().rows().to_vec();
    rows.reverse();
    let schedule = FeeSchedule::new(token.generation(), rows).unwrap();
    let outcome = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        Ok(schedule),
        &arena,
        1_000,
    );
    assert_eq!(outcome.status, SnapshotStatus::Blocked);
    assert!(!ready.load(Ordering::Acquire));
    assert!(arena.config.live_fee_schedule.load_full().is_none());
}

#[tokio::test]
async fn a_new_token_after_universe_change_cannot_inherit_old_authorization() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        1_000,
    );
    assert!(ready.load(Ordering::Acquire));
    quantum_arena::symbols::update_dynamic_universe(vec!["ETHUSDT".into(), "BTCUSDT".into()]);
    let new_token = session.snapshot_token();
    let transport = Transport::default();
    assert!(
        session
            .dispatch_entry(&new_token, &transport, &request(), &arena)
            .await
            .is_err()
    );
    assert_eq!(transport.configured.load(Ordering::Relaxed), 0);
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 0);
    assert!(!ready.load(Ordering::Acquire));
    session.apply_snapshot(
        &new_token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&new_token),
        &arena,
        2_000,
    );
    session
        .dispatch_entry(&new_token, &transport, &request(), &arena)
        .await
        .unwrap();
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 1);
}

#[test]
fn close_receipt_is_used_only_for_its_exact_event_and_quantity_without_fabricating_fees() {
    use god_engine_core::close_accounting::LocalCloseAccounting;
    use quantum_engine::host_oms::matching_local_close;
    let receipt = LocalCloseAccounting {
        event_time_ms: 1_000,
        quantity: 2.0,
        entry_price: 100.0,
        exit_price: 110.0,
        gross_pnl: 20.0,
        entry_fee: -0.01,
        exit_fee: 0.0,
    };
    let matched = matching_local_close(Some(receipt), 1_000, 2.0).unwrap();
    assert_eq!(matched.gross_pnl, 20.0);
    assert_eq!(matched.total_fees(), -0.01);
    assert!(matching_local_close(Some(receipt), 1_001, 2.0).is_none());
    assert!(matching_local_close(Some(receipt), 1_000, 3.0).is_none());
    assert!(matching_local_close(None, 1_000, 2.0).is_none());
}

#[test]
fn commission_cash_flow_preserves_rebates_in_the_host_cost_convention() {
    use quantum_engine::host_oms::signed_commission_cost;
    let expense = signed_commission_cost(-1.0).unwrap();
    let rebate = signed_commission_cost(0.25).unwrap();
    assert_eq!(expense, 1.0);
    assert_eq!(rebate, -0.25);
    let fees = expense + rebate;
    assert_eq!(fees, 0.75);
    assert_eq!(10.0 - 0.5 - fees, 8.75);
    assert_eq!(signed_commission_cost(0.0), Some(-0.0));
    assert!(signed_commission_cost(f64::NAN).is_none());
    assert!(signed_commission_cost(f64::INFINITY).is_none());
}

#[test]
fn balance_publication_requires_current_executor_and_finite_nonnegative_evidence() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    assert!(!ready.load(Ordering::Acquire));
    let mut published = Vec::new();
    for balance in [0.0, 37.25] {
        assert!(session.publish_current_balance(&token, balance, |value| published.push(value)));
    }
    assert_eq!(published, vec![0.0, 37.25]);
    for balance in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            !session.publish_current_balance(&token, balance, |_| panic!(
                "invalid balance reached callback"
            ))
        );
    }
    session
        .replace_executor(executor(false, false), &arena)
        .unwrap();
    assert!(!session.publish_current_balance(&token, 123.0, |_| panic!(
        "old account changed current capital"
    )));
    assert!(
        session.publish_current_balance(&session.snapshot_token(), 0.0, |value| published
            .push(value))
    );
    assert_eq!(published, vec![0.0, 37.25, 0.0]);
}

#[test]
fn snapshot_and_core_publication_share_a_gate_so_a_new_reservation_is_never_phantom_cleaned() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        1_000,
    );
    assert!(ready.load(Ordering::Acquire));
    let publication = arena.config.entry_publication_gate.lock().unwrap();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let done = Arc::new(AtomicBool::new(false));
    let worker = {
        let session = Arc::clone(&session);
        let arena = Arc::clone(&arena);
        let done = Arc::clone(&done);
        let token = token.clone();
        std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let outcome = session.apply_snapshot(
                &token,
                Ok(flat_positions()),
                Ok(Vec::new()),
                fees(&token),
                &arena,
                2_000,
            );
            done.store(true, Ordering::Release);
            outcome
        })
    };
    started_rx.recv().unwrap();
    assert!(!done.load(Ordering::Acquire));
    let slot = arena.coins[0].positions.get_slot(2);
    assert!(slot.open_with_horizon(
        true,
        100.0,
        0.01,
        1.0,
        1_000,
        0.0,
        0.0,
        PositionHorizon::Continuous
    ));
    arena.used_margin.store(1.0, Ordering::Relaxed);
    drop(publication);
    let outcome = worker.join().unwrap();
    assert_eq!(outcome.status, SnapshotStatus::Blocked);
    assert_eq!(
        outcome.error.as_deref(),
        Some("OMS_UNCONFIRMED_LOCAL_RESERVATION")
    );
    assert!(slot.is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 1.0);
    assert!(!ready.load(Ordering::Acquire));
    assert!(
        arena
            .config
            .entry_publication_paused
            .load(Ordering::Acquire)
    );
}

#[test]
fn replaced_session_rejects_old_success_and_old_error_before_any_state_mutation() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let old = session.snapshot_token();
    session.apply_snapshot(
        &old,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&old),
        &arena,
        1_000,
    );
    session
        .replace_executor(executor(false, false), &arena)
        .unwrap();
    assert!(!ready.load(Ordering::Acquire));
    assert!(session.check_entry(&old, &arena).is_err());
    let current = session.snapshot_token();
    session.apply_snapshot(
        &current,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&current),
        &arena,
        2_000,
    );
    assert!(ready.load(Ordering::Acquire));

    // A stale flat observation would close this position and release its margin if applied.
    let slot = arena.coins[0].positions.get_slot(2);
    assert!(slot.open_with_horizon(
        true,
        100.0,
        1.0,
        5.0,
        2_000,
        0.0,
        0.0,
        PositionHorizon::Continuous
    ));
    arena.used_margin.store(5.0, Ordering::Relaxed);
    let stale = session.apply_snapshot(
        &old,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&old),
        &arena,
        3_000,
    );
    assert_eq!(stale.status, SnapshotStatus::Stale);
    assert!(slot.is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 5.0);
    assert!(old.executor().registry().active_orders().is_empty());
    assert!(current.executor().registry().active_orders().is_empty());
    let mut old_open = flat_positions();
    let symbol = old_open[0].symbol.clone();
    old_open[0].position_amt = 4.0;
    old_open[0].entry_price = 100.0;
    old_open[0].leverage = 2.0;
    old_open[0].update_time = 123;
    let stale_open = session.apply_snapshot(
        &old,
        Ok(old_open),
        Ok(Vec::new()),
        fees(&old),
        &arena,
        3_500,
    );
    assert_eq!(stale_open.status, SnapshotStatus::Stale);
    let adopted_id = format!("adopted_{symbol}_123");
    assert!(old.executor().registry().get(&adopted_id).is_none());
    assert!(current.executor().registry().get(&adopted_id).is_none());
    assert_eq!(slot.quantity.load(Ordering::Relaxed), 1.0);
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 5.0);
    let stale_error = session.apply_snapshot(
        &old,
        Err("old account outage".into()),
        Ok(Vec::new()),
        fees(&old),
        &arena,
        4_000,
    );
    assert_eq!(stale_error.status, SnapshotStatus::Stale);
    assert!(
        ready.load(Ordering::Acquire),
        "old error cannot revoke a newer successful observation"
    );
    assert!(session.check_entry(&current, &arena).is_ok());
}

#[derive(Default)]
struct Transport {
    configured: AtomicUsize,
    submitted: AtomicUsize,
    pause: bool,
    pause_submit: bool,
    entered: Notify,
    resume: Notify,
}

impl EntryTransport for Transport {
    fn supports_native_iceberg(&self) -> bool {
        false
    }

    async fn configure_leverage(&self, _symbol: &str, _leverage: u32) -> Result<(), String> {
        self.configured.fetch_add(1, Ordering::Relaxed);
        if self.pause {
            self.entered.notify_one();
            self.resume.notified().await;
        }
        Ok(())
    }

    async fn submit_entry(&self, _request: &EntryRequest<'_>) -> Result<(), String> {
        self.submitted.fetch_add(1, Ordering::Relaxed);
        if self.pause_submit {
            self.entered.notify_one();
            self.resume.notified().await;
        }
        Ok(())
    }
}

#[tokio::test]
async fn unreconciled_entry_never_mutates_transport_and_ready_entry_uses_real_dispatcher() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, _) = session(true, false);
    let token = session.snapshot_token();
    let transport = Transport::default();
    assert!(
        session
            .dispatch_entry(&token, &transport, &request(), &arena)
            .await
            .is_err()
    );
    assert_eq!(transport.configured.load(Ordering::Relaxed), 0);
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 0);
    session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        1_000,
    );
    session
        .dispatch_entry(&token, &transport, &request(), &arena)
        .await
        .unwrap();
    assert_eq!(transport.configured.load(Ordering::Relaxed), 1);
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn revocation_during_leverage_await_withholds_submission() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, _) = session(true, false);
    let token = session.snapshot_token();
    session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        1_000,
    );
    let transport = Transport {
        pause: true,
        ..Default::default()
    };
    let req = request();
    let (result, ()) = tokio::join!(
        session.dispatch_entry(&token, &transport, &req, &arena),
        async {
            transport.entered.notified().await;
            session.apply_snapshot(
                &token,
                Err("current outage".into()),
                Ok(Vec::new()),
                fees(&token),
                &arena,
                2_000,
            );
            transport.resume.notify_one();
        }
    );
    assert!(result.is_err());
    assert_eq!(transport.configured.load(Ordering::Relaxed), 1);
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn replacement_during_leverage_await_withholds_old_entry_until_new_session_reconciles() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    let pinned_executor = Arc::clone(token.executor());
    session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        1_000,
    );
    let transport = Transport {
        pause: true,
        ..Default::default()
    };
    let req = request();
    let (result, ()) = tokio::join!(
        session.dispatch_entry(&token, &transport, &req, &arena),
        async {
            transport.entered.notified().await;
            let busy = session
                .replace_executor(executor(false, false), &arena)
                .unwrap_err();
            assert!(busy.starts_with("OMS_EXECUTOR_BUSY"));
            assert!(Arc::ptr_eq(
                session.snapshot_token().executor(),
                &pinned_executor
            ));
            let current = session.snapshot_token();
            session.apply_snapshot(
                &current,
                Ok(flat_positions()),
                Ok(Vec::new()),
                fees(&current),
                &arena,
                2_000,
            );
            assert!(
                !ready.load(Ordering::Acquire),
                "old account transport has not drained"
            );
            transport.resume.notify_one();
        }
    );
    assert!(result.is_err());
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 0);
    session
        .replace_executor(executor(false, false), &arena)
        .unwrap();
    let current = session.snapshot_token();
    session.apply_snapshot(
        &current,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&current),
        &arena,
        3_000,
    );
    assert!(ready.load(Ordering::Acquire));
    assert!(session.check_entry(&token, &arena).is_err());
    assert!(Arc::ptr_eq(token.executor(), &pinned_executor));
    assert!(!Arc::ptr_eq(
        session.snapshot_token().executor(),
        &pinned_executor
    ));
}

#[tokio::test]
async fn owned_entry_task_retains_old_account_through_post_ack_settlement_until_drop() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        1_000,
    );
    let task = session.begin_entry_task(&token, &arena).unwrap();
    let transport = Transport::default();
    let req = request();
    session
        .dispatch_entry(&token, &transport, &req, &arena)
        .await
        .unwrap();
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 1);

    // Submission returned, but its task still owns fill attribution/protection work.
    let registry = token.executor().registry();
    let replacement = Arc::new(OrderExecutor::new_with_shared(
        String::new(),
        "offline-contract".into(),
        false,
        Arc::clone(&registry),
        Some(Arc::clone(&arena)),
    ));
    assert!(
        session
            .replace_executor(Arc::clone(&replacement), &arena)
            .unwrap_err()
            .starts_with("OMS_EXECUTOR_BUSY")
    );
    assert!(!ready.load(Ordering::Acquire));
    assert!(Arc::ptr_eq(
        session.snapshot_token().executor(),
        token.executor()
    ));
    let fill = OrderAck {
        order_id: 42,
        symbol: req.symbol.into(),
        client_order_id: req.client_order_id.into(),
        status: "FILLED".into(),
        side: "BUY".into(),
        order_type: "MARKET".into(),
        orig_qty: req.quantity,
        executed_qty: req.quantity,
        avg_price: 100.0,
        price: 100.0,
        cum_quote: req.quantity * 100.0,
        update_time: 1_100,
        ..Default::default()
    };
    registry.apply_ack(&fill, 1_100);
    assert_eq!(
        registry.get(req.client_order_id).unwrap().executed_qty,
        req.quantity
    );
    drop(task);
    session.replace_executor(replacement, &arena).unwrap();
    assert!(!Arc::ptr_eq(
        session.snapshot_token().executor(),
        token.executor()
    ));
    assert_eq!(
        session
            .snapshot_token()
            .executor()
            .registry()
            .get(req.client_order_id)
            .unwrap()
            .executed_qty,
        req.quantity
    );
}

#[tokio::test]
async fn refresh_freezes_new_entries_and_drains_submission_without_closing_its_unconfirmed_reservation()
 {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    let token = session.snapshot_token();
    session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        1_000,
    );
    let slot = arena.coins[0].positions.get_slot(2);
    assert!(slot.open_with_horizon(
        true,
        100.0,
        0.01,
        1.0,
        1_000,
        0.0,
        0.0,
        PositionHorizon::Continuous
    ));
    arena.used_margin.store(1.0, Ordering::Relaxed);
    let transport = Transport {
        pause_submit: true,
        ..Default::default()
    };
    let req = request();
    let (entry_result, ()) = tokio::join!(
        session.dispatch_entry(&token, &transport, &req, &arena),
        async {
            transport.entered.notified().await;
            assert!(session.begin_refresh(&token, &arena));
            assert!(!ready.load(Ordering::Acquire));
            assert!(
                session
                    .check_entry(&session.snapshot_token(), &arena)
                    .is_err()
            );
            let waiting = session.await_quiescence(&token);
            tokio::pin!(waiting);
            assert!(futures_util::poll!(waiting.as_mut()).is_pending());
            let while_pending = session.apply_snapshot(
                &token,
                Ok(flat_positions()),
                Ok(Vec::new()),
                fees(&token),
                &arena,
                2_000,
            );
            assert_eq!(while_pending.status, SnapshotStatus::Blocked);
            assert!(slot.is_open());
            assert_eq!(arena.used_margin.load(Ordering::Relaxed), 1.0);
            transport.resume.notify_one();
            assert!(waiting.await);
            let after_ack = session.apply_snapshot(
                &token,
                Ok(flat_positions()),
                Ok(Vec::new()),
                fees(&token),
                &arena,
                3_000,
            );
            assert_eq!(
                after_ack.status,
                SnapshotStatus::Blocked,
                "unconfirmed reservation is not proof of a phantom position"
            );
            assert!(slot.is_open());
            assert_eq!(arena.used_margin.load(Ordering::Relaxed), 1.0);
        }
    );
    assert!(
        entry_result.is_ok(),
        "an already admitted request is not retracted"
    );
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 1);
}
