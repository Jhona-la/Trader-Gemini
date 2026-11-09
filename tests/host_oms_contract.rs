//! Actual host authorization, reconciliation and dispatch functions with synthetic observations.
//! No REST requests, live credentials, training, tapes or model loading.
use arc_swap::ArcSwap;
use execution_engine::{
    entry_dispatch::{EntryRequest, EntryRoute, EntryTransport},
    executor::{
        EDAD_MINIMA_INEXISTENCIA_MS, HORIZONTE_CONSULTA_ORDEN_MS, OrderExecutor,
        error_es_orden_inexistente, inexistencia_probada,
    },
    order_types::OrderAck,
    reconciliation::PositionRiskEntry,
};
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees, position::PositionHorizon};
use god_engine_core::{ReservedEntryOrder, entry_reservation::EntryReservation};
use quantum_engine::host_oms::{
    EntrySettlement, OmsSession, SessionToken, SnapshotStatus, entry_execution_for_host,
    release_confirmed_side,
};
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
    let rejected = session
        .replace_executor(executor(false, false), &arena)
        .unwrap_err();
    assert!(rejected.starts_with("OMS_EXECUTOR_BUSY"));
    assert_eq!(session.snapshot_token().generation(), token.generation());
    assert!(Arc::ptr_eq(
        session.snapshot_token().executor(),
        token.executor()
    ));
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
    ids: Mutex<Vec<String>>,
    fail_first_submit: bool,
    pause: bool,
    pause_submit: bool,
    entered: Notify,
    resume: Notify,
    oco: AtomicUsize,
}

impl Transport {
    fn complete_oco(&self) {
        self.oco.fetch_add(1, Ordering::Relaxed);
    }
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

    async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {
        self.ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(request.client_order_id.to_string());
        let submitted = self.submitted.fetch_add(1, Ordering::Relaxed) + 1;
        if self.pause_submit {
            self.entered.notify_one();
            self.resume.notified().await;
        }
        if self.fail_first_submit && submitted == 1 {
            return Err("synthetic first proposal rejection".into());
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

/// LG-32: cuerpo real del -2013 tal como lo formatea el cliente HTTP.
const ORDEN_INEXISTENTE: &str =
    r#"Binance API Error: status=400 Bad Request body={"code":-2013,"msg":"Order does not exist."}"#;

/// LG-32: una intención pura (sin orderId ni fills) ausente de openOrders bloqueaba
/// las entradas para siempre: nada la consultaba y `unresolved_orders` no bajaba de 1.
/// Ahora una consulta que responde -2013 pasado el plazo de la ventana de Binance la
/// cierra y el OMS vuelve a autorizar. Un -2013 demasiado pronto o cualquier otro
/// error no cierran nada.
#[test]
fn lg32_intencion_pura_con_inexistencia_probada_deja_de_bloquear() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    session.initialize_arena(&arena);
    let token = session.snapshot_token();
    let executor = token.executor();
    let registry = executor.registry();
    let creada = 1_000_000;
    registry.register_intent("lg32_nunca_aceptada", "BTCUSDT", "BUY", "LONG", "MARKET", 0.01, creada);
    let bloqueado = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        creada + 1,
    );
    assert_eq!(bloqueado.status, SnapshotStatus::Blocked);
    assert_eq!(bloqueado.unresolved_orders, 1);

    let orden = registry.get("lg32_nunca_aceptada").expect("intención registrada");
    let pronto = creada + EDAD_MINIMA_INEXISTENCIA_MS - 1;
    assert!(!executor.aplicar_consulta_de_intencion_ausente(&orden, pronto, Err(ORDEN_INEXISTENTE)));
    assert!(!executor.aplicar_consulta_de_intencion_ausente(
        &orden,
        creada + 60_000,
        Err("Network Error: operation timed out")
    ));
    assert_eq!(registry.active_orders().len(), 1, "sin prueba la intención sigue activa");

    let a_tiempo = creada + EDAD_MINIMA_INEXISTENCIA_MS;
    assert!(executor.aplicar_consulta_de_intencion_ausente(&orden, a_tiempo, Err(ORDEN_INEXISTENTE)));
    let listo = session.apply_snapshot(
        &token,
        Ok(flat_positions()),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        a_tiempo + 1,
    );
    assert_eq!(listo.status, SnapshotStatus::Ready);
    assert_eq!(listo.unresolved_orders, 0);
    assert!(ready.load(Ordering::Acquire));
}

/// LG-32: una orden que el exchange ya reconoció (orderId) no se cierra con un -2013:
/// existió, y el registro local no prueba que no tuviera fills. Un ack terminal válido
/// sin ejecución sí la cierra, por la misma validación que el resto de consultas.
#[test]
fn lg32_orden_reconocida_solo_se_cierra_con_un_ack_terminal_valido() {
    let _universe = universe();
    let (session, _ready) = session(true, false);
    let token = session.snapshot_token();
    let executor = token.executor();
    let registry = executor.registry();
    let creada = 2_000_000;
    registry.register_intent("lg32_reconocida", "BTCUSDT", "BUY", "LONG", "LIMIT", 0.01, creada);
    let nueva = OrderAck {
        symbol: "BTCUSDT".into(),
        client_order_id: "lg32_reconocida".into(),
        order_id: 77,
        side: "BUY".into(),
        status: "NEW".into(),
        order_type: "LIMIT".into(),
        orig_qty: 0.01,
        ..Default::default()
    };
    registry.apply_ack(&nueva, creada + 10);
    let orden = registry.get("lg32_reconocida").expect("orden reconocida");
    assert!(!executor.aplicar_consulta_de_intencion_ausente(&orden, creada + 60_000, Err(ORDEN_INEXISTENTE)));
    assert_eq!(registry.active_orders().len(), 1);

    let cancelada = OrderAck {
        status: "CANCELED".into(),
        ..nueva
    };
    assert!(executor.aplicar_consulta_de_intencion_ausente(&orden, creada + 60_000, Ok(&cancelada)));
    assert!(registry.active_orders().is_empty());
}

/// LG-32: sólo el código -2013 leído del JSON cuenta, y sólo dentro del horizonte en
/// que Binance garantiza encontrar las órdenes con fills (90 días).
#[test]
fn lg32_la_inexistencia_exige_el_codigo_y_el_horizonte_documentados() {
    assert!(error_es_orden_inexistente(ORDEN_INEXISTENTE));
    assert!(!error_es_orden_inexistente("-2013"));
    assert!(!error_es_orden_inexistente(
        r#"Binance API Error: status=400 Bad Request body={"code":-2011,"msg":"Unknown order sent."}"#
    ));
    assert!(!error_es_orden_inexistente("Binance API Error: status=502 body=<html>"));
    assert!(!inexistencia_probada(0, EDAD_MINIMA_INEXISTENCIA_MS));
    assert!(inexistencia_probada(1, 1 + EDAD_MINIMA_INEXISTENCIA_MS));
    assert!(!inexistencia_probada(1, EDAD_MINIMA_INEXISTENCIA_MS));
    assert!(!inexistencia_probada(1, 1 + HORIZONTE_CONSULTA_ORDEN_MS));
    assert!(!inexistencia_probada(10_000, 5_000), "reloj hacia atrás");
}

fn reserved_order(
    arena: &Arc<GlobalArena>,
    slot: usize,
    is_long: bool,
    tau_ms: u64,
) -> ReservedEntryOrder {
    let position = arena.coins[0].positions.get_slot(slot);
    let (tp, sl) = if is_long {
        (110.0, 90.0)
    } else {
        (90.0, 110.0)
    };
    assert!(position.open_with_tau_and_fee(
        is_long,
        100.0,
        1.0,
        10.0,
        1_000,
        tp,
        sl,
        PositionHorizon::Continuous,
        0.7,
        0.8,
        0.01,
        tau_ms,
    ));
    arena.used_margin.fetch_add(10.0, Ordering::Relaxed);
    let generation = position.generation.load(Ordering::Acquire);
    ReservedEntryOrder {
        order: (is_long, 100.0, 1.0, tp, sl),
        reservation: EntryReservation {
            coin_id: 0,
            slot,
            generation,
            symbol: "BTCUSDT".into(),
        },
    }
}

async fn dispatch_batch(
    session: &Arc<OmsSession>,
    arena: &Arc<GlobalArena>,
    transport: &Transport,
    entries: Vec<ReservedEntryOrder>,
) -> Vec<String> {
    let mut client_ids = Vec::with_capacity(entries.len());
    for reserved in entries {
        let token = session.snapshot_token();
        let _entry_task = session.begin_entry_task(&token, arena).unwrap();
        let settlement = EntrySettlement::acquire(reserved.reservation.clone(), true).await;
        let client_id = execution_engine::entry_dispatch::new_entry_client_id(true);
        let request = EntryRequest {
            client_order_id: &client_id,
            ..request()
        };
        let outcome = session
            .dispatch_entry(&token, transport, &request, arena)
            .await;
        match outcome {
            Ok(()) => {
                reserved
                    .reservation
                    .confirmar_llenado(arena, Some(reserved.order.2))
                    .unwrap();
                transport.complete_oco();
                settlement.release();
            }
            Err(_) => {
                if reserved.reservation.cancel(arena).is_ok() {
                    settlement.release();
                }
            }
        }
        client_ids.push(client_id);
    }
    client_ids
}

#[tokio::test]
async fn f28_batch_reaches_oms_twice_with_independent_sessions_and_uuids() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, _) = session(false, true);
    let transport = Transport::default();
    let entries = vec![
        reserved_order(&arena, 0, true, 15_000),
        reserved_order(&arena, 1, true, 900_000),
    ];
    let ids = dispatch_batch(&session, &arena, &transport, entries).await;

    assert_eq!(transport.configured.load(Ordering::Relaxed), 2);
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 2);
    assert_eq!(transport.oco.load(Ordering::Relaxed), 2);
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    let recorded = transport.ids.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(recorded.as_slice(), ids.as_slice());
    for slot in [0, 1] {
        let position = arena.coins[0].positions.get_slot(slot);
        assert!(position.is_open());
        assert!(position.exchange_confirmed.load(Ordering::Acquire));
    }
}

#[tokio::test]
async fn f28_rejection_of_one_proposal_does_not_cancel_its_sibling() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, _) = session(false, true);
    let transport = Transport {
        fail_first_submit: true,
        ..Default::default()
    };
    let entries = vec![
        reserved_order(&arena, 0, true, 15_000),
        reserved_order(&arena, 1, true, 900_000),
    ];
    let ids = dispatch_batch(&session, &arena, &transport, entries).await;

    assert_eq!(ids.len(), 2);
    assert_eq!(transport.submitted.load(Ordering::Relaxed), 2);
    assert_eq!(transport.oco.load(Ordering::Relaxed), 1);
    assert!(!arena.coins[0].positions.get_slot(0).is_open());
    let sibling = arena.coins[0].positions.get_slot(1);
    assert!(sibling.is_open());
    assert!(sibling.exchange_confirmed.load(Ordering::Acquire));
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 10.0);
}

#[tokio::test]
async fn f28_partial_fill_is_not_total_and_pending_settlement_does_not_release_early() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, _) = session(false, true);
    let first = reserved_order(&arena, 0, true, 15_000);
    let second = reserved_order(&arena, 1, true, 900_000);
    let settlement = EntrySettlement::acquire(first.reservation.clone(), true).await;
    let waiting_reservation = second.reservation.clone();
    let waiter = tokio::spawn(async move {
        let next = EntrySettlement::acquire(waiting_reservation, true).await;
        next.release();
    });
    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
    assert!(
        !waiter.is_finished(),
        "el hermano espera el settlement LONG"
    );

    let client_id = "f28_partial_entry";
    let registry = session.snapshot_token().executor().registry();
    registry.register_intent(client_id, "BTCUSDT", "BUY", "LONG", "LIMIT", 1.0, 1_000);
    let partial = OrderAck {
        order_id: 28,
        symbol: "BTCUSDT".into(),
        client_order_id: client_id.into(),
        status: "PARTIALLY_FILLED".into(),
        side: "BUY".into(),
        order_type: "LIMIT".into(),
        orig_qty: 1.0,
        executed_qty: 0.4,
        avg_price: 100.0,
        price: 100.0,
        cum_quote: 40.0,
        update_time: 1_100,
        ..Default::default()
    };
    registry.apply_ack(&partial, 1_100);
    assert_eq!(
        entry_execution_for_host(registry.get(client_id).as_ref(), false, 1.0),
        None,
        "un parcial activo no acredita cantidad terminal ni fill total"
    );
    let position = arena.coins[0].positions.get_slot(0);
    assert!(!position.exchange_confirmed.load(Ordering::Acquire));
    assert_eq!(position.quantity.load(Ordering::Relaxed), 1.0);
    drop(settlement);
    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
    assert!(
        !waiter.is_finished(),
        "salir de la rama pendiente no puede liberar settlement"
    );

    let terminal = OrderAck {
        status: "EXPIRED".into(),
        ..partial
    };
    registry.apply_ack(&terminal, 1_200);
    assert_eq!(
        entry_execution_for_host(registry.get(client_id).as_ref(), false, 1.0),
        Some(0.4)
    );
    first
        .reservation
        .confirmar_llenado(&arena, Some(0.4))
        .unwrap();
    assert_eq!(position.quantity.load(Ordering::Relaxed), 0.4);
    assert!(position.exchange_confirmed.load(Ordering::Acquire));
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 14.0);
    assert_eq!(entry_execution_for_host(None, true, 1.0), Some(1.0));

    EntrySettlement::release_for(&first.reservation);
    waiter.await.unwrap();
}


fn remote_leg(symbol: &str, side: &str, amount: f64) -> PositionRiskEntry {
    PositionRiskEntry {
        symbol: symbol.into(),
        position_side: side.into(),
        position_amt: amount,
        entry_price: 100.0,
        leverage: 20.0,
        update_time: 1_200,
        ..Default::default()
    }
}

fn confirm_batch(arena: &Arc<GlobalArena>, directions: &[bool]) {
    for (slot, is_long) in directions.iter().copied().enumerate() {
        let reserved = reserved_order(arena, slot, is_long, 15_000 + slot as u64 * 885_000);
        reserved
            .reservation
            .confirmar_llenado(arena, Some(1.0))
            .unwrap();
    }
}

fn slot_identities(arena: &GlobalArena) -> [(u64, u64, f64); 2] {
    core::array::from_fn(|slot| {
        let position = arena.coins[0].positions.get_slot(slot);
        (
            position.generation.load(Ordering::Acquire),
            position.entry_tau_ms.load(Ordering::Relaxed),
            position.quantity.load(Ordering::Relaxed),
        )
    })
}

#[test]
fn f28_refresh_after_long_long_batch_preserves_both_slots_without_duplicate_adoption() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    confirm_batch(&arena, &[true, true]);
    let before = slot_identities(&arena);
    let token = session.snapshot_token();
    let outcome = session.apply_snapshot(
        &token,
        Ok(vec![remote_leg("BTCUSDT", "LONG", 2.0), remote_leg("ETHUSDT", "BOTH", 0.0)]),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        2_000,
    );

    assert_eq!(outcome.status, SnapshotStatus::Ready);
    assert_eq!(outcome.unresolved, 0);
    assert!(ready.load(Ordering::Acquire));
    assert!(!arena.config.entry_publication_paused.load(Ordering::Acquire));
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 20.0);
    assert!(arena.coins[0].positions.get_slot(0).is_open());
    assert!(arena.coins[0].positions.get_slot(1).is_open());
    assert_eq!(slot_identities(&arena), before);
    assert!(token.executor().registry().active_orders().is_empty());
}

#[test]
fn f28_refresh_after_long_short_batch_accepts_hedge_and_preserves_both_sides() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    confirm_batch(&arena, &[true, false]);
    let token = session.snapshot_token();
    let outcome = session.apply_snapshot(
        &token,
        Ok(vec![
            remote_leg("BTCUSDT", "LONG", 1.0),
            remote_leg("BTCUSDT", "SHORT", -1.0),
            remote_leg("ETHUSDT", "BOTH", 0.0),
        ]),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        2_000,
    );

    assert_eq!(outcome.status, SnapshotStatus::Ready);
    assert_eq!(outcome.unresolved, 0);
    assert!(ready.load(Ordering::Acquire));
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 20.0);
    assert!(arena.coins[0].positions.get_slot(0).is_long.load(Ordering::Relaxed));
    assert!(!arena.coins[0].positions.get_slot(1).is_long.load(Ordering::Relaxed));
    assert!(token.executor().registry().active_orders().is_empty());
}

#[test]
fn f28_refresh_with_ambiguous_multi_slot_drift_preserves_identities_and_blocks() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    confirm_batch(&arena, &[true, true]);
    let before = slot_identities(&arena);
    let token = session.snapshot_token();
    let outcome = session.apply_snapshot(
        &token,
        Ok(vec![remote_leg("BTCUSDT", "LONG", 3.0), remote_leg("ETHUSDT", "BOTH", 0.0)]),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        2_000,
    );

    assert_eq!(outcome.status, SnapshotStatus::Blocked);
    assert!(outcome.unresolved > 0);
    assert!(!ready.load(Ordering::Acquire));
    assert!(arena.config.entry_publication_paused.load(Ordering::Acquire));
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 20.0);
    assert_eq!(slot_identities(&arena), before);
}

#[test]
fn f28_refresh_flat_short_leg_preserves_the_long_leg() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    confirm_batch(&arena, &[true, false]);
    let long = arena.coins[0].positions.get_slot(0);
    let long_generation = long.generation.load(Ordering::Acquire);
    let long_tau = long.entry_tau_ms.load(Ordering::Relaxed);
    let token = session.snapshot_token();
    let outcome = session.apply_snapshot(
        &token,
        Ok(vec![
            remote_leg("BTCUSDT", "LONG", 1.0),
            remote_leg("BTCUSDT", "SHORT", 0.0),
            remote_leg("ETHUSDT", "BOTH", 0.0),
        ]),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        2_000,
    );

    assert_eq!(outcome.status, SnapshotStatus::Ready);
    assert!(ready.load(Ordering::Acquire));
    assert!(long.is_open());
    assert_eq!(long.generation.load(Ordering::Acquire), long_generation);
    assert_eq!(long.entry_tau_ms.load(Ordering::Relaxed), long_tau);
    assert!(!arena.coins[0].positions.get_slot(1).is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 10.0);
}

#[test]
fn f28_refresh_missing_short_evidence_does_not_close_either_leg() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    confirm_batch(&arena, &[true, false]);
    let before = slot_identities(&arena);
    let token = session.snapshot_token();
    let outcome = session.apply_snapshot(
        &token,
        Ok(vec![remote_leg("BTCUSDT", "LONG", 1.0), remote_leg("ETHUSDT", "BOTH", 0.0)]),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        2_000,
    );

    assert_eq!(outcome.status, SnapshotStatus::Blocked);
    assert!(!ready.load(Ordering::Acquire));
    assert!(arena.coins[0].positions.get_slot(0).is_open());
    assert!(arena.coins[0].positions.get_slot(1).is_open());
    assert_eq!(slot_identities(&arena), before);
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 20.0);
}

#[test]
fn f28_duplicate_remote_legs_on_same_side_remain_ambiguous_and_block_admission() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = session(true, false);
    confirm_batch(&arena, &[true, true]);
    let token = session.snapshot_token();
    let outcome = session.apply_snapshot(
        &token,
        Ok(vec![
            remote_leg("BTCUSDT", "LONG", 1.0),
            remote_leg("BTCUSDT", "LONG", 1.0),
            remote_leg("ETHUSDT", "BOTH", 0.0),
        ]),
        Ok(Vec::new()),
        fees(&token),
        &arena,
        2_000,
    );

    assert_eq!(outcome.status, SnapshotStatus::Blocked);
    assert!(outcome.unresolved > 0);
    assert!(!ready.load(Ordering::Acquire));
    assert!(arena.config.entry_publication_paused.load(Ordering::Acquire));
}


#[tokio::test]
async fn f28_terminal_partial_release_requires_protection_guard_and_drains_quiescence() {
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
    let first = reserved_order(&arena, 0, false, 15_000);
    let second = reserved_order(&arena, 1, false, 900_000);
    let entry_task = session.begin_entry_task(&token, &arena).unwrap();
    let settlement = EntrySettlement::acquire(first.reservation.clone(), false).await;
    let waiting_reservation = second.reservation.clone();
    let waiter = tokio::spawn(async move {
        let next = EntrySettlement::acquire(waiting_reservation, false).await;
        next.release();
    });
    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
    assert!(!waiter.is_finished());

    let client_id = "f28_terminal_short";
    let registry = token.executor().registry();
    registry.register_intent(client_id, "BTCUSDT", "SELL", "SHORT", "LIMIT", 1.0, 1_000);
    let partial = OrderAck {
        order_id: 29,
        symbol: "BTCUSDT".into(),
        client_order_id: client_id.into(),
        status: "PARTIALLY_FILLED".into(),
        side: "SELL".into(),
        order_type: "LIMIT".into(),
        orig_qty: 1.0,
        executed_qty: 0.4,
        avg_price: 100.0,
        price: 100.0,
        cum_quote: 40.0,
        update_time: 1_100,
        ..Default::default()
    };
    registry.apply_ack(&partial, 1_100);
    assert_eq!(release_confirmed_side(&arena, "BTCUSDT", false, false), 0);
    assert!(!waiter.is_finished());

    registry.apply_ack(
        &OrderAck {
            status: "EXPIRED".into(),
            ..partial
        },
        1_200,
    );
    first
        .reservation
        .confirmar_llenado(&arena, Some(0.4))
        .unwrap();
    drop(settlement);
    assert!(!waiter.is_finished());
    assert_eq!(release_confirmed_side(&arena, "BTCUSDT", false, false), 0);
    // Este booleano representa la guarda que el watchdog deriva de
    // ensure_position_protected; no acredita un round-trip OCO HTTP.
    assert_eq!(release_confirmed_side(&arena, "BTCUSDT", false, true), 1);
    drop(entry_task);
    assert!(session.await_quiescence(&token).await);
    waiter.await.unwrap();

    let stale_reservation = EntryReservation {
        generation: first.reservation.generation - 1,
        ..first.reservation.clone()
    };
    let stale = EntrySettlement::acquire(stale_reservation, false).await;
    assert_eq!(
        release_confirmed_side(&arena, "BTCUSDT", false, true),
        0,
        "la protección de una generación nueva no libera el settlement anterior"
    );
    stale.release();
}


#[tokio::test]
async fn f28_deadline_cancels_reservation_revokes_admission_and_allows_quiescence() {
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
    let first = reserved_order(&arena, 0, false, 15_000);
    let second = reserved_order(&arena, 1, false, 900_000);
    let entry_task = session.begin_entry_task(&token, &arena).unwrap();
    let _settlement = EntrySettlement::acquire(first.reservation.clone(), false).await;
    let waiting_reservation = second.reservation.clone();
    let waiter = tokio::spawn(async move {
        let next = EntrySettlement::acquire(waiting_reservation, false).await;
        next.release();
    });
    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
    assert!(!waiter.is_finished());

    assert!(quantum_engine::host_oms::cancel_entry_after_deadline(
        &session,
        &token,
        &arena,
        &first.reservation,
    ));
    assert!(!arena.coins[0].positions.get_slot(0).is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 10.0);
    assert!(!ready.load(Ordering::Acquire));
    assert!(arena.config.entry_publication_paused.load(Ordering::Acquire));
    assert!(session.check_entry(&token, &arena).is_err());
    drop(entry_task);
    assert!(session.await_quiescence(&token).await);
    waiter.await.unwrap();
}


mod f28_oco_roundtrip {
    use super::*;
    use execution_engine::{executor::ExecutionProvider, ioc_evidence::DestinoReserva};
    use std::collections::HashMap;
    use tokio::{
        io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
        net::TcpStream,
    };

    #[derive(Debug)]
    struct RecordedRequest {
        line: String,
        params: HashMap<String, String>,
    }

    async fn read_request(socket: &mut TcpStream) -> RecordedRequest {
        let mut reader = BufReader::new(socket);
        let mut line = String::new();
        reader.read_line(&mut line).await.expect("request line");
        let request_line = line.trim_end().to_string();
        let mut content_length = 0usize;
        loop {
            line.clear();
            let read = reader.read_line(&mut line).await.expect("request headers");
            if read == 0 || line == "\r\n" || line == "\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    content_length = value.trim().parse().expect("numeric content-length");
                }
            }
        }
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body).await.expect("request body");
        let params = request_line
            .split_whitespace()
            .nth(1)
            .and_then(|target| target.split_once('?'))
            .map(|(_, query)| {
                url::form_urlencoded::parse(query.as_bytes())
                    .into_owned()
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        RecordedRequest {
            line: request_line,
            params,
        }
    }

    async fn respond(mut socket: TcpStream, status: &str, body: String) {
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("loopback response");
        socket.shutdown().await.expect("loopback close");
    }

    #[tokio::test]
    async fn host_rest_terminal_oco_receipt_releases_and_reaches_quiescence() {
        let _universe = universe();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback listener");
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let server = tokio::spawn(async move {
            let mut entry_queries = 0usize;
            let mut oco_receipts = 0usize;
            for _ in 0..4 {
                let (mut socket, _) = listener.accept().await.expect("accept request");
                let request = read_request(&mut socket).await;
                let line = request.line.clone();
                let params = request.params.clone();
                recorded
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(request);
                if line.starts_with("GET /fapi/v1/order?") {
                    entry_queries += 1;
                    let status = if entry_queries == 1 {
                        "PARTIALLY_FILLED"
                    } else {
                        "EXPIRED"
                    };
                    let body = format!(
                        "{{\"symbol\":\"BTCUSDT\",\"clientOrderId\":\"f28_oco_roundtrip\",\"orderId\":28028,\"side\":\"SELL\",\"status\":\"{status}\",\"origQty\":\"1.0\",\"executedQty\":\"0.4\",\"avgPrice\":\"100.0\",\"price\":\"100.0\"}}"
                    );
                    respond(socket, "200 OK", body).await;
                } else if line.starts_with("POST /fapi/v1/algoOrder?") {
                    oco_receipts += 1;
                    let client_algo_id = params
                        .get("clientAlgoId")
                        .cloned()
                        .unwrap_or_default();
                    let body = format!(
                        "{{\"algoId\":{},\"clientAlgoId\":\"{}\",\"algoStatus\":\"NEW\"}}",
                        90_000 + oco_receipts,
                        client_algo_id
                    );
                    respond(socket, "200 OK", body).await;
                } else {
                    respond(
                        socket,
                        "400 Bad Request",
                        "{\"code\":-9999,\"msg\":\"unexpected loopback route\"}".into(),
                    )
                    .await;
                }
            }
        });

        let executor = OrderExecutor::new(String::new(), "offline-contract".into(), false)
            .con_base_de_prueba(&base);
        executor.is_hedge_mode.store(false, Ordering::Relaxed);
        let executor = Arc::new(executor);
        let ready = Arc::new(AtomicBool::new(true));
        let session = Arc::new(OmsSession::new(
            Arc::new(ArcSwap::from(Arc::clone(&executor))),
            Arc::clone(&ready),
        ));
        let arena = GlobalArena::build_in_own_stack(100.0);
        let token = session.snapshot_token();
        session.apply_snapshot(
            &token,
            Ok(flat_positions()),
            Ok(Vec::new()),
            fees(&token),
            &arena,
            1_000,
        );
        let first = reserved_order(&arena, 0, false, 15_000);
        let second = reserved_order(&arena, 1, false, 900_000);
        let entry_task = session.begin_entry_task(&token, &arena).unwrap();
        let _settlement = EntrySettlement::acquire(first.reservation.clone(), false).await;
        let waiting_reservation = second.reservation.clone();
        let waiter = tokio::spawn(async move {
            let next = EntrySettlement::acquire(waiting_reservation, false).await;
            next.release();
        });
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        assert!(!waiter.is_finished());

        let client_id = "f28_oco_roundtrip";
        executor.registry().register_intent(
            client_id,
            "BTCUSDT",
            "SELL",
            "BOTH",
            "LIMIT",
            1.0,
            1_000,
        );
        let destination = execution_engine::entry_dispatch::resolve_terminal_entry_until(
            executor.as_ref(),
            "BTCUSDT",
            client_id,
            "AMBIGUOUS: loopback round-trip",
            std::time::Duration::from_secs(5),
        )
        .await;
        assert_eq!(destination, DestinoReserva::Confirmar { ejecutada: 0.4 });
        first
            .reservation
            .confirmar_llenado(&arena, Some(0.4))
            .unwrap();
        let position = arena.coins[0].positions.get_slot(0);
        assert_eq!(position.quantity.load(Ordering::Relaxed), 0.4);
        assert!(position.exchange_confirmed.load(Ordering::Acquire));
        assert!(!waiter.is_finished(), "no release before the OCO receipt");

        let oco_result = executor
            .execute_oco_order(
                "BTCUSDT",
                false,
                0.4,
                90.0,
                110.0,
                0.001,
                0.1,
                client_id,
            )
            .await;
        assert!(oco_result.is_ok(), "{oco_result:?}");
        assert_eq!(
            release_confirmed_side(&arena, "BTCUSDT", false, oco_result.is_ok()),
            1
        );
        drop(entry_task);
        assert!(session.await_quiescence(&token).await);
        waiter.await.unwrap();
        server.await.unwrap();

        let requests = requests.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(requests.len(), 4);
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.line.starts_with("GET /fapi/v1/order?"))
                .count(),
            2
        );
        assert!(requests.iter().all(|request| {
            !request.line.starts_with("POST /fapi/v1/order?")
        }), "no new entry POST is allowed during recovery");
        let oco: Vec<_> = requests
            .iter()
            .filter(|request| request.line.starts_with("POST /fapi/v1/algoOrder?"))
            .collect();
        assert_eq!(oco.len(), 2);
        for request in oco {
            let params = &request.params;
            assert_eq!(params.get("symbol").map(String::as_str), Some("BTCUSDT"));
            assert_eq!(params.get("side").map(String::as_str), Some("BUY"));
            assert_eq!(params.get("reduceOnly").map(String::as_str), Some("true"));
            assert!(!params.contains_key("positionSide"));
            assert_eq!(
                params
                    .get("quantity")
                    .and_then(|value| value.parse::<f64>().ok()),
                Some(0.4)
            );
            assert_eq!(params.get("algoType").map(String::as_str), Some("CONDITIONAL"));
            let order_type = params.get("type").map(String::as_str).unwrap_or("");
            let (suffix, trigger) = match order_type {
                "STOP_MARKET" => ("_SL", 110.0),
                "TAKE_PROFIT_MARKET" => ("_TP", 90.0),
                other => panic!("unexpected OCO type {other}"),
            };
            assert_eq!(
                params.get("clientAlgoId").map(String::as_str),
                Some(format!("{client_id}{suffix}").as_str())
            );
            assert_eq!(
                params
                    .get("triggerPrice")
                    .and_then(|value| value.parse::<f64>().ok()),
                Some(trigger)
            );
        }
    }
}
