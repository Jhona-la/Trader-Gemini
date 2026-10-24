//! OMS authorization used by the host, including its executor replacement and entry path.
//!
//! A successful regular-open-orders query is not proof that orphan orders are resolved.
//! Session replacement and snapshot application share a short synchronous lock; no network
//! future holds that lock. Admission is checked immediately before each entry transport stage.
//! Revocation cannot retract a request already admitted to the transport or an exchange fill.
use arc_swap::ArcSwap;
use execution_engine::{
    entry_dispatch::{self, EntryRequest, EntryTransport},
    executor::{ExecutionProvider, OrderExecutor},
    order_types::OrderAck,
    reconciliation::{self, PositionRiskEntry},
};
use god_engine_core::close_accounting::LocalCloseAccounting;
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees};
use std::collections::HashSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::Notify;

pub struct OmsSession {
    executors: Arc<ArcSwap<OrderExecutor>>,
    ready: Arc<AtomicBool>,
    state: Mutex<SessionState>,
    quiescent: Notify,
    refresh_serial: tokio::sync::Mutex<()>,
}

struct SessionState {
    generation: u64,
    approved_universe: Option<Arc<Vec<String>>>,
    in_flight: usize,
}

#[derive(Clone)]
pub struct SessionToken {
    generation: u64,
    executor: Arc<OrderExecutor>,
}

impl SessionToken {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn executor(&self) -> &Arc<OrderExecutor> {
        &self.executor
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotStatus {
    Ready,
    Blocked,
    Stale,
}

#[derive(Debug)]
pub struct SnapshotOutcome {
    pub status: SnapshotStatus,
    pub adopted: usize,
    pub adjustments: usize,
    pub unresolved: usize,
    pub orphan_orders: usize,
    pub unresolved_orders: usize,
    pub pending_regular_orders: usize,
    pub pruned: usize,
    pub error: Option<String>,
}

impl SnapshotOutcome {
    fn empty(status: SnapshotStatus) -> Self {
        Self {
            status,
            adopted: 0,
            adjustments: 0,
            unresolved: 0,
            orphan_orders: 0,
            unresolved_orders: 0,
            pending_regular_orders: 0,
            pruned: 0,
            error: None,
        }
    }
}

impl OmsSession {
    pub fn new(executors: Arc<ArcSwap<OrderExecutor>>, ready: Arc<AtomicBool>) -> Self {
        // Demo/Testnet is a live exchange session. Only explicit local paper needs no REST proof.
        ready.store(executors.load().is_paper_trading(), Ordering::Release);
        Self {
            executors,
            ready,
            state: Mutex::new(SessionState {
                generation: 0,
                approved_universe: None,
                in_flight: 0,
            }),
            quiescent: Notify::new(),
            refresh_serial: tokio::sync::Mutex::new(()),
        }
    }

    pub fn snapshot_token(&self) -> SessionToken {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        SessionToken {
            generation: state.generation,
            executor: self.executors.load_full(),
        }
    }

    fn token_is_current(&self, generation: u64, token: &SessionToken) -> bool {
        generation == token.generation && Arc::ptr_eq(&self.executors.load_full(), &token.executor)
    }

    /// Called immediately after the host creates its arena, before any core can use it.
    pub fn initialize_arena(&self, arena: &GlobalArena) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.approved_universe = None;
        let paper = self.executors.load().is_paper_trading();
        self.reset_fee_requirement(arena, paper);
        self.ready.store(paper, Ordering::Release);
    }

    fn reset_fee_requirement(&self, arena: &GlobalArena, paper: bool) {
        arena
            .config
            .entry_publication_paused
            .store(!paper, Ordering::Release);
        arena
            .config
            .observed_fees_required
            .store(!paper, Ordering::Release);
        arena.config.live_fee_schedule.store(None);
    }

    /// Every host executor replacement goes through this operation before publishing the executor.
    pub fn replace_executor(
        &self,
        executor: Arc<OrderExecutor>,
        arena: &GlobalArena,
    ) -> Result<u64, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Also serialize with a core proposal not yet handed to its queued task.
        let _publication = arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.ready.store(false, Ordering::Release);
        state.approved_universe = None;
        self.reset_fee_requirement(arena, self.executors.load().is_paper_trading());
        arena
            .config
            .entry_publication_paused
            .store(true, Ordering::Release);
        if state.in_flight > 0
            || arena.coins.iter().any(|coin| {
                coin.positions
                    .slots()
                    .iter()
                    .any(|slot| slot.is_open() && !slot.exchange_confirmed.load(Ordering::Acquire))
            })
        {
            return Err(
                "OMS_EXECUTOR_BUSY: entry transport or settlement still owns this session".into(),
            );
        }
        let next = state
            .generation
            .checked_add(1)
            .ok_or_else(|| "OMS_GENERATION_EXHAUSTED".to_string())?;
        state.generation = next;
        let paper = executor.is_paper_trading();
        self.reset_fee_requirement(arena, paper);
        self.executors.store(executor);
        self.ready.store(paper, Ordering::Release);
        self.quiescent.notify_waiters();
        Ok(next)
    }

    pub fn check_entry(&self, token: &SessionToken, arena: &GlobalArena) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.check_entry_locked(&mut state, token, arena)
    }

    fn check_entry_locked(
        &self,
        state: &mut SessionState,
        token: &SessionToken,
        arena: &GlobalArena,
    ) -> Result<(), String> {
        if !self.token_is_current(state.generation, token) {
            return Err("OMS_ENTRY_WITHHELD: stale executor session".into());
        }
        let current = self.ready.load(Ordering::Acquire)
            && !arena
                .config
                .entry_publication_paused
                .load(Ordering::Acquire);
        let fee_evidence = if token.executor.is_paper_trading() {
            !arena.config.observed_fees_required.load(Ordering::Acquire)
        } else {
            let universe = quantum_arena::symbols::get_active_universe_ref();
            state
                .approved_universe
                .as_ref()
                .is_some_and(|approved| Arc::ptr_eq(approved, &universe))
                && arena
                    .config
                    .live_fee_schedule
                    .load()
                    .as_ref()
                    .is_some_and(|book| {
                        book.generation() == token.generation
                            && book.rows().len() == universe.len()
                            && !universe.is_empty()
                            && book.rows().iter().zip(universe.iter()).enumerate().all(
                                |(i, (row, symbol))| {
                                    row.symbol.eq_ignore_ascii_case(symbol)
                                        && arena.config.fee_rates_for(i)
                                            == Some((row.maker, row.taker))
                                },
                            )
                    })
        };
        if current && fee_evidence {
            Ok(())
        } else {
            if !token.executor.is_paper_trading() && !fee_evidence {
                state.approved_universe = None;
                self.ready.store(false, Ordering::Release);
                self.reset_fee_requirement(arena, false);
            }
            Err("OMS_ENTRY_WITHHELD: session is stale or unreconciled".into())
        }
    }

    /// Freeze new admission before any potentially slow private query. An old task may
    /// finish its already admitted transport, but cannot submit another entry stage.
    pub fn begin_refresh(&self, token: &SessionToken, arena: &GlobalArena) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !self.token_is_current(state.generation, token) {
            return false;
        }
        self.ready.store(false, Ordering::Release);
        state.approved_universe = None;
        self.reset_fee_requirement(arena, token.executor.is_paper_trading());
        arena
            .config
            .entry_publication_paused
            .store(true, Ordering::Release);
        true
    }

    /// The callback is synchronous and must not reenter this session. Current zero is a valid
    /// account balance; an old executor's observation never writes the new session's capital.
    pub fn publish_current_balance<F: FnOnce(f64)>(
        &self,
        token: &SessionToken,
        balance: f64,
        publish: F,
    ) -> bool {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if self.token_is_current(state.generation, token) && balance.is_finite() && balance >= 0.0 {
            publish(balance);
            true
        } else {
            false
        }
    }

    pub async fn await_quiescence(&self, token: &SessionToken) -> bool {
        loop {
            let notified = self.quiescent.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if !self.token_is_current(state.generation, token) {
                    return false;
                }
                if state.in_flight == 0 {
                    return true;
                }
            }
            notified.await;
        }
    }

    fn admit_transport<'a>(
        &'a self,
        token: &SessionToken,
        arena: &GlobalArena,
    ) -> Result<TransportPermit<'a>, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.check_entry_locked(&mut state, token, arena)?;
        state.in_flight = state
            .in_flight
            .checked_add(1)
            .ok_or("OMS_IN_FLIGHT_OVERFLOW")?;
        Ok(TransportPermit { session: self })
    }

    /// Held by the complete host task, including post-ack fill attribution, protection and
    /// rollback. Executor replacement cannot move that settlement onto a different account.
    pub fn begin_entry_task(
        self: &Arc<Self>,
        token: &SessionToken,
        arena: &GlobalArena,
    ) -> Result<OwnedEntryPermit, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.check_entry_locked(&mut state, token, arena)?;
        state.in_flight = state
            .in_flight
            .checked_add(1)
            .ok_or("OMS_IN_FLIGHT_OVERFLOW")?;
        Ok(OwnedEntryPermit {
            session: Arc::clone(self),
        })
    }

    fn release_admission(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.in_flight -= 1;
        if state.in_flight == 0 {
            self.quiescent.notify_waiters();
        }
    }

    /// Applies only an observation obtained with the current executor. Stale successes AND
    /// stale errors leave the new session, registry and arena untouched.
    pub fn apply_snapshot(
        &self,
        token: &SessionToken,
        positions: Result<Vec<PositionRiskEntry>, String>,
        orders: Result<Vec<OrderAck>, String>,
        fees: Result<FeeSchedule, String>,
        arena: &GlobalArena,
        now_ms: u64,
    ) -> SnapshotOutcome {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !self.token_is_current(state.generation, token) {
            return SnapshotOutcome::empty(SnapshotStatus::Stale);
        }
        // Lock order is always session -> publication. Core uses only publication, and
        // rechecks the pause/book under it before publishing a position or debiting capital.
        let _publication = arena
            .config
            .entry_publication_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.approved_universe = None;
        self.ready.store(false, Ordering::Release);
        if state.in_flight > 0 {
            self.reset_fee_requirement(arena, token.executor.is_paper_trading());
            let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
            outcome.error = Some("OMS_TRANSPORT_STILL_IN_FLIGHT".into());
            return outcome;
        }
        if token.executor.is_paper_trading() {
            self.reset_fee_requirement(arena, true);
            self.ready.store(true, Ordering::Release);
            return SnapshotOutcome::empty(SnapshotStatus::Ready);
        }
        self.ready.store(false, Ordering::Release);
        self.reset_fee_requirement(arena, false);
        let (positions, orders, fees) = match (positions, orders, fees) {
            (Ok(positions), Ok(orders), Ok(fees)) => (positions, orders, fees),
            (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
                let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
                outcome.error = Some(error);
                return outcome;
            }
        };
        let universe = quantum_arena::symbols::get_active_universe_ref();
        let expected: HashSet<_> = universe.iter().map(|s| s.to_ascii_uppercase()).collect();
        if fees.generation() != token.generation
            || expected.is_empty()
            || expected.len() != universe.len()
            || fees.rows().len() != universe.len()
            || fees
                .rows()
                .iter()
                .zip(universe.iter())
                .any(|(row, symbol)| !row.symbol.eq_ignore_ascii_case(symbol))
        {
            let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
            outcome.error = Some("OMS_FEES_INCOMPLETE_OR_WRONG_SESSION".into());
            return outcome;
        }
        // Publish this session's complete symbol schedule before authorization. Global legacy
        // maker/taker scalars are not evidence that every symbol has those account rates.
        arena.config.live_fee_schedule.store(Some(Arc::new(fees)));
        let registry = token.executor.registry();
        let report = reconciliation::reconcile_with_orders(&positions, &orders, &registry);
        let remote_ids: HashSet<_> = orders.iter().map(|o| o.client_order_id.as_str()).collect();
        // Known resting orders may have no position yet. An active local intent absent
        // from openOrders instead has unknown terminal status: retain it and block entries.
        let unresolved_orders = registry
            .active_orders()
            .iter()
            .filter(|o| !remote_ids.contains(o.client_order_id.as_str()))
            .count();
        // This endpoint lacks a proven pending-entry margin reservation contract. Do not
        // close a flat remote position's local reservation while a resting order may fill.
        // Conditional algo protection is a separate endpoint; no regular order is canceled here.
        if !orders.is_empty() || unresolved_orders > 0 {
            let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
            outcome.orphan_orders = report.orphan_remote_orders.len();
            outcome.unresolved_orders = unresolved_orders;
            outcome.pending_regular_orders = orders.len();
            return outcome;
        }
        if arena.coins.iter().any(|coin| {
            coin.positions
                .slots()
                .iter()
                .any(|p| p.is_open() && !p.exchange_confirmed.load(Ordering::Acquire))
        }) {
            let mut outcome = SnapshotOutcome::empty(SnapshotStatus::Blocked);
            outcome.error = Some("OMS_UNCONFIRMED_LOCAL_RESERVATION".into());
            return outcome;
        }
        let adopted = report.apply_to_registry_with_rates(&registry, |symbol| {
            quantum_arena::symbols::get_coin_id(symbol)
                .and_then(|ci| arena.config.fee_rates_for(ci))
                .map(|(_, taker)| taker)
        });
        let arena_report = reconciliation::reconcile_arena_checked(&positions, arena, now_ms);
        // Do not manufacture CANCELED acknowledgements merely because no position exists:
        // a regular resting order can be live before it creates a position.
        let pruned = registry.prune_terminated(now_ms.saturating_sub(600_000));
        let ready = arena_report.unresolved.is_empty()
            && report.orphan_remote_orders.is_empty()
            && unresolved_orders == 0;
        if ready {
            state.approved_universe = Some(universe);
        }
        arena
            .config
            .entry_publication_paused
            .store(!ready, Ordering::Release);
        self.ready.store(ready, Ordering::Release);
        SnapshotOutcome {
            status: if ready {
                SnapshotStatus::Ready
            } else {
                SnapshotStatus::Blocked
            },
            adopted,
            adjustments: arena_report.adjustments,
            unresolved: arena_report.unresolved.len(),
            orphan_orders: report.orphan_remote_orders.len(),
            unresolved_orders,
            pending_regular_orders: orders.len(),
            pruned,
            error: None,
        }
    }

    pub async fn refresh(&self, arena: &GlobalArena) -> SnapshotOutcome {
        // Boot/periodic/transition refreshes must not overlap and publish a ready flag
        // while another pass still holds an older exposure snapshot.
        let _refresh = self.refresh_serial.lock().await;
        let token = self.snapshot_token();
        if !self.begin_refresh(&token, arena) || !self.await_quiescence(&token).await {
            return SnapshotOutcome::empty(SnapshotStatus::Stale);
        }
        if token.executor.is_paper_trading() {
            return self.apply_snapshot(
                &token,
                Ok(Vec::new()),
                Ok(Vec::new()),
                Err("local paper scenario".into()),
                arena,
                now_ms(),
            );
        }
        // Position-side assumptions are authorization evidence too. An error leaves
        // the account mode unknown and cannot be repaired by successful fee/position queries.
        if let Err(error) = token.executor.ensure_hedge_mode().await {
            return self.apply_snapshot(
                &token,
                Ok(Vec::new()),
                Ok(Vec::new()),
                Err(format!("OMS_MODE_UNCONFIRMED: {error}")),
                arena,
                now_ms(),
            );
        }
        // Obtain the potentially long per-symbol schedule before the final exposure snapshot.
        let universe = quantum_arena::symbols::get_active_universe_ref();
        let mut rows = Vec::with_capacity(universe.len());
        for symbol in universe.iter() {
            match token.executor.fetch_commission_rate(symbol).await {
                Ok((maker, taker)) => rows.push(SymbolFees {
                    symbol: symbol.to_string(),
                    maker,
                    taker,
                }),
                Err(error) => {
                    return self.apply_snapshot(
                        &token,
                        Ok(Vec::new()),
                        Ok(Vec::new()),
                        Err(format!("OMS_FEES {symbol}: {error}")),
                        arena,
                        now_ms(),
                    );
                }
            }
        }
        let fees = FeeSchedule::new(token.generation, rows).map_err(|e| format!("OMS_FEES: {e:?}"));
        if fees.is_err() {
            return self.apply_snapshot(
                &token,
                Ok(Vec::new()),
                Ok(Vec::new()),
                fees,
                arena,
                now_ms(),
            );
        }
        let positions = token.executor.fetch_position_risk().await;
        if positions.is_err() {
            return self.apply_snapshot(&token, positions, Ok(Vec::new()), fees, arena, now_ms());
        }
        let orders = token.executor.fetch_all_open_orders().await;
        self.apply_snapshot(&token, positions, orders, fees, arena, now_ms())
    }

    pub async fn dispatch_entry<T: EntryTransport>(
        &self,
        token: &SessionToken,
        transport: &T,
        request: &EntryRequest<'_>,
        arena: &GlobalArena,
    ) -> Result<(), String> {
        let guarded = GuardedEntry {
            session: self,
            token,
            transport,
            arena,
        };
        entry_dispatch::dispatch_entry(&guarded, request).await
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A local proposal receipt belongs only to the exact event and quantity that produced it.
/// It does not assert that the exchange settled this close.
pub fn matching_local_close(
    receipt: Option<LocalCloseAccounting>,
    event_time_ms: u64,
    quantity: f64,
) -> Option<LocalCloseAccounting> {
    receipt.filter(|r| {
        r.event_time_ms == event_time_ms
            && r.quantity == quantity
            && quantity.is_finite()
            && quantity > 0.0
            && r.entry_price.is_finite()
            && r.entry_price > 0.0
            && r.exit_price.is_finite()
            && r.exit_price > 0.0
            && r.gross_pnl.is_finite()
            && r.entry_fee.is_finite()
            && r.exit_fee.is_finite()
            && r.total_fees().is_finite()
    })
}

/// Binance income is cash flow: negative COMMISSION is a cost, positive is a rebate.
pub fn signed_commission_cost(income: f64) -> Option<f64> {
    income.is_finite().then_some(-income)
}

struct GuardedEntry<'a, T> {
    session: &'a OmsSession,
    token: &'a SessionToken,
    transport: &'a T,
    arena: &'a GlobalArena,
}

struct TransportPermit<'a> {
    session: &'a OmsSession,
}

#[must_use = "retain this permit until the whole entry task, including settlement, finishes"]
pub struct OwnedEntryPermit {
    session: Arc<OmsSession>,
}

impl Drop for OwnedEntryPermit {
    fn drop(&mut self) {
        self.session.release_admission();
    }
}

impl Drop for TransportPermit<'_> {
    fn drop(&mut self) {
        self.session.release_admission();
    }
}

impl<T: EntryTransport> EntryTransport for GuardedEntry<'_, T> {
    fn supports_native_iceberg(&self) -> bool {
        self.transport.supports_native_iceberg()
    }

    async fn configure_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {
        let _permit = self.session.admit_transport(self.token, self.arena)?;
        self.transport.configure_leverage(symbol, leverage).await
    }

    async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {
        // Leverage confirmation may have awaited while OMS was revoked or replaced.
        let _permit = self.session.admit_transport(self.token, self.arena)?;
        self.transport.submit_entry(request).await
    }
}
