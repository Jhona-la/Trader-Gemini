//! B15-12: terminal valuation uses a mark from an accepted replay prefix.
//!
//! Prepared against b67469077a76652dd818271e18e91bed7f8324a5 using existing
//! public APIs. Execution results and source identity belong to the audit receipt.
//!
//! These are valuation contracts, not evidence that a strategy can enter or
//! profit. The first four cases inject one position after the row under test,
//! immediately before an invalid sentinel. Every remaining row is invalid,
//! so neither the core's exit management nor the host envelope sees that slot.
//! All processed rows precede W=64: no kill switch or risk policy is changed.
//! The final break fixture deliberately exercises management of an open slot;
//! its survival preconditions are strict and remain pending actual execution.
//!
//! Nonfinite book quantities are intentionally outside this contract. At this
//! base, Core sanitizes an invalid trade quantity to zero before features and
//! can accept its price; it is not an established whole-event rejection.
//! Likewise, equal timestamps are admitted in caller order, not deduplicated.

use backtest_engine::booktick_replay::{
    ReplayConfig, ReplayStats, ReplayTick, run_booktick_replay_with_observer,
};
use quantum_arena::GlobalArena;
use quantum_arena::genome::SuperGenotype;
use std::sync::Arc;
use std::sync::atomic::Ordering;

const ROWS: usize = 80;
const WARMUP: usize = 64;
const VALID_PREFIX: usize = 8;
const CASE_ROW: usize = VALID_PREFIX;
const INJECTION_ROW: usize = CASE_ROW + 1;
const BASE_TS: u64 = 1_700_000_000_000;
const LAST_PREFIX_TS: u64 = BASE_TS + (VALID_PREFIX as u64 - 1) * 100;
const CAPITAL: f64 = 1_000.0;
const QUANTITY: f64 = 2.0;
const MARGIN: f64 = 100.0;
const PREFIX_MARK: f64 = 100.0;
const CASE_MARK: f64 = 102.0;

#[derive(Clone, Copy, Debug)]
enum TerminalCase {
    InvalidPriceControl,
    CrossedBook,
    BackwardClock,
    EqualClockAccepted,
}

fn invalid_row(ts_ms: u64) -> ReplayTick {
    ReplayTick {
        ts_ms,
        bid: f64::NAN,
        ask: f64::NAN,
        bid_qty: 2.0,
        ask_qty: 2.0,
    }
}

fn priced_row(ts_ms: u64, mid: f64) -> ReplayTick {
    ReplayTick {
        ts_ms,
        bid: mid - 0.01,
        ask: mid + 0.01,
        bid_qty: 2.0,
        ask_qty: 2.0,
    }
}

fn tape(case: TerminalCase) -> Vec<ReplayTick> {
    let mut rows: Vec<_> = (0..ROWS)
        .map(|i| invalid_row(BASE_TS + i as u64 * 100))
        .collect();
    for (i, row) in rows.iter_mut().take(VALID_PREFIX).enumerate() {
        *row = priced_row(BASE_TS + i as u64 * 100, PREFIX_MARK);
    }
    rows[CASE_ROW] = match case {
        TerminalCase::InvalidPriceControl => invalid_row(LAST_PREFIX_TS + 100),
        TerminalCase::CrossedBook => ReplayTick {
            bid: CASE_MARK + 1.0,
            ask: CASE_MARK - 1.0,
            ..priced_row(LAST_PREFIX_TS + 100, CASE_MARK)
        },
        TerminalCase::BackwardClock => priced_row(LAST_PREFIX_TS - 1, CASE_MARK),
        TerminalCase::EqualClockAccepted => priced_row(LAST_PREFIX_TS, CASE_MARK),
    };
    rows
}

fn assert_valuation(case: TerminalCase, trade_only: bool, is_long: bool, entry: f64) {
    let rows = tape(case);
    // Exercise the real replay, including its short-tape guard.
    assert!(ROWS - WARMUP > 10);
    assert!(CASE_ROW < WARMUP);
    let cfg = ReplayConfig {
        initial_capital: CAPITAL,
        warmup_ticks: WARMUP,
        trade_only,
        shift_atr_frac: 0.0,
    };
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let mut final_arena: Option<Arc<GlobalArena>> = None;
    let mut visited = 0usize;
    let mut injected = false;
    let mut admitted_at_injection = None;
    let mut admitted_at_end = None;
    let stats: ReplayStats = run_booktick_replay_with_observer(
        &rows,
        &genome,
        None,
        &cfg,
        |i, core| {
            visited += 1;
            let fe = &core.feature_engines[0];
            let state = (fe.tick_count, fe.current_ts, fe.last_price);
            if i == INJECTION_ROW {
                assert!(!injected);
                assert!(core.arena.coins.iter().all(|coin| !coin.positions.is_any_open()));
                assert_eq!(
                    core.arena.unified_capital.load(Ordering::Relaxed).to_bits(),
                    CAPITAL.to_bits(),
                    "the warmup prefix must not have spent or earned cash",
                );
                admitted_at_injection = Some(state);
                // Position::open does not represent an exchange fill or debit
                // an entry fee. The oracle below starts from this known cash
                // balance and charges only the existing estimated exit cost.
                let (tp, sl) = if is_long { (1_000.0, 1.0) } else { (1.0, 1_000.0) };
                core.arena.coins[0].positions.slot_0.open(
                    is_long, entry, QUANTITY, MARGIN, LAST_PREFIX_TS, tp, sl,
                );
                assert!(core.arena.coins[0].positions.slot_0.is_open());
                final_arena = Some(Arc::clone(&core.arena));
                injected = true;
            }
            if i == ROWS - 1 {
                admitted_at_end = Some(state);
            }
        },
    );

    assert_eq!(visited, ROWS, "this fixture must reach the terminal observer");
    assert!(injected, "the open-position precondition is mandatory");
    let arena = final_arena.expect("observer must retain the actual final arena");
    let (accepted_count, accepted_clock, accepted_price) =
        admitted_at_injection.expect("injection observer was not reached");
    assert_eq!(admitted_at_end, admitted_at_injection);
    let additional_accepted = u64::from(matches!(case, TerminalCase::EqualClockAccepted));
    let events_per_row = if trade_only { 1u64 } else { 2u64 };
    assert_eq!(accepted_count, (VALID_PREFIX as u64 + additional_accepted) * events_per_row);
    assert_eq!(accepted_clock, LAST_PREFIX_TS);
    let expected_mark = if additional_accepted == 1 { CASE_MARK } else { PREFIX_MARK };
    assert!((accepted_price - expected_mark).abs() < 1e-10);

    let cash = arena.unified_capital.load(Ordering::Relaxed);
    assert_eq!(cash.to_bits(), CAPITAL.to_bits());
    assert_eq!(stats.trades, 0, "the fixture must not rely on an executed close");
    assert_eq!(stats.net_pnl.to_bits(), 0.0f64.to_bits());
    assert_eq!(stats.open_positions_end, 1);
    let slot = arena.coins[0].positions.slot_0.snapshot().expect("slot must remain open");
    assert_eq!(slot.is_long, is_long);
    assert_eq!(slot.entry_price.to_bits(), entry.to_bits());
    assert_eq!(slot.quantity.to_bits(), QUANTITY.to_bits());

    let taker = arena.config.live_taker_fee.load(Ordering::Relaxed);
    let slippage = arena.config.base_slippage_floor.load(Ordering::Relaxed);
    assert!(taker.is_finite() && taker >= 0.0);
    assert!(slippage.is_finite() && slippage >= 0.0);
    // Independent position identity: signed price change times quantity,
    // less one estimated exit charge on the marked notional.
    let direction = if is_long { 1.0 } else { -1.0 };
    let expected_pnl = direction * QUANTITY * (expected_mark - entry)
        - expected_mark * QUANTITY * (taker + slippage);
    let expected_equity = cash + expected_pnl;
    assert!(stats.mtm_unrealized_pnl.is_finite());
    assert!(stats.final_capital.is_finite());
    assert!(
        (stats.mtm_unrealized_pnl - expected_pnl).abs() < 1e-10,
        "{case:?}, trade_only={trade_only}, long={is_long}, entry={entry}: MTM {} != {expected_pnl}",
        stats.mtm_unrealized_pnl,
    );
    assert!(
        (stats.final_capital - expected_equity).abs() < 1e-10,
        "{case:?}, trade_only={trade_only}, long={is_long}, entry={entry}: equity {} != {expected_equity}",
        stats.final_capital,
    );
}

fn exercise(case: TerminalCase) {
    // Match the large-stack convention of existing backtest integration tests;
    // the worker is joined before another case starts.
    std::thread::Builder::new()
        .name("booktick-terminal-mark-contract".into())
        .stack_size(64 << 20)
        .spawn(move || {
            for trade_only in [true, false] {
                for is_long in [true, false] {
                    for entry in [90.0, 110.0] {
                        assert_valuation(case, trade_only, is_long, entry);
                    }
                }
            }
        })
        .expect("spawn contract worker")
        .join()
        .expect("terminal mark contract failed");
}

#[test]
fn terminal_mark_values_an_open_position_from_the_accepted_prefix() {
    exercise(TerminalCase::InvalidPriceControl);
}

#[test]
fn crossed_terminal_book_cannot_supply_the_mark() {
    exercise(TerminalCase::CrossedBook);
}

#[test]
fn backward_terminal_timestamp_cannot_supply_the_mark() {
    exercise(TerminalCase::BackwardClock);
}

#[test]
fn equal_terminal_timestamp_can_supply_a_new_accepted_mark() {
    exercise(TerminalCase::EqualClockAccepted);
}

// Identity and economic exposure of the same open occupant. Defensive TP/SL
// levels may evolve naturally; this fixture never disables that management.
fn open_slot_identity(arena: &GlobalArena) -> [u64; 8] {
    let slot = arena.coins[0].positions.slot_0.snapshot().expect("break fixture requires an open slot");
    [
        slot.generation,
        u64::from(slot.is_long),
        slot.entry_price.to_bits(),
        slot.quantity.to_bits(),
        slot.margin_used.to_bits(),
        slot.entry_time_ms,
        slot.entry_fee.to_bits(),
        slot.entry_tau_ms,
    ]
}

fn replay_until_cash_break(trade_only: bool, suffix_mark: f64, suffix_step_ms: u64) -> ReplayStats {
    const BREAK_QUANTITY: f64 = 0.01;
    const BREAK_MARGIN: f64 = 1.0;
    const BREAK_CASH: f64 = -1.0;
    let mut rows: Vec<_> = (0..ROWS)
        .map(|i| priced_row(BASE_TS + i as u64 * suffix_step_ms, suffix_mark))
        .collect();
    // Both tapes have exactly the same processed prefix. Only the unvisited
    // suffix changes, including its timestamps and terminal market price.
    rows[0] = priced_row(BASE_TS, PREFIX_MARK);
    rows[1] = priced_row(BASE_TS + 100, PREFIX_MARK);
    let cfg = ReplayConfig {
        initial_capital: CAPITAL,
        warmup_ticks: WARMUP,
        trade_only,
        shift_atr_frac: 0.0,
    };
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let mut visited = 0usize;
    let mut final_arena: Option<Arc<GlobalArena>> = None;
    let mut injected_identity = None;
    let mut identity_before_break = None;
    let stats = run_booktick_replay_with_observer(
        &rows,
        &genome,
        None,
        &cfg,
        |i, core| {
            visited += 1;
            assert!(i <= 1, "the replay consumed a row after the cash break");
            if i == 0 {
                assert!(core.arena.coins.iter().all(|coin| !coin.positions.is_any_open()));
                assert_eq!(core.arena.unified_capital.load(Ordering::Relaxed).to_bits(), CAPITAL.to_bits());
                core.arena.coins[0].positions.slot_0.open(
                    true, PREFIX_MARK, BREAK_QUANTITY, BREAK_MARGIN, BASE_TS, 110.0, 90.0,
                );
                core.arena.used_margin.store(BREAK_MARGIN, Ordering::Relaxed);
                injected_identity = Some(open_slot_identity(&core.arena));
                final_arena = Some(Arc::clone(&core.arena));
            } else {
                // Mandatory fixture validation: event zero and its host gate
                // must have preserved this exact occupant and exposure. An
                // exit or rollback is a fixture failure, never a passing case.
                let identity = open_slot_identity(&core.arena);
                assert_eq!(Some(identity), injected_identity);
                assert_eq!(core.arena.unified_capital.load(Ordering::Relaxed).to_bits(), CAPITAL.to_bits());
                assert_eq!(core.arena.used_margin.load(Ordering::Relaxed).to_bits(), BREAK_MARGIN.to_bits());
                assert_eq!(core.feature_engines[0].tick_count, if trade_only { 1 } else { 2 });
                assert_eq!(core.feature_engines[0].current_ts, BASE_TS);
                identity_before_break = Some(identity);
                // A controlled accounting shock exercises the existing cash
                // stop. It is not a claim of naturally generated trading loss.
                // No position is opened here and no defense is reconfigured.
                core.arena.unified_capital.store(BREAK_CASH, Ordering::Relaxed);
            }
        },
    );

    assert_eq!(visited, 2, "the fixture must terminate at the intended row");
    let arena = final_arena.expect("the first event must retain the actual arena");
    assert!(identity_before_break.is_some());
    assert_eq!(stats.open_positions_end, 1, "an empty position set cannot prove terminal MTM");
    assert_eq!(Some(open_slot_identity(&arena)), identity_before_break);
    assert_eq!(arena.used_margin.load(Ordering::Relaxed).to_bits(), BREAK_MARGIN.to_bits());
    let cash = arena.unified_capital.load(Ordering::Relaxed);
    assert_eq!(cash.to_bits(), BREAK_CASH.to_bits());
    assert_eq!(stats.trades, 0, "the fixture must retain the injected position");
    assert_eq!(stats.net_pnl.to_bits(), 0.0f64.to_bits());

    let taker = arena.config.live_taker_fee.load(Ordering::Relaxed);
    let slippage = arena.config.base_slippage_floor.load(Ordering::Relaxed);
    assert!(taker.is_finite() && taker >= 0.0);
    assert!(slippage.is_finite() && slippage >= 0.0);
    // Entry and the last processed mark are both 100: gross PnL is zero.
    // The suffix must supply neither a price gain nor a different exit fee.
    let expected_pnl = -PREFIX_MARK * BREAK_QUANTITY * (taker + slippage);
    assert!(stats.mtm_unrealized_pnl.is_finite());
    assert!(stats.final_capital.is_finite());
    assert!(
        (stats.mtm_unrealized_pnl - expected_pnl).abs() < 1e-12,
        "unvisited suffix mark {suffix_mark} changed MTM: {} != {expected_pnl}",
        stats.mtm_unrealized_pnl,
    );
    assert!((stats.final_capital - (cash + expected_pnl)).abs() < 1e-12);
    stats
}

#[test]
fn unvisited_suffix_after_cash_break_cannot_change_terminal_mark() {
    // FIXTURE PENDING EXECUTION: public APIs were checked against the base;
    // compilation and survival through the real core/envelope remain unverified.
    // Keep every precondition above: never replace one with an empty-slot
    // branch, ignore, disabled defense, or a successful early return.
    std::thread::Builder::new()
        .name("booktick-terminal-break-contract".into())
        .stack_size(64 << 20)
        .spawn(|| {
            for trade_only in [true, false] {
                let unchanged_suffix = replay_until_cash_break(trade_only, 100.0, 100);
                let changed_suffix = replay_until_cash_break(trade_only, 200.0, 10_000);
                assert_eq!(unchanged_suffix.open_positions_end, 1);
                assert_eq!(changed_suffix.open_positions_end, 1);
                assert_eq!(unchanged_suffix.mtm_unrealized_pnl.to_bits(), changed_suffix.mtm_unrealized_pnl.to_bits());
                assert_eq!(unchanged_suffix.final_capital.to_bits(), changed_suffix.final_capital.to_bits());
            }
        })
        .expect("spawn cash-break contract worker")
        .join()
        .expect("cash-break fixture or terminal mark contract failed");
}
