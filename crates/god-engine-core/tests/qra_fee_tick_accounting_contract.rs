//! QRA fees through the production event/position/capital paths.
//! Synthetic in-memory model and bounded T-1 tape; no training or exchange I/O.
//! Only the first test admits an entry through process_event -> process_tick_dual.
//! The shorter close/reservation cases declare their manually seeded preconditions.
//! Accounting is local simulation in quote units, not host/exchange settlement.
use god_engine_core::{
    entry_reservation::EntryReservation, outcome_context::OutcomeContext, GodEngineCore,
};
use quantum_arena::{
    fee_schedule::{FeeSchedule, SymbolFees},
    genome::SuperGenotype,
    position::{PositionHorizon, PositionSnapshot},
    symbol_registry, symbols, GlobalArena,
};
use std::{
    error::Error,
    path::PathBuf,
    sync::{atomic::Ordering::Relaxed, Arc, Mutex},
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
static ENVIRONMENT: Mutex<()> = Mutex::new(());
const CAPITAL: f64 = 1_000.0;
const SYMBOL: &str = "BTCUSDT";
const START_MS: u64 = 1_700_000_000_000;
const ENTRY_RATES: (f64, f64) = (1.0 / 8192.0, 1.0 / 2048.0);

struct FixtureDirectory {
    root: PathBuf,
    previous: PathBuf,
    temporary_root: PathBuf,
}

impl FixtureDirectory {
    fn new() -> TestResult<Self> {
        let temporary_root = std::env::temp_dir().canonicalize()?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = temporary_root.join(format!("qra-fee-tick-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&root)?;
        std::fs::create_dir(root.join("data"))?;
        let previous = std::env::current_dir()?;
        std::env::set_current_dir(&root)?;
        Ok(Self {
            root,
            previous,
            temporary_root,
        })
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.previous);
        if let Ok(exact) = self.root.canonicalize() {
            let owned_name = exact
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("qra-fee-tick-"));
            if exact == self.root && exact.starts_with(&self.temporary_root) && owned_name {
                let _ = std::fs::remove_dir_all(exact);
            }
        }
    }
}

fn close_enough(actual: f64, expected: f64, label: &str) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{label}: {actual}, {expected}"
    );
    let tolerance = 1e-10 * (1.0 + actual.abs().max(expected.abs()));
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn install_synthetic_model() -> TestResult {
    let data = god_engine_core::ml_inference::NanoForestData {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![0, -1, -1],
        threshold: vec![0.0, 0.0, 0.0],
        value: vec![0.0, -3.0, 3.0],
        tree_offsets: vec![0, 3],
        init_score: 0.0,
    };
    let forest = god_engine_core::ml_inference::NanoForest::from_data(data)?;
    god_engine_core::ml_inference::NanoForest::store_global("BTCUSDT_MOTOR", forest);
    Ok(())
}

fn fresh_core(rates: (f64, f64)) -> TestResult<GodEngineCore> {
    fresh_core_for_symbols(&[SYMBOL], rates)
}

fn fresh_core_for_symbols(universe: &[&str], rates: (f64, f64)) -> TestResult<GodEngineCore> {
    symbols::update_dynamic_universe(universe.iter().map(|symbol| (*symbol).into()).collect());
    symbol_registry::update_registry(
        universe
            .iter()
            .map(|symbol| symbol_registry::get_official_binance_spec(symbol))
            .collect(),
    );
    quantum_arena::feed_health::clear();
    install_synthetic_model()?;
    let baseline = SuperGenotype::new_baseline(0.0002, 0.0005);
    let genome = SuperGenotype::from_vector(&baseline.to_vector());
    let arena = GlobalArena::from_genome_in_own_stack(CAPITAL, &genome);
    arena.config.live_maker_fee.store(rates.0, Relaxed);
    arena.config.live_taker_fee.store(rates.1, Relaxed);
    let mut core =
        GodEngineCore::new_with_outcome_context(arena, OutcomeContext::IsolatedSimulation);
    core.swing_nn = None;
    Ok(core)
}

// Same fixed-seed series, directional model and 12 ticks/minute as
// cf5d_medida_al_cerrar. Stop at the FIRST admission, not at the end of a backtest.
fn t1_prices() -> Vec<f64> {
    let mut seed = 0x5DEECE66Du64;
    let mut price = 60_000.0_f64;
    let mut previous = price;
    let mut ticks = Vec::with_capacity(4 * 24 * 60 * 12);
    for index in 0..4 * 24 * 60 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
        price *= 1.0 + (index as f64 / 180.0).sin() * 0.0030 + u * 0.0040 + 0.00004;
        // The legacy series starts interpolation at its first generated close.
        if index == 0 {
            previous = price;
        }
        for part in 0..12 {
            let weight = (part + 1) as f64 / 12.0;
            ticks.push((previous.ln() * (1.0 - weight) + price.ln() * weight).exp());
        }
        previous = price;
    }
    ticks
}

fn market_event(core: &mut GodEngineCore, prices: &[f64], index: usize) -> bool {
    let price = prices[index];
    let previous = if index == 0 { price } else { prices[index - 1] };
    let (entry, _) = core.process_event(
        0,
        true,
        index % 12 == 11,
        true,
        price,
        1.0,
        price * 0.9998,
        price * 1.0002,
        5.0,
        5.0,
        0.0,
        0.0,
        START_MS + (index as u64 + 1) * 5_000,
        false,
        &[0.0; 54],
        price < previous,
    );
    entry.is_some()
}

fn open_snapshot(core: &GodEngineCore, slot: usize) -> TestResult<PositionSnapshot> {
    core.arena.coins[0]
        .positions
        .get_slot(slot)
        .snapshot()
        .ok_or_else(|| "position must be open".into())
}

fn assert_closed_accounting(
    core: &mut GodEngineCore,
    before: PositionSnapshot,
    returned_net: f64,
    exit_rate: f64,
    capital_before_close: f64,
    event_time_ms: u64,
) {
    let coin = &core.arena.coins[0];
    let exit_price = coin.last_close_exit_price.load(Relaxed);
    assert!(exit_price.is_finite() && exit_price > 0.0);
    let direction = if before.is_long { 1.0 } else { -1.0 };
    // The observed physical prices already include the local spread/slippage
    // estimate. Deduct only commissions here, never those price costs twice.
    let gross = direction * before.quantity * (exit_price - before.entry_price);
    let exit_fee = before.quantity * exit_price * exit_rate;
    let net = gross - before.entry_fee - exit_fee;
    close_enough(
        returned_net,
        net,
        "returned trade PnL includes both signed fees",
    );
    close_enough(
        core.arena.unified_capital.load(Relaxed),
        capital_before_close + gross - exit_fee,
        "cash debits the exit once; entry was already charged",
    );
    close_enough(
        coin.metrics.pnl_realized.load(Relaxed),
        net,
        "position outcome ledger",
    );
    close_enough(core.diag_pnl_sum, net, "core diagnostic ledger");
    close_enough(
        core.diag_notional_sum,
        before.quantity * exit_price,
        "actual exit notional",
    );
    // This is the LOCAL fill journal, not the independent host/exchange channel.
    // With one losing close, its reconstructed drawdown includes both fees.
    if net < 0.0 {
        let metrics = core.auditor_interno.compute_independent_metrics();
        close_enough(
            metrics.calculated_max_drawdown,
            -net / CAPITAL,
            "local fill journal drawdown",
        );
    }
    assert_eq!(core.diag_close_total, 1);
    assert_eq!(coin.metrics.trade_count.load(Relaxed), 1);
    let receipt = core.take_local_close_accounting(0);
    assert!(
        receipt.is_some(),
        "actual close must publish accounting for the host"
    );
    if let Some(receipt) = receipt {
        assert_eq!(receipt.event_time_ms, event_time_ms);
        close_enough(receipt.quantity, before.quantity, "receipt closed quantity");
        close_enough(
            receipt.entry_price,
            before.entry_price,
            "receipt entry price",
        );
        close_enough(
            receipt.exit_price,
            exit_price,
            "receipt physical exit price",
        );
        close_enough(receipt.gross_pnl, gross, "receipt gross outcome");
        close_enough(
            receipt.entry_fee,
            before.entry_fee,
            "receipt preserves entry-time fee",
        );
        close_enough(receipt.exit_fee, exit_fee, "receipt exit-time fee");
        close_enough(
            receipt.total_fees(),
            before.entry_fee + exit_fee,
            "receipt signed fee sum",
        );
        close_enough(
            receipt.gross_pnl - receipt.total_fees(),
            returned_net,
            "host receipt net",
        );
    }
    assert!(
        core.take_local_close_accounting(0).is_none(),
        "receipt is consumed once"
    );
}

#[test]
fn real_entry_refresh_close_and_unknown_admission_use_actual_notional() -> TestResult {
    let _serial = ENVIRONMENT.lock().map_err(|_| "fixture mutex poisoned")?;
    let _directory = FixtureDirectory::new()?;
    let prices = t1_prices();
    let mut core = fresh_core(ENTRY_RATES)?;
    let mut admitted = None;
    for index in 0..prices.len() {
        if market_event(&mut core, &prices, index) {
            admitted = Some(index);
            break;
        }
    }
    let index = admitted.ok_or("bounded T-1 fixture must admit a real entry")?;
    let token = core
        .entry_reservation(0)
        .ok_or("actual entry must publish its reservation")?;
    let before = open_snapshot(&core, token.slot)?;
    assert_eq!(before.generation, token.generation);
    assert_eq!(
        core.arena.coins[0]
            .positions
            .slots()
            .iter()
            .filter(|p| p.is_open())
            .count(),
        1,
        "fixture must isolate one entry for the accounting oracle"
    );
    close_enough(
        before.entry_fee,
        before.quantity * before.entry_price * ENTRY_RATES.1,
        "entry fee uses filled grid quantity and physical price",
    );
    close_enough(
        core.arena.unified_capital.load(Relaxed),
        CAPITAL - before.entry_fee,
        "real admission charges the entry exactly once",
    );
    close_enough(
        core.arena.used_margin.load(Relaxed),
        before.margin_used,
        "entry margin reservation",
    );
    assert_eq!(core.diag_opened, 1);

    let updated = (-1.0 / 4096.0, 1.0 / 1024.0);
    core.arena.config.live_maker_fee.store(updated.0, Relaxed);
    core.arena.config.live_taker_fee.store(updated.1, Relaxed);
    let capital_before_close = core.arena.unified_capital.load(Relaxed);
    let stop = if before.is_long {
        before.sl_price * 0.999
    } else {
        before.sl_price * 1.001
    };
    assert!(stop > 0.0, "admitted entry must carry a stop");
    let close_ms = START_MS + (index as u64 + 2) * 5_000;
    let (entry, close, maker) = core.process_tick_dual(
        0,
        stop,
        stop * 1.0001,
        5.0,
        5.0,
        close_ms,
        &[0.0; 54],
        true,
        false,
    );
    assert!(
        entry.is_none() && maker.is_none(),
        "closing tick forbids reentry"
    );
    let (is_long, net, quantity) = close.ok_or("real stop close must reach caller")?;
    assert_eq!(is_long, before.is_long);
    close_enough(quantity, before.quantity, "closed quantity");
    assert_eq!(
        core.reality_fees(),
        updated,
        "refresh occurs through the tick, never manual sync"
    );
    assert_closed_accounting(
        &mut core,
        before,
        net,
        updated.1,
        capital_before_close,
        close_ms,
    );
    assert!(!core.arena.coins[0].positions.get_slot(token.slot).is_open());
    close_enough(
        core.arena.used_margin.load(Relaxed),
        0.0,
        "close releases its margin",
    );
    close_enough(
        core.arena.unified_capital.load(Relaxed),
        CAPITAL + net,
        "whole local trade cash",
    );
    drop(core);

    // Replay the exact same prefix: only the fee observation changes just
    // before the tick that the positive control ACTUALLY admitted.
    for invalid_case in 0..3 {
        let mut rejected = fresh_core(ENTRY_RATES)?;
        for event in 0..=index {
            if event == index {
                match invalid_case {
                    0 => rejected
                        .arena
                        .config
                        .live_maker_fee
                        .store(f64::NAN, Relaxed),
                    1 => rejected
                        .arena
                        .config
                        .live_taker_fee
                        .store(f64::INFINITY, Relaxed),
                    _ => {
                        rejected
                            .arena
                            .config
                            .observed_fees_required
                            .store(true, Relaxed);
                        rejected.arena.config.live_fee_schedule.store(Some(Arc::new(
                            FeeSchedule::new(
                                71,
                                vec![SymbolFees {
                                    symbol: "ETHUSDT".into(),
                                    maker: 0.0,
                                    taker: 0.0,
                                }],
                            )?,
                        )));
                        assert_eq!(
                            rejected.arena.config.fee_rates_for(0),
                            None,
                            "another symbol's finite row cannot authorize this entry"
                        );
                    }
                }
            }
            assert!(
                !market_event(&mut rejected, &prices, event),
                "unknown fees must veto the admitted control tick"
            );
        }
        assert!(rejected.arena.coins[0]
            .positions
            .slots()
            .iter()
            .all(|p| !p.is_open()));
        assert_eq!(rejected.diag_opened, 0);
        close_enough(
            rejected.arena.unified_capital.load(Relaxed),
            CAPITAL,
            "rejection cannot charge fees",
        );
        close_enough(
            rejected.arena.used_margin.load(Relaxed),
            0.0,
            "rejection cannot reserve margin",
        );
        assert_eq!(
            rejected.reality_fees(),
            ENTRY_RATES,
            "last finite values are close estimates only"
        );
    }
    Ok(())
}

#[test]
fn per_symbol_ticks_use_their_own_observation_or_matching_close_history() -> TestResult {
    let _serial = ENVIRONMENT.lock().map_err(|_| "fixture mutex poisoned")?;
    let _directory = FixtureDirectory::new()?;
    // Three independent runs: refreshed evidence, unavailable book, and a
    // replacement symbol in the same slot. Positions are MANUALLY seeded;
    // this case proves close-rate selection, not admission of a real entry.
    for state in 0..3 {
        let mut core = fresh_core_for_symbols(&[SYMBOL, "ETHUSDT"], (0.01, 0.02))?;
        let first = (1.0 / 128.0, 1.0 / 64.0);
        let second = (-1.0 / 4096.0, 1.0 / 2048.0);
        let publish = |second_rates: (f64, f64)| -> TestResult<Arc<FeeSchedule>> {
            Ok(Arc::new(FeeSchedule::new(
                72,
                vec![
                    SymbolFees {
                        symbol: SYMBOL.into(),
                        maker: first.0,
                        taker: first.1,
                    },
                    SymbolFees {
                        symbol: "ETHUSDT".into(),
                        maker: second_rates.0,
                        taker: second_rates.1,
                    },
                ],
            )?))
        };
        core.arena
            .config
            .observed_fees_required
            .store(true, Relaxed);
        core.arena
            .config
            .live_fee_schedule
            .store(Some(publish(second)?));
        // Establish ETH history, then make BTC the latest shared physics
        // estimate. A later ETH close must not inherit this BTC observation.
        let _ = core.process_tick_dual(
            1,
            100.0,
            100.01,
            5.0,
            5.0,
            START_MS - 10_000,
            &[0.0; 54],
            true,
            false,
        );
        assert_eq!(core.reality_fees(), second);
        let _ = core.process_tick_dual(
            0,
            100.0,
            100.01,
            5.0,
            5.0,
            START_MS - 5_000,
            &[0.0; 54],
            true,
            false,
        );
        assert_eq!(core.reality_fees(), first);
        assert!(core.arena.coins[1]
            .positions
            .get_slot(0)
            .open_with_tau_and_fee(
                true,
                100.0,
                1.0,
                10.0,
                START_MS,
                101.0,
                99.0,
                PositionHorizon::Continuous,
                0.6,
                0.7,
                -0.125,
                30_000
            ));
        core.arena.used_margin.store(10.0, Relaxed);
        core.arena.unified_capital.store(CAPITAL + 0.125, Relaxed);
        let expected = match state {
            0 => {
                let refreshed = (0.0, -1.0 / 1024.0);
                core.arena
                    .config
                    .live_fee_schedule
                    .store(Some(publish(refreshed)?));
                refreshed
            }
            1 => {
                core.arena.config.live_fee_schedule.store(None);
                assert_eq!(core.arena.config.fee_rates_for(1), None);
                second // Matching ETH history is only a defensive estimate.
            }
            _ => {
                symbols::update_dynamic_universe(vec![SYMBOL.into(), "BNBUSDT".into()]);
                symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
                    "BNBUSDT",
                )]);
                assert_eq!(core.arena.config.fee_rates_for(1), None);
                // The existing position is deliberately orphaned by this
                // replacement. The core's explicit close-only default
                // scenario is neither ETH history nor observed BNB fees.
                let scenario = god_engine_core::reality_physics::RealityPhysics::default();
                (scenario.maker_fee(), scenario.taker_fee())
            }
        };
        let (entry, close, quote) = core.process_tick_dual(
            1,
            97.0,
            97.01,
            5.0,
            5.0,
            START_MS + 5_000,
            &[0.0; 54],
            true,
            false,
        );
        assert!(entry.is_none() && quote.is_none());
        let (_, net, quantity) = close.ok_or("coin1 stop must reach the real close path")?;
        assert_eq!(core.reality_fees(), expected);
        close_enough(quantity, 1.0, "coin1 close quantity");
        let exit = core.arena.coins[1].last_close_exit_price.load(Relaxed);
        let gross = exit - 100.0;
        let exit_fee = exit * expected.1;
        close_enough(
            net,
            gross + 0.125 - exit_fee,
            "coin1 signed net with its own rate",
        );
        close_enough(
            core.arena.unified_capital.load(Relaxed),
            CAPITAL + net,
            "coin1 cash ledger",
        );
        close_enough(
            core.arena.used_margin.load(Relaxed),
            0.0,
            "coin1 released margin",
        );
        assert_eq!(
            core.diag_opened, 0,
            "manually seeded case never proves admission"
        );
        let receipt = core
            .take_local_close_accounting(1)
            .ok_or("coin1 close accounting")?;
        close_enough(receipt.entry_fee, -0.125, "coin1 historical rebate");
        close_enough(receipt.exit_fee, exit_fee, "coin1 selected exit rate");
        close_enough(
            receipt.gross_pnl - receipt.total_fees(),
            net,
            "coin1 host receipt",
        );
        assert!(
            core.take_local_close_accounting(0).is_none(),
            "coin0 has no close receipt"
        );
        assert!(
            core.take_local_close_accounting(1).is_none(),
            "coin1 receipt consumed once"
        );
    }
    Ok(())
}

#[test]
fn manually_seeded_closes_refresh_zero_and_signed_maker_taker_fees() -> TestResult {
    let _serial = ENVIRONMENT.lock().map_err(|_| "fixture mutex poisoned")?;
    let _directory = FixtureDirectory::new()?;
    for is_long in [true, false] {
        for maker_exit in [true, false] {
            for rate in [0.0, -1.0 / 8192.0, 1.0 / 2048.0] {
                let mut core = fresh_core(ENTRY_RATES)?;
                let position = core.arena.coins[0].positions.get_slot(0);
                assert!(position.open_with_tau_and_fee(
                    is_long,
                    100.0,
                    1.0,
                    10.0,
                    START_MS,
                    if is_long { 101.0 } else { 99.0 },
                    if is_long { 99.0 } else { 101.0 },
                    PositionHorizon::Continuous,
                    0.6,
                    0.7,
                    0.125,
                    30_000
                ));
                core.arena.used_margin.store(10.0, Relaxed);
                core.arena.unified_capital.store(CAPITAL - 0.125, Relaxed);
                let before = open_snapshot(&core, 0)?;
                // Different rates make choosing the wrong side observable.
                let rates = if maker_exit {
                    (rate, 1.0 / 1024.0)
                } else {
                    (1.0 / 1024.0, rate)
                };
                core.arena.config.live_maker_fee.store(rates.0, Relaxed);
                core.arena.config.live_taker_fee.store(rates.1, Relaxed);
                let bid = match (is_long, maker_exit) {
                    (true, true) => 102.0,
                    (false, true) => 98.0,
                    (true, false) => 97.0,
                    (false, false) => 103.0,
                };
                let (entry, close, quote) = core.process_tick_dual(
                    0,
                    bid,
                    bid + 0.01,
                    5.0,
                    5.0,
                    START_MS + 5_000,
                    &[0.0; 54],
                    true,
                    false,
                );
                assert!(entry.is_none() && quote.is_none());
                let (_, net, _) = close.ok_or("seeded TP/SL must close through production tick")?;
                assert_eq!(
                    core.arena.coins[0].last_close_reason.load(Relaxed),
                    if maker_exit { 1 } else { 2 }
                );
                assert_closed_accounting(
                    &mut core,
                    before,
                    net,
                    rate,
                    CAPITAL - 0.125,
                    START_MS + 5_000,
                );
                assert_eq!(core.reality_fees(), rates);
                close_enough(
                    core.arena.used_margin.load(Relaxed),
                    0.0,
                    "seeded close releases margin",
                );
                assert!(!core.arena.coins[0].positions.get_slot(0).is_open());
            }
        }
    }
    Ok(())
}

#[test]
fn unknown_fee_observation_preserves_a_last_known_defensive_close_estimate() -> TestResult {
    let _serial = ENVIRONMENT.lock().map_err(|_| "fixture mutex poisoned")?;
    let _directory = FixtureDirectory::new()?;
    let mut core = fresh_core(ENTRY_RATES)?;
    // A valid earlier tick establishes this symbol's history. Constructor
    // scenario values alone are not observed evidence for a defensive close.
    let (_, close, _) = core.process_tick_dual(
        0,
        100.0,
        100.01,
        5.0,
        5.0,
        START_MS - 5_000,
        &[0.0; 54],
        true,
        false,
    );
    assert!(close.is_none());
    assert!(core.arena.coins[0]
        .positions
        .get_slot(0)
        .open_with_tau_and_fee(
            true,
            100.0,
            1.0,
            10.0,
            START_MS,
            101.0,
            99.0,
            PositionHorizon::Continuous,
            0.6,
            0.7,
            0.125,
            30_000
        ));
    core.arena.used_margin.store(10.0, Relaxed);
    core.arena.unified_capital.store(CAPITAL - 0.125, Relaxed);
    let before = open_snapshot(&core, 0)?;
    core.arena.config.live_maker_fee.store(f64::NAN, Relaxed);
    core.arena
        .config
        .live_taker_fee
        .store(f64::NEG_INFINITY, Relaxed);
    let (entry, close, quote) = core.process_tick_dual(
        0,
        97.0,
        97.01,
        5.0,
        5.0,
        START_MS + 5_000,
        &[0.0; 54],
        true,
        false,
    );
    assert!(entry.is_none() && quote.is_none());
    let (_, net, _) = close.ok_or("unknown current fees must not suppress defensive stop")?;
    assert_eq!(
        core.reality_fees(),
        ENTRY_RATES,
        "retained estimate is not a fresh observation"
    );
    assert!(core.arena.config.live_maker_fee.load(Relaxed).is_nan());
    assert_eq!(
        core.arena.config.live_taker_fee.load(Relaxed),
        f64::NEG_INFINITY
    );
    assert_closed_accounting(
        &mut core,
        before,
        net,
        ENTRY_RATES.1,
        CAPITAL - 0.125,
        START_MS + 5_000,
    );
    Ok(())
}

#[test]
fn signed_entry_reservation_cancel_and_partial_fill_preserve_cash_and_other_slots() -> TestResult {
    let _serial = ENVIRONMENT.lock().map_err(|_| "fixture mutex poisoned")?;
    let _directory = FixtureDirectory::new()?;
    for partial in [false, true] {
        let core = fresh_core(ENTRY_RATES)?;
        let arena = &core.arena;
        for (slot, quantity, margin, fee) in [(0, 1.0, 10.0, -0.25), (2, 2.0, 20.0, 0.5)] {
            assert!(arena.coins[0]
                .positions
                .get_slot(slot)
                .open_with_tau_and_fee(
                    true,
                    100.0,
                    quantity,
                    margin,
                    START_MS,
                    101.0,
                    99.0,
                    PositionHorizon::Continuous,
                    0.6,
                    0.7,
                    fee,
                    30_000
                ));
        }
        arena.used_margin.store(30.0, Relaxed);
        arena.unified_capital.store(CAPITAL - 0.5 + 0.25, Relaxed);
        let position = arena.coins[0].positions.get_slot(0);
        let token = EntryReservation {
            coin_id: 0,
            slot: 0,
            generation: position.generation.load(Relaxed),
            symbol: SYMBOL.into(),
        };
        close_enough(
            position.entry_fee.load(Relaxed),
            -0.25,
            "signed entry fee survives position publication",
        );
        if partial {
            assert_eq!(token.confirmar_llenado(arena, Some(0.4)), Ok(()));
            close_enough(position.quantity.load(Relaxed), 0.4, "partial quantity");
            close_enough(
                position.entry_fee.load(Relaxed),
                -0.1,
                "rebate scales with actual fill",
            );
            close_enough(arena.used_margin.load(Relaxed), 24.0, "partial margin");
            close_enough(
                arena.unified_capital.load(Relaxed),
                CAPITAL - 0.5 + 0.1,
                "unfilled rebate is removed, never refunded as a positive fee",
            );
            let capital = arena.unified_capital.load(Relaxed);
            assert_eq!(token.confirmar_llenado(arena, Some(0.2)), Ok(()));
            assert_eq!(
                arena.unified_capital.load(Relaxed),
                capital,
                "confirmation is idempotent"
            );
        } else {
            assert_eq!(token.cancel(arena), Ok(()));
            close_enough(
                arena.unified_capital.load(Relaxed),
                CAPITAL - 0.5,
                "cancel removes unrealized rebate",
            );
            close_enough(
                arena.used_margin.load(Relaxed),
                20.0,
                "cancel releases only its margin",
            );
            assert!(!position.is_open());
            assert!(
                token.cancel(arena).is_err(),
                "second cancel cannot mutate cash"
            );
            close_enough(
                arena.unified_capital.load(Relaxed),
                CAPITAL - 0.5,
                "second cancel cash",
            );
            close_enough(
                arena.used_margin.load(Relaxed),
                20.0,
                "second cancel margin",
            );
        }
        let other = arena.coins[0].positions.get_slot(2);
        assert!(other.is_open());
        close_enough(other.quantity.load(Relaxed), 2.0, "other slot quantity");
        close_enough(other.margin_used.load(Relaxed), 20.0, "other slot margin");
        close_enough(other.entry_fee.load(Relaxed), 0.5, "other slot fee");
    }
    Ok(())
}
