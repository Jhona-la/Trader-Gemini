use quantum_arena::{FeeSchedule, FeeScheduleError, QuantumConfig, SuperGenotype, SymbolFees};
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

fn universe(symbols: &[&str]) -> UniverseGuard {
    let lock = UNIVERSE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let previous = quantum_arena::symbols::get_active_universe();
    quantum_arena::symbols::update_dynamic_universe(
        symbols.iter().map(|symbol| (*symbol).to_owned()).collect(),
    );
    UniverseGuard {
        previous,
        _lock: lock,
    }
}

fn config() -> QuantumConfig {
    QuantumConfig::from_genome(100.0, &SuperGenotype::default())
}

fn row(symbol: &str, maker: f64, taker: f64) -> SymbolFees {
    SymbolFees {
        symbol: symbol.to_owned(),
        maker,
        taker,
    }
}

#[test]
fn two_symbols_use_their_own_observed_rates() {
    let _universe = universe(&["BTCUSDT", "ETHUSDT"]);
    let config = config();
    config.observed_fees_required.store(true, Ordering::Release);
    let schedule = FeeSchedule::new(
        17,
        vec![
            row("BTCUSDT", 0.0001, 0.0002),
            row("ETHUSDT", 0.0003, 0.0007),
        ],
    )
    .unwrap();
    assert_eq!(schedule.generation(), 17);
    assert_eq!(schedule.rows().len(), 2);
    config.live_fee_schedule.store(Some(Arc::new(schedule)));
    assert_eq!(config.fee_rates_for(0), Some((0.0001, 0.0002)));
    assert_eq!(config.fee_rates_for(1), Some((0.0003, 0.0007)));
    assert_eq!(config.fee_rates_for(usize::MAX), None);
}

#[test]
fn reordered_universe_invalidates_old_rows_until_republished() {
    let _universe = universe(&["BTCUSDT", "ETHUSDT"]);
    let config = config();
    config.observed_fees_required.store(true, Ordering::Release);
    config.live_fee_schedule.store(Some(Arc::new(
        FeeSchedule::new(
            1,
            vec![
                row("BTCUSDT", 0.0001, 0.0002),
                row("ETHUSDT", 0.0003, 0.0007),
            ],
        )
        .unwrap(),
    )));
    quantum_arena::symbols::update_dynamic_universe(vec!["ETHUSDT".into(), "BTCUSDT".into()]);
    assert_eq!(config.fee_rates_for(0), None);
    assert_eq!(config.fee_rates_for(1), None);
    config.live_fee_schedule.store(Some(Arc::new(
        FeeSchedule::new(
            2,
            vec![
                row("ETHUSDT", 0.0003, 0.0007),
                row("BTCUSDT", 0.0001, 0.0002),
            ],
        )
        .unwrap(),
    )));
    assert_eq!(config.fee_rates_for(0), Some((0.0003, 0.0007)));
    assert_eq!(config.fee_rates_for(1), Some((0.0001, 0.0002)));
}

#[test]
fn nonfinite_rates_fail_in_either_field() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for candidate in [row("BTCUSDT", invalid, 0.0), row("BTCUSDT", 0.0, invalid)] {
            assert!(matches!(
                FeeSchedule::new(1, vec![candidate]),
                Err(FeeScheduleError::NonFiniteRate { .. })
            ));
        }
    }
}

#[test]
fn duplicate_symbol_identity_is_rejected_even_with_matching_rates() {
    for duplicate in ["BTCUSDT", "btcusdt"] {
        assert!(matches!(
            FeeSchedule::new(1, vec![row("BTCUSDT", 0.0, 0.0), row(duplicate, 0.0, 0.0)]),
            Err(FeeScheduleError::DuplicateSymbol { .. })
        ));
    }
    assert!(matches!(
        FeeSchedule::new(1, vec![row("", 0.0, 0.0)]),
        Err(FeeScheduleError::EmptySymbol { .. })
    ));
}

#[test]
fn required_evidence_never_falls_back_for_missing_book_or_symbol() {
    let _universe = universe(&["BTCUSDT", "ETHUSDT"]);
    let config = config();
    config.observed_fees_required.store(true, Ordering::Release);
    assert_eq!(config.fee_rates_for(0), None);
    config.live_fee_schedule.store(Some(Arc::new(
        FeeSchedule::new(1, vec![row("BTCUSDT", 0.0, 0.0)]).unwrap(),
    )));
    assert_eq!(config.fee_rates_for(0), Some((0.0, 0.0)));
    assert_eq!(config.fee_rates_for(1), None);
    config
        .live_fee_schedule
        .store(Some(Arc::new(FeeSchedule::new(2, vec![]).unwrap())));
    assert_eq!(config.fee_rates_for(0), None);
}

#[test]
fn scenario_defaults_are_explicit_finite_and_overridden_by_observations() {
    let _universe = universe(&["BTCUSDT", "ETHUSDT"]);
    let config = config();
    assert!(!config.observed_fees_required.load(Ordering::Acquire));
    assert!(config.live_fee_schedule.load().is_none());
    config.live_maker_fee.store(-0.0002, Ordering::Release);
    config.live_taker_fee.store(0.0, Ordering::Release);
    assert_eq!(config.fee_rates_for(0), Some((-0.0002, 0.0)));
    config.live_fee_schedule.store(Some(Arc::new(
        FeeSchedule::new(1, vec![row("btcusdt", 0.0, 0.0001)]).unwrap(),
    )));
    assert_eq!(config.fee_rates_for(0), Some((0.0, 0.0001)));
    assert_eq!(config.fee_rates_for(1), Some((-0.0002, 0.0)));
    config.live_fee_schedule.store(None);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        config.live_maker_fee.store(invalid, Ordering::Release);
        assert_eq!(config.fee_rates_for(0), None);
        config.live_maker_fee.store(0.0, Ordering::Release);
        config.live_taker_fee.store(invalid, Ordering::Release);
        assert_eq!(config.fee_rates_for(0), None);
        config.live_taker_fee.store(0.0, Ordering::Release);
    }
}

#[test]
fn zero_sign_rebates_and_finite_rates_are_preserved_without_a_cap() {
    let _universe = universe(&["BTCUSDT", "ETHUSDT"]);
    let config = config();
    config.observed_fees_required.store(true, Ordering::Release);
    config.live_fee_schedule.store(Some(Arc::new(
        FeeSchedule::new(
            0,
            vec![
                row("BTCUSDT", -0.0, -0.0003),
                row("ETHUSDT", f64::MAX, -f64::MAX),
            ],
        )
        .unwrap(),
    )));
    let (maker, taker) = config.fee_rates_for(0).unwrap();
    assert_eq!(maker.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(taker, -0.0003);
    assert_eq!(config.fee_rates_for(1), Some((f64::MAX, -f64::MAX)));
}

#[test]
fn schedules_are_per_config_and_clearing_required_evidence_is_not_a_default() {
    let _universe = universe(&["BTCUSDT"]);
    let first = config();
    let second = config();
    first.observed_fees_required.store(true, Ordering::Release);
    second.observed_fees_required.store(true, Ordering::Release);
    first.live_fee_schedule.store(Some(Arc::new(
        FeeSchedule::new(1, vec![row("BTCUSDT", 0.0, -0.0001)]).unwrap(),
    )));
    assert_eq!(first.fee_rates_for(0), Some((0.0, -0.0001)));
    assert_eq!(second.fee_rates_for(0), None);
    first.live_fee_schedule.store(None);
    assert_eq!(first.fee_rates_for(0), None);
}
