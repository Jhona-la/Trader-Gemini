//! CX: metamorphic contracts on the real replay, not a duplicate simulator.
//! In-memory fixtures only; no tapes, training, promotion or exchange calls.
use super::*;

fn ticks(n: usize) -> Vec<ReplayTick> {
    let mut seed = 0x5DEECE66Du64;
    let mut price = 60_000.0;
    (0..n)
        .map(|i| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
            price *= 1.0 + (i as f64 / 180.0).sin() * 0.003 + u * 0.004 + 0.00004;
            ReplayTick {
                ts_ms: 1_700_000_000_000 + i as u64 * 60_000,
                bid: price * 0.9998,
                ask: price * 1.0002,
                bid_qty: 450.0 + u.abs() * 1000.0,
                ask_qty: 450.0 + (1.0 - u.abs()) * 1000.0,
            }
        })
        .collect()
}

fn cfg(trade_only: bool, warmup_ticks: usize) -> ReplayConfig {
    ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks,
        trade_only,
        shift_atr_frac: 0.10,
    }
}

fn trace(tape: &[ReplayTick], cfg: &ReplayConfig, prefix: usize) -> Vec<Vec<u64>> {
    crate::asegurar_spec_nativo("BTCUSDT");
    let mut states = Vec::new();
    run_booktick_replay_observed(
        tape,
        &SuperGenotype::new_baseline(0.0002, 0.0005),
        None,
        cfg,
        |i, core| {
            if i < prefix {
                let fe = &core.feature_engines[0];
                let mut state: Vec<u64> = fe
                    .get_universal_features()
                    .iter()
                    .map(|v| v.to_bits() as u64)
                    .collect();
                state.extend([
                    fe.last_price.to_bits(),
                    fe.v_t.to_bits(),
                    fe.hurst.current().to_bits(),
                    fe.kline_ema_fast.to_bits(),
                    fe.kline_ema_slow.to_bits(),
                    fe.tick_count,
                    core.arena.unified_capital.load(Ordering::Relaxed).to_bits(),
                    core.arena.used_margin.load(Ordering::Relaxed).to_bits(),
                    core.arena.coins[0].positions.is_any_open() as u64,
                ]);
                states.push(state);
            }
        },
    );
    assert_eq!(
        states.len(),
        prefix,
        "fixture must actually reach the compared prefix"
    );
    states
}

#[test]
fn cx_first_event_has_no_future_feature_history() {
    let mut observed = false;
    run_booktick_replay_observed(
        &ticks(650),
        &SuperGenotype::new_baseline(0.0002, 0.0005),
        None,
        &cfg(true, 100),
        |i, core| {
            if i == 0 {
                observed = true;
                assert_eq!(
                    core.feature_engines[0].last_price, 0.0,
                    "future closes reached initial state"
                );
            }
        },
    );
    assert!(observed);
}

#[test]
fn cx_future_suffix_cannot_change_prefix_in_either_mode() {
    let original = ticks(650);
    let mut changed = original.clone();
    for t in &mut changed[40..] {
        t.bid *= 1.7;
        t.ask *= 1.7;
    }
    for mode in [true, false] {
        let a = trace(&original, &cfg(mode, 100), 40);
        let b = trace(&changed, &cfg(mode, 100), 40);
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                a, b,
                "future suffix changed prefix at {i}, trade_only={mode}"
            );
        }
    }
}

#[test]
fn cx_appending_data_cannot_change_existing_prefix() {
    let all = ticks(800);
    for mode in [true, false] {
        let a = trace(&all[..180], &cfg(mode, 100), 120);
        let b = trace(&all, &cfg(mode, 100), 120);
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            assert_eq!(a, b, "appending data changed event {i}, trade_only={mode}");
        }
    }
}

#[test]
fn cx_warmup_observes_but_never_opens_or_spends_capital() {
    crate::asegurar_spec_nativo("BTCUSDT");
    for mode in [true, false] {
        let config = cfg(mode, 800);
        let mut reached = false;
        run_booktick_replay_observed(
            &ticks(830),
            &SuperGenotype::new_baseline(0.0002, 0.0005),
            None,
            &config,
            |i, core| {
                if i <= config.warmup_ticks {
                    assert!(
                        !core.arena.coins[0].positions.is_any_open(),
                        "warmup opened at {i}, mode={mode}"
                    );
                    assert_eq!(
                        core.arena.unified_capital.load(Ordering::Relaxed),
                        config.initial_capital
                    );
                    assert_eq!(core.arena.used_margin.load(Ordering::Relaxed), 0.0);
                }
                if i == config.warmup_ticks {
                    reached = true;
                    assert!(core.feature_engines[0].tick_count > 0);
                }
            },
        );
        assert!(reached);
    }
}

#[test]
fn cx_zero_warmup_does_not_disable_the_entry_path() {
    crate::asegurar_spec_nativo("BTCUSDT");
    let mut saw_open = false;
    run_booktick_replay_observed(
        &ticks(830),
        &SuperGenotype::new_baseline(0.0002, 0.0005),
        None,
        &cfg(true, 0),
        |_, core| {
            saw_open |= core.arena.coins[0].positions.is_any_open();
        },
    );
    assert!(
        saw_open,
        "positive control: the fixture must allow entries without warmup"
    );
}

#[test]
fn cx_extreme_warmup_is_rejected_without_overflow() {
    let mut visited = 0;
    let result = run_booktick_replay_observed(
        &ticks(20),
        &SuperGenotype::new_baseline(0.0002, 0.0005),
        None,
        &cfg(true, usize::MAX),
        |_, _| visited += 1,
    );
    assert_eq!(result.trades, 0);
    assert_eq!(
        visited, 0,
        "an impossible warmup must not wrap and run the engine"
    );
}

#[test]
fn cx_macro_before_first_observation_is_missing_not_future() {
    let mut series: [Vec<(i64, f64)>; 6] = Default::default();
    series[0] = vec![(100, 123.0), (103, 125.0)];
    let history = OmniHistory { series };
    assert_eq!(
        history.value_at(0, 99),
        0.0,
        "legacy neutral missing sentinel, never day100"
    );
    assert_eq!(history.value_at(0, 100), 123.0);
    assert_eq!(history.value_at(0, 102), 123.0);
}

// Same timestamp as the first valid row isolates price admission from the
// separate raw-row minute-boundary issue (CX-07). W=0 keeps the trading
// boundary identical despite the extra rejected row. Trade-only and shift=0
// are controls; NaN * 0 must not corrupt an ostensibly unshifted book either.
fn assert_rejected_prefix_is_inert(bid: f64, ask: f64) {
    let valid = ticks(80);
    let mut prefixed = vec![ReplayTick {
        bid,
        ask,
        ..valid[0].clone()
    }];
    prefixed.extend_from_slice(&valid);
    for trade_only in [true, false] {
        for shift_atr_frac in [0.0, 0.10] {
            let config = ReplayConfig {
                shift_atr_frac,
                ..cfg(trade_only, 0)
            };
            let expected = trace(&valid, &config, valid.len());
            let observed = trace(&prefixed, &config, prefixed.len());
            assert_eq!(observed[0], observed[1], "rejected row mutated core");
            for (i, (a, b)) in expected.iter().zip(&observed[1..]).enumerate() {
                assert_eq!(
                    a, b,
                    "rejected prefix changed accepted event {i}, trade_only={trade_only}, shift={shift_atr_frac}"
                );
            }
        }
    }
}

#[test]
fn cx_nan_first_row_cannot_poison_accepted_replay() {
    assert_rejected_prefix_is_inert(f64::NAN, f64::NAN);
}

#[test]
fn cx_infinite_first_row_cannot_poison_accepted_replay() {
    assert_rejected_prefix_is_inert(f64::INFINITY, f64::INFINITY);
}

#[test]
fn cx_nonpositive_first_row_cannot_poison_accepted_replay() {
    assert_rejected_prefix_is_inert(-100.0, -90.0);
}

#[test]
fn cx_crossed_first_row_cannot_poison_accepted_replay() {
    assert_rejected_prefix_is_inert(900_000.0, 600_000.0);
}

#[test]
fn cx_only_rejected_prices_leave_core_cold_and_capital_intact() {
    let mut invalid = ticks(20);
    for (i, tick) in invalid.iter_mut().enumerate() {
        (tick.bid, tick.ask) = match i % 4 {
            0 => (f64::NAN, f64::NAN),
            1 => (f64::INFINITY, f64::INFINITY),
            2 => (-100.0, -90.0),
            _ => (900_000.0, 600_000.0),
        };
    }
    for trade_only in [true, false] {
        let config = cfg(trade_only, 0);
        let mut visited = 0;
        let result = run_booktick_replay_observed(
            &invalid,
            &SuperGenotype::new_baseline(0.0002, 0.0005),
            None,
            &config,
            |_, core| {
                visited += 1;
                assert_eq!(core.feature_engines[0].tick_count, 0);
                assert_eq!(core.feature_engines[0].last_price, 0.0);
                assert!(!core.arena.coins[0].positions.is_any_open());
                assert_eq!(core.arena.used_margin.load(Ordering::Relaxed), 0.0);
            },
        );
        assert_eq!(visited, invalid.len());
        assert_eq!(result.trades, 0);
        assert_eq!(result.final_capital, config.initial_capital);
    }
}

#[test]
fn cx_merge_zero_warmup_has_no_hidden_600_row_floor() {
    crate::asegurar_spec_nativo("BTCUSDT");
    for trade_only in [true, false] {
        let mut checked = false;
        run_booktick_replay_observed(
            &ticks(80),
            &SuperGenotype::new_baseline(0.0002, 0.0005),
            None,
            &cfg(trade_only, 0),
            |i, core| {
                if i == 1 {
                    checked = true;
                    assert_eq!(
                        core.feature_engines[0].tick_count,
                        if trade_only { 1 } else { 2 },
                        "first accepted row was silently skipped"
                    );
                }
            },
        );
        assert!(checked, "the short tape must actually be replayed");
    }
}

#[test]
fn cx_merge_short_warmup_observes_each_row_once_without_entries() {
    crate::asegurar_spec_nativo("BTCUSDT");
    for trade_only in [true, false] {
        for warmup in [10, 60, 69] {
            let config = cfg(trade_only, warmup);
            let events_per_row = if trade_only { 1 } else { 2 };
            let mut reached_boundary = false;
            run_booktick_replay_observed(
                &ticks(80),
                &SuperGenotype::new_baseline(0.0002, 0.0005),
                None,
                &config,
                |i, core| {
                    if i <= warmup {
                        assert_eq!(
                            core.feature_engines[0].tick_count,
                            i as u64 * events_per_row,
                            "warmup skipped or duplicated history at {i}, W={warmup}, trade_only={trade_only}"
                        );
                        assert!(!core.arena.coins[0].positions.is_any_open());
                        assert_eq!(
                            core.arena.unified_capital.load(Ordering::Relaxed),
                            config.initial_capital
                        );
                        assert_eq!(core.arena.used_margin.load(Ordering::Relaxed), 0.0);
                    }
                    reached_boundary |= i == warmup;
                },
            );
            assert!(reached_boundary);
        }
    }
}
