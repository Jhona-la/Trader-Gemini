use god_engine_core::stateful_engine::StatefulEngine;

fn snapshot(e: &StatefulEngine) -> Vec<u64> {
    let mut v = vec![
        e.tick_count,
        e.current_ts,
        e.kline_start_ms,
        e.last_trade_is_sell as u64,
    ];
    v.extend(
        [
            e.last_price,
            e.ema_fast,
            e.ema_slow,
            e.fair_price,
            e.v_t,
            e.a_t,
            e.last_inst_v,
            e.dir_velocity,
            e.kline_high,
            e.kline_low,
            e.kline_volume,
            e.last_hawkes_ratio,
        ]
        .map(f64::to_bits),
    );
    v.extend(e.get_universal_features().map(|x| x.to_bits() as u64));
    v.extend(e.get_spectral_ml_features().map(|x| x.to_bits() as u64));
    v
}
fn seeded() -> StatefulEngine {
    let mut e = StatefulEngine::new();
    e.process_tick(100.0, 2.0, 1000);
    e.process_tick(99.0, 3.0, 1100);
    e
}
#[test]
fn warmup_seeds_tick_emas_without_artificial_trend() {
    for price in [0.001, 100.0, 100_000.0] {
        let mut e = StatefulEngine::new();
        e.process_kline(price, price, price, price, 1.0);
        e.process_tick(price, 1.0, 1000);
        assert_eq!(e.ema_fast, price);
        assert_eq!(e.ema_slow, price);
        assert_eq!(e.get_micro_trend(), 0.0);
    }
}
#[test]
fn warmup_seeds_kalman_from_first_tick_not_zero() {
    let mut e = StatefulEngine::new();
    e.process_kline(100.0, 100.0, 100.0, 100.0, 1.0);
    e.process_tick(100.0, 1.0, 1000);
    assert_eq!(e.fair_price, 100.0);
}
#[test]
fn warmup_does_not_change_tick_ema_kernel() {
    let mut warm = StatefulEngine::new();
    warm.process_kline(80.0, 90.0, 70.0, 85.0, 4.0);
    let mut fresh = StatefulEngine::new();
    for (i, p) in [100.0, 99.0, 102.0, 101.0].into_iter().enumerate() {
        warm.process_tick(p, 1.0, 1000 + i as u64);
        fresh.process_tick(p, 1.0, 1000 + i as u64);
        assert_eq!(
            (warm.ema_fast, warm.ema_slow),
            (fresh.ema_fast, fresh.ema_slow)
        );
    }
}
#[test]
fn reset_clears_kinematics_and_tick_rule_memory() {
    let mut e = seeded();
    assert!(e.last_inst_v > 0.0 && e.dir_velocity > 0.0 && e.last_trade_is_sell);
    e.reset();
    assert_eq!(e.last_inst_v, 0.0);
    assert_eq!(e.dir_velocity, 0.0);
    assert!(!e.last_trade_is_sell);
}
#[test]
fn reset_followed_by_flat_ticks_matches_fresh_engine() {
    let mut e = seeded();
    e.reset();
    let mut fresh = StatefulEngine::new();
    for t in [2000, 2100, 2200] {
        e.process_tick(100.0, 1.0, t);
        fresh.process_tick(100.0, 1.0, t);
        assert_eq!(snapshot(&e), snapshot(&fresh));
    }
}
#[test]
fn backward_tick_is_rejected_without_prefix_mutation() {
    let mut e = seeded();
    let before = snapshot(&e);
    e.process_tick(200.0, 7.0, 900);
    assert_eq!(snapshot(&e), before);
}
#[test]
fn invalid_tick_volume_does_not_mutate_state() {
    for volume in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let mut e = seeded();
        let before = snapshot(&e);
        e.process_tick(102.0, volume, 1200);
        assert_eq!(snapshot(&e), before, "volume={volume}");
    }
}
#[test]
fn overflowing_notional_does_not_mutate_state() {
    let mut e = seeded();
    let before = snapshot(&e);
    e.process_tick(1e200, 1e200, 1200);
    assert_eq!(snapshot(&e), before);
}
#[test]
fn overflowing_relative_return_does_not_mutate_state() {
    let mut e = StatefulEngine::new();
    e.process_tick(1e-300, 0.0, 1000);
    let before = snapshot(&e);
    e.process_tick(1e300, 0.0, 1100);
    assert_eq!(snapshot(&e), before);
}
#[test]
fn invalid_kline_is_rejected_before_any_state_change() {
    let base = [100.0, 102.0, 98.0, 101.0, 4.0];
    let mut invalid = vec![
        [103.0, 102.0, 98.0, 101.0, 4.0],
        [100.0, 97.0, 98.0, 101.0, 4.0],
        [100.0, 102.0, 98.0, 97.0, 4.0],
        [100.0, 102.0, 0.0, 101.0, 4.0],
        [100.0, 102.0, 98.0, 101.0, -1.0],
    ];
    for i in 0..5 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut row = base;
            row[i] = bad;
            invalid.push(row);
        }
    }
    for [o, h, l, c, v] in invalid {
        let mut e = seeded();
        let before = snapshot(&e);
        e.process_kline(o, h, l, c, v);
        assert_eq!(snapshot(&e), before, "row={o},{h},{l},{c},{v}");
    }
}
#[test]
fn zero_timestamp_is_a_valid_initialized_clock() {
    let mut e = StatefulEngine::new();
    e.process_tick(100.0, 2.0, 0);
    e.process_tick(101.0, 3.0, 1);
    assert_eq!(e.kline_start_ms, 0);
    assert_eq!(e.kline_volume, 5.0);
}
#[test]
fn equal_timestamps_and_zero_volume_remain_accepted() {
    let mut e = StatefulEngine::new();
    e.process_tick(100.0, 0.0, 1000);
    e.process_tick(101.0, 0.0, 1000);
    assert_eq!(e.tick_count, 2);
    assert_eq!(e.last_price, 101.0);
}

#[test]
fn cooldown_large_minimum_is_respected_without_panicking() {
    let mut e = StatefulEngine::new();
    e.record_trade_outcome(30_000.0, true, true, 1, 1000);
    e.current_ts = 1001;
    assert!(!e.can_open_at_tau(30_000.0, 600_000));
}
#[test]
fn cooldown_multiplication_does_not_wrap_or_panic() {
    let mut e = StatefulEngine::new();
    e.record_trade_outcome(30_000.0, true, true, 1, 1000);
    e.current_ts = 1001;
    assert!(!e.can_open_at_tau(30_000.0, u64::MAX));
}
#[test]
fn cooldown_rejects_nonphysical_horizon() {
    let e = StatefulEngine::new();
    for tau in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        assert!(!e.can_open_at_tau(tau, 0), "tau={tau}");
    }
}
#[test]
fn cooldown_fallback_clock_does_not_overflow() {
    let mut e = StatefulEngine::new();
    e.tick_count = 1u64 << 62;
    assert!(e.can_open_at_tau(30_000.0, 1));
}

#[test]
fn fallible_tick_reports_rejection_reason() {
    use god_engine_core::stateful_engine::FeatureInputError as E;
    let mut e = seeded();
    assert_eq!(
        e.try_process_tick(f64::NAN, 1.0, 1200),
        Err(E::InvalidPrice)
    );
    assert_eq!(e.try_process_tick(100.0, -1.0, 1200), Err(E::InvalidVolume));
    assert_eq!(
        e.try_process_tick(100.0, 1.0, 1000),
        Err(E::BackwardTimestamp)
    );
    assert_eq!(
        e.try_process_tick(1e200, 1e200, 1200),
        Err(E::NonFiniteDerivedValue)
    );
    e.tick_count = u64::MAX;
    assert_eq!(
        e.try_process_tick(100.0, 1.0, 1200),
        Err(E::CounterExhausted)
    );
}
#[test]
fn cumulative_volume_overflow_is_rejected_before_mutation() {
    use god_engine_core::stateful_engine::FeatureInputError as E;
    let mut e = StatefulEngine::new();
    e.try_process_tick(0.1, 1e308, 0).unwrap();
    let before = snapshot(&e);
    assert_eq!(
        e.try_process_tick(0.1, 1e308, 1),
        Err(E::NonFiniteDerivedValue)
    );
    assert_eq!(snapshot(&e), before);
}
#[test]
fn fallible_kline_rejects_relative_overflow_and_invalid_volume() {
    use god_engine_core::stateful_engine::FeatureInputError as E;
    let mut e = StatefulEngine::new();
    e.try_process_kline(1e-300, 1e-300, 1e-300, 1e-300, 0.0)
        .unwrap();
    let before = snapshot(&e);
    assert_eq!(
        e.try_process_kline(1e300, 1e300, 1e300, 1e300, 0.0),
        Err(E::NonFiniteDerivedValue)
    );
    assert_eq!(
        e.try_process_kline(1.0, 1.0, 1.0, 1.0, -1.0),
        Err(E::InvalidVolume)
    );
    assert_eq!(snapshot(&e), before);
}
#[test]
fn rejection_then_valid_continuation_matches_clean_prefix() {
    let mut dirty = seeded();
    let mut clean = seeded();
    dirty.process_tick(102.0, f64::NAN, 1200);
    dirty.process_tick(102.0, 1.0, 900);
    dirty.process_kline(102.0, 101.0, 100.0, 100.5, 1.0);
    for (ts, p) in [(1200, 101.0), (1300, 100.0), (61_000, 102.0)] {
        dirty.try_process_tick(p, 1.0, ts).unwrap();
        clean.try_process_tick(p, 1.0, ts).unwrap();
        assert_eq!(snapshot(&dirty), snapshot(&clean));
    }
}
#[test]
fn per_asset_instances_do_not_share_feature_state() {
    let mut a = seeded();
    let mut b = StatefulEngine::new();
    let before = snapshot(&b);
    a.process_tick(110.0, 2.0, 1300);
    assert_eq!(snapshot(&b), before);
    b.process_tick(0.01, 3.0, 5);
    assert_eq!((a.last_price, b.last_price), (110.0, 0.01));
}
#[test]
fn cooldown_never_undercuts_caller_minimum_across_streaks() {
    for n in 0..5 {
        for minimum in [0, 1000, 600_000, 10_000_000, u64::MAX] {
            let mut e = StatefulEngine::new();
            for i in 0..n {
                e.record_trade_outcome(30_000.0, true, true, i, 1);
            }
            e.current_ts = minimum;
            if minimum > 0 {
                assert!(!e.can_open_at_tau(30_000.0, minimum));
            }
        }
    }
}
