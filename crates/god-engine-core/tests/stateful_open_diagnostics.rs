//! Passing diagnostics reproduce OPEN limitations; they are not repair tests.
use feature_engine::HawkesProcessEngine;
use god_engine_core::stateful_engine::StatefulEngine;

#[test]
fn open_cooldown_has_a_hard_band_boundary() {
    let mut e = StatefulEngine::new();
    e.record_trade_outcome(59_999.0, true, true, 1, 1000);
    e.record_trade_outcome(3_600_000.0, false, true, 2, 1100);
    e.current_ts = 20_000;
    assert!(!e.can_open_at_tau(59_999.0, 10_000));
    assert!(e.can_open_at_tau(60_000.0, 10_000));
}
#[test]
fn open_hawkes_direct_api_accepts_late_impulse() {
    let mut h = HawkesProcessEngine::default();
    let before = h.update(1000, 1.0, 100.0, 100.0).0;
    let after = h.update(900, 1.0, 100.0, 100.0).0;
    assert!(after > before);
    assert_eq!(h.last_update_ms, 1000);
}
#[test]
fn open_hawkes_epoch_zero_loses_first_decay_interval() {
    let mut h = HawkesProcessEngine::default();
    let before = h.update(0, 1.0, 100.0, 100.0).0;
    let after = h.update(1000, 0.0, 100.0, 100.0).0;
    assert_eq!(after, before);
}
#[test]
fn open_hawkes_empty_batch_returns_false_neutral_ratio() {
    let mut h = HawkesProcessEngine::default();
    let before = h.update(1000, 1.0, 100.0, 100.0);
    assert!(before.2 > 0.0);
    let after = h.update_batch(&[]);
    assert_eq!((after.0, after.1), (before.0, before.1));
    assert_eq!(after.2, 0.0);
}
#[test]
fn open_internal_minute_is_anchored_to_arrival_not_regular_grid() {
    let mut e = StatefulEngine::new();
    e.process_tick(100.0, 1.0, 0);
    e.process_tick(101.0, 1.0, 60_001);
    assert_eq!(e.kline_start_ms, 60_001);
}
