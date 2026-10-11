//! Epigenetic feedback is a causal state transition, with the existing 15-minute decay.
use quantum_arena::{state::CoinArena, GlobalArena};
use std::sync::atomic::Ordering::Relaxed;

fn state(coin: &CoinArena) -> (u64, u64, u64, u64, u64, bool) {
    (
        coin.epigenetic_bias.load(Relaxed).to_bits(),
        coin.epigenetic_threshold_modifier.load(Relaxed).to_bits(),
        coin.last_close_ts.load(Relaxed),
        coin.last_scalp_close_ts.load(Relaxed),
        coin.last_swing_close_ts.load(Relaxed),
        coin.last_close_was_win.load(Relaxed),
    )
}

fn seed(coin: &CoinArena, threshold: f64) {
    coin.epigenetic_bias.store(1.1, Relaxed);
    coin.epigenetic_threshold_modifier.store(threshold, Relaxed);
    coin.last_close_ts.store(1_000, Relaxed);
    coin.last_scalp_close_ts.store(901, Relaxed);
    coin.last_swing_close_ts.store(902, Relaxed);
    coin.last_close_was_win.store(false, Relaxed);
}

#[test]
fn winning_feedback_does_not_resurrect_a_decayed_penalty() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    seed(coin, 1.25);
    let now = 9_001_000; // Ten existing homeostatic time constants.
    assert!((coin.get_active_epigenetic_threshold(now) - 1.0).abs() < 0.00002);
    coin.apply_spectral_epigenetic_feedback_with_time(0.002, 10_000, 30_000.0, now);
    // Existing fast-win shift: -0.006 - 0.015*(1 - 10/90).
    let expected = 1.0 + 0.25 * (-10.0_f64).exp() - 0.019333333333333334;
    let actual = coin.get_active_epigenetic_threshold(now);
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    assert!(actual < 1.0, "a win must not restore an expired penalty");
}

#[test]
fn losing_feedback_does_not_resurrect_a_decayed_discount() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    seed(coin, 0.80);
    let now = 9_001_000;
    coin.apply_spectral_epigenetic_feedback_with_time(-0.002, 10_000, 30_000.0, now);
    let expected = 1.0 - 0.20 * (-10.0_f64).exp() + 0.015;
    let actual = coin.get_active_epigenetic_threshold(now);
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    assert!(actual > 1.0, "a loss must not restore an expired discount");
}

#[test]
fn nonfinite_feedback_preserves_all_feedback_state() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    for pnl in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        seed(coin, 1.08);
        let before = state(coin);
        coin.apply_spectral_epigenetic_feedback_with_time(pnl, 10_000, 30_000.0, 2_000);
        assert_eq!(state(coin), before, "invalid PnL {pnl} changed state");
    }
}

#[test]
fn regressive_feedback_preserves_all_feedback_state() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    seed(coin, 1.08);
    let before = state(coin);
    coin.apply_spectral_epigenetic_feedback_with_time(0.002, 10_000, 30_000.0, 999);
    assert_eq!(state(coin), before, "older feedback rewound the state");
}

#[test]
fn distinct_closes_at_the_same_timestamp_both_apply() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    seed(coin, 1.0);
    for _ in 0..2 {
        coin.apply_spectral_epigenetic_feedback_with_time(0.002, 10_000, 30_000.0, 1_000);
    }
    assert!((coin.epigenetic_bias.load(Relaxed) - 1.26).abs() < 1e-12);
    let expected = 1.0 - 2.0 * 0.019333333333333334;
    assert!((coin.get_active_epigenetic_threshold(1_000) - expected).abs() < 1e-12);
    assert_eq!(coin.last_close_ts.load(Relaxed), 1_000);
    assert!(coin.last_close_was_win.load(Relaxed));
}

#[test]
fn first_feedback_retains_the_existing_update_policy() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    coin.apply_spectral_epigenetic_feedback_with_time(0.002, 10_000, 30_000.0, 123_000);
    assert!((coin.epigenetic_bias.load(Relaxed) - 1.08).abs() < 1e-12);
    assert!((coin.get_active_epigenetic_threshold(123_000) - 0.9806666666666667).abs() < 1e-12);
    assert_eq!(coin.last_close_ts.load(Relaxed), 123_000);
    assert_eq!(coin.last_scalp_close_ts.load(Relaxed), 123_000);
    assert_eq!(coin.last_swing_close_ts.load(Relaxed), 123_000);
}
