//! Tests of the actual consumer wrapper, without a trading engine or arena.
use god_engine_core::math_kernels::RecursiveHurst;

#[test]
fn deterministic_return_trend_is_not_published_as_valid_hurst() {
    let mut h = RecursiveHurst::new();
    for i in 0..2049 {
        let t = i as f64;
        h.update((1e-6 * t * t).exp());
    }
    assert_eq!(h.samples(), 1024);
    assert_eq!(h.current(), 0.5);
    assert_eq!(h.confidence(), 0.0);
}

#[test]
fn constant_prices_remain_an_unidentified_exponent_not_a_confident_regime() {
    let mut h = RecursiveHurst::new();
    for _ in 0..2049 {
        h.update(100.0);
    }
    assert_eq!(h.current(), 0.5);
    assert_eq!(h.confidence(), 0.0);
}
