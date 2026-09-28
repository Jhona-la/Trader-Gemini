//! Numerical/input contracts. These do not certify predictive correlation.
use quantum_arena::state::CompactTick;
use risk_engine::correlation_guard::{
    correlacion_de_retornos, hayashi_yoshida_correlation, pearson,
};

fn tick(timestamp: u64, p: f64) -> CompactTick {
    CompactTick {
        timestamp,
        bid_price: p,
        ask_price: p,
        bid_qty: 1.0,
        ask_qty: 1.0,
    }
}

#[test]
fn pearson_is_scale_invariant_at_finite_extremes() {
    for scale in [1e-300, 1.0, 1e150, 1e300, f64::MAX] {
        let a = [-scale, scale];
        assert!(
            (pearson(&a, &a).unwrap() - 1.0).abs() < 1e-12,
            "scale={scale}"
        );
        assert!((pearson(&a, &[scale, -scale]).unwrap() + 1.0).abs() < 1e-12);
    }
}

#[test]
fn pearson_never_silently_truncates_or_drops_invalid_samples() {
    assert!(pearson(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_none());
    assert!(pearson(&[1.0, 2.0, f64::NAN], &[1.0, 2.0]).is_none());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(pearson(&[1.0, bad], &[1.0, 2.0]).is_none());
    }
    assert!(pearson(&[1.0, 1.0], &[2.0, 3.0]).is_none());
}

#[test]
fn pearson_preserves_translation_and_positive_rescaling() {
    let a = [1.0, -3.0, 4.0, 2.0, 0.5];
    let b = [2.0, 1.0, -1.0, 3.0, 0.2];
    let expected = pearson(&a, &b).unwrap();
    let x: Vec<_> = a.iter().map(|v| v * 1e200 + 1e201).collect();
    let y: Vec<_> = b.iter().map(|v| v * 1e-200 - 1e-199).collect();
    assert!((pearson(&x, &y).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn pearson_constant_series_never_gain_rounding_variance() {
    for n in 2..=512 {
        let a = vec![0.3; n];
        let b: Vec<_> = (0..n).map(|i| i as f64).collect();
        assert!(pearson(&a, &a).is_none(), "constant n={n}");
        assert!(pearson(&a, &b).is_none(), "constant left n={n}");
        assert!(pearson(&b, &a).is_none(), "constant right n={n}");
    }
}

#[test]
fn hy_rejects_nonincreasing_event_times() {
    for ts in [[1, 2, 2, 3], [1, 3, 2, 4]] {
        let a: Vec<_> = ts
            .iter()
            .enumerate()
            .map(|(i, &t)| tick(t, 100.0 + i as f64))
            .collect();
        assert!(hayashi_yoshida_correlation(&a, &a).is_none());
    }
}

#[test]
fn hy_rejects_invalid_quotes_instead_of_skipping_their_intervals() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut a = vec![
            tick(0, 100.0),
            tick(1, 101.0),
            tick(2, 100.0),
            tick(3, 102.0),
        ];
        let b = a.clone();
        a[1].bid_price = bad;
        assert!(hayashi_yoshida_correlation(&a, &b).is_none(), "quote={bad}");
    }
}

#[test]
fn hy_disjoint_history_does_not_dilute_common_covariation() {
    let b = vec![tick(10, 100.0), tick(11, 101.0), tick(12, 100.0)];
    let mut a = vec![tick(0, 1e-100)];
    a.extend(b.iter().cloned());
    a.push(tick(20, 1e100));
    assert!((hayashi_yoshida_correlation(&a, &b).unwrap() - 1.0).abs() < 1e-12);
    assert!((hayashi_yoshida_correlation(&b, &a).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn hy_log_returns_do_not_overflow_for_positive_finite_prices() {
    for values in [
        [1e-300, 1e300, 1e-300],
        [f64::MAX, f64::MAX / 2.0, f64::MAX],
    ] {
        let a: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &p)| tick(i as u64, p))
            .collect();
        assert!((hayashi_yoshida_correlation(&a, &a).unwrap() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn hy_tiny_but_nonzero_variation_is_not_an_arbitrary_zero() {
    let a = vec![tick(0, 100.0), tick(1, 100.00000001), tick(2, 100.0)];
    assert!((hayashi_yoshida_correlation(&a, &a).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn grid_fallback_does_not_launder_invalid_timestamps() {
    let mut a: Vec<_> = (0..200)
        .map(|i| tick(1000 + i * 10, 100.0 + (i as f64).sin()))
        .collect();
    let b = a.clone();
    a[100].timestamp = a[99].timestamp;
    assert!(correlacion_de_retornos(&a, &b, 0.5).is_none());
}
