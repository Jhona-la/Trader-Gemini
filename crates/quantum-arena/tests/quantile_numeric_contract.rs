use quantum_arena::P2Quantile;

#[test]
#[should_panic(expected = "quantile probability")]
fn nan_probability_cannot_construct_an_inert_estimator() {
    let _ = P2Quantile::new(f64::NAN);
}

#[test]
fn finite_extreme_observations_preserve_finite_ordered_markers() {
    let mut p = P2Quantile::new(0.8);
    let values = [-1.0, -0.5, 0.0, 0.5, 1.0];
    for i in 0..500 {
        p.update(values[i % 5] * 1e308);
        assert!(p.q.iter().all(|q| q.is_finite()), "i={i}, q={:?}", p.q);
        if p.initialized {
            assert!(p.q.windows(2).all(|q| q[0] <= q[1]), "i={i}, q={:?}", p.q);
        }
    }
}

#[test]
fn multiplicative_units_do_not_change_the_quantile_geometry() {
    for p in [0.1, 0.5, 0.8, 0.95] {
        let mut base = P2Quantile::new(p);
        let mut large = P2Quantile::new(p);
        let values = [-1.0, -0.5, 0.0, 0.5, 1.0];
        for i in 0..500 {
            let x = values[i % 5];
            base.update(x);
            large.update(x * 1e308);
        }
        assert!(
            (base.value() - large.value() / 1e308).abs() < 1e-10,
            "p={p}, base={}, scaled={}",
            base.value(),
            large.value()
        );
    }
}

#[test]
fn checked_constructor_rejects_invalid_probabilities_without_panicking() {
    for p in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.1,
        0.0,
        1.0,
        1.1,
    ] {
        assert!(P2Quantile::try_new(p).is_err());
    }
    for p in [0.001, 0.01, 0.5, 0.99, 0.999] {
        assert_eq!(P2Quantile::try_new(p).unwrap().p, p);
    }
}

#[test]
fn legacy_constructor_keeps_its_explicit_clipping_policy() {
    assert_eq!(P2Quantile::new(0.001).p, 0.01);
    assert_eq!(P2Quantile::new(0.999).p, 0.99);
    assert_eq!(P2Quantile::new(f64::INFINITY).p, 0.99);
    assert_eq!(P2Quantile::new(f64::NEG_INFINITY).p, 0.01);
}

#[test]
fn invalid_samples_do_not_change_any_marker_or_sample_count() {
    let mut p = P2Quantile::new(0.8);
    for i in 0..50 {
        p.update((i % 7) as f64);
    }
    let before = format!("{p:?}");
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        p.update(x);
    }
    assert_eq!(format!("{p:?}"), before);
}

#[test]
fn warmup_keeps_the_existing_order_statistic_without_allocation() {
    let mut p = P2Quantile::new(0.8);
    assert_eq!(p.value(), 0.0);
    for (x, expected) in [(3.0, 3.0), (1.0, 1.0), (4.0, 3.0), (2.0, 3.0)] {
        p.update(x);
        assert_eq!(p.value(), expected);
    }
}

#[test]
fn opposite_extreme_endpoints_cannot_poison_the_linear_fallback() {
    for p in [0.1, 0.5, 0.9] {
        let mut estimator = P2Quantile::new(p);
        for i in 0..200 {
            let x = if i % 5 < 2 { -f64::MAX } else { f64::MAX };
            estimator.update(x);
            assert!(estimator.value().is_finite());
            if estimator.initialized {
                assert!(estimator.q.windows(2).all(|q| q[0] <= q[1]));
            }
        }
    }
}

#[test]
fn cumulative_history_is_not_a_recent_window_after_a_regime_change() {
    let mut p = P2Quantile::new(0.8);
    for _ in 0..10_000 {
        p.update(0.0);
    }
    for _ in 0..100 {
        p.update(1.0);
    }
    // The last 100 samples have exact quantile 1, the lifetime data does not.
    // This documents the estimand; it is not a promise of fast adaptation.
    assert!(p.value() < 0.5, "lifetime estimate={}", p.value());
    assert_eq!(p.count, 10_100);
}
