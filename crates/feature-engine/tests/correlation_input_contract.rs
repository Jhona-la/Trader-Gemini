use feature_engine::MarketCorrelationHeatmap;

fn seeded() -> MarketCorrelationHeatmap {
    let mut model = MarketCorrelationHeatmap::new(2, 10.0);
    for prices in [[100.0, 100.0], [110.0, 90.0], [99.0, 105.0], [113.0, 100.0]] {
        model.update(&prices);
    }
    model
}

#[test]
fn incomplete_or_invalid_cross_sections_do_not_change_future_estimates() {
    for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        let mut actual = seeded();
        let mut control = seeded();
        actual.update(&[invalid, 150.0]);
        let got = actual.update(&[120.0, 110.0]);
        let expected = control.update(&[120.0, 110.0]);
        assert!(
            (got - expected).abs() < 1e-12,
            "input {invalid}: {got} != {expected}"
        );
    }
}

#[test]
fn unrepresentable_returns_do_not_commit_a_price_baseline() {
    let mut actual = MarketCorrelationHeatmap::new(2, 10.0);
    let mut control = MarketCorrelationHeatmap::new(2, 10.0);
    actual.update(&[1e-300, 100.0]);
    control.update(&[1e-300, 100.0]);
    actual.update(&[1e300, 150.0]);
    let got = actual.update(&[2e-300, 120.0]);
    let expected = control.update(&[2e-300, 120.0]);
    assert!((got - expected).abs() < 1e-12, "{got} != {expected}");
}

#[test]
fn quality_and_constructor_domains_are_explicit() {
    assert!(MarketCorrelationHeatmap::try_new(0, 10.0).is_err());
    assert!(MarketCorrelationHeatmap::try_new(2, f64::NAN).is_err());
    let mut model = MarketCorrelationHeatmap::new(2, 10.0);
    assert!(model.try_update(&[100.0]).is_err());
    assert_eq!(model.try_update(&[100.0, 100.0]), Ok(None));
    assert_eq!(model.try_update(&[100.0, 100.0]), Ok(None));
    assert!(model.try_update(&[f64::NAN, 100.0]).is_err());
}

#[test]
fn overflowed_second_moments_do_not_commit_partial_state() {
    let mut actual = seeded();
    let mut control = seeded();
    assert!(actual.try_update(&[1e200, 110.0]).is_err());
    assert_eq!(
        actual.try_update(&[120.0, 110.0]),
        control.try_update(&[120.0, 110.0])
    );
}

#[test]
fn identical_nonconstant_returns_remain_fully_correlated() {
    let mut model = MarketCorrelationHeatmap::new(2, 10.0);
    let mut price = 100.0;
    for i in 0..20 {
        price *= if i % 2 == 0 { 1.01 } else { 0.99 };
        let result = model.try_update(&[price, 2.0 * price]).unwrap();
        if i > 2 {
            assert!((result.unwrap() - 1.0).abs() < 1e-12);
        }
    }
}
