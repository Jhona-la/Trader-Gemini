use god_engine_core::slippage_predictor::BookDepthSlippagePredictor as Predictor;

#[test]
fn impact_multiplier_is_continuous_at_neutral_orderbook_imbalance() {
    let at = Predictor::predict_slippage_bps(1000.0, 10000.0, 0.001, 0.0, true);
    let left = Predictor::predict_slippage_bps(1000.0, 10000.0, 0.001, -1e-12, true);
    let right = Predictor::predict_slippage_bps(1000.0, 10000.0, 0.001, 1e-12, true);
    assert!((left - at).abs() < 1e-9, "left={left} neutral={at}");
    assert!((right - at).abs() < 1e-9);
}

#[test]
fn negative_imbalance_multiplier_matches_the_stated_exponential() {
    let base = Predictor::predict_slippage_bps(1000.0, 10000.0, 0.001, 0.0, true);
    let adverse = Predictor::predict_slippage_bps(1000.0, 10000.0, 0.001, -0.8, true);
    assert!((adverse / base - 1.6_f64.exp()).abs() < 1e-12);
}

#[test]
fn impact_is_side_symmetric_and_nonincreasing_with_favorable_imbalance() {
    let mut previous = f64::INFINITY;
    for obi in [-1.0, -0.8, -0.1, 0.0, 0.1, 0.8, 1.0] {
        let long = Predictor::predict_slippage_bps(1000.0, 10000.0, 0.001, obi, true);
        let short = Predictor::predict_slippage_bps(1000.0, 10000.0, 0.001, -obi, false);
        assert_eq!(long, short);
        assert!(long <= previous);
        previous = long;
    }
}

#[test]
fn open_missing_depth_still_becomes_a_finite_cost_instead_of_missing_evidence() {
    assert_eq!(
        Predictor::predict_slippage_bps(1000.0, 0.0, 0.001, 0.0, true),
        1.5
    );
}

#[test]
fn open_fixed_depth_floor_breaks_common_notional_unit_scaling() {
    let a = Predictor::predict_slippage_bps(1.0, 10.0, 0.01, 0.0, true);
    let b = Predictor::predict_slippage_bps(100.0, 1000.0, 0.01, 0.0, true);
    assert!(b > a * 3.0); // Same Q/D but fixed 100-unit depth floor changes the result.
}
