use god_engine_core::darwin::meets_promotion_margin;

#[test]
fn invalid_or_missing_score_on_either_side_cannot_authorize_a_comparison() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!meets_promotion_margin(1.0, invalid));
        assert!(!meets_promotion_margin(invalid, 1.0));
    }
}

#[test]
fn existing_finite_margin_retains_positive_negative_and_zero_boundaries() {
    assert!(!meets_promotion_margin(1.05, 1.0));
    assert!(meets_promotion_margin(1.06, 1.0));
    assert!(!meets_promotion_margin(-0.95, -1.0));
    assert!(meets_promotion_margin(-0.94, -1.0));
    assert!(!meets_promotion_margin(0.0001, 0.0));
    assert!(meets_promotion_margin(0.0002, 0.0));
}

#[test]
fn clearing_margin_does_not_mean_positive_utility_or_statistical_significance() {
    assert!(meets_promotion_margin(-0.5, -1.0));
    assert!(!meets_promotion_margin(1.0, 1.0));
}
