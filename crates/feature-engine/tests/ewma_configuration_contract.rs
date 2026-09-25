use feature_engine::Ewma;

#[test]
#[should_panic(expected = "alpha")]
fn nan_alpha_is_rejected_at_construction() {
    let _ = Ewma::new(f64::NAN);
}

#[test]
#[should_panic(expected = "alpha")]
fn extrapolating_gain_is_not_an_ewma() {
    let _ = Ewma::new(2.0);
}

#[test]
#[should_panic(expected = "period")]
fn negative_period_is_rejected_at_construction() {
    let _ = Ewma::from_period(-1.0);
}

#[test]
fn legacy_valid_event_updates_are_preserved() {
    let mut s = Ewma::from_period(9.0);
    assert_eq!(s.update(100.0), 100.0);
    assert_eq!(s.update(110.0), 102.0);
    assert_eq!(s.update(f64::NAN), 102.0);
}
