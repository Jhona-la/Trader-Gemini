//! Characterizations of still-open duplicate APIs (FMT-140), not fixes.
#[allow(dead_code)]
#[path = "../../../src/features/ewma.rs"]
mod legacy_ewma;
#[allow(dead_code)]
#[path = "../../../src/features/welford.rs"]
mod legacy_welford;

#[test]
fn legacy_welford_still_accepts_nan_and_poisoning_persists() {
    let mut legacy = legacy_welford::WelfordOnline::new();
    let mut current = feature_engine::WelfordOnline::new();
    for x in [10.0, f64::NAN, 20.0] {
        legacy.update(x);
        current.update(x);
    }
    assert!(legacy.mean().is_nan());
    assert_eq!(current.mean(), 15.0);
}

#[test]
fn legacy_gain_fallback_still_differs_from_checked_crate_api() {
    assert_eq!(legacy_ewma::Ewma::new(f64::NAN).alpha, 0.1);
    assert_eq!(legacy_ewma::Ewma::from_period(-1.0).alpha, 2.0 / 15.0);
    assert!(feature_engine::Ewma::try_new(f64::NAN).is_err());
    assert!(feature_engine::Ewma::try_from_period(-1.0).is_err());
}
