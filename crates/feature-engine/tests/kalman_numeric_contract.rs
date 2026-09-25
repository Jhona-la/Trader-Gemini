use feature_engine::KalmanFilter1D;

#[test]
fn finite_large_variances_do_not_overflow_the_gain_denominator() {
    let mut filter = KalmanFilter1D::new(0.0, 1e308, 0.0, 1e308);
    assert!((filter.update(10.0) - 5.0).abs() < 1e-12);
    assert!((filter.p / 5e307 - 1.0).abs() < 1e-12);
}

#[test]
fn convex_update_does_not_overflow_the_innovation() {
    let mut filter = KalmanFilter1D::new(-1e308, 1.0, 0.0, 1.0);
    assert_eq!(filter.update(1e308), 0.0);
}

#[test]
fn small_covariances_preserve_unit_scaling_instead_of_using_a_floor() {
    let mut filter = KalmanFilter1D::new(0.0, 1e-30, 0.0, 1e-30);
    assert!((filter.update(10.0) - 5.0).abs() < 1e-12);
    assert!((filter.p / 5e-31 - 1.0).abs() < 1e-12);
}

#[test]
fn invalid_public_state_does_not_create_an_extrapolating_gain() {
    let mut filter = KalmanFilter1D::new(10.0, 1.0, 0.0, 1.0);
    filter.r = -0.5;
    assert_eq!(filter.update(20.0), 10.0);
    assert_eq!(filter.p, 1.0);
}

#[test]
fn rejected_measurement_does_not_change_dynamic_noise() {
    let mut filter = KalmanFilter1D::new(10.0, 1.0, 0.0, 1.0);
    assert_eq!(filter.update_with_dynamic_r(f64::NAN, 100.0), 10.0);
    assert_eq!(filter.r, 1.0);
}

#[test]
fn fallible_constructor_and_update_reject_invalid_domains() {
    assert!(KalmanFilter1D::try_new(f64::NAN, 1.0, 0.0, 1.0).is_err());
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(KalmanFilter1D::try_new(0.0, bad, 0.0, 1.0).is_err());
        assert!(KalmanFilter1D::try_new(0.0, 1.0, bad, 1.0).is_err());
        assert!(KalmanFilter1D::try_new(0.0, 1.0, 0.0, bad).is_err());
    }
    let mut singular = KalmanFilter1D::new(10.0, 0.0, 0.0, 0.0);
    assert!(singular.try_update(20.0).is_err());
    assert_eq!((singular.x, singular.p), (10.0, 0.0));
    let mut overflow = KalmanFilter1D::new(10.0, 1e308, 1e308, 1.0);
    assert!(overflow.try_update(20.0).is_err());
    assert_eq!((overflow.x, overflow.p), (10.0, 1e308));
}

#[test]
fn dynamic_noise_has_no_absolute_unit_floor() {
    let mut filter = KalmanFilter1D::new(0.0, 1e-30, 0.0, 1.0);
    assert!((filter.update_with_dynamic_r(10.0, 1e-30) - 5.0).abs() < 1e-12);
    assert_eq!(filter.r, 1e-30);
}

#[test]
fn nominal_update_matches_scalar_posterior_and_zero_prior_stays_certain() {
    let mut filter = KalmanFilter1D::new(10.0, 2.0, 1.0, 4.0);
    assert!((filter.try_update(17.0).unwrap() - 13.0).abs() < 1e-12);
    assert!((filter.p - 12.0 / 7.0).abs() < 1e-12);
    let mut fixed = KalmanFilter1D::new(10.0, 0.0, 0.0, 4.0);
    assert_eq!(fixed.try_update(17.0).unwrap(), 10.0);
    assert_eq!(fixed.p, 0.0);
}
