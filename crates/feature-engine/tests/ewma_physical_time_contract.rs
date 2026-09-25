use feature_engine::Ewma;

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 2e-14 * a.abs().max(b.abs()).max(f64::MIN_POSITIVE),
        "{a} != {b}"
    );
}

#[test]
fn checked_constructors_do_not_invent_a_gain() {
    for a in [f64::NAN, f64::INFINITY, 0.0, -1.0, 1.01] {
        assert!(Ewma::try_new(a).is_err());
    }
    for p in [f64::NAN, f64::INFINITY, 0.0, -1.0, 0.99] {
        assert!(Ewma::try_from_period(p).is_err());
    }
    assert_eq!(Ewma::try_new(0.00001).unwrap().alpha, 0.00001);
    assert_eq!(Ewma::try_from_period(9.0).unwrap().alpha, 0.2);
}

#[test]
fn exact_constant_input_solution() {
    let mut s = Ewma::new(0.2);
    s.update_elapsed(2.0, 0.0, 10.0).unwrap();
    close(
        s.update_elapsed(5.0, 10.0, 10.0).unwrap(),
        5.0 - 3.0 * (-1.0f64).exp(),
    );
    assert_eq!(s.alpha, 0.2);
}

#[test]
fn partitioning_a_held_signal_preserves_solution() {
    let mut one = Ewma::new(0.2);
    one.update(2.0);
    let mut split = one;
    one.update_elapsed(9.0, 100.0, 37.0).unwrap();
    for dt in [1.0, 2.0, 7.0, 30.0, 60.0] {
        split.update_elapsed(9.0, dt, 37.0).unwrap();
    }
    close(one.value, split.value);
}

#[test]
fn nanosecond_update_at_century_scale_is_not_rounded_to_zero() {
    let mut s = Ewma::new(0.2);
    s.update(0.0);
    let tau = 100.0 * 365.25 * 86400.0 * 1000.0;
    let result = s.update_elapsed(1.0, 1e-6, tau).unwrap();
    assert!(result > 0.0);
    close(result, 1e-6 / tau);
}

#[test]
fn invalid_physical_inputs_are_atomic_and_zero_elapsed_does_not_move() {
    let mut s = Ewma::new(0.2);
    s.update(3.0);
    for (x, dt, tau) in [
        (f64::NAN, 1.0, 1.0),
        (4.0, -1.0, 1.0),
        (4.0, 1.0, 0.0),
        (4.0, f64::INFINITY, 1.0),
        (4.0, 1.0, f64::NAN),
    ] {
        assert!(s.update_elapsed(x, dt, tau).is_err());
        assert_eq!((s.value, s.alpha, s.is_initialized), (3.0, 0.2, true));
    }
    assert_eq!(s.update_elapsed(100.0, 0.0, 1.0), Ok(3.0));
}

#[test]
fn time_unit_rescaling_and_long_gap() {
    let mut a = Ewma::new(0.2);
    a.update(3.0);
    let mut b = a;
    close(
        a.update_elapsed(8.0, 2.0, 7.0).unwrap(),
        b.update_elapsed(8.0, 2000.0, 7000.0).unwrap(),
    );
    assert_eq!(
        a.update_elapsed(-4.0, f64::MAX, f64::MIN_POSITIVE),
        Ok(-4.0)
    );
}

#[test]
fn event_clock_is_not_physical_clock() {
    let mut one = Ewma::new(0.2);
    one.update(0.0);
    let mut many = one;
    one.update(1.0);
    for _ in 0..10 {
        many.update(1.0);
    }
    assert!((one.value - many.value).abs() > 0.6);
}
