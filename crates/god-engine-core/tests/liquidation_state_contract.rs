use god_engine_core::liquidation_feed::{
    LiquidationObservation as Obs, LiquidationState, ObservationUpdate as Update,
};

fn observation(time: u64, notional: f64) -> Obs {
    Obs {
        symbol: "BTCUSDT".into(),
        event_time_ms: time,
        trade_time_ms: time - 1,
        is_buy: false,
        reported_filled_notional: notional,
    }
}
fn lambda() -> f64 {
    std::f64::consts::LN_2 / 1000.0
}

#[test]
fn views_are_nonconsuming_and_decay_by_exchange_elapsed_time() {
    let s = LiquidationState::new(observation(1000, 1e6), lambda()).unwrap();
    assert_eq!(s.severity_at(1000), Ok(1.0));
    for _ in 0..3 {
        assert_eq!(s.severity_at(2000), Ok(0.5));
    }
    assert_eq!(s.severity_at(3000), Ok(0.25));
    assert_eq!(s.latest().event_time_ms, 1000);
}

#[test]
fn repeated_cumulative_snapshots_form_max_envelope_not_sum() {
    let mut s = LiquidationState::new(observation(1000, 1e6), lambda()).unwrap();
    assert_eq!(
        s.observe(observation(2000, 1.0), lambda()),
        Ok(Update::Accepted)
    );
    assert_eq!(s.severity_at(2000), Ok(0.5));
    assert_eq!(
        s.observe(observation(3000, 1000.0), lambda()),
        Ok(Update::Accepted)
    );
    assert_eq!(s.severity_at(3000), Ok(0.5));
    assert_eq!(s.severity_at(4000), Ok(0.25));
}

#[test]
fn duplicates_and_older_do_not_refresh_and_same_time_uses_max_without_double_counting() {
    let mut s = LiquidationState::new(observation(1000, 1000.0), lambda()).unwrap();
    assert_eq!(
        s.observe(observation(1000, 1000.0), lambda()),
        Ok(Update::Duplicate)
    );
    assert_eq!(
        s.observe(observation(999, 1e6), lambda()),
        Ok(Update::Older)
    );
    assert_eq!(s.severity_at(2000), Ok(0.25));
    assert_eq!(
        s.observe(observation(1000, 1e6), lambda()),
        Ok(Update::ConflictingTimestamp)
    );
    assert_eq!(s.severity_at(2000), Ok(0.5));
    assert_eq!(
        s.observe(observation(1000, 1000.0), lambda()),
        Ok(Update::ConflictingTimestamp)
    );
    assert_eq!(s.severity_at(2000), Ok(0.5));
}

#[test]
fn invalid_observations_rates_and_future_asof_never_contaminate_state() {
    let mut s = LiquidationState::new(observation(1000, 1e6), lambda()).unwrap();
    for n in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(s.observe(observation(2000, n), lambda()).is_err());
    }
    for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(LiquidationState::new(observation(1000, 1e6), rate).is_err());
        assert!(s.observe(observation(2000, 1e6), rate).is_err());
    }
    let mut other = observation(2000, 1e6);
    other.symbol = "ETHUSDT".into();
    assert!(s.observe(other, lambda()).is_err());
    assert!(s.severity_at(999).is_err());
    assert_eq!(s.severity_at(2000), Ok(0.5));
}

#[test]
fn huge_elapsed_time_has_finite_zero_limit_not_an_arbitrary_ttl() {
    let s = LiquidationState::new(observation(1000, 1e6), lambda()).unwrap();
    assert_eq!(s.severity_at(u64::MAX), Ok(0.0));
}

#[test]
fn changed_decay_rate_applies_prospectively_at_the_next_observation() {
    let mut s = LiquidationState::new(observation(1000, 1e6), lambda()).unwrap();
    s.observe(observation(2000, 0.0), lambda() / 2.0).unwrap();
    assert_eq!(s.severity_at(2000), Ok(0.5));
    assert_eq!(s.severity_at(4000), Ok(0.25));
}

#[test]
fn repaired_legacy_decay_kernel_does_not_rewind_on_an_older_event() {
    use god_engine_core::math_kernels::ExponentialDecayTensor;
    let mut k = ExponentialDecayTensor::new(1000.0);
    k.apply_event(1.0, 2000);
    k.apply_event(0.0, 1000);
    assert_eq!(k.last_timestamp_ms, 2000);
    assert_eq!(k.decay_to(2000), 1.0); // XXXIV: old input no longer ages evidence twice.
}
