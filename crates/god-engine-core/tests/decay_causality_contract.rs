use god_engine_core::math_kernels::DecayError;
use god_engine_core::{
    math_kernels::ExponentialDecayTensor as Decay, stateful_engine::StatefulEngine,
};

#[test]
fn old_event_cannot_rewind_the_clock_or_change_the_observed_state() {
    let mut d = Decay::new(1000.0);
    d.apply_event(1.0, 2000);
    d.apply_event(9.0, 1000);
    assert_eq!((d.last_timestamp_ms, d.current_severity), (2000, 1.0));
    assert_eq!(d.decay_to(2000), 1.0);
}

#[test]
fn nonfinite_and_negative_impulses_do_not_mutate_valid_state() {
    for impulse in [f64::NAN, f64::INFINITY, -1.0] {
        let mut d = Decay::new(1000.0);
        d.apply_event(1.0, 1000);
        d.apply_event(impulse, 2000);
        assert_eq!((d.last_timestamp_ms, d.current_severity), (1000, 1.0));
    }
}

#[test]
fn finite_impulse_overflow_does_not_publish_partial_state() {
    let mut d = Decay::new(1000.0);
    d.apply_event(f64::MAX, 1000);
    d.apply_event(f64::MAX, 1001);
    assert_eq!((d.last_timestamp_ms, d.current_severity), (1000, f64::MAX));
}

#[test]
fn invalid_macro_flow_cannot_poison_dark_alpha() {
    let mut e = StatefulEngine::new();
    e.update_macro_flow(0.001, 1.0, 1000);
    e.update_macro_flow(0.001, f64::NAN, 2000);
    assert_eq!(
        (
            e.dark_alpha.last_timestamp_ms,
            e.dark_alpha.current_severity
        ),
        (1000, 1.0)
    );
}

#[test]
fn invalid_funding_cannot_advance_other_macro_evidence() {
    let mut e = StatefulEngine::new();
    e.update_macro_flow(0.001, 1.0, 1000);
    e.update_macro_flow(f64::NAN, 3.0, 2000);
    assert_eq!(
        (
            e.dark_alpha.last_timestamp_ms,
            e.dark_alpha.current_severity
        ),
        (1000, 1.0)
    );
}

#[test]
fn checked_constructor_rejects_invalid_or_unrepresentable_half_lives() {
    for h in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::from_bits(1)] {
        assert!(Decay::try_new(h).is_err(), "half-life={h}");
    }
    assert!(Decay::try_new(f64::MAX).is_ok());
}

#[test]
fn checked_api_returns_specific_reasons_without_moving_the_clock() {
    let mut d = Decay::try_new(1000.0).unwrap();
    d.try_apply_event(2.0, 1000).unwrap();
    assert_eq!(
        d.try_apply_event(-1.0, 2000),
        Err(DecayError::InvalidImpulse)
    );
    assert_eq!(d.try_decay_to(999), Err(DecayError::OutOfOrder));
    assert_eq!((d.last_timestamp_ms, d.current_severity), (1000, 2.0));
    d.decay_lambda = f64::NAN;
    assert_eq!(d.try_apply_event(1.0, 2000), Err(DecayError::InvalidRate));
    d.decay_lambda = std::f64::consts::LN_2 / 1000.0;
    d.current_severity = f64::INFINITY;
    assert_eq!(d.try_decay_to(2000), Err(DecayError::InvalidState));
    assert_eq!(d.last_timestamp_ms, 1000);
}

#[test]
fn pure_decay_satisfies_semigroup_and_elapsed_time_unit_equivalence() {
    let mut direct = Decay::new(1000.0);
    direct.try_apply_event(3.0, 1000).unwrap();
    let mut partitioned = direct.clone();
    partitioned.try_decay_to(1250).unwrap();
    partitioned.try_decay_to(1700).unwrap();
    let one = direct.try_decay_to(2500).unwrap();
    let split = partitioned.try_decay_to(2500).unwrap();
    assert!((one - split).abs() < 1e-12);
    let mut scaled = Decay::new(1_000_000.0);
    scaled.try_apply_event(3.0, 1_000_000).unwrap();
    assert!((scaled.try_decay_to(2_500_000).unwrap() - one).abs() < 1e-12);
}

#[test]
fn equal_timestamp_impulses_are_additive_and_huge_age_converges_to_zero() {
    let mut d = Decay::new(1000.0);
    d.try_apply_event(1.0, 1000).unwrap();
    d.try_apply_event(2.0, 1000).unwrap();
    assert_eq!(d.try_decay_to(1000), Ok(3.0));
    assert_eq!(d.try_decay_to(u64::MAX), Ok(0.0));
}

#[test]
fn rejected_macro_transition_preserves_all_observed_macro_fields_and_counts_rejection() {
    let mut e = StatefulEngine::new();
    e.update_macro_features(0.2, 0.001, 1.0, 2000);
    let prior = (
        e.obi_accel.prev_obi,
        e.obi_accel.prev_obi_velocity,
        e.fr_elasticity.prev_funding_rate,
        e.dark_alpha.current_severity,
    );
    e.update_macro_features(0.9, 0.5, 2.0, 1000);
    assert_eq!(
        (
            e.obi_accel.prev_obi,
            e.obi_accel.prev_obi_velocity,
            e.fr_elasticity.prev_funding_rate,
            e.dark_alpha.current_severity
        ),
        prior
    );
    assert_eq!(e.rejected_macro_updates, 1);
    e.update_macro_flow(f64::NAN, 0.0, 3000);
    assert_eq!(e.rejected_macro_updates, 2);
    e.reset();
    assert_eq!(e.rejected_macro_updates, 0);
}
