use audit_engine::drift_auditor::{DriftAuditError as E, DriftRecovery};
use std::num::NonZeroU32;

fn recovery() -> DriftRecovery {
    DriftRecovery::new(NonZeroU32::new(10).unwrap())
}

#[test]
fn releases_exactly_on_tenth_clean_observation_not_eleventh() {
    let mut r = recovery();
    r.observe(&Err(E::NonfinitePnl));
    for n in 1..10 {
        r.observe(&Ok(0.0));
        assert!(r.is_blocked());
        assert_eq!(r.clean_count(), n);
    }
    r.observe(&Ok(0.0));
    assert!(!r.is_blocked());
    assert_eq!(r.clean_count(), 0);
}

#[test]
fn new_failure_discards_all_prior_clean_observations() {
    let mut r = recovery();
    r.observe(&Err(E::InvalidLimit));
    for _ in 0..9 {
        r.observe(&Ok(0.0));
    }
    r.observe(&Err(E::Exceeded {
        drift: 0.1,
        limit: 0.05,
    }));
    assert_eq!(r.clean_count(), 0);
    r.observe(&Ok(0.0));
    assert!(r.is_blocked());
    assert_eq!(r.clean_count(), 1);
}

#[test]
fn all_invalid_results_reset_recovery_including_forged_ok_nan() {
    for result in [
        Err(E::InvalidLimit),
        Err(E::IncompatibleTrade),
        Err(E::NonfinitePnl),
        Err(E::NonfiniteDifference),
        Err(E::InvalidAccumulator),
        Err(E::AccumulatorOverflow),
        Ok(f64::NAN),
        Ok(f64::INFINITY),
    ] {
        let mut r = recovery();
        r.observe(&Err(E::InvalidLimit));
        r.observe(&Ok(0.0));
        r.observe(&result);
        assert!(r.is_blocked());
        assert_eq!(r.clean_count(), 0);
    }
}

#[test]
fn clean_history_before_incident_does_not_preapprove_recovery() {
    let mut r = recovery();
    for _ in 0..100 {
        r.observe(&Ok(0.0));
    }
    assert_eq!(r.clean_count(), 0);
    r.observe(&Err(E::NonfinitePnl));
    r.observe(&Ok(0.0));
    assert_eq!(r.clean_count(), 1);
    assert!(r.is_blocked());
}

#[test]
fn one_observation_policy_and_repeated_cycles_have_no_stale_state() {
    let mut r = DriftRecovery::new(NonZeroU32::new(1).unwrap());
    for _ in 0..3 {
        r.observe(&Err(E::InvalidLimit));
        assert!(r.is_blocked());
        r.observe(&Ok(0.0));
        assert!(!r.is_blocked());
    }
}
