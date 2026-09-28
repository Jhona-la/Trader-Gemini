//! Public-state diagnostics for temporal observation integrity, not a profitability test.
use god_engine_core::W1ChangepointObserver;

#[test]
fn invalid_observation_does_not_consume_a_timestamp() {
    let mut reference = W1ChangepointObserver::new();
    let mut challenged = W1ChangepointObserver::new();
    for t in 1..=40 {
        reference.observe_at(t, 0.2);
        challenged.observe_at(t, f64::NAN);
        challenged.observe_at(t, 0.2);
    }
    assert_eq!(
        challenged.transition_probability(),
        reference.transition_probability()
    );
}

#[test]
fn event_time_must_not_rewind_the_observer() {
    let mut reference = W1ChangepointObserver::new();
    let mut challenged = W1ChangepointObserver::new();
    for t in 1..=40 {
        reference.observe_at(t, 0.2);
        challenged.observe_at(t, 0.2);
    }
    challenged.observe_at(1, 3.0);
    reference.observe_at(41, 0.2);
    challenged.observe_at(41, 0.2);
    assert_eq!(
        challenged.transition_probability(),
        reference.transition_probability()
    );
}

#[test]
fn unrepresentable_likelihood_must_not_freeze_all_future_evidence() {
    let mut observer = W1ChangepointObserver::new();
    for _ in 0..40 {
        observer.observe(0.2);
    }
    observer.observe(f64::MAX);
    for _ in 0..40 {
        observer.observe(0.2);
    }
    let p = observer.observe(3.0).expect("warm detector");
    assert!(
        p > 0.5,
        "an out-of-domain numerical excursion froze posterior: {p}"
    );
}

#[test]
fn small_representable_likelihood_is_normalized_not_clipped_to_zero() {
    let mut observer = W1ChangepointObserver::new();
    for _ in 0..40 {
        observer.observe(0.2);
    }
    // Full log-scale transport spans 31*ln(4) > 40, not just four log units.
    let p = observer.observe(40.0).unwrap();
    assert!(p > 0.99 && p <= 1.0, "tiny but representable evidence: {p}");
    for _ in 0..40 {
        observer.observe(0.2);
    }
    assert!(observer.observe(3.0).unwrap() > 0.5);
}

#[test]
fn numerical_failure_does_not_commit_the_event_clock() {
    let mut reference = W1ChangepointObserver::new();
    let mut challenged = W1ChangepointObserver::new();
    for t in 1..=40 {
        reference.observe_at(t, 0.2);
        challenged.observe_at(t, 0.2);
    }
    assert!(challenged.observe_at(41, f64::MAX).is_none());
    assert_eq!(challenged.last_w1(), 0.2);
    reference.observe_at(41, 0.2);
    challenged.observe_at(41, 0.2);
    assert_eq!(
        challenged.transition_probability(),
        reference.transition_probability()
    );
}
