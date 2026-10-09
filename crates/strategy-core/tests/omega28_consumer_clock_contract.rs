use strategy_core::{MultivariateCointegrationEngine, SignalType, TradeHorizon};

#[test]
fn continuous_ou_mature_clock_rejection_abstains_without_state_change() {
    let mut engine =
        MultivariateCointegrationEngine::new([1.0, 0.0, 0.0, 0.0], 2.0)
            .with_continuous_ou();

    // A first observation at t=0 is valid: count carries initialization.
    assert!(engine.update_and_evaluate(&[1.0; 4], 0).is_none());
    let initial = engine.physical_sde.as_ref().expect("opt-in SDE");
    assert_eq!(initial.count, 1);
    assert_eq!(initial.last_ts_ms, 0);
    assert_eq!(initial.last_value, 0.0);

    for t in 1..9 {
        assert!(engine
            .update_and_evaluate(&[1.0; 4], t * 1_000)
            .is_none());
    }

    // This is the tenth accepted observation. Its score is posterior,
    // so a rejected duplicate would return that same score without the gate.
    let first = engine
        .update_and_evaluate(&[(0.1_f64).exp(), 1.0, 1.0, 1.0], 9_000)
        .expect("mature causal shock must emit before checking rejection");
    assert_eq!(first.signal, SignalType::Short);
    assert_eq!(first.horizon, TradeHorizon::Continuous);
    assert!(first.expected_duration_ms > 0);
    let mature = engine.physical_sde.as_ref().expect("opt-in SDE");
    assert_eq!(mature.count, 10);
    assert_eq!(mature.last_ts_ms, 9_000);
    assert!(
        mature.stationary_zscore(mature.last_value) > engine.z_score_threshold,
        "fixture must expose the stale-score signal, not vacuous cold abstention"
    );

    // Derived Debug covers both structs, including the SDE's private moments.
    // The fixture is finite, and exact before/after formatting detects mutations.
    let before = format!("{engine:?}");

    // Different valid prices cannot justify reusing an old timestamp/score.
    assert!(
        engine.update_and_evaluate(&[1.0; 4], 9_000).is_none(),
        "a duplicate observation must not emit a new intent"
    );
    assert_eq!(format!("{engine:?}"), before, "duplicate mutated state");

    assert!(
        engine
            .update_and_evaluate(&[(-0.1_f64).exp(), 1.0, 1.0, 1.0], 8_000)
            .is_none(),
        "a retrograde observation must not emit a new intent"
    );
    assert_eq!(format!("{engine:?}"), before, "retrograde tick mutated state");

    // A fresh, larger positive deviation must still be processed and emit.
    let next = engine
        .update_and_evaluate(&[(0.2_f64).exp(), 1.0, 1.0, 1.0], 10_000)
        .expect("causal advancement must retain healthy emission");
    assert_eq!(next.signal, SignalType::Short);
    assert_eq!(next.horizon, TradeHorizon::Continuous);
    assert!(next.expected_duration_ms > 0);
    let advanced = engine.physical_sde.as_ref().expect("opt-in SDE");
    assert_eq!(advanced.count, 11);
    assert_eq!(advanced.last_ts_ms, 10_000);
    assert_eq!(advanced.last_value, (0.2_f64).exp().ln());
    assert_ne!(format!("{engine:?}"), before, "causal tick did not advance state");
}
