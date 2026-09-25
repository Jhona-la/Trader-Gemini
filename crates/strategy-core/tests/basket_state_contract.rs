use strategy_core::{MultivariateCointegrationEngine, SignalType};

#[test]
fn overflow_during_second_update_does_not_partially_commit() {
    let mut engine = MultivariateCointegrationEngine::new([1e155, 0.0, 0.0, 0.0], 2.0);
    engine.update_and_evaluate(&[1.0; 4], 1);
    let before = (
        engine.count,
        engine.mean_spread,
        engine.var_spread,
        engine.theta_reversion_speed,
    );
    assert!(engine
        .update_and_evaluate(&[2.0, 1.0, 1.0, 1.0], 2)
        .is_none());
    assert_eq!(
        (
            engine.count,
            engine.mean_spread,
            engine.var_spread,
            engine.theta_reversion_speed
        ),
        before
    );
}

#[test]
fn invalid_public_weight_abstains_in_allocations_and_updates() {
    let mut engine = MultivariateCointegrationEngine::new([1.0; 4], 2.0);
    engine.weights[0] = f64::NAN;
    assert_eq!(engine.get_basket_allocations(SignalType::Long), [0.0; 4]);
    assert!(engine.update_and_evaluate(&[100.0; 4], 1).is_none());
    assert_eq!(engine.count, 0);
    engine.weights = [1.0; 4];
    engine.count = usize::MAX;
    assert!(engine.update_and_evaluate(&[100.0; 4], 1).is_none());
    assert_eq!(engine.count, usize::MAX);
}

#[test]
fn open_debt_jump_filter_can_freeze_after_a_persistent_level_change() {
    let mut engine = MultivariateCointegrationEngine::new([1.0, -1.0, 0.0, 0.0], 2.0);
    for t in 0..20 {
        engine.update_and_evaluate(&[100.0; 4], t);
    }
    for t in 20..200 {
        engine.update_and_evaluate(&[10000.0, 100.0, 100.0, 100.0], t);
    }
    assert_eq!(engine.count, 20);
    assert_eq!(engine.mean_spread, 0.0);
}

#[test]
fn open_debt_physical_time_has_no_effect_on_ou_statistics() {
    let mut a = MultivariateCointegrationEngine::new([1.0, -1.0, 0.0, 0.0], 2.0);
    let mut b = a.clone();
    for t in 0..100 {
        let prices = [100.0 + (t % 3) as f64, 100.0, 100.0, 100.0];
        a.update_and_evaluate(&prices, t);
        b.update_and_evaluate(&prices, t * 1_000_000);
    }
    assert_eq!(
        (a.mean_spread, a.var_spread, a.half_life()),
        (b.mean_spread, b.var_spread, b.half_life())
    );
}

#[test]
fn rejected_jump_does_not_increment_accepted_count() {
    let mut engine = MultivariateCointegrationEngine::new([1.0, -1.0, 0.0, 0.0], 2.0);
    for t in 0..20 {
        engine.update_and_evaluate(&[100.0; 4], t);
    }
    let before = (
        engine.count,
        engine.mean_spread,
        engine.var_spread,
        engine.theta_reversion_speed,
    );
    assert!(engine
        .update_and_evaluate(&[10000.0, 100.0, 100.0, 100.0], 21)
        .is_none());
    assert_eq!(
        (
            engine.count,
            engine.mean_spread,
            engine.var_spread,
            engine.theta_reversion_speed
        ),
        before
    );
}

#[test]
fn nonrepresentable_spread_does_not_poison_public_state() {
    let mut engine = MultivariateCointegrationEngine::new([f64::MAX; 4], 2.0);
    assert!(engine.update_and_evaluate(&[100.0; 4], 1).is_none());
    assert_eq!(engine.count, 0);
    assert!(engine.mean_spread.is_finite());
}

#[test]
fn basket_normalization_is_invariant_to_weight_magnitude() {
    for magnitude in [1e-200, 1.0, 1e308] {
        let engine = MultivariateCointegrationEngine::new(
            [magnitude, -magnitude, magnitude, -magnitude],
            2.0,
        );
        assert_eq!(
            engine.get_basket_allocations(SignalType::Long),
            [0.25, -0.25, 0.25, -0.25]
        );
        assert_eq!(engine.get_basket_allocations(SignalType::Flat), [0.0; 4]);
    }
}
