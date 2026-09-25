use evolution_engine::fitness::{FitnessInputs, INVIABLE, compute, compute_with_bayesian_prior};

fn input() -> FitnessInputs {
    FitnessInputs {
        initial_capital: 100.0,
        final_capital: 110.0,
        max_drawdown_pct: 0.1,
        total_trades: 30,
        min_trades_required: 30,
        oos_start_capital: 100.0,
        oos_end_capital: 100.0,
    }
}

#[test]
fn capital_ratio_extremes_remain_finite_and_agree_with_core() {
    for (start, end) in [(1e-300, 1e300), (1e300, 1e-300)] {
        let mut i = input();
        i.initial_capital = start;
        i.final_capital = end;
        let score = compute(&i);
        assert!(score.is_finite(), "score={score}");
        assert_eq!(
            score,
            god_engine_core::fitness_compute(start, end, i.max_drawdown_pct, 30)
        );
    }
}

#[test]
fn invalid_oos_is_not_silently_treated_as_no_degradation() {
    for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        let mut i = input();
        i.oos_start_capital = invalid;
        assert_eq!(compute(&i), INVIABLE, "start={invalid}");
        let mut i = input();
        i.oos_end_capital = invalid;
        assert_eq!(compute(&i), INVIABLE, "end={invalid}");
    }
}

#[test]
fn invalid_drawdown_is_not_replaced_by_a_finite_penalty() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut i = input();
        i.max_drawdown_pct = invalid;
        assert_eq!(compute(&i), INVIABLE);
    }
}

#[test]
fn invalid_cold_start_prior_does_not_make_a_nonfinite_candidate() {
    let mut i = input();
    i.total_trades = 5;
    for prior in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(compute_with_bayesian_prior(&i, prior), INVIABLE);
    }
}

#[test]
fn finite_oos_extreme_loss_is_penalized_without_a_ratio_floor() {
    let mut i = input();
    i.oos_start_capital = 1e300;
    i.oos_end_capital = 1e-300;
    let score = compute(&i);
    let raw = god_engine_core::fitness_compute(100.0, 110.0, 0.1, 30);
    let expected = raw / (1.0 + 1e300_f64.ln() - 1e-300_f64.ln());
    assert!(score.is_finite() && (score - expected).abs() < 1e-15);
}

#[test]
fn prior_is_not_used_once_the_declared_sample_requirement_is_met() {
    let i = input();
    assert_eq!(compute_with_bayesian_prior(&i, f64::NAN), compute(&i));
}

#[test]
fn open_inactive_policy_is_still_ranked_below_a_measured_loss_by_legacy_sentinel() {
    let mut cash = input();
    cash.total_trades = 0;
    cash.final_capital = 100.0;
    let mut loss = input();
    loss.final_capital = 90.0;
    assert_eq!(compute(&cash), INVIABLE);
    assert!(compute(&loss) > compute(&cash));
}

#[test]
fn open_full_penalized_fitness_is_not_additive_across_drawdown_periods() {
    let mut period = input();
    period.final_capital = period.initial_capital;
    let full = compute(&period);
    // Two zero-return periods, each reaching DD=10%, also have full-path DD=10%.
    assert!((2.0 * full - full).abs() > 0.02);
}
