use god_engine_core::fitness_compute;
use god_engine_core::fitness_contract::{FitnessError, checked_fitness, log_capital_growth};

#[test]
fn finite_capitals_do_not_overflow_before_log_growth() {
    let value = fitness_compute(1e-300, 1e300, 0.0, 30);
    assert!(
        value.is_finite(),
        "finite wealth endpoints produced {value}"
    );
    assert!((value - (1e300_f64.ln() - 1e-300_f64.ln())).abs() < 1e-10);
}

#[test]
fn finite_capitals_do_not_underflow_to_ruin_before_log_growth() {
    let value = fitness_compute(1e300, 1e-300, 0.0, 30);
    assert!(
        value.is_finite(),
        "positive terminal wealth produced {value}"
    );
}

#[test]
fn nonfinite_drawdown_is_inadmissible_not_nan_or_a_valid_score() {
    for dd in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(fitness_compute(100.0, 110.0, dd, 30), f64::NEG_INFINITY);
    }
}

#[test]
fn changing_capital_units_does_not_change_fitness() {
    let a = fitness_compute(1.0, 1.125, 0.2, 30);
    for scale in [1e-250, 1e-100, 1e100, 1e250] {
        assert!((a - fitness_compute(scale, scale * 1.125, 0.2, 30)).abs() < 1e-12);
    }
}

#[test]
fn checked_fitness_preserves_distinct_invalid_and_insufficient_evidence_reasons() {
    assert_eq!(
        checked_fitness(100.0, 110.0, 0.0, 0, 30),
        Err(FitnessError::InsufficientTrades {
            observed: 0,
            required: 30
        })
    );
    assert_eq!(
        checked_fitness(100.0, 110.0, f64::NAN, 30, 30),
        Err(FitnessError::InvalidDrawdown)
    );
    assert_eq!(
        log_capital_growth(0.0, 1.0),
        Err(FitnessError::InvalidInitialCapital)
    );
    assert_eq!(
        log_capital_growth(1.0, 0.0),
        Err(FitnessError::InvalidFinalCapital)
    );
}

#[test]
fn extreme_positive_endpoints_have_finite_antisymmetric_growth() {
    let tiny = f64::from_bits(1);
    let forward = log_capital_growth(tiny, f64::MAX).unwrap();
    let reverse = log_capital_growth(f64::MAX, tiny).unwrap();
    assert!(forward.is_finite() && forward > 0.0);
    assert_eq!(forward, -reverse);
    assert_eq!(log_capital_growth(tiny, tiny), Ok(0.0));
}

#[test]
fn finite_legacy_drawdown_clamps_and_minimum_count_remain_compatible() {
    assert_eq!(
        fitness_compute(100.0, 110.0, 1.7, 30),
        fitness_compute(100.0, 110.0, 1.0, 30)
    );
    assert_eq!(
        fitness_compute(100.0, 110.0, -0.2, 30),
        fitness_compute(100.0, 110.0, 0.0, 30)
    );
    assert_eq!(fitness_compute(100.0, 110.0, 0.0, 29), f64::NEG_INFINITY);
}
