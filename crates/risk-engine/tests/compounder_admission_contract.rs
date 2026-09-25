use risk_engine::capital_compounder::CapitalCompounderEngine as C;

fn size(p: f64, pf: f64, corr: f64, min: f64, max: f64) -> f64 {
    C::calculate_compounding_position_notional(
        100.0, p, pf, 0.8, 0.6, 0.0, 0, 0, corr, 1.0, min, max,
    )
}

#[test]
fn nonpositive_edge_cannot_become_positive_by_profit_factor_floor() {
    for pf in [-2.0, 0.0, 0.5, 1.0] {
        assert_eq!(size(0.6, pf, 1.0, 0.01, 0.25), 0.0);
    }
}

#[test]
fn low_hit_rate_with_positive_binary_edge_is_not_rejected_by_fixed_forty_percent_gate() {
    assert!(size(0.2, 2.0, 1.0, 0.0, 0.25) > 0.0);
}

#[test]
fn risk_floor_cannot_reinflate_a_penalized_size() {
    assert_eq!(size(0.6, 2.0, 1e-9, 0.01, 0.25), 0.0);
}

#[test]
fn invalid_correlation_or_bounds_cannot_authorize_size() {
    for (corr, min, max) in [
        (f64::NAN, 0.01, 0.25),
        (1.0, 0.3, 0.1),
        (1.0, 0.0, 2.0),
        (1.0, 0.0, f64::NAN),
    ] {
        assert_eq!(size(0.6, 2.0, corr, min, max), 0.0);
    }
}

#[test]
fn invalid_capital_cannot_be_substituted_with_funded_capacity() {
    for capital in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            C::get_capital_regime_metrics(capital, 0.0, 13.0).max_concurrent_positions,
            0
        );
    }
}

#[test]
fn capital_metrics_are_invariant_to_monetary_units() {
    let dollars = C::get_capital_regime_metrics(0.52, 0.0, 0.13);
    let cents = C::get_capital_regime_metrics(52.0, 0.0, 13.0);
    assert_eq!(dollars.wealth_factor, cents.wealth_factor);
    assert_eq!(
        dollars.max_concurrent_positions,
        cents.max_concurrent_positions
    );
}

#[test]
fn admissible_size_never_increases_when_correlation_penalty_strengthens() {
    let mut previous = 0.0;
    for n in 0..=100 {
        let value = size(0.6, 2.0, n as f64 / 100.0, 0.01, 0.25);
        assert!(value >= previous);
        assert!(value <= 25.0);
        previous = value;
    }
}

#[test]
fn invalid_probability_confidence_hurst_or_drawdown_cannot_authorize_size() {
    for (p, c, h, dd) in [
        (1.1, 0.8, 0.6, 0.0),
        (0.6, f64::NAN, 0.6, 0.0),
        (0.6, 0.8, f64::INFINITY, 0.0),
        (0.6, 0.8, 0.6, -0.1),
        (0.6, 0.8, 0.6, 1.1),
    ] {
        assert_eq!(
            C::calculate_compounding_position_notional(
                100.0, p, 2.0, c, h, dd, 0, 0, 1.0, 1.0, 0.01, 0.25
            ),
            0.0
        );
    }
}

#[test]
fn sizing_scales_with_capital_units_without_changing_the_fraction() {
    let a = C::calculate_compounding_position_notional(
        10.0, 0.6, 2.0, 0.8, 0.6, 0.0, 0, 0, 1.0, 1.0, 0.0, 0.25,
    );
    assert!((size(0.6, 2.0, 1.0, 0.0, 0.25) - a * 10.0).abs() < 1e-12);
}
