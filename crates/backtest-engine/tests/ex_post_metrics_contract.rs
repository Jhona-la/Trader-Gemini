//! MX: independent closed-form oracles; no replay, network or fitness changes.
use backtest_engine::metrics::{ex_post_metrics, ExPostMetrics};

const DAY: u64 = 86_400_000;
const YEAR: u64 = 31_557_600_000;

fn mixed(scale: f64) -> Vec<f64> {
    (0..20)
        .map(|i| if i % 2 == 0 { 2.0 * scale } else { -scale })
        .collect()
}
fn near(actual: f64, expected: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= 2e-12 * expected.abs().max(1.0),
        "actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn zero_span_is_unknown_annualization_not_ruin() {
    let m = ex_post_metrics(&mixed(1.0), 200.0, 100.0, 110.0, 0.1, 0);
    assert!(m.cagr.is_nan() && m.sharpe_ann.is_nan() && m.sortino_ann.is_nan());
    assert!(m.calmar.is_nan() && m.turnover_per_day.is_nan());
    near(m.profit_factor, 2.0);
    near(m.win_rate, 0.5);
    near(m.cvar_95, 1.0);
}

#[test]
fn nonfinite_pnl_invalidates_trade_metrics_without_panicking_or_dropping_rows() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut pnls = mixed(1.0);
        pnls[7] = bad;
        let m = std::panic::catch_unwind(|| ex_post_metrics(&pnls, 100.0, 100.0, 110.0, 0.1, YEAR))
            .expect("invalid observations must not panic");
        assert_eq!(m.n_trades, 20);
        assert!(m.sharpe_ann.is_nan() && m.sortino_ann.is_nan() && m.cvar_95.is_nan());
        assert!(m.win_rate.is_nan() && m.profit_factor.is_nan());
        near(m.cagr, 0.1); // independent endpoint input remains usable
    }
}

#[test]
fn invalid_initial_capital_does_not_invent_zero_growth() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let m = ex_post_metrics(&mixed(1.0), 100.0, bad, 110.0, 0.1, YEAR);
        assert!(m.cagr.is_nan() && m.calmar.is_nan() && m.turnover_per_day.is_nan());
        near(m.win_rate, 0.5);
    }
}

#[test]
fn nonfinite_final_capital_is_not_ruin_or_infinite_growth() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let m = ex_post_metrics(&mixed(1.0), 100.0, 100.0, bad, 0.1, YEAR);
        assert!(m.cagr.is_nan() && m.calmar.is_nan());
        near(m.profit_factor, 2.0);
    }
}

#[test]
fn invalid_drawdown_is_not_zero_risk() {
    for bad in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let m = ex_post_metrics(&mixed(1.0), 100.0, 100.0, 110.0, bad, YEAR);
        assert!(m.max_dd.is_nan() && m.calmar.is_nan());
        near(m.cagr, 0.1);
    }
}

#[test]
fn invalid_notional_is_not_valid_turnover() {
    for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let m = ex_post_metrics(&mixed(1.0), bad, 100.0, 110.0, 0.1, YEAR);
        assert!(m.turnover_per_day.is_nan());
        near(m.profit_factor, 2.0);
    }
}

#[test]
fn empty_sample_preserves_endpoint_growth_but_not_trade_statistics() {
    let m = ex_post_metrics(&[], 0.0, 100.0, 110.0, 0.1, YEAR);
    near(m.cagr, 0.1);
    assert!(m.sharpe_ann.is_nan() && m.sortino_ann.is_nan() && m.cvar_95.is_nan());
    assert!(m.win_rate.is_nan() && m.profit_factor.is_nan());
    assert_eq!(m.turnover_per_day, 0.0);
}

#[test]
fn default_has_no_measured_performance() {
    let m = ExPostMetrics::default();
    assert_eq!(m.n_trades, 0);
    for value in [
        m.cagr,
        m.sharpe_ann,
        m.sortino_ann,
        m.calmar,
        m.max_dd,
        m.cvar_95,
        m.turnover_per_day,
        m.win_rate,
        m.profit_factor,
    ] {
        assert!(value.is_nan());
    }
}

#[test]
fn all_zero_trades_have_undefined_risk_ratios_not_fabricated_zeros() {
    let m = ex_post_metrics(&[0.0; 20], 0.0, 100.0, 100.0, 0.0, YEAR);
    assert!(m.sharpe_ann.is_nan() && m.sortino_ann.is_nan());
    assert!(m.profit_factor.is_nan() && m.calmar.is_nan());
    assert_eq!((m.cagr, m.cvar_95, m.win_rate), (0.0, 0.0, 0.0));
}

#[test]
fn zero_dispersion_has_no_finite_sharpe() {
    for pnl in [-1.0, 1.0] {
        let m = ex_post_metrics(&[pnl; 20], 0.0, 100.0, 100.0, 0.1, YEAR);
        assert!(m.sharpe_ann.is_nan());
    }
}

#[test]
fn dimensionless_ratios_are_invariant_to_currency_scale() {
    for scale in [1e-300, 1e-20, 1.0, 1e150, 1e300] {
        let m = ex_post_metrics(&mixed(scale), 0.0, 100.0, 110.0, 0.1, YEAR);
        near(m.sharpe_ann, 20.0_f64.sqrt() / 3.0);
        near(m.sortino_ann, 10.0_f64.sqrt());
        near(m.profit_factor, 2.0);
        near(m.cvar_95 / scale, 1.0);
    }
}

#[test]
fn tiny_valid_drawdown_is_not_clamped_to_zero() {
    let m = ex_post_metrics(&mixed(1.0), 0.0, 100.0, 110.0, 1e-15, YEAR);
    near(m.calmar / 1e14, 1.0);
}

#[test]
fn empirical_tail_uses_fractional_mass_not_ceil_average() {
    let mut pnls = vec![1.0; 19];
    pnls.extend([-100.0, -20.0]);
    let m = ex_post_metrics(&pnls, 0.0, 100.0, 1.0, 0.99, YEAR);
    // 1.05 observations: (100 + .05*20)/1.05 = 2020/21.
    near(m.cvar_95, 2020.0 / 21.0);
}

#[test]
fn empirical_tail_handles_ties_and_sample_floor() {
    let mut pnls = vec![1.0; 37];
    pnls.extend([-100.0, -100.0]);
    near(
        ex_post_metrics(&pnls, 0.0, 100.0, 1.0, 0.99, YEAR).cvar_95,
        100.0,
    );
    assert!(ex_post_metrics(&[-1.0; 19], 0.0, 100.0, 81.0, 0.19, YEAR)
        .cvar_95
        .is_nan());
}

#[test]
fn large_finite_tail_does_not_overflow_intermediate_sum() {
    let m = ex_post_metrics(&[-1e308; 40], 0.0, 100.0, 0.0, 1.0, YEAR);
    near(m.cvar_95 / 1e308, 1.0);
    near(m.sortino_ann, -40.0_f64.sqrt());
}

#[test]
fn cagr_avoids_intermediate_ratio_overflow_and_underflow() {
    for (initial, final_capital, expected) in [
        (1e-300, 1e300, 10.0_f64.powf(0.6) - 1.0),
        (1e300, 1e-300, 10.0_f64.powf(-0.6) - 1.0),
    ] {
        near(
            ex_post_metrics(&[1.0], 0.0, initial, final_capital, 0.1, 1000 * YEAR).cagr,
            expected,
        );
    }
}

#[test]
fn turnover_avoids_intermediate_ratio_overflow() {
    let m = ex_post_metrics(&[1.0], 1e300, 1e-10, 1.0, 0.1, 1_000_000 * DAY);
    near(m.turnover_per_day / 1e304, 1.0);
}

#[test]
fn finite_ruin_remains_minus_one_only_with_valid_clock() {
    for final_capital in [0.0, -10.0] {
        assert_eq!(
            ex_post_metrics(&[-1.0], 0.0, 100.0, final_capital, 1.0, YEAR).cagr,
            -1.0
        );
        assert!(ex_post_metrics(&[-1.0], 0.0, 100.0, final_capital, 1.0, 0)
            .cagr
            .is_nan());
    }
}

#[test]
fn panel_exposes_cash_pnl_basis_and_proxy_annualization() {
    let line = ex_post_metrics(&mixed(1.0), 0.0, 100.0, 110.0, 0.1, YEAR).panel_line();
    assert!(line.contains("basis=cash_pnl_per_trade"));
    assert!(line.contains("annualization=iid_proxy"));
}

#[test]
fn downside_does_not_underflow_before_a_representable_sortino() {
    let m = ex_post_metrics(&[1.0, -1e-200], 0.0, 100.0, 101.0, 0.1, YEAR);
    // mean ~1/2, downside=1e-200/sqrt(2), annual factor=sqrt(2).
    near(m.sortino_ann / 1e200, 1.0);
}

#[test]
fn signed_zero_drawdown_cannot_reverse_calmar_sign() {
    let m = ex_post_metrics(&mixed(1.0), 0.0, 100.0, 110.0, -0.0, YEAR);
    assert_eq!(m.calmar, f64::INFINITY);
}
