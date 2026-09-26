//! Regression contracts from FMT-096/098/099. No exchange, files, or live engine.
use risk_engine::kelly_envelope::{EdgePosterior, RiskEnvelope};
use risk_engine::tp_sl::{compute_tp_sl, compute_tp_sl_with_target_rr, TpSlInputs};

fn geometry() -> TpSlInputs {
    TpSlInputs {
        tau_ms: 60_000.0,
        atr_ratio: 0.002,
        hurst: 0.5,
        roundtrip_fee: 0.001,
        sl_atr_multiplier: 1.0,
        sigma_forecast: None,
    }
}

fn positive_edge() -> RiskEnvelope {
    RiskEnvelope {
        posterior: EdgePosterior {
            alpha: 300.5,
            beta: 100.5,
        },
        avg_win: 2.0,
        avg_loss: 1.0,
        payoff_ratio: 2.0,
    }
}

#[test]
fn contracts_target_rr_cannot_reduce_baseline_or_change_probability_contract() {
    let i = geometry();
    let baseline = compute_tp_sl(i);
    let targeted = compute_tp_sl_with_target_rr(i, 2.0);
    assert!(targeted.tp_pct >= baseline.tp_pct);
    assert_eq!(targeted.sl_pct, baseline.sl_pct);
    assert_eq!(targeted.rr_required, baseline.rr_required);
    let p = quantum_arena::genome::SuperGenotype::WORST_TOLERATED_WR;
    let ev = p * targeted.tp_pct - (1.0 - p) * targeted.sl_pct - i.roundtrip_fee;
    assert!(ev >= -1e-12, "conditional EV changed: {ev}");
}

#[test]
fn contracts_target_rr_is_monotone_over_continuous_horizons() {
    // Representation domain, not a claim of observability or tradability at every tau.
    for j in 0..129 {
        let mut i = geometry();
        i.tau_ms = (1e-6_f64.ln() + j as f64 / 128.0 * (3.15576e12_f64 / 1e-6).ln()).exp();
        let baseline = compute_tp_sl(i);
        let mut previous = baseline.tp_pct;
        for target in [0.0, 1.0, 1.9, 2.0, 2.5, 3.0, 5.0, 10.0] {
            let out = compute_tp_sl_with_target_rr(i, target);
            assert!(out.tp_pct >= previous, "tau={} target={target}", i.tau_ms);
            assert_eq!(out.rr_required, baseline.rr_required);
            assert_eq!(out.below_tradeable_floor, baseline.below_tradeable_floor);
            previous = out.tp_pct;
        }
    }
}

#[test]
fn contracts_nonrepresentable_target_does_not_poison_finite_geometry() {
    let mut i = geometry();
    i.atr_ratio = 2.0;
    let baseline = compute_tp_sl(i);
    assert!(baseline.tp_pct.is_finite());
    let out = compute_tp_sl_with_target_rr(i, f64::MAX);
    assert!(out.tp_pct.is_finite());
    assert_eq!(out.tp_pct, baseline.tp_pct);
}

/// D-750 (fusión PR #5): sin edge probado la exposición financiada es
/// EXACTAMENTE la orden mínima ejecutable del símbolo (mn·stop/capital de
/// fracción, es decir mn/capital de techo nocional), ni un céntimo más, y
/// sólo en régimen micro. En régimen estándar no hay exploración financiada.
#[test]
fn contracts_exploration_without_evidence_is_exactly_the_minimum_executable() {
    let env = RiskEnvelope::new();
    assert_eq!(env.risk_fraction(0.85, 10.0), 0.0);
    assert_eq!(
        env.max_leverage(13.0, 0.005, 5.0, 0.85, 10.0),
        (5.0 / 13.0, true)
    );
    assert_eq!(env.max_leverage(100.0, 0.005, 5.0, 0.85, 10.0), (0.0, false));
}

/// D-750: la evidencia NEGATIVA mata el dimensionado (risk_fraction = 0)
/// pero no el sondeo mínimo, que no puede violar el tope de ruina por
/// construcción (una sola orden mínima no arruina la cuenta). La evidencia
/// modula el TECHO, no la existencia de la sonda.
#[test]
fn contracts_mature_negative_evidence_kills_sizing_not_the_minimal_probe() {
    let mut env = RiskEnvelope::new();
    for _ in 0..500 {
        env.record_trade(false, 0.0, -1.0);
    }
    assert_eq!(env.risk_fraction(0.85, 10.0), 0.0);
    assert_eq!(
        env.max_leverage(13.0, 0.005, 5.0, 0.85, 10.0),
        (5.0 / 13.0, true)
    );
}

#[test]
fn contracts_exchange_minimum_cannot_override_positive_risk_budget() {
    let env = RiskEnvelope {
        posterior: EdgePosterior {
            alpha: 500.5,
            beta: 500.5,
        },
        avg_win: 1.0 / 0.97,
        avg_loss: 1.0,
        payoff_ratio: 1.0 / 0.97,
    };
    let f = env.risk_fraction(0.0, 0.0);
    assert!((f - 0.015).abs() < 1e-12);
    assert!(13.0 * f < 5.0 * 0.1);
    assert_eq!(env.max_leverage(13.0, 0.1, 5.0, 0.0, 0.0), (0.0, false));
}

#[test]
fn contracts_invalid_capital_stop_and_exchange_minimum_fail_closed() {
    let env = positive_edge();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        assert_eq!(env.max_leverage(bad, 0.01, 5.0, 1.64, 50.0), (0.0, false));
        assert_eq!(env.max_leverage(100.0, bad, 5.0, 1.64, 50.0), (0.0, false));
        assert_eq!(env.max_leverage(100.0, 0.01, bad, 1.64, 50.0), (0.0, false));
    }
}

#[test]
fn contracts_invalid_posterior_or_uncertainty_cannot_create_edge() {
    let env = positive_edge();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert_eq!(env.risk_fraction(bad, 50.0), 0.0);
        assert_eq!(env.risk_fraction(1.64, bad), 0.0);
        let mut damaged = env.clone();
        damaged.posterior.alpha = bad;
        assert_eq!(damaged.risk_fraction(1.64, 50.0), 0.0);
        damaged = env.clone();
        damaged.posterior.beta = bad;
        assert_eq!(damaged.risk_fraction(1.64, 50.0), 0.0);
        damaged = env.clone();
        damaged.payoff_ratio = bad;
        assert_eq!(damaged.risk_fraction(1.64, 50.0), 0.0);
    }
}

#[test]
fn projection_limits_quantity_and_includes_roundtrip_cost() {
    let env = positive_edge();
    let budget = env.exposure_budget(100.0, 0.10, 0.01, 1.64, 50.0).unwrap();
    let q = budget
        .project_quantity(10.0, 100.0, 0.01, 0.01, 5.0)
        .unwrap();
    assert!(q < 10.0);
    assert!(q * 100.0 <= budget.max_notional());
    assert!(q * 100.0 * 0.11 <= budget.risk_budget_usd());
    assert_eq!(budget.loss_per_notional(), 0.11);
}

#[test]
fn projection_checks_lot_feasibility_after_continuous_minimum() {
    let env = positive_edge();
    let budget = env.exposure_budget(20.0, 0.1, 0.0, 1.64, 50.0).unwrap();
    assert_eq!(budget.max_notional(), 50.0);
    // Continuous N=50 exists; at price=30 and step=1 only N=30 or 60 exist.
    assert!(budget
        .project_quantity(10.0, 30.0, 1.0, 1.0, 50.0)
        .is_none());
    assert_eq!(
        budget.project_quantity(10.0, 30.0, 1.0, 1.0, 30.0),
        Some(1.0)
    );
}

#[test]
fn projection_never_raises_candidate_to_an_exchange_minimum() {
    let budget = positive_edge()
        .exposure_budget(100.0, 0.01, 0.001, 1.64, 50.0)
        .unwrap();
    assert!(budget
        .project_quantity(0.01, 100.0, 0.001, 0.001, 5.0)
        .is_none());
    assert!(budget
        .project_quantity(0.01, 100.0, 0.001, 0.1, 0.5)
        .is_none());
}

#[test]
fn projection_cost_increase_cannot_increase_exposure() {
    let env = positive_edge();
    let mut previous = f64::INFINITY;
    for cost in [0.0, 0.0001, 0.001, 0.01, 0.10, 1.0] {
        let budget = env.exposure_budget(100.0, 0.01, cost, 1.64, 50.0).unwrap();
        let q = budget
            .project_quantity(1000.0, 100.0, 0.001, 0.001, 1.0)
            .unwrap_or(0.0);
        assert!(q <= previous);
        previous = q;
    }
}

#[test]
fn projection_is_invariant_to_currency_units() {
    let env = positive_edge();
    let baseline = env
        .exposure_budget(100.0, 0.10, 0.01, 1.64, 50.0)
        .unwrap()
        .project_quantity(10.0, 100.0, 0.01, 0.01, 5.0)
        .unwrap();
    for k in [0.01, 1.0, 10.0, 1000.0] {
        let budget = env
            .exposure_budget(100.0 * k, 0.10, 0.01, 1.64, 50.0)
            .unwrap();
        assert_eq!(
            budget.project_quantity(10.0, 100.0 * k, 0.01, 0.01, 5.0 * k),
            Some(baseline)
        );
    }
}

#[test]
fn projection_post_rounding_invariants_on_scale_grid() {
    let env = positive_edge();
    let mut accepted = 0;
    for capital in [0.01, 13.0, 1000.0, 1e9] {
        for stop in [1e-6, 0.001, 0.1, 1.0] {
            let budget = env
                .exposure_budget(capital, stop, 0.0008, 1.64, 50.0)
                .unwrap();
            for price in [0.01, 0.3, 100.0, 12_345.678] {
                for step in [1e-8, 0.001, 0.1, 1.0] {
                    for proposed in [0.03, 1.0, 1000.0] {
                        if let Some(q) = budget.project_quantity(proposed, price, step, 0.0, 0.001)
                        {
                            accepted += 1;
                            assert!(q <= proposed);
                            assert!(q * price >= 0.001);
                            assert!(q * price <= budget.max_notional());
                            assert!(
                                q * price * budget.loss_per_notional() <= budget.risk_budget_usd()
                            );
                            let lots = q / step;
                            assert!(
                                (lots - lots.round()).abs()
                                    <= 2.0 * f64::EPSILON * lots.abs().max(1.0)
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(accepted > 100, "non-vacuous property grid");
}

#[test]
fn projection_rejects_invalid_market_constraints_and_unrepresentable_lots() {
    let budget = positive_edge()
        .exposure_budget(100.0, 0.1, 0.01, 1.64, 50.0)
        .unwrap();
    for bad in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert!(budget
            .project_quantity(bad, 100.0, 0.01, 0.01, 5.0)
            .is_none());
        assert!(budget
            .project_quantity(10.0, bad, 0.01, 0.01, 5.0)
            .is_none());
        assert!(budget
            .project_quantity(10.0, 100.0, bad, 0.01, 5.0)
            .is_none());
        assert!(budget
            .project_quantity(10.0, 100.0, 0.01, 0.01, bad)
            .is_none());
        if bad != 0.0 {
            assert!(budget
                .project_quantity(10.0, 100.0, 0.01, bad, 5.0)
                .is_none());
        }
    }
    assert!(budget
        .project_quantity(1e100, 1e-10, 1e-20, 0.0, 5.0)
        .is_none());
}

#[test]
fn budget_rejects_missing_evidence_and_nonfinite_or_overflowing_inputs() {
    assert!(RiskEnvelope::new()
        .exposure_budget(100.0, 0.01, 0.001, 1.64, 50.0)
        .is_none());
    let env = positive_edge();
    for bad in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert!(env.exposure_budget(bad, 0.01, 0.001, 1.64, 50.0).is_none());
        assert!(env.exposure_budget(100.0, bad, 0.001, 1.64, 50.0).is_none());
        if bad != 0.0 {
            assert!(env.exposure_budget(100.0, 0.01, bad, 1.64, 50.0).is_none());
        }
    }
    assert!(env
        .exposure_budget(f64::MAX, f64::MIN_POSITIVE, 0.0, 1.64, 50.0)
        .is_none());
    assert!(env
        .exposure_budget(100.0, f64::MAX, f64::MAX, 1.64, 50.0)
        .is_none());
}

#[test]
fn exposure_ratio_and_budget_agree_when_minimum_is_feasible() {
    let env = positive_edge();
    for capital in [13.0, 100.0, 10_000.0] {
        for stop in [0.0001_f64, 0.005, 0.1] {
            let budget = env
                .exposure_budget(capital, stop.max(0.0005), 0.0, 1.64, 50.0)
                .unwrap();
            let (ratio, allowed) = env.max_leverage(capital, stop, 5.0, 1.64, 50.0);
            if allowed {
                assert_eq!(ratio, budget.max_notional() / capital);
                assert!(5.0 * budget.loss_per_notional() <= budget.risk_budget_usd());
            } else {
                assert_eq!(ratio, 0.0);
            }
        }
    }
}

#[test]
fn contracts_minimum_notional_rejects_invalid_filter_and_product_overflow() {
    use risk_engine::guard::enforce_minimum_notional;
    for minimum in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        assert_eq!(enforce_minimum_notional(10.0, minimum, 2.0), (false, 0.0));
    }
    assert_eq!(enforce_minimum_notional(f64::MAX, 5.0, 2.0), (false, 0.0));
    assert_eq!(
        enforce_minimum_notional(f64::from_bits(1), 5.0, 0.5),
        (false, 0.0)
    );
    assert_eq!(enforce_minimum_notional(2.5, 5.0, 2.0), (true, 2.5));
}

#[test]
fn contracts_a_small_payoff_cannot_be_promoted_into_an_edge() {
    let env = RiskEnvelope {
        posterior: EdgePosterior {
            alpha: 99900.5,
            beta: 100.5,
        },
        avg_win: 0.0001,
        avg_loss: 1.0,
        payoff_ratio: 0.0001,
    };
    // Even p>99% cannot pay for the loss size. b must not be floored to .05.
    assert_eq!(env.risk_fraction(1.64, 50.0), 0.0);
}

#[test]
fn contracts_a_rare_win_cannot_be_floored_to_one_percent() {
    let env = RiskEnvelope {
        posterior: EdgePosterior {
            alpha: 100.5,
            beta: 99900.5,
        },
        avg_win: 200.0,
        avg_loss: 1.0,
        payoff_ratio: 200.0,
    };
    // Actual posterior mean about .001005 is below breakeven 1/201.
    assert_eq!(env.risk_fraction(1.64, 50.0), 0.0);
    let expected = 100.5 / 100001.0;
    assert_eq!(env.posterior.mean(), expected);
    assert!(env.posterior.lcb(1.64) < expected);
}

#[test]
fn contracts_beta_variance_does_not_overflow_intermediate_products() {
    let posterior = EdgePosterior {
        alpha: 1e200,
        beta: 1e200,
    };
    assert_eq!(posterior.mean(), 0.5);
    let sigma = posterior.sd();
    assert!(sigma.is_finite() && sigma > 0.0);
    assert!((sigma - 3.5355339059327376e-101).abs() < 1e-115);
}
