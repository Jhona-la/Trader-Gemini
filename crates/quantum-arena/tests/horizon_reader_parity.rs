//! FMT-134: isolated genotype/configuration parity, without live arena or disk.
use quantum_arena::genome::SuperGenotype;
use quantum_arena::temporal_spectrum::{HorizonCurve, SPECTRUM_SCALES_MS};
use quantum_arena::QuantumConfig;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn fixture() -> SuperGenotype {
    let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
    g.tp_horizon_curve = HorizonCurve { a: -5.0, b: 0.1 };
    g.sl_horizon_curve = HorizonCurve { a: -6.0, b: 0.1 };
    g.scalp_kelly_fraction = 0.1;
    g.swing_kelly_fraction = 0.3;
    g.scalp_obi_threshold = 0.075;
    g.swing_obi_threshold = 0.5;
    g.scalp_trail_atr_mult_base = 1.0;
    g.swing_trail_atr_mult_base = 10.0;
    g.scalp_trail_act_atr = 0.2;
    g.swing_trail_act_atr = 10.0;
    g.scalp_trail_step_atr = 0.1;
    g.swing_trail_step_atr = 10.0;
    g.derive_anchors_from_curves();
    g.sync_continuous_curves();
    g
}

#[test]
fn tp_sl_readers_share_the_runtime_domain() {
    let g = fixture();
    let c = QuantumConfig::from_genome(13.0, &g);
    for tau in SPECTRUM_SCALES_MS {
        close(g.tp_at_tau(tau), c.tp_at_tau(tau));
        close(g.sl_at_tau(tau), c.sl_at_tau(tau));
    }
}

#[test]
fn kelly_readers_share_the_runtime_domain_and_bounds() {
    let g = fixture();
    let c = QuantumConfig::from_genome(13.0, &g);
    for tau in SPECTRUM_SCALES_MS {
        close(g.kelly_at_tau(tau), c.kelly_at_tau(tau));
    }
}

#[test]
fn obi_readers_share_the_runtime_domain_and_bounds() {
    let g = fixture();
    let c = QuantumConfig::from_genome(13.0, &g);
    for tau in SPECTRUM_SCALES_MS {
        close(g.obi_threshold_at_tau(tau), c.obi_threshold_at_tau(tau));
    }
}

#[test]
fn trailing_readers_share_the_runtime_domain_and_bounds() {
    let g = fixture();
    let c = QuantumConfig::from_genome(13.0, &g);
    for tau in SPECTRUM_SCALES_MS {
        let (gm, ga, gs) = g.trail_params_at_tau(tau);
        let (cm, ca, cs, _) = c.trail_params_at_tau(tau);
        close(gm, cm);
        close(ga, ca);
        close(gs, cs);
    }
}

#[test]
fn derived_curve_cache_cannot_override_authoritative_genes() {
    let mut g = fixture();
    // Serialized derived values can be stale; runtime already resynchronizes.
    g.kelly_horizon_curve = HorizonCurve::flat(0.99);
    g.obi_horizon_curve = HorizonCurve::flat(0.99);
    g.trail_mult_horizon_curve = HorizonCurve::flat(15.0);
    g.trail_act_horizon_curve = HorizonCurve::flat(15.0);
    g.trail_step_horizon_curve = HorizonCurve::flat(15.0);
    let c = QuantumConfig::from_genome(13.0, &g);
    let tau = 600_000.0;
    close(g.kelly_at_tau(tau), c.kelly_at_tau(tau));
    close(g.obi_threshold_at_tau(tau), c.obi_threshold_at_tau(tau));
    let (gm, ga, gs) = g.trail_params_at_tau(tau);
    let (cm, ca, cs, _) = c.trail_params_at_tau(tau);
    close(gm, cm);
    close(ga, ca);
    close(gs, cs);
}

#[test]
fn config_retains_its_existing_formulas_for_valid_queries() {
    let g = fixture();
    let c = QuantumConfig::from_genome(13.0, &g);
    for i in 0..=256 {
        let tau = (SPECTRUM_SCALES_MS[0].ln()
            + i as f64 / 256.0 * (SPECTRUM_SCALES_MS[31] / SPECTRUM_SCALES_MS[0]).ln())
        .exp();
        close(c.tp_at_tau(tau), g.tp_horizon_curve.eval(tau));
        close(c.sl_at_tau(tau), g.sl_horizon_curve.eval(tau));
        close(
            c.kelly_at_tau(tau),
            g.kelly_horizon_curve.eval(tau).clamp(0.01, 3.0),
        );
        close(
            c.obi_threshold_at_tau(tau),
            g.obi_horizon_curve.eval(tau).clamp(0.10, 0.95),
        );
        let expected_act = g.trail_act_horizon_curve.eval(tau).clamp(1.5, 6.0);
        let (m, a, s, maximum) = c.trail_params_at_tau(tau);
        close(m, g.trail_mult_horizon_curve.eval(tau).clamp(1.5, 6.0));
        close(a, expected_act);
        close(s, g.trail_step_horizon_curve.eval(tau).clamp(0.5, 4.0));
        close(maximum, (expected_act * 1.5).clamp(2.0, 7.0));
    }
}
