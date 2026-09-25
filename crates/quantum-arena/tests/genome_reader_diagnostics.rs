//! FMT-134: former diagnostic witnesses, now regression tests for shared
//! runtime read policy. Historical results are retained in audit X.
//! No disk loading or live arena.
use quantum_arena::genome::SuperGenotype;
use quantum_arena::temporal_spectrum::{HorizonCurve, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};
use quantum_arena::QuantumConfig;

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn identical_tp_sl_coefficients_have_the_same_query_domain() {
    let mut genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    genome.tp_horizon_curve = HorizonCurve { a: -5.0, b: 0.1 };
    genome.sl_horizon_curve = HorizonCurve { a: -6.0, b: 0.1 };
    genome.derive_anchors_from_curves();
    let config = QuantumConfig::from_genome(13.0, &genome);
    for tau in [TAU_ANCHOR_FAST_MS, 600_000.0, TAU_ANCHOR_SLOW_MS] {
        close(config.tp_at_tau(tau), genome.tp_at_tau(tau));
        close(config.sl_at_tau(tau), genome.sl_at_tau(tau));
    }
    for tau in [1e-6, 3_155_760_000_000.0] {
        close(config.tp_at_tau(tau), genome.tp_horizon_curve.eval(tau));
        close(config.sl_at_tau(tau), genome.sl_horizon_curve.eval(tau));
        close(config.tp_at_tau(tau), genome.tp_at_tau(tau));
        close(config.sl_at_tau(tau), genome.sl_at_tau(tau));
    }
}

#[test]
fn obi_floor_agrees_inside_shared_domain() {
    let mut genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    genome.scalp_obi_threshold = 0.075;
    genome.swing_obi_threshold = 0.075;
    genome.sync_continuous_curves();
    let config = QuantumConfig::from_genome(13.0, &genome);
    close(genome.obi_threshold_at_tau(600_000.0), 0.10);
    close(config.obi_threshold_at_tau(600_000.0), 0.10);
}

#[test]
fn trailing_limits_agree_inside_shared_domain() {
    let mut genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    genome.scalp_trail_atr_mult_base = 1.0;
    genome.swing_trail_atr_mult_base = 1.0;
    genome.scalp_trail_act_atr = 0.2;
    genome.swing_trail_act_atr = 0.2;
    genome.scalp_trail_step_atr = 0.1;
    genome.swing_trail_step_atr = 0.1;
    genome.sync_continuous_curves();
    let config = QuantumConfig::from_genome(13.0, &genome);
    let (mult, act, step) = genome.trail_params_at_tau(600_000.0);
    close(mult, 1.5);
    close(act, 1.5);
    close(step, 0.5);
    let (mult, act, step, _) = config.trail_params_at_tau(600_000.0);
    close(mult, 1.5);
    close(act, 1.5);
    close(step, 0.5);
}
