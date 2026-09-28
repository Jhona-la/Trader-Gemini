//! Regressions for the repaired numerical/input contracts. These assertions
//! do not certify the statistical model or the entire portfolio-risk path.
use quantum_arena::state::CompactTick;
use risk_engine::correlation_guard::{
    hayashi_yoshida_correlation, pearson, rho_promedio, CorrelationGuardEngine,
};

fn tick(timestamp: u64, price: f64) -> CompactTick {
    CompactTick {
        timestamp,
        bid_price: price,
        ask_price: price,
        bid_qty: 1.0,
        ask_qty: 1.0,
    }
}

#[test]
fn hy_must_reject_duplicate_timestamps() {
    let a = vec![
        tick(1, 100.0),
        tick(2, 101.0),
        tick(2, 102.0),
        tick(3, 103.0),
    ];
    assert!(hayashi_yoshida_correlation(&a, &a).is_none());
}

#[test]
fn hy_common_window_must_not_depend_on_disjoint_history() {
    let b = vec![tick(10, 100.0), tick(11, 101.0), tick(12, 100.0)];
    let mut a = vec![tick(0, 100.0 * (-50.0_f64).exp())];
    a.extend(b.iter().cloned());
    let common = hayashi_yoshida_correlation(&b, &b).unwrap();
    let extended = hayashi_yoshida_correlation(&a, &b).unwrap();
    assert!((extended - common).abs() < 1e-10, "{extended} != {common}");
}

#[test]
fn unknown_pairs_must_not_be_dropped_from_the_mean() {
    let c = vec![
        vec![1.0, 0.2, f64::NAN],
        vec![0.2, 1.0, 0.2],
        vec![f64::NAN, 0.2, 1.0],
    ];
    assert!(rho_promedio(&c).is_none());
}

#[test]
fn impossible_average_correlation_must_not_relax_the_veto() {
    let cap = risk_engine::ruin::clamp_ruin(1.0, 0.5);
    assert!(CorrelationGuardEngine::veto_por_exposicion_estructural(
        4,
        cap / 2.0,
        0.5,
        Some(-0.4)
    ));
}

#[test]
fn pearson_must_preserve_scale_invariance_for_finite_samples() {
    let a = [-1e150, 1e150];
    assert!((pearson(&a, &a).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn synchronous_hy_is_not_centered_pearson_in_general() {
    // Integrated co-variation uses uncentered increments; Pearson centers.
    let a = vec![
        tick(0, 100.0),
        tick(1, 100.0 * 0.1_f64.exp()),
        tick(2, 100.0 * 0.3_f64.exp()),
    ];
    let b = vec![
        tick(0, 100.0),
        tick(1, 100.0 * 0.2_f64.exp()),
        tick(2, 100.0 * 0.3_f64.exp()),
    ];
    assert!((hayashi_yoshida_correlation(&a, &b).unwrap() - 0.8).abs() < 1e-12);
    assert!((pearson(&[0.1, 0.2], &[0.2, 0.1]).unwrap() + 1.0).abs() < 1e-12);
}
