//! OPEN design defects: passing characterizes the current limitation.
use quantum_arena::position::{Position, PositionHorizon, PositionManager};
use std::sync::atomic::Ordering;

#[test]
fn counterexample_log_scale_separation_does_not_imply_orthogonality() {
    // For normalized causal exponentials h_tau(t)=sqrt(2/tau)*exp(-t/tau),
    // the L2 inner product is 2*sqrt(tau1*tau2)/(tau1+tau2)=sech(delta/2).
    // Algebraic counterexample to the interpretation, not a model in production.
    let delta_log_tau: f64 = 0.80;
    let inner_product = 1.0 / (delta_log_tau / 2.0).cosh();
    assert!(inner_product > 0.92);
}
fn open(p: &Position) {
    assert!(p.open_with_horizon(
        true,
        100.0,
        1.0,
        10.0,
        1,
        110.0,
        90.0,
        PositionHorizon::Continuous
    ));
}
#[test]
fn open_debt_entry_tensor_survives_a_new_generation_without_features() {
    let p = Position::default();
    open(&p);
    *p.nn_entry_tensor.lock().unwrap() = vec![7.0; 54];
    let previous = p.generation.load(Ordering::Acquire);
    p.close();
    open(&p);
    assert!(p.generation.load(Ordering::Acquire) > previous);
    assert_eq!(*p.nn_entry_tensor.lock().unwrap(), vec![7.0; 54]);
}
#[test]
fn open_debt_invalid_scale_is_replaced_by_fixed_admission_scale() {
    let p = PositionManager::default();
    for t in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e-6] {
        assert_eq!(p.find_resonant_slot(t, true), Some(0));
    }
}
#[test]
fn open_debt_public_mutation_does_not_version_the_snapshot() {
    let p = Position::default();
    open(&p);
    let before = p.snapshot().unwrap();
    p.quantity.store(2.0, Ordering::Relaxed);
    let after = p.snapshot().unwrap();
    assert_eq!(before.generation, after.generation);
    assert_ne!(before.quantity, after.quantity);
}
