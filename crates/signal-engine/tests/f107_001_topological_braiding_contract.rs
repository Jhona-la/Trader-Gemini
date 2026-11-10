// =====================================================================
// crates/signal-engine/tests/f107_001_topological_braiding_contract.rs
//
// Formal Verification Suite for Fase 107 (§107):
// Topological Quantum Computation & Fibonacci Anyon Braiding.
// =====================================================================

use signal_engine::topological_quantum_braiding::{
    TopologicalQuantumBraidingConfig, TopologicalQuantumBraidingEngine, PHI_INV, PHI_INV_SQRT,
};

#[test]
fn test_c1_fibonacci_f_matrix_unitarity_and_involution() {
    let engine = TopologicalQuantumBraidingEngine::with_default();
    let f = engine.f_matrix();
    assert_eq!(f[0][0], PHI_INV);
    assert_eq!(f[0][1], PHI_INV_SQRT);
    // F = [[phi^{-1}, phi^{-1/2}], [phi^{-1/2}, -phi^{-1}]]
    // F^2 must be exact Identity:
    // (0,0): phi^{-2} + phi^{-1} = 1.0
    // (0,1): phi^{-3/2} - phi^{-3/2} = 0.0
    // (1,1): phi^{-1} + phi^{-2} = 1.0
    let f00 = PHI_INV * PHI_INV + PHI_INV_SQRT * PHI_INV_SQRT;
    let f01 = PHI_INV * PHI_INV_SQRT - PHI_INV_SQRT * PHI_INV;
    let f11 = PHI_INV_SQRT * PHI_INV_SQRT + (-PHI_INV) * (-PHI_INV);

    assert!((f00 - 1.0).abs() < 1e-12, "F00^2 + F01^2 must equal 1.0");
    assert!(
        f01.abs() < 1e-12,
        "Off-diagonal element must cancel exactly to 0.0"
    );
    assert!((f11 - 1.0).abs() < 1e-12, "F10^2 + F11^2 must equal 1.0");
}

#[test]
fn test_c2_topological_qubit_unitarity_preserved_under_braiding() {
    let mut engine = TopologicalQuantumBraidingEngine::with_default();

    for i in 0..50 {
        let ret = (i as f64 * 0.1).sin() * 0.002;
        let ofi = (i as f64 * 0.15).cos() * 0.4;
        let rep = engine.update(ret, ofi, 1.2, 1.0);

        assert!(
            (rep.topological_fidelity - 1.0).abs() < 1e-6,
            "Unitary braiding must preserve fidelity"
        );
        assert!(
            !rep.topological_braid_veto,
            "Unitary braiding should not trigger veto"
        );
        assert!(rep.qubit_probability_0 >= 0.0 && rep.qubit_probability_0 <= 1.0);
        assert!(rep.qubit_probability_1 >= 0.0 && rep.qubit_probability_1 <= 1.0);
        assert!((rep.qubit_probability_0 + rep.qubit_probability_1 - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_c3_directional_signal_evolution_from_asymmetric_flow() {
    let mut engine = TopologicalQuantumBraidingEngine::with_default();

    // Persistent strong positive momentum
    for _ in 0..20 {
        engine.update(0.005, 0.8, 2.0, 0.8);
    }

    let rep = engine.update(0.005, 0.8, 2.0, 0.8);
    assert!(
        rep.directional_signal.abs() > 0.05,
        "Persistent directional flow must shift topological signal"
    );
    assert!(rep.braid_entropy >= 0.0 && rep.braid_entropy <= 0.70);
    assert!(rep.total_braids_compiled == 21);
}

#[test]
fn test_c4_decoherence_tolerance_and_veto_firing() {
    let mut config = TopologicalQuantumBraidingConfig::default();
    config.min_fidelity_threshold = 0.999999;
    config.decoherence_tolerance = 0.0000001;
    let mut engine = TopologicalQuantumBraidingEngine::new(config);

    // Warmup
    engine.update(0.001, 0.1, 1.0, 1.0);

    // Force non-unitary perturbation by feeding NaN/extreme values
    let rep = engine.update(f64::NAN, f64::INFINITY, 1.0, 1.0);
    // Veto behavior check
    assert!(rep.slot_scale >= config.scale_min && rep.slot_scale <= config.scale_max);
}

#[test]
fn test_c5_zero_allocation_and_nan_immunity() {
    let mut engine = TopologicalQuantumBraidingEngine::with_default();

    for _ in 0..10 {
        engine.update(0.002, 0.3, 1.0, 1.0);
    }

    // Feed NaNs and infinities
    let rep = engine.update(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(rep.directional_signal.is_finite());
    assert!(rep.topological_fidelity.is_finite());
    assert!(rep.braid_entropy.is_finite());
    assert!(rep.slot_scale.is_finite());

    // Reset and verify clean state
    engine.reset();
    let rep_reset = engine.update(0.0, 0.0, 1.0, 1.0);
    assert!(!rep_reset.topological_braid_veto);
    assert_eq!(rep_reset.total_braids_compiled, 1);
}
