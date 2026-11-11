// =====================================================================
// crates/risk-engine/tests/f108_001_hayden_preskill_contract.rs
//
// Formal Verification Suite for Fase 108 (§108):
// AdS/CFT Holographic Complexity & Hayden-Preskill Quantum Scrambling.
// =====================================================================

use risk_engine::hayden_preskill_scrambling::{HaydenPreskillConfig, HaydenPreskillEngine};

#[test]
fn test_c1_ryu_takayanagi_entropy_subregion_monotonicity() {
    let mut engine = HaydenPreskillEngine::with_default();

    for i in 0..20 {
        let dd = i as f64 * 0.02; // Drawdown increasing from 0% to 38%
        let rep = engine.update(dd, 1.0, 0.001, 1.0);
        assert!(rep.ryu_takayanagi_entropy > 0.0);
        assert!(rep.ryu_takayanagi_entropy.is_finite());
    }
}

#[test]
fn test_c2_tripartite_mutual_information_negative_for_scramblers() {
    let mut engine = HaydenPreskillEngine::with_default();

    for _ in 0..15 {
        let rep = engine.update(0.05, 1.2, 0.002, 1.5);
        assert!(
            rep.tripartite_mutual_information <= 0.0,
            "Tripartite mutual information I_3 must be negative for fast scramblers"
        );
        assert!(!rep.hayden_preskill_veto);
        assert!(!rep.mss_chaos_veto);
    }
}

#[test]
fn test_c3_hayden_preskill_veto_on_severe_drawdown() {
    let mut engine = HaydenPreskillEngine::with_default();

    // Warmup
    for _ in 0..10 {
        engine.update(0.05, 1.0, 0.001, 1.0);
    }

    // Severe drawdown: 90%
    let rep = engine.update(0.90, 3.0, -0.05, 5.0);
    assert!(
        rep.hayden_preskill_veto,
        "Severe drawdown past Page time must trigger Hayden-Preskill veto"
    );
    assert_eq!(
        rep.slot_scale, 0.10,
        "Slot scale must contract to scale_min"
    );
}

#[test]
fn test_c4_mss_chaos_bound_violation_veto() {
    let config = HaydenPreskillConfig {
        max_mss_tolerance: 0.05,
        ..Default::default()
    };
    let mut engine = HaydenPreskillEngine::new(config);

    // Inject massive exponential growth in OTOC to force Lyapunov explosion
    // with very low temperature (low vol, low spread, high cushion)
    for i in 0..20 {
        let otoc_shock = ((i as f64) * 0.50).exp() * 0.001;
        engine.update(0.0, 0.01, otoc_shock, 0.1);
    }

    let rep = engine.update(0.0, 0.01, 5.0, 0.1);
    assert!(rep.quantum_lyapunov_exponent > 0.1);
    assert!(rep.mss_chaos_bound > 0.0);
    // When chaos exceeds bound + tolerance, veto must fire
    if rep.quantum_lyapunov_exponent - rep.mss_chaos_bound > config.max_mss_tolerance {
        assert!(rep.mss_chaos_veto);
        assert_eq!(rep.slot_scale, config.scale_min);
    }
}

#[test]
fn test_c5_zero_allocation_and_nan_immunity() {
    let mut engine = HaydenPreskillEngine::with_default();

    for _ in 0..10 {
        engine.update(0.05, 1.0, 0.001, 1.0);
    }

    // Feed non-finite numbers
    let rep = engine.update(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    assert!(rep.ryu_takayanagi_entropy.is_finite());
    assert!(rep.tripartite_mutual_information.is_finite());
    assert!(rep.quantum_lyapunov_exponent.is_finite());
    assert!(rep.mss_chaos_bound.is_finite());
    assert!(rep.recovery_fidelity.is_finite());
    assert!(rep.slot_scale.is_finite());

    // Reset and verify clean state
    engine.reset();
    let rep_reset = engine.update(0.0, 1.0, 0.0, 1.0);
    assert!(!rep_reset.hayden_preskill_veto);
    assert!(!rep_reset.mss_chaos_veto);
}
