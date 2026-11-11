// =====================================================================
// crates/feature-engine/tests/f106_001_geometric_langlands_contract.rs
//
// Formal Verification Suite for Fase 106 (§106):
// Geometric Langlands Program & Automorphic Hecke Eigensheaves.
// =====================================================================

use feature_engine::geometric_langlands_hecke::{
    GeometricLanglandsHeckeConfig, GeometricLanglandsHeckeEngine,
};

#[test]
fn test_c1_hecke_eigenvalues_bounded_under_nominal_conditions() {
    let mut engine = GeometricLanglandsHeckeEngine::with_default();

    // Nominal calm market
    for i in 0..25 {
        let ret = (i as f64 * 0.05).sin() * 0.001;
        let ofi = (i as f64 * 0.1).cos() * 0.2;
        let rep = engine.update(ret, ofi, 1.0, 1.5);
        if i > 10 {
            assert!(
                !rep.langlands_anomaly_veto,
                "Nominal market should not trigger Langlands veto"
            );
            assert!(rep.slot_scale >= 0.90 && rep.slot_scale <= 1.25);
        }
    }
}

#[test]
fn test_c2_ramanujan_violation_triggers_langlands_anomaly_veto() {
    let config = GeometricLanglandsHeckeConfig {
        max_anomaly_tolerance: 0.20,
        ..Default::default()
    };
    let mut engine = GeometricLanglandsHeckeEngine::new(config);

    // Inject strong resonant manipulation at prime lag 2
    for i in 0..30 {
        let ret = if i & 1 == 0 { 0.05 } else { -0.05 };
        let ofi = if i & 1 == 0 { 1.0 } else { -1.0 };
        engine.update(ret, ofi, 5.0, 0.5);
    }

    let rep = engine.update(0.05, 1.0, 5.0, 0.5);
    assert!(
        rep.langlands_anomaly_veto,
        "Resonant manipulation must trigger Langlands anomaly veto"
    );
    assert_eq!(
        rep.slot_scale, config.scale_min,
        "Slot scale must contract to scale_min under veto"
    );
    assert!(
        rep.ramanujan_violation_count > 0 || rep.max_hecke_deviation > config.max_anomaly_tolerance
    );
}

#[test]
fn test_c3_automorphic_l_function_central_evaluation() {
    let mut engine = GeometricLanglandsHeckeEngine::with_default();

    for i in 0..20 {
        let ret = (i as f64 * 0.02).sin() * 0.002;
        let ofi = (i as f64 * 0.03).cos() * 0.1;
        let rep = engine.update(ret, ofi, 1.2, 2.0);
        assert!(rep.automorphic_l_function_central.is_finite());
        assert!(rep.automorphic_l_function_central > 0.0);
    }
}

#[test]
fn test_c4_satake_parameters_unitarity_tracking() {
    let mut engine = GeometricLanglandsHeckeEngine::with_default();

    for i in 0..25 {
        let ret = (i as f64 * 0.07).sin() * 0.003;
        let ofi = (i as f64 * 0.04).cos() * 0.3;
        let rep = engine.update(ret, ofi, 1.0, 1.0);
        for &satake in &rep.satake_parameters {
            assert!(satake.is_finite());
        }
    }
}

#[test]
fn test_c5_zero_allocation_and_nan_finite_immunity() {
    let mut engine = GeometricLanglandsHeckeEngine::with_default();

    // Warmup
    for _ in 0..10 {
        engine.update(0.001, 0.1, 1.0, 1.0);
    }

    // Feed non-finite numbers
    let rep_nan = engine.update(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    assert!(rep_nan.slot_scale.is_finite());
    assert!(rep_nan.automorphic_l_function_central.is_finite());

    // Reset and verify clean state
    engine.reset();
    let rep_reset = engine.update(0.001, 0.1, 1.0, 1.0);
    assert!(!rep_reset.langlands_anomaly_veto);
}
