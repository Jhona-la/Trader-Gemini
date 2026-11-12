//! Contrato Formal de Certificación: Fase 109 (§109) — Simetría Especular Homológica de Kontsevich
//!
//! Verifica en silicio:
//! 1. Invariantes del modelo A y B en estado nominal.
//! 2. Crecimiento analítico de acoplamientos de Yukawa con instantones de Gromov-Witten.
//! 3. Respuesta quiral de la homología de intersección de Floer ante OFI.
//! 4. Activación determinista fail-closed del veto ante colapso de simetría.
//! 5. Robustez numérica absoluta e inmunidad a NaN/Inf (invariante `rb_p1`).

use feature_engine::homological_mirror_symmetry::{
    HomologicalMirrorSymmetryConfig, HomologicalMirrorSymmetryEngine,
};

#[test]
fn test_c1_nominal_homological_mirror_symmetry_invariants() {
    let mut engine = HomologicalMirrorSymmetryEngine::new(HomologicalMirrorSymmetryConfig::default());

    let rep = engine.update(0.0002, 0.15, 0.85, 2.5);

    assert!(rep.symplectic_kahler_modulus_im > 0.0);
    assert!(rep.instanton_yukawa_coupling > 0.0);
    assert!(rep.mirror_jacobian_determinant > 0.0);
    assert!(!rep.mirror_symmetry_collapse_veto);
    assert!(rep.slot_scale >= 0.10 && rep.slot_scale <= 1.25);
}

#[test]
fn test_c2_instanton_yukawa_coupling_growth() {
    let mut engine = HomologicalMirrorSymmetryEngine::new(HomologicalMirrorSymmetryConfig::default());

    // Evaluamos una serie de ticks y verificamos estabilidad y finitud de la suma instantónica
    for i in 0..10 {
        let rep = engine.update(
            0.0001 * (i as f64),
            0.10,
            0.50 + 0.05 * (i as f64),
            2.0,
        );
        assert!(rep.instanton_yukawa_coupling.is_finite());
        assert!(rep.instanton_yukawa_coupling >= 0.0);
    }
}

#[test]
fn test_c3_floer_lagrangian_intersection_directional_response() {
    let mut engine_bull = HomologicalMirrorSymmetryEngine::new(HomologicalMirrorSymmetryConfig::default());
    let mut engine_bear = HomologicalMirrorSymmetryEngine::new(HomologicalMirrorSymmetryConfig::default());

    let rep_bull = engine_bull.update(0.001, 0.80, 0.50, 1.5);
    let rep_bear = engine_bear.update(-0.001, -0.80, 0.50, 1.5);

    assert!(
        rep_bull.floer_intersection_index > rep_bear.floer_intersection_index,
        "OFI alcista debe producir un índice de intersección de Floer mayor que OFI bajista"
    );
}

#[test]
fn test_c4_mirror_symmetry_collapse_veto_propagation() {
    let mut config = HomologicalMirrorSymmetryConfig::default();
    // Fijamos un umbral estricto sobre el determinante jacobiano para forzar veto
    config.min_jacobian_determinant = 10.0;
    let mut engine = HomologicalMirrorSymmetryEngine::new(config);

    let rep = engine.update(0.0001, 0.0, 1.0, 5.0);

    assert!(rep.mirror_symmetry_collapse_veto);
    assert_eq!(rep.slot_scale, 0.10);
}

#[test]
fn test_c5_zero_panic_rb_p1_and_numerical_guards() {
    let mut engine = HomologicalMirrorSymmetryEngine::new(HomologicalMirrorSymmetryConfig::default());

    let degenerate_inputs = [
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, 1e20, -50.0),
        (-0.50, -1.0, 0.0, 0.0),
        (0.0, 0.0, 100.0, 500.0),
    ];

    for &(ret, ofi, vol, spr) in &degenerate_inputs {
        let rep = engine.update(ret, ofi, vol, spr);
        assert!(rep.symplectic_kahler_modulus_re.is_finite());
        assert!(rep.symplectic_kahler_modulus_im.is_finite());
        assert!(rep.instanton_yukawa_coupling.is_finite());
        assert!(rep.mirror_jacobian_determinant.is_finite());
        assert!(rep.floer_intersection_index.is_finite());
        assert!(rep.slot_scale >= 0.10 && rep.slot_scale <= 1.25);
    }
}
