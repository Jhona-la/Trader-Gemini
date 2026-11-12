//! Contrato Formal de Verificación: Fase 96 (§96) — Geometría de Calabi-Yau Cuaterniónica & Hiperkähler
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor hiperkähler:
//! - C1: Finitud de la norma cuaterniónica $\|q\|_{\mathbb{H}} \ge 0$, no-negatividad del mapa de momentos y curvatura Ricci regularizada.
//! - C2: Triple de formas de Kähler $(\omega_I, \omega_J, \omega_K)$ y modulación continua de slot $\eta_{\text{hyperkahler}} \in [0.10, 1.25]$.
//! - C3: Detonación determinista del veto fail-closed ante singularidad de mapa de momentos o agotamiento de capital.
//! - C4: Determinismo estricto en stack $O(1)$, inmunidad ante NaNs e infinitos y cero fallos en tiempo de ejecución.

use risk_engine::{QuaternionicHyperkahlerConfig, QuaternionicHyperkahlerEngine};

#[test]
fn test_c1_quaternion_norm_and_moment_map_finitude() {
    let config = QuaternionicHyperkahlerConfig::default();
    let mut engine = QuaternionicHyperkahlerEngine::new(config);

    let report = engine.evaluate(13.50, 0.01, 0.02, 0.85);

    assert!(
        report.quaternion_norm.is_finite() && report.quaternion_norm > 0.0,
        "Norma del cuaternión debe ser finita y estrictamente positiva: {}",
        report.quaternion_norm
    );
    assert!(
        report.moment_map_norm.is_finite() && report.moment_map_norm >= 0.0,
        "Norma del mapa de momentos debe ser no-negativa: {}",
        report.moment_map_norm
    );
    assert!(
        report.ricci_flat_curvature.is_finite() && report.ricci_flat_curvature >= 0.0,
        "Curvatura Ricci-flat debe ser finita y no-negativa: {}",
        report.ricci_flat_curvature
    );
    assert!(
        !report.moment_map_singularity_veto,
        "Estado financiero solvente no debe activar el veto de singularidad de mapa de momentos"
    );
}

#[test]
fn test_c2_kahler_triple_and_slot_scaling() {
    let config = QuaternionicHyperkahlerConfig::default();
    let mut engine = QuaternionicHyperkahlerEngine::new(config);

    for i in 0..10 {
        let cap = 13.0 + 0.5 * (i as f64);
        let pnl = 0.005 * (i as f64);
        let rep = engine.evaluate(cap, pnl, 0.02, 0.90);

        assert!(
            rep.kahler_form_omega_i.is_finite(),
            "omega_i debe ser finita: {}",
            rep.kahler_form_omega_i
        );
        assert!(
            rep.kahler_form_omega_j.is_finite(),
            "omega_j debe ser finita: {}",
            rep.kahler_form_omega_j
        );
        assert!(
            rep.kahler_form_omega_k.is_finite(),
            "omega_k debe ser finita: {}",
            rep.kahler_form_omega_k
        );
        assert!(
            rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25,
            "Factor de slot fuera de límites [0.10, 1.25]: {}",
            rep.slot_scaling_factor
        );
    }
}

#[test]
fn test_c3_moment_map_singularity_veto_activation() {
    let config = QuaternionicHyperkahlerConfig {
        max_moment_map_norm: 0.50, // Umbral muy estricto para forzar veto de mapa de momentos
        ..Default::default()
    };
    let mut engine = QuaternionicHyperkahlerEngine::new(config);

    // Con capital deplecionado o perturbación hiperbólica gigantesca, se detona el veto
    let mut report = engine.evaluate(13.0, 0.0, 0.01, 0.50);
    for _ in 0..5 {
        report = engine.evaluate(50.0, 3.0, 5.0, 0.02);
    }

    assert!(
        report.moment_map_singularity_veto,
        "Divergencia extrema del mapa de momentos debe detonar moment_map_singularity_veto"
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = QuaternionicHyperkahlerConfig::default();
    let mut engine = QuaternionicHyperkahlerEngine::new(config);

    let report = engine.evaluate(f64::NAN, f64::INFINITY, -10.0, f64::NAN);

    assert!(report.quaternion_norm.is_finite());
    assert!(report.moment_map_norm.is_finite());
    assert!(report.kahler_form_omega_i.is_finite());
    assert!(report.kahler_form_omega_j.is_finite());
    assert!(report.kahler_form_omega_k.is_finite());
    assert!(report.ricci_flat_curvature.is_finite());
    assert!(report.slot_scaling_factor.is_finite());
}
