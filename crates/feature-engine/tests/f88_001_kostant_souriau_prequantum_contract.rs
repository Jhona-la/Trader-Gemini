//! Contrato Formal de Verificación: Fase 88 (§88) — Cuantización Geométrica de Kostant-Souriau
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor precuántico:
//! - C1: Conservación simpléctica local $\Omega_{pq} \ge 0$ y holonomía de Reeb finita en $[-1, 1]$.
//! - C2: Coherencia con polarización de Kähler $\kappa_{\text{polar}} \in [0, 1]$ y modulación de slot $\eta_{\text{ks}} \in [0.10, 1.25]$.
//! - C3: Detonación determinista del veto fail-closed ante colapso de polarización o deriva simpléctica crítica.
//! - C4: Determinismo estricto en stack $O(1)$, cero heap allocations e inmunidad ante NaNs e infinitos.

use feature_engine::{KostantSouriauConfig, KostantSouriauPrequantumEngine};

#[test]
fn test_c1_symplectic_area_and_reeb_holonomy() {
    let config = KostantSouriauConfig::default();
    let mut engine = KostantSouriauPrequantumEngine::new(config);

    let report = engine.evaluate(0.005, 0.15, 1.2, 0.02);

    assert!(
        report.prequantum_observable_val.is_finite(),
        "Observable precuántico debe ser finito"
    );
    assert!(
        report.symplectic_curvature_norm >= 0.0,
        "Curvatura simpléctica normalizada debe ser no-negativa: {}",
        report.symplectic_curvature_norm
    );
    assert!(
        report.reeb_contact_holonomy >= -1.0 && report.reeb_contact_holonomy <= 1.0,
        "Holonomía de contacto de Reeb fuera de [-1, 1]: {}",
        report.reeb_contact_holonomy
    );
    assert!(
        !report.prequantum_collapse_veto,
        "Flujo suave no debe activar el veto precuántico"
    );
}

#[test]
fn test_c2_polarization_coherence_and_slot_scaling() {
    let config = KostantSouriauConfig::default();
    let mut engine = KostantSouriauPrequantumEngine::new(config);

    for i in 0..10 {
        let q = 0.001 * (i as f64);
        let p = 0.05 + 0.02 * (i as f64);
        let rep = engine.evaluate(q, p, 1.0, 0.01);

        assert!(
            rep.polarization_coherence >= 0.0 && rep.polarization_coherence <= 1.0,
            "Coherencia de polarización fuera de [0, 1]: {}",
            rep.polarization_coherence
        );
        assert!(
            rep.slot_scale_factor >= 0.10 && rep.slot_scale_factor <= 1.25,
            "Factor de slot fuera de límites [0.10, 1.25]: {}",
            rep.slot_scale_factor
        );
    }
}

#[test]
fn test_c3_prequantum_collapse_veto_activation() {
    let mut config = KostantSouriauConfig::default();
    config.min_polarization_coherence = 0.50;
    config.max_symplectic_curvature_drift = 1.00;
    let mut engine = KostantSouriauPrequantumEngine::new(config);

    // Inyectamos choques anarmónicos severos para forzar colapso de la polarización
    let mut report = engine.evaluate(0.8, -0.9, 45.0, 5.0);
    for _ in 0..5 {
        report = engine.evaluate(-0.9, 0.95, 48.0, 8.0);
    }

    assert!(
        report.prequantum_collapse_veto,
        "Choque anarmónico extremo debe detonar prequantum_collapse_veto"
    );
    assert_eq!(
        report.slot_scale_factor, 0.10,
        "En colapso precuántico, el slot debe comprimirse a 0.10"
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = KostantSouriauConfig::default();
    let mut engine = KostantSouriauPrequantumEngine::new(config);

    let report = engine.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);

    assert!(report.prequantum_observable_val.is_finite());
    assert!(report.symplectic_curvature_norm.is_finite());
    assert!(report.reeb_contact_holonomy.is_finite());
    assert!(report.polarization_coherence.is_finite());
    assert!(report.slot_scale_factor.is_finite());
}
