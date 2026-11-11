//! Contrato Formal de Certificación: Fase 112 (§112) — Teorema del Índice de Atiyah-Patodi-Singer & Invariante Eta
//!
//! Pruebas formales de validación matemática:
//! 1. `test_c1_nominal_equilibrium_and_a_roof_regularity`: Régimen estacionario con índice acotado y $\hat{A} \approx 1$.
//! 2. `test_c2_aps_index_divergence_veto`: Desborde del índice analítico $|\operatorname{ind}_{\text{APS}}| > 4.5$ con veto fail-closed.
//! 3. `test_c3_eta_invariant_boundary_veto`: Asimetría espectral de frontera $\eta(0) > 0.85$ detonando veto.
//! 4. `test_c4_curvature_a_roof_collapse_veto`: Singularidad de curvatura $p_1(TM)$ con colapso de género $\hat{A}$.
//! 5. `test_c5_nan_infinity_immunity_and_zero_panic_guarantee`: Inmunidad total contra NaN/$\pm\infty$ y garantía zero-panic.

use feature_engine::{AtiyahPatodiSingerConfig, AtiyahPatodiSingerIndexEngine};

#[test]
fn test_c1_nominal_equilibrium_and_a_roof_regularity() {
    let mut engine = AtiyahPatodiSingerIndexEngine::new(AtiyahPatodiSingerConfig::default());

    let mut report = engine.update(0.0001, 0.05, 1.5, 0.02);
    for i in 1..20 {
        let ret = if i % 2 == 0 { 0.0002 } else { -0.0001 };
        report = engine.update(ret, 0.02, 1.2, 0.015);
    }

    assert!(
        report.aps_index.abs() < 2.0,
        "El índice APS analítico debe permanecer acotado"
    );
    assert!(
        report.a_roof_genus >= 0.80 && report.a_roof_genus <= 1.20,
        "El género A-roof debe rondar la unidad en régimen nominal"
    );
    assert!(
        report.eta_invariant.abs() < 0.50,
        "El invariante eta de frontera debe ser bajo en equilibrio"
    );
    assert!(
        !report.atiyah_patodi_singer_collapse_veto,
        "No debe haber veto en mercado regular"
    );
    assert!(
        report.slot_scaling_factor >= 0.50 && report.slot_scaling_factor <= 1.25,
        "El factor de slot debe mantenerse saludable"
    );
}

#[test]
fn test_c2_aps_index_divergence_veto() {
    let mut engine = AtiyahPatodiSingerIndexEngine::new(AtiyahPatodiSingerConfig {
        max_index_divergence: 0.05,
        ..Default::default()
    });

    // Inyectamos retornos y desbalances consistentes para inflar la segunda clase de Chern c_2(E)
    let mut report = engine.update(0.50, 1.0, 50.0, 10.0);
    for _ in 0..25 {
        report = engine.update(0.50, 1.0, 50.0, 10.0);
    }

    assert!(
        report.atiyah_patodi_singer_collapse_veto,
        "La divergencia del índice APS debe detonar el veto fail-closed"
    );
    assert_eq!(
        report.slot_scaling_factor, 0.10,
        "El slot debe colapsar a 0.10 ante veto"
    );
}

#[test]
fn test_c3_eta_invariant_boundary_veto() {
    let mut engine = AtiyahPatodiSingerIndexEngine::new(AtiyahPatodiSingerConfig {
        max_eta_invariant_threshold: 0.40,
        ..Default::default()
    });

    // Imbalance agresor unilateral extremo persistente en la frontera
    let mut report = engine.update(0.02, 1.0, 5.0, 0.50);
    for _ in 0..15 {
        report = engine.update(0.03, 1.0, 5.0, 0.60);
    }

    assert!(
        report.eta_invariant.abs() > 0.40,
        "El invariante eta de frontera debe superar el umbral sensible"
    );
    assert!(
        report.atiyah_patodi_singer_collapse_veto,
        "La asimetría espectral excesiva debe activar el veto"
    );
    assert_eq!(report.slot_scaling_factor, 0.10);
}

#[test]
fn test_c4_curvature_a_roof_collapse_veto() {
    let mut engine = AtiyahPatodiSingerIndexEngine::new(AtiyahPatodiSingerConfig {
        pontryagin_scale: 1000.0,
        ..Default::default()
    });

    // Salto discontinuo severo de volatilidad y spread que infla el conmutador de curvatura p_1
    engine.update(0.0, 0.0, 1.0, 0.01);
    let report = engine.update(0.80, -0.90, 500.0, 50.0);

    assert!(report.pontryagin_number_p1.abs() > 0.0);
    assert!(
        report.atiyah_patodi_singer_collapse_veto,
        "Distorsión severa de curvatura debe forzar veto de índice"
    );
    assert_eq!(report.slot_scaling_factor, 0.10);
}

#[test]
fn test_c5_nan_infinity_immunity_and_zero_panic_guarantee() {
    let mut engine = AtiyahPatodiSingerIndexEngine::new(AtiyahPatodiSingerConfig::default());

    let report_nan = engine.update(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(report_nan.aps_index.is_finite());
    assert!(report_nan.a_roof_genus.is_finite());
    assert!(report_nan.eta_invariant.is_finite());
    assert!(report_nan.slot_scaling_factor >= 0.10 && report_nan.slot_scaling_factor <= 1.25);

    let report_inf = engine.update(
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    );
    assert!(report_inf.aps_index.is_finite());
    assert!(report_inf.a_roof_genus.is_finite());
    assert!(report_inf.slot_scaling_factor >= 0.10);
}
