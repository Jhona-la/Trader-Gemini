//! # Contrato de Pruebas: Fase 104 (§104) — Entrelazamiento Cuántico y Fractones X-Cube
//!
//! Verifica las invariantes matemáticas de la entropía de Levin-Wen/Kitaev-Preskill,
//! la dinámica de excitaciones subdimensionales y la resiliencia fail-closed bajo el estándar RB-P1.

use signal_engine::fracton_topological_order::{
    FractonTopologicalConfig, FractonTopologicalOrderEngine, FRACTON_BUFFER_CAPACITY, FRACTON_SITES,
};

#[test]
fn test_c1_fracton_initialization_and_purity() {
    let mut engine = FractonTopologicalOrderEngine::new(FractonTopologicalConfig::default());
    assert_eq!(FRACTON_SITES, 8);
    assert_eq!(FRACTON_BUFFER_CAPACITY, 16);

    let report = engine.update(0.0001, 0.05, 1.5, 0.02);
    assert!(report.topological_entanglement_entropy >= 0.0);
    assert!(report.cube_operator_expectation.abs() <= 1.0);
    assert!(report.star_operator_expectation.abs() <= 1.0);
    assert!(report.subdimensional_dipole_density >= 0.0);
    assert!(report.subdimensional_dipole_density <= 1.0);
    assert!(report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.25);
}

#[test]
fn test_c2_cube_and_star_operator_algebra() {
    let mut engine = FractonTopologicalOrderEngine::default();
    for i in 0..20 {
        let ret = (i as f64 * 0.0001) - 0.001;
        let ofi = (i as f64 * 0.05).sin() * 0.30;
        let rep = engine.update(ret, ofi, 2.0, 0.10);
        assert!(rep.cube_operator_expectation >= -1.0 && rep.cube_operator_expectation <= 1.0);
        assert!(rep.star_operator_expectation >= -1.0 && rep.star_operator_expectation <= 1.0);
        assert!(rep.subdimensional_dipole_density >= 0.0);
    }
}

#[test]
fn test_c3_topological_entanglement_entropy_extraction() {
    let mut engine = FractonTopologicalOrderEngine::default();
    // Microestructura coherente produce entropía de entrelazamiento topológico positiva
    let mut reports = Vec::with_capacity(10);
    for _ in 0..10 {
        reports.push(engine.update(0.0002, 0.15, 1.2, 0.08));
    }
    let last = reports.last().unwrap();
    assert!(last.topological_entanglement_entropy.is_finite());
    assert!(last.topological_entanglement_entropy >= 0.0);
}

#[test]
fn test_c4_fracton_collapse_veto_activation() {
    // Configuración con umbral estricto para forzar colapso
    let strict_config = FractonTopologicalConfig {
        topological_entropy_threshold: 1.50, // Alto para inducir veto
        fracton_mobility_threshold: 0.10,    // Bajo para inducir veto por exceso de dipolos
        subdimensional_scale: 0.10,
    };
    let mut engine = FractonTopologicalOrderEngine::new(strict_config);
    let report = engine.update(0.05, 0.90, 50.0, -0.80);
    assert!(report.fracton_collapse_veto);
    assert!((report.slot_scaling_factor - 0.10).abs() < 1e-6);
}

#[test]
fn test_c5_fracton_rb_p1_resilience() {
    let mut engine = FractonTopologicalOrderEngine::default();
    // Entrada patológica con NaN e Infinito
    let rep_nan = engine.update(f64::NAN, f64::INFINITY, -100.0, f64::NAN);
    assert!(rep_nan.topological_entanglement_entropy.is_finite());
    assert!(rep_nan.cube_operator_expectation.is_finite());
    assert!(rep_nan.star_operator_expectation.is_finite());
    assert!(rep_nan.subdimensional_dipole_density.is_finite());
    assert!(rep_nan.slot_scaling_factor >= 0.10);

    // Entrada con subdesbordamiento extremo
    let rep_neg = engine.update(-1e300, 1e300, 0.0, -1e300);
    assert!(rep_neg.topological_entanglement_entropy.is_finite());
    assert!(rep_neg.slot_scaling_factor <= 1.25);
}
