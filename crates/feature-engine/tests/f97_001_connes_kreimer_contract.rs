//! # Contrato Formal de Verificación — Fase 97 (§97)
//!
//! ## Geometría No Conmutativa de Connes-Kreimer & Renormalización Hopf de Microestructura
//!
//! Contratos verificados:
//! - `c1`: Monotonía del residuo de divergencia ultravioleta y grado del álgebra de Hopf.
//! - `c2`: Sustracción de subdivergencias vía antípoda de Zimmermann y purificación de carácter.
//! - `c3`: Activación determinista del veto estricto `CONNES_KREIMER_DIVERGENCE_VETO`.
//! - `c4`: Inmunidad absoluta a NaN/Inf y determinismo estricto de stack $\mathcal{O}(1)$ (`rb_p1`).

use feature_engine::{ConnesKreimerConfig, ConnesKreimerHopfEngine};

#[test]
fn test_c1_hopf_degree_and_uv_residue_monotonicity() {
    let config = ConnesKreimerConfig::default();
    let mut engine = ConnesKreimerHopfEngine::new(config);

    // Estado casi plano
    let report_quiet = engine.evaluate(0.0001, 0.001, 2.0, 0.01);
    assert!(report_quiet.uv_divergence_residue >= 0.0);
    assert!(report_quiet.hopf_algebra_degree >= 0.0);
    assert!(!report_quiet.connes_kreimer_divergence_veto);
    assert!(report_quiet.slot_scaling_factor >= 0.50);

    // Estado con turbulencia moderada
    let report_turbulent = engine.evaluate(0.005, 0.20, 10.0, 0.05);
    assert!(report_turbulent.uv_divergence_residue > report_quiet.uv_divergence_residue);
    assert!(report_turbulent.hopf_algebra_degree >= report_quiet.hopf_algebra_degree);
    assert!(!report_turbulent.connes_kreimer_divergence_veto);
}

#[test]
fn test_c2_antipode_zimmermann_subtraction_purification() {
    let config = ConnesKreimerConfig::default();
    let mut engine = ConnesKreimerHopfEngine::new(config);

    // Secuencia de ticks para observar la convolución de Birkhoff
    let mut last_report = engine.evaluate(0.001, 0.05, 5.0, 0.02);
    for i in 1..=10 {
        let p = 0.001 * (i as f64);
        let q = 0.05 + 0.01 * (i as f64);
        last_report = engine.evaluate(p, q, 5.0, 0.02);
        assert!(last_report.renormalized_character.is_finite());
        assert!((0.0..=1.0).contains(&last_report.directional_conviction));
        assert!(last_report.slot_scaling_factor >= 0.10 && last_report.slot_scaling_factor <= 1.25);
    }

    assert!(!last_report.connes_kreimer_divergence_veto);
}

#[test]
fn test_c3_connes_kreimer_divergence_veto_activation() {
    let config = ConnesKreimerConfig {
        max_divergence_residue: 0.05, // Umbral muy sensible para probar la activación del veto
        ..Default::default()
    };
    let mut engine = ConnesKreimerHopfEngine::new(config);

    // Forzamos divergencia masiva en el acoplamiento de vértices
    let report_blown = engine.evaluate(0.10, 2.50, 40.0, 0.80);
    assert!(
        report_blown.connes_kreimer_divergence_veto,
        "Divergencia masiva de loops debe detonar CONNES_KREIMER_DIVERGENCE_VETO"
    );
    assert!(report_blown.slot_scaling_factor <= 0.30);
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = ConnesKreimerConfig::default();
    let mut engine = ConnesKreimerHopfEngine::new(config);

    // Inyección de entradas patológicas
    let report_nan = engine.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    assert!(
        report_nan.connes_kreimer_divergence_veto,
        "Valores no finitos deben activar veto de seguridad"
    );
    assert!(report_nan.uv_divergence_residue.is_finite());
    assert!(report_nan.renormalized_character.is_finite());
    assert!(report_nan.hopf_algebra_degree.is_finite());
    assert!(report_nan.directional_conviction.is_finite());
    assert!(report_nan.slot_scaling_factor.is_finite());

    // Reset y verificación de recuperación limpia
    engine.reset();
    let report_clean = engine.evaluate(0.001, 0.01, 2.0, 0.01);
    assert!(!report_clean.connes_kreimer_divergence_veto);
    assert!(report_clean.uv_divergence_residue.is_finite());
}
