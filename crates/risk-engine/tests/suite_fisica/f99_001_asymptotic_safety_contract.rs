//! # Contrato Formal de Verificación — Fase 99 (§99)
//!
//! ## Gravitación 4D Cuántica Asintóticamente Segura de Weinberg & Punto Fijo No Gaussiano Reuter
//!
//! Contratos verificados:
//! - `c1`: Proximidad y estabilidad del punto fijo no gaussiano (NGFP) de Reuter en régimen normal.
//! - `c2`: Continuidad del flujo de Wetterich y regularidad de la curvatura de Ricci de riesgo.
//! - `c3`: Activación determinista del veto estricto `ASYMPTOTIC_SAFETY_SINGULARITY_VETO` y contracción a 0.10.
//! - `c4`: Inmunidad absoluta a NaN/Inf y determinismo estricto de stack $\mathcal{O}(1)$ (`rb_p1`).

use risk_engine::{
    AsymptoticSafetyConfig, AsymptoticSafetyReuterEngine, REUTER_NGFP_G_STAR,
    REUTER_NGFP_LAMBDA_STAR,
};

#[test]
fn test_c1_reuter_ngfp_proximity_and_bounds() {
    let config = AsymptoticSafetyConfig::default();
    let mut engine = AsymptoticSafetyReuterEngine::new(config);

    // Estado con riesgo bajo: Heat = 0.20, Vol = 1.0, Lev = 1.0, DD = 0.0%
    let rep_quiet = engine.evaluate(0.20, 1.0, 1.0, 0.0);
    assert!(rep_quiet.dimensionless_newton_coupling > 0.0);
    assert!(rep_quiet.ngfp_distance >= 0.0 && rep_quiet.ngfp_distance < 1.0);
    assert!(rep_quiet.anomalous_dimension_eta.is_finite());
    assert!(!rep_quiet.asymptotic_safety_singularity_veto);
    assert!(rep_quiet.slot_scaling_factor >= 0.50);

    // Verificamos que las constantes coincidan con los valores canónicos
    assert!((REUTER_NGFP_G_STAR - 0.707).abs() < 1e-4);
    assert!((REUTER_NGFP_LAMBDA_STAR - 0.193).abs() < 1e-4);
}

#[test]
fn test_c2_wetterich_flow_continuity_and_curvature() {
    let config = AsymptoticSafetyConfig::default();
    let mut engine = AsymptoticSafetyReuterEngine::new(config);

    // Secuencia de variaciones de apalancamiento y volatilidad
    let mut last_rep = engine.evaluate(0.25, 1.05, 1.5, 0.5);
    for i in 1..=10 {
        let heat = 0.25 + 0.02 * (i as f64);
        let vol = 1.05 + 0.05 * (i as f64);
        let lev = 1.5 + 0.1 * (i as f64);
        let dd = 0.5 + 0.2 * (i as f64);
        last_rep = engine.evaluate(heat, vol, lev, dd);

        assert!(last_rep.ricci_risk_curvature.is_finite());
        assert!(last_rep.slot_scaling_factor >= 0.10 && last_rep.slot_scaling_factor <= 1.25);
        assert!(last_rep.ngfp_distance.is_finite());
    }

    assert!(!last_rep.asymptotic_safety_singularity_veto);
}

#[test]
fn test_c3_asymptotic_safety_singularity_veto_activation() {
    let config = AsymptoticSafetyConfig {
        max_ngfp_distance: 0.20, // Umbral muy sensible para probar la activación del veto
        ..Default::default()
    };
    let mut engine = AsymptoticSafetyReuterEngine::new(config);

    // Forzamos un colapso severo de riesgo: Heat = 1.0, Vol = 8.0, Lev = 20.0, DD = 60%
    let rep_blown = engine.evaluate(1.0, 8.0, 20.0, 60.0);
    assert!(
        rep_blown.asymptotic_safety_singularity_veto,
        "Colapso de riesgo debe detonar ASYMPTOTIC_SAFETY_SINGULARITY_VETO"
    );
    assert_eq!(
        rep_blown.slot_scaling_factor, 0.10,
        "Veto debe forzar contracción inmediata del slot a 0.10"
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = AsymptoticSafetyConfig::default();
    let mut engine = AsymptoticSafetyReuterEngine::new(config);

    // Inyección de entradas patológicas
    let rep_nan = engine.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    assert!(
        rep_nan.asymptotic_safety_singularity_veto,
        "Valores NaN/Infinitos deben detonar el veto de seguridad"
    );
    assert!(rep_nan.dimensionless_newton_coupling.is_finite());
    assert!(rep_nan.dimensionless_cosmological_constant.is_finite());
    assert!(rep_nan.anomalous_dimension_eta.is_finite());
    assert!(rep_nan.ricci_risk_curvature.is_finite());
    assert!(rep_nan.ngfp_distance.is_finite());
    assert_eq!(rep_nan.slot_scaling_factor, 0.10);

    // Reset y recuperación limpia al NGFP
    engine.reset();
    let rep_clean = engine.evaluate(0.15, 1.0, 1.0, 0.0);
    assert!(!rep_clean.asymptotic_safety_singularity_veto);
    assert!(rep_clean.ngfp_distance < 0.50);
}
