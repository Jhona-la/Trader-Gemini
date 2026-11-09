//! # Contrato Formal F74-001: Dualidad AdS/CFT Holográfica de Maldacena y Tensión de Cuerda Cuántica
//!
//! Valida la correspondencia AdS/CFT, dimensión conforme $\Delta \ge 1$, acoplamiento de 't Hooft $\lambda$,
//! tensión de cuerda cuántica $T_{\text{string}}$, alerta de pantalla holográfica RG y resiliencia `rb_p1`.

use signal_engine::ads_cft_holographic::{AdSCftHolographicEngine, AdSCftParams};

#[test]
fn test_c1_maldacena_conformal_dimension_and_string_tension() {
    let params = AdSCftParams::default();
    let mut engine = AdSCftHolographicEngine::new(params);

    let rep = engine.update(0.002, 0.015, 0.58, 1.5);

    // Dimensión conforme de operador primario \Delta >= 1.0 (cota unitaria de CFT)
    assert!(
        rep.conformal_dimension_delta >= 1.0,
        "La dimensión conforme del operador primario debe ser >= 1.0 (obtenido: {})",
        rep.conformal_dimension_delta
    );

    // Acoplamiento de 't Hooft \lambda >= 0.0
    assert!(
        rep.thooft_coupling_lambda >= 0.0,
        "El acoplamiento de 't Hooft debe ser no negativo"
    );

    // Tensión de cuerda fundamental T_string > 0.0
    assert!(
        rep.string_tension > 0.0,
        "La tensión de cuerda cuántica debe ser estrictamente positiva (obtenido: {})",
        rep.string_tension
    );

    // Profundidad radial z en el bulto de Poincaré dentro de rango
    assert!(
        rep.radial_depth_z >= 0.01 && rep.radial_depth_z <= 10.0,
        "La coordenada radial en el bulto debe estar acotada físicamente"
    );
}

#[test]
fn test_c2_calm_persistent_regime_and_constructive_conviction() {
    let params = AdSCftParams::default();
    let mut engine = AdSCftHolographicEngine::new(params);

    // Régimen altamente persistente y ordenado: H = 0.70, spread ultra-ajustado (0.5 bps), baja volatilidad
    let rep = engine.update(0.001, 0.008, 0.70, 0.5);

    assert!(
        !rep.holographic_rg_screen_alert,
        "No debe haber alerta de pantalla holográfica en régimen persistente clásico"
    );
    assert!(
        rep.holographic_conviction_modulator >= 0.90,
        "La convicción holográfica debe ser constructiva (obtenido: {})",
        rep.holographic_conviction_modulator
    );
    assert!(
        rep.thooft_coupling_lambda >= params.min_thooft_coupling_threshold,
        "El acoplamiento de 't Hooft debe superar el umbral de supergravedad clásica"
    );
}

#[test]
fn test_c3_turbulent_decoherence_triggers_rg_screen_alert() {
    let params = AdSCftParams::default();
    let mut engine = AdSCftHolographicEngine::new(params);

    // Colapso de acoplamiento: spread descomunal (50.0 bps) y antipersistencia / volatilidad ruidosa (H = 0.10)
    let rep = engine.update(-0.05, 0.35, 0.10, 50.0);

    assert!(
        rep.holographic_rg_screen_alert,
        "Debe activarse la alerta de pantalla holográfica RG ante dispersión extrema"
    );
    assert_eq!(
        rep.holographic_conviction_modulator, 0.50,
        "La convicción debe deprimirse rígidamente al mínimo defensivo (0.50)"
    );
}

#[test]
fn test_c4_zero_panic_rb_p1_resilience() {
    let params = AdSCftParams::default();
    let mut engine = AdSCftHolographicEngine::new(params);

    let pathological_inputs = [
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::INFINITY),
        (1e-308, -1e-308, 0.0, -50.0),
        (-0.5, 1.0, 500.0, 200.0),
    ];

    for (ret, vol, h, spread) in pathological_inputs {
        let rep = engine.update(ret, vol, h, spread);
        assert!(rep.conformal_dimension_delta.is_finite());
        assert!(rep.thooft_coupling_lambda.is_finite());
        assert!(rep.string_tension.is_finite());
        assert!(rep.radial_depth_z.is_finite());
        assert!(rep.holographic_conviction_modulator.is_finite());
    }
}
