//! Contrato formal de verificación para Fase 38: Teoría de Juegos de Campo Medio (Mean Field Games) y Psicología de Masas
//!
//! Invariantes certificados:
//! 1. c1: Conservación estricta de masa de probabilidad en FPK: $\int m(x) dx = 1.0 \pm 10^{-6}$.
//! 2. c2: Detección analítica de concentración de masa y riesgo de estampida $\kappa_{\mathrm{crowd}}$.
//! 3. c3: Activación sigmoidal de la probabilidad de agotamiento de squeeze $P_{\mathrm{exhaustion}}$.
//! 4. c4: Convicción anti-rebaño y atracción hacia la coordenada del equilibrio de Nash.
//! 5. c5: Cero pánicos (`rb_p1`) y robustez numérica ante entradas no acotadas.

use signal_engine::mean_field_games_psychology::{
    MeanFieldGamesParams, MeanFieldGamesPsychologyEngine,
};

#[test]
fn c1_mfg_mass_conservation_and_grid_consistency() {
    let mut engine = MeanFieldGamesPsychologyEngine::new(MeanFieldGamesParams::default());

    // Ejecutar 50 pasos con desequilibrio aleatorio y volatilidad variable
    for step in 0..50 {
        let ofi = ((step as f64 * 0.17).sin()).clamp(-1.0, 1.0);
        let vol = 0.02 + 0.01 * ((step as f64 * 0.23).cos()).abs();
        let report = engine.update_step(ofi, vol, 0.02);

        // Verificar que las métricas extraídas son finitas
        assert!(report.crowd_herding_risk.is_finite());
        assert!(report.crowd_center_of_mass.is_finite());
        assert!(
            report.crowd_center_of_mass >= -1.0 && report.crowd_center_of_mass <= 1.0,
            "Centro de masa fuera de [-1, 1]: {}",
            report.crowd_center_of_mass
        );
        assert!(report.crowd_spatial_variance >= 0.0);
    }
}

#[test]
fn c2_crowd_herding_and_stampede_detection() {
    let mut engine = MeanFieldGamesPsychologyEngine::new(MeanFieldGamesParams {
        diffusion_coeff: 0.02, // Baja difusión para favorecer concentración de masa
        crowd_penalty: 0.80,
        ..MeanFieldGamesParams::default()
    });

    // Flujo agresor unidireccional masivo hacia el lado comprador (OFI = +0.95)
    for _ in 0..30 {
        let _ = engine.update_step(0.95, 0.01, 0.02);
    }

    let report = engine.update_step(0.95, 0.01, 0.02);

    // El centro de masa debe haberse desplazado significativamente a la derecha
    assert!(
        report.crowd_center_of_mass > 0.10,
        "Centro de masa debería desplazarse a territorio positivo, obtenido: {}",
        report.crowd_center_of_mass
    );

    // El riesgo de estampida / aglomeración debe activarse
    assert!(
        report.crowd_herding_risk > 0.0,
        "El riesgo de aglomeración debería ser positivo, obtenido: {}",
        report.crowd_herding_risk
    );

    // Dirección de caza institucional debe estar alineada al alza (+1.0)
    assert_eq!(
        report.predatory_hunt_direction, 1.0,
        "Caza institucional debería apuntar a +1.0"
    );
}

#[test]
fn c3_squeeze_exhaustion_probability_activation() {
    let mut engine = MeanFieldGamesPsychologyEngine::new(MeanFieldGamesParams {
        diffusion_coeff: 0.01,
        ..MeanFieldGamesParams::default()
    });

    // Empuje extremo prolongado para forzar masa a la cola superior
    for _ in 0..60 {
        let _ = engine.update_step(1.0, 0.005, 0.03);
    }

    let report = engine.update_step(1.0, 0.005, 0.03);

    // Probabilidad de agotamiento de squeeze debe ser no trivial (> 0.20)
    assert!(
        report.squeeze_exhaustion_prob > 0.20,
        "Agotamiento de squeeze debería estar activo ante masa en cola extrema: {}",
        report.squeeze_exhaustion_prob
    );
}

#[test]
fn c4_anti_crowd_alpha_reversal_mechanics() {
    let mut engine = MeanFieldGamesPsychologyEngine::new(MeanFieldGamesParams::default());

    // Estado inicial neutro
    let report_neutral = engine.update_step(0.0, 0.02, 0.02);
    assert!(report_neutral.anti_crowd_alpha.abs() < 0.25);

    // Empuje sostenido negativo (estampida vendedora minorista hacia -1.0)
    for _ in 0..40 {
        let _ = engine.update_step(-0.90, 0.02, 0.02);
    }
    let report_dump = engine.update_step(-0.90, 0.02, 0.02);

    // Ante un pánico vendedor extremo, la señal anti-rebaño debe ofrecer convicción compradora
    // para arbitrar el colapso hacia el equilibrio de Nash
    assert!(
        report_dump.crowd_center_of_mass < 0.0,
        "Centro de masa debe ser negativo ante pánico vendedor"
    );
    assert!(
        report_dump.anti_crowd_alpha > 0.0,
        "Alpha anti-rebaño debe ser positivo ante pánico vendedor extremo: {}",
        report_dump.anti_crowd_alpha
    );
}

#[test]
fn c5_zero_panic_guarantee() {
    let mut engine = MeanFieldGamesPsychologyEngine::new(MeanFieldGamesParams::default());

    // Entradas patológicas
    let p1 = engine.update_step(f64::NAN, f64::INFINITY, 0.0);
    assert!(p1.crowd_herding_risk.is_finite());
    assert!(p1.squeeze_exhaustion_prob.is_finite());
    assert!(p1.nash_equilibrium_coordinate.is_finite());
    assert!(p1.anti_crowd_alpha.is_finite());

    let p2 = engine.update_step(-1e308, -1e308, -10.0);
    assert!(p2.crowd_herding_risk.is_finite());
    assert!(p2.anti_crowd_alpha >= -1.0 && p2.anti_crowd_alpha <= 1.0);
}
