//! CONTRATO FORMAL: DESACOPLAMIENTO DINÁMICO DE UMBRALES ML Y FACTORES METACOGNITIVOS (Ola Ω117).
//!
//! # Por qué existe
//! Erradica la petrificación heurística de los asientos SeniorML y SeniorMetacognitivo.
//! Anteriormente:
//! 1. `SeniorML` utilizaba constantes fijas congeladas (`DEAD_ZONE = 0.05`, `FULL_EDGE = 0.20`),
//!    impidiendo que regímenes de alta eficiencia o micro-scalping estrecharan la zona muerta
//!    o que mercados volátiles exigieran mayor convicción.
//! 2. `SeniorMetacognitivo` utilizaba factores congelados en código (`dd_penalty_mult = 5.0`,
//!    `max_dd_penalty = 0.50`, `majority_factor = 0.70`, `solitary_factor = 0.40`), haciendo imposible
//!    modular la aversión al drawdown o la disciplina de consenso según el estado del portafolio o el régimen.
//!
//! # Solución (Ola Ω117)
//! 1. Parámetros desacoplados expuestos en `MarketSnapshotPayload` y `CouncilParams`.
//! 2. Funciones puras `evaluate_ml_opinion_with_bounds` y `evaluate_metacognitive_opinion`
//!    con sanitización estricta fail-closed (garantía de no pánico ante NaN / Inf / cotas invertidas).
//! 3. Propagación `Cow` sin asignación de heap en la ruta caliente cuando los parámetros del consejo difieren de los canónicos.

use metacortex_engine::consejo_seniors::{
    evaluate_metacognitive_opinion, evaluate_ml_opinion_with_bounds, ConsejoDeliberacion,
    CouncilParams, MarketSnapshotPayload, SeniorRole, DEFAULT_META_DD_PENALTY_MULT,
    DEFAULT_META_MAJORITY_FACTOR, DEFAULT_META_MAX_DD_PENALTY, DEFAULT_META_SOLITARY_FACTOR,
    DEFAULT_ML_DEAD_ZONE, DEFAULT_ML_FULL_EDGE,
};

#[test]
fn test_ml_opinion_bounds_canonical_parity() {
    // 1. Zona muerta canónica [-0.05, 0.05] sobre base 0.50 -> prob in [0.45, 0.55]
    let (sig_neutral, conf_neutral) = evaluate_ml_opinion_with_bounds(
        0.52,
        0.50,
        DEFAULT_ML_DEAD_ZONE,
        DEFAULT_ML_FULL_EDGE,
    );
    assert_eq!(sig_neutral, 0.0, "Dentro de dead_zone 0.05 la señal debe ser 0.0");
    assert_eq!(conf_neutral, 0.0);

    // 2. Exactamente en borde de dead zone (edge = 0.05)
    let (sig_edge, _) = evaluate_ml_opinion_with_bounds(
        0.55,
        0.50,
        DEFAULT_ML_DEAD_ZONE,
        DEFAULT_ML_FULL_EDGE,
    );
    assert!(sig_edge.abs() < 1e-9, "En el límite de dead_zone debe ser 0.0");

    // 3. Saturación plena en full_edge (edge = 0.20 -> prob = 0.70)
    let (sig_full_bull, conf_full_bull) = evaluate_ml_opinion_with_bounds(
        0.70,
        0.50,
        DEFAULT_ML_DEAD_ZONE,
        DEFAULT_ML_FULL_EDGE,
    );
    assert!((sig_full_bull - 1.0).abs() < 1e-9, "Edge 0.20 debe saturar señal en +1.0");
    assert!((conf_full_bull - 1.0).abs() < 1e-9);

    // 4. Saturación plena bajista (edge = -0.20 -> prob = 0.30)
    let (sig_full_bear, conf_full_bear) = evaluate_ml_opinion_with_bounds(
        0.30,
        0.50,
        DEFAULT_ML_DEAD_ZONE,
        DEFAULT_ML_FULL_EDGE,
    );
    assert!((sig_full_bear - (-1.0)).abs() < 1e-9, "Edge -0.20 debe saturar señal en -1.0");
    assert!((conf_full_bear - 1.0).abs() < 1e-9);

    // 5. Escalamiento lineal intermedio (edge = 0.125 -> (0.125 - 0.05) / (0.20 - 0.05) = 0.075 / 0.15 = 0.50)
    let (sig_half, conf_half) = evaluate_ml_opinion_with_bounds(
        0.625,
        0.50,
        DEFAULT_ML_DEAD_ZONE,
        DEFAULT_ML_FULL_EDGE,
    );
    assert!((sig_half - 0.50).abs() < 1e-6);
    assert!((conf_half - 0.50).abs() < 1e-6);
}

#[test]
fn test_ml_opinion_bounds_dynamic_overrides() {
    // Zona muerta ultra-estrecha para micro-scalping (dead_zone = 0.01, full_edge = 0.05)
    let custom_dz = 0.01;
    let custom_fe = 0.05;

    // Edge de 0.02 ahora es VÁLIDO y activo (fuera de la nueva dead_zone 0.01)
    let (sig_active, conf_active) = evaluate_ml_opinion_with_bounds(
        0.52,
        0.50,
        custom_dz,
        custom_fe,
    );
    // (0.02 - 0.01) / (0.05 - 0.01) = 0.01 / 0.04 = 0.25
    assert!((sig_active - 0.25).abs() < 1e-6);
    assert!((conf_active - 0.25).abs() < 1e-6);

    // Con los defaults, 0.52 era 0.0
    let (sig_def, _) = evaluate_ml_opinion_with_bounds(
        0.52,
        0.50,
        DEFAULT_ML_DEAD_ZONE,
        DEFAULT_ML_FULL_EDGE,
    );
    assert_eq!(sig_def, 0.0);
}

#[test]
fn test_metacognitive_opinion_drawdown_penalty_and_consensus_scaling() {
    // Escenario: Perspectivas alineadas (ML, Espectral, Flujo alcistas)
    // ml_prob = 0.70, ml_base = 0.50 -> ml_sig = 1.0
    // fused_score = 0.50, persistence = 0.60 -> spec_sig = 0.50
    // book_imbalance = 0.50 -> flow_sig = 0.50
    // Unanimidad: (1.0 + 0.50 + 0.50) / 3 = 0.6667

    // 1. Sin drawdown
    let expected_spec = (0.50 * (0.60 * 2.0_f64).tanh()).clamp(-1.0, 1.0);
    let expected_unanimous_mean = (1.0 + expected_spec + 0.50) / 3.0;

    let (sig_clean, conf_clean, _, _, _) = evaluate_metacognitive_opinion(
        0.70, 0.50, 0.50, 0.60, 0.50,
        0.0, 0.65,
        DEFAULT_ML_DEAD_ZONE, DEFAULT_ML_FULL_EDGE,
        DEFAULT_META_DD_PENALTY_MULT, DEFAULT_META_MAX_DD_PENALTY,
        DEFAULT_META_MAJORITY_FACTOR, DEFAULT_META_SOLITARY_FACTOR,
    );
    assert!((sig_clean - expected_unanimous_mean).abs() < 1e-4);
    assert!((conf_clean - 0.65).abs() < 1e-4);

    // 2. Con drawdown 0.06 y multiplicador canónico (5.0 -> penalización = 0.30)
    let (_, conf_dd_def, _, _, _) = evaluate_metacognitive_opinion(
        0.70, 0.50, 0.50, 0.60, 0.50,
        0.06, 0.65,
        DEFAULT_ML_DEAD_ZONE, DEFAULT_ML_FULL_EDGE,
        DEFAULT_META_DD_PENALTY_MULT, DEFAULT_META_MAX_DD_PENALTY,
        DEFAULT_META_MAJORITY_FACTOR, DEFAULT_META_SOLITARY_FACTOR,
    );
    assert!((conf_dd_def - (0.65 - 0.30)).abs() < 1e-4);

    // 3. Con multiplicador dinámico más agresivo (mult = 8.0 -> penalización 0.48)
    let (_, conf_dd_aggr, _, _, _) = evaluate_metacognitive_opinion(
        0.70, 0.50, 0.50, 0.60, 0.50,
        0.06, 0.65,
        DEFAULT_ML_DEAD_ZONE, DEFAULT_ML_FULL_EDGE,
        8.0, 0.50,
        DEFAULT_META_MAJORITY_FACTOR, DEFAULT_META_SOLITARY_FACTOR,
    );
    assert!((conf_dd_aggr - (0.65 - 0.48)).abs() < 1e-4);

    // 4. Mayoría simple (2 a favor, 1 neutro/cero) con majority_factor dinámico
    // ML = 1.0, Espectral = expected_spec > 0, Flujo = 0.0
    let expected_maj_mean = (1.0 + expected_spec + 0.0) / 3.0;

    let (sig_maj_def, _, _, div_def, _) = evaluate_metacognitive_opinion(
        0.70, 0.50, 0.50, 0.60, 0.0,
        0.0, 0.60,
        DEFAULT_ML_DEAD_ZONE, DEFAULT_ML_FULL_EDGE,
        DEFAULT_META_DD_PENALTY_MULT, DEFAULT_META_MAX_DD_PENALTY,
        0.70, DEFAULT_META_SOLITARY_FACTOR,
    );
    assert!(!div_def);
    assert!((sig_maj_def - expected_maj_mean * 0.70).abs() < 1e-4);

    let (sig_maj_dyn, _, _, _, _) = evaluate_metacognitive_opinion(
        0.70, 0.50, 0.50, 0.60, 0.0,
        0.0, 0.60,
        DEFAULT_ML_DEAD_ZONE, DEFAULT_ML_FULL_EDGE,
        DEFAULT_META_DD_PENALTY_MULT, DEFAULT_META_MAX_DD_PENALTY,
        0.85, DEFAULT_META_SOLITARY_FACTOR,
    );
    assert!((sig_maj_dyn - expected_maj_mean * 0.85).abs() < 1e-4);
}

#[test]
fn test_ml_metacognitive_opinion_fail_closed_sanitization() {
    // 1. ML con NaN y cotas invertidas
    let (sig_nan, conf_nan) = evaluate_ml_opinion_with_bounds(
        f64::NAN,
        f64::INFINITY,
        -0.05, // Negativo
        0.02,  // Menor que dead_zone corregido
    );
    assert!(sig_nan.is_finite());
    assert!(conf_nan.is_finite());
    assert_eq!(sig_nan, 0.0);
    assert_eq!(conf_nan, 0.0);

    // 2. Metacognitivo con NaN en todo
    let (sig_meta, conf_meta, weight_meta, _, _) = evaluate_metacognitive_opinion(
        f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
        f64::NAN, f64::NAN,
        f64::NAN, f64::NAN,
        f64::NAN, f64::NAN,
        f64::NAN, f64::NAN,
    );
    assert!(sig_meta.is_finite());
    assert!(conf_meta.is_finite());
    assert!(weight_meta.is_finite());
}

#[test]
fn test_deliberar_traced_cow_propagation() {
    let mut params = CouncilParams::default();
    params.ml_dead_zone = 0.02;
    params.ml_full_edge = 0.10;
    params.meta_dd_penalty_mult = 8.0;
    params.meta_max_dd_penalty = 0.40;
    params.meta_majority_factor = 0.85;
    params.meta_solitary_factor = 0.50;

    let consejo = ConsejoDeliberacion::with_params(params).expect("parámetros válidos");

    let payload = MarketSnapshotPayload {
        ml_prob: 0.54, // Edge 0.04 sobre base 0.50. Con dead_zone canónico 0.05 -> 0.0; con 0.02 -> activo
        ml_model_base: 0.50,
        intended_direction: 1.0,
        fused_score: 0.40,
        persistence: 0.50,
        book_imbalance: 0.40,
        ..Default::default()
    };

    let trace = consejo.deliberar_traced(&payload, 0.60, None);
    assert!(!trace.opinions.is_empty(), "Deliberación debe producir opiniones sin error de integridad");

    // Verificar que la opinión de SeniorML fue activa debido a la propagación Cow
    let ml_op = trace
        .opinions
        .iter()
        .find(|o| o.role == SeniorRole::Ml)
        .expect("debe existir opinión ML");
    assert!(
        ml_op.signal_direction > 0.0,
        "ML debe ser activo con dead_zone adaptado a 0.02: signal={}",
        ml_op.signal_direction
    );
}
