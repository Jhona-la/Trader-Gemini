//! CONTRATO FORMAL: DESACOPLAMIENTO DINÁMICO Y CONTINUO DE TOLERANCIAS CAUSALES (Ola Ω115).
//!
//! # Por qué existe
//! En `SeniorCausal` (`metacortex-engine`), las reglas de permiso/veto por manipulación Pearl do-calculus
//! y la excepción de breakout institucional alineado contenían literales fijos congelados en código:
//! - `causal_veto_threshold.clamp(0.60, 0.90)` (piso arbitrario de 0.60 y techo de 0.90).
//! - `book_imbalance.abs() > 0.25` (desequilibrio de libro rígido del 25%).
//! - `do_calculus_risk < 0.88` (techo de toxicidad máximo en breakout congelado al 88%).
//!
//! En un universo multivariante continuo con micro-capital ($13 USD) y activos con microestructuras dispares
//! (e.g. BTC vs altcoins de alta beta), estas cotas rígidas generaban:
//! 1. Veto espurio en activos líquidos donde un desequilibrio de 20% ya es una absorción institucional masiva.
//! 2. Entrada vulnerable a selección adversa en activos de baja liquidez donde un desequilibrio de 25% es ruido normal.
//! 3. Imposibilidad de calibración dinámica desde `CouncilParams` o por activo desde `arena.registry`.
//!
//! # Solución (Ola Ω115)
//! - Función pura `evaluate_causal_manipulation_veto` con sanitización estricta fail-closed.
//! - Campos dinámicos `causal_min_book_imbalance` y `causal_aligned_max_risk` en `MarketSnapshotPayload`.
//! - Campos correspondientes en `CouncilParams` con builder `with_causal_bounds`.
//! - Propagación lock-free `Cow` en `deliberar_traced`.
//! - Sincronización atómica en `QuantumConfig` (`council_causal_veto_threshold`, etc.).
//! - Resolución omnisciente por activo en `council_snapshot` (`c{id}:causal_veto_threshold`, etc.).

use metacortex_engine::consejo_seniors::{CouncilParams, SeniorAgent, SeniorCausal};
use metacortex_engine::{
    evaluate_causal_manipulation_veto, ConsejoDeliberacion, MarketSnapshotPayload, SeniorRole,
    DEFAULT_CAUSAL_ALIGNED_MAX_RISK, DEFAULT_CAUSAL_MIN_BOOK_IMBALANCE,
    DEFAULT_CAUSAL_VETO_THRESHOLD,
};

fn base_causal_payload() -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        intended_direction: 1.0,
        book_imbalance: 0.50,
        do_calculus_risk: 0.20,
        dominant_tau_ms: 30_000.0, // Fast scalping (s = 0.0 < 0.5)
        ..Default::default()
    }
}

#[test]
fn test_causal_opinion_canonical_defaults_exact_parity() {
    let mut payload = base_causal_payload();
    payload.do_calculus_risk = 0.80; // Mayor que threshold 0.75
    payload.book_imbalance = 0.50; // Mayor que min_book_imbalance 0.25, alineado con dir 1.0
    payload.dominant_tau_ms = 30_000.0; // s = 0.0 < 0.5

    // Caso 1: Breakout alineado con riesgo 0.80 < 0.88 -> NO debe vetar
    let op = SeniorCausal.evaluate(&payload, 0.55);
    assert_eq!(op.role, SeniorRole::Causal);
    assert!(!op.is_veto, "Breakout alineado institucional no debe ser vetado con riesgo 0.80");

    let (is_veto_pure, safe_risk, safe_thresh, is_aligned) = evaluate_causal_manipulation_veto(
        payload.do_calculus_risk,
        payload.causal_veto_threshold,
        payload.book_imbalance,
        payload.intended_direction,
        payload.spectral_s(),
        payload.causal_min_book_imbalance,
        payload.causal_aligned_max_risk,
    );
    assert_eq!(op.is_veto, is_veto_pure);
    assert!((op.confidence - (1.0 - safe_risk)).abs() < 1e-12);
    assert_eq!(safe_thresh, DEFAULT_CAUSAL_VETO_THRESHOLD);
    assert_eq!(payload.causal_min_book_imbalance, DEFAULT_CAUSAL_MIN_BOOK_IMBALANCE);
    assert_eq!(payload.causal_aligned_max_risk, DEFAULT_CAUSAL_ALIGNED_MAX_RISK);
    assert!(is_aligned);

    // Caso 2: Sin alineación de libro (desequilibrio en contra) -> DEBE vetar
    payload.book_imbalance = -0.50;
    let op_contrary = SeniorCausal.evaluate(&payload, 0.55);
    assert!(op_contrary.is_veto, "Libro en contra con riesgo 0.80 debe vetar");

    // Caso 3: Breakout alineado pero con toxicidad extrema > 0.88 (ej. 0.92) -> DEBE vetar
    payload.book_imbalance = 0.50;
    payload.do_calculus_risk = 0.92;
    let op_toxic = SeniorCausal.evaluate(&payload, 0.55);
    assert!(op_toxic.is_veto, "Toxicidad extrema > 0.88 debe vetar incluso con libro alineado");
}

#[test]
fn test_causal_opinion_dynamic_payload_override() {
    let mut payload = base_causal_payload();
    payload.do_calculus_risk = 0.70;
    payload.book_imbalance = 0.30;
    payload.intended_direction = 1.0;

    // Con defaults: threshold 0.75 -> riesgo 0.70 < 0.75 -> no veta
    assert!(!SeniorCausal.evaluate(&payload, 0.55).is_veto);

    // Override para activo sensible/volátil: veto más estricto a 0.65 y mayor exigencia de imbalance 0.40
    payload.causal_veto_threshold = 0.65;
    payload.causal_min_book_imbalance = 0.40;

    // Ahora riesgo 0.70 > 0.65, y book_imbalance 0.30 < 0.40 (no califica como breakout alineado) -> DEBE vetar
    let op_strict = SeniorCausal.evaluate(&payload, 0.55);
    assert!(op_strict.is_veto, "Payload dinámico con cota estricta debe vetar");

    // Si el libro se ensancha a 0.50 > 0.40 y riesgo 0.70 < aligned_max_risk 0.88 -> debe autorizar
    payload.book_imbalance = 0.50;
    let op_aligned = SeniorCausal.evaluate(&payload, 0.55);
    assert!(!op_aligned.is_veto, "Breakout institucional adaptativo debe autorizar");
}

#[test]
fn test_causal_opinion_council_params_propagation() {
    // Configurar política estricta en CouncilParams
    let custom_params = CouncilParams::default()
        .with_causal_bounds(0.60, 0.35, 0.80)
        .expect("Parámetros causales válidos");

    let mut council = ConsejoDeliberacion::with_params(custom_params)
        .expect("Consejo con custom params");
    council.agents = vec![Box::new(SeniorCausal)];

    // Payload con defaults canónicos
    let mut payload = base_causal_payload();
    payload.do_calculus_risk = 0.68;
    payload.book_imbalance = 0.28; // < 0.35 (no califica como breakout alineado bajo custom)

    let trace = council.deliberar_traced(&payload, 0.55, None);
    assert_eq!(trace.opinions.len(), 1);
    assert_eq!(trace.opinions[0].role, SeniorRole::Causal);
    assert!(
        trace.opinions[0].is_veto,
        "CouncilParams debe propagar causal_veto_threshold=0.60 y min_imbalance=0.35 hacia SeniorCausal"
    );
}

#[test]
fn test_causal_opinion_bounds_validation_and_fail_closed() {
    // 1. Resiliencia contra no finitos en evaluate_causal_manipulation_veto
    let (is_veto_nan, safe_risk_nan, safe_thresh_nan, _) = evaluate_causal_manipulation_veto(
        f64::NAN,
        0.75,
        0.50,
        1.0,
        0.0,
        0.25,
        0.88,
    );
    assert!(is_veto_nan, "Riesgo NaN debe fallar cerrado y vetar");
    assert_eq!(safe_risk_nan, 1.0);
    assert_eq!(safe_thresh_nan, 0.75);

    // Parámetros corruptos en función pura caen a defaults seguros
    let (_, _, safe_thresh_corrupt, _) = evaluate_causal_manipulation_veto(
        0.50,
        f64::NAN,
        0.50,
        1.0,
        0.0,
        -1.0,
        2.5,
    );
    assert_eq!(safe_thresh_corrupt, DEFAULT_CAUSAL_VETO_THRESHOLD);

    // 2. Validación en MarketSnapshotPayload
    let mut p = base_causal_payload();
    p.causal_min_book_imbalance = -0.01;
    assert!(p.validate().is_err());
    p.causal_min_book_imbalance = 1.05;
    assert!(p.validate().is_err());

    p.causal_min_book_imbalance = 0.25;
    p.causal_aligned_max_risk = 1.05;
    assert!(p.validate().is_err());

    // 3. Validación en CouncilParams
    assert!(CouncilParams::default().with_causal_bounds(0.0, 0.25, 0.88).is_err());
    assert!(CouncilParams::default().with_causal_bounds(1.05, 0.25, 0.88).is_err());
    assert!(CouncilParams::default().with_causal_bounds(0.75, -0.1, 0.88).is_err());
    assert!(CouncilParams::default().with_causal_bounds(0.75, 0.25, 1.5).is_err());
}

#[test]
fn test_causal_opinion_directional_conjunction_and_symmetry() {
    // Verificación de simetría en Long y Short para breakout alineado
    for dir in [-1.0, 1.0] {
        let mut p = base_causal_payload();
        p.intended_direction = dir;
        p.do_calculus_risk = 0.80; // Mayor a threshold 0.75

        // Libro alineado con la dirección
        p.book_imbalance = 0.40 * dir;
        p.dominant_tau_ms = 30_000.0; // s < 0.5
        assert!(
            !SeniorCausal.evaluate(&p, 0.5).is_veto,
            "Dirección {dir} con libro alineado no debe ser vetada"
        );

        // Libro opuesto a la dirección (selección adversa)
        p.book_imbalance = -0.40 * dir;
        assert!(
            SeniorCausal.evaluate(&p, 0.5).is_veto,
            "Dirección {dir} con libro opuesto DEBE ser vetada"
        );

        // Intención plana (dir = 0.0) -> no hay dirección para alinear, debe vetar por riesgo
        p.intended_direction = 0.0;
        p.book_imbalance = 0.40;
        assert!(
            SeniorCausal.evaluate(&p, 0.5).is_veto,
            "Intención plana con riesgo 0.80 debe vetar"
        );
    }
}
