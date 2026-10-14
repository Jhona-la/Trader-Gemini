//! CONTRATO FORMAL: DESACOPLAMIENTO DINÁMICO DE UMBRALES Y AMORTIGUAMIENTOS DE ENTE DEL MERCADO (Ola Ω118).
//!
//! # Por qué existe
//! Erradica la petrificación heurística del asiento SeniorEnteMercado (ballenas, cascadas, psicología de masas).
//! Anteriormente:
//! 1. `SeniorEnteMercado` utilizaba constantes fijas congeladas en el código:
//!    - Veto hardcodeado: `let is_veto = liq > 0.85;` ignorando cualquier parámetro adaptativo del consejo o payload.
//!    - Amortiguación de ballenas: `(whale_z / 4.0).powi(2)` con piso rígido `0.60`.
//!    - Amortiguación de cascadas: `(-1.2 * liq * liq).exp()` con piso rígido `0.40`.
//!    - Amortiguación de masas contrarian: `(1.0 + 0.25 * crowd^2)` con piso rígido `0.70`.
//! 2. `MarketSnapshotPayload` carecía de campos para calibrar estos umbrales por moneda o régimen.
//! 3. `deliberar_traced` tenía que parchar `op.is_veto` a posteriori en lugar de que el asiento
//!    evaluara directamente con el breaker configurado.
//!
//! # Solución (Ola Ω118)
//! 1. Parámetros desacoplados expuestos en `MarketSnapshotPayload`, `CouncilParams` y `QuantumConfig`:
//!    - `cascade_severity_breaker`
//!    - `whale_z_threshold`
//!    - `whale_min_floor`
//!    - `liq_decay_mult`
//!    - `liq_min_floor`
//!    - `crowd_imbalance_decay`
//!    - `crowd_min_floor`
//! 2. Función pura `evaluate_ente_mercado_opinion` con sanitización estricta fail-closed C¹
//!    (garantía matemática de no pánico ante NaN / Inf / cotas fuera de rango).
//! 3. Propagación `Cow` sin asignación de heap redundante en la ruta caliente cuando los parámetros
//!    del consejo difieren de los canónicos.

use metacortex_engine::consejo_seniors::{
    evaluate_ente_mercado_opinion, ConsejoDeliberacion, CouncilParams, MarketSnapshotPayload,
    SeniorRole, DEFAULT_CASCADE_SEVERITY_BREAKER, DEFAULT_CROWD_IMBALANCE_DECAY,
    DEFAULT_CROWD_MIN_FLOOR, DEFAULT_LIQ_DECAY_MULT, DEFAULT_LIQ_MIN_FLOOR,
    DEFAULT_WHALE_MIN_FLOOR, DEFAULT_WHALE_Z_THRESHOLD,
};

#[test]
fn test_ente_mercado_opinion_canonical_parity() {
    // 1. Mercado en calma total: sin ballenas, sin cascadas, sin sesgo de masas
    let (dir, conf_calm, is_veto_calm, just_calm) = evaluate_ente_mercado_opinion(
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0, // Dirección alcista solicitada
        0,
        0.0,
        DEFAULT_CASCADE_SEVERITY_BREAKER,
        DEFAULT_WHALE_Z_THRESHOLD,
        DEFAULT_WHALE_MIN_FLOOR,
        DEFAULT_LIQ_DECAY_MULT,
        DEFAULT_LIQ_MIN_FLOOR,
        DEFAULT_CROWD_IMBALANCE_DECAY,
        DEFAULT_CROWD_MIN_FLOOR,
    );
    assert_eq!(dir, 1.0);
    assert!((conf_calm - 1.0).abs() < 1e-9, "Mercado calmo debe tener convicción 1.0");
    assert!(!is_veto_calm, "Mercado calmo no debe vetar");
    assert!(!just_calm.contains("[VETO CASCADA]"));

    // 2. Cascada mayor canónica (liq = 0.90 > 0.85): debe activar veto estricto
    let (_dir_shock, _conf_shock, is_veto_shock, just_shock) = evaluate_ente_mercado_opinion(
        0.0,
        0.90,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0.0,
        DEFAULT_CASCADE_SEVERITY_BREAKER,
        DEFAULT_WHALE_Z_THRESHOLD,
        DEFAULT_WHALE_MIN_FLOOR,
        DEFAULT_LIQ_DECAY_MULT,
        DEFAULT_LIQ_MIN_FLOOR,
        DEFAULT_CROWD_IMBALANCE_DECAY,
        DEFAULT_CROWD_MIN_FLOOR,
    );
    assert!(is_veto_shock, "Liquidaciones 0.90 con breaker canónico 0.85 debe vetar");
    assert!(just_shock.contains("[VETO CASCADA]"));

    // 3. Shock de ballena moderado (whale_z = 4.0): amortiguación 1 / (1 + (4/4)^2) = 0.50 -> clamp al piso canónico 0.60
    let (_dir_whale, conf_whale, is_veto_whale, _) = evaluate_ente_mercado_opinion(
        4.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0.0,
        DEFAULT_CASCADE_SEVERITY_BREAKER,
        DEFAULT_WHALE_Z_THRESHOLD,
        DEFAULT_WHALE_MIN_FLOOR,
        DEFAULT_LIQ_DECAY_MULT,
        DEFAULT_LIQ_MIN_FLOOR,
        DEFAULT_CROWD_IMBALANCE_DECAY,
        DEFAULT_CROWD_MIN_FLOOR,
    );
    assert!(!is_veto_whale);
    assert!((conf_whale - 0.60).abs() < 1e-9, "Whale z=4.0 debe clampear al piso canónico 0.60");
}

#[test]
fn test_dynamic_cascade_severity_breaker_adaptation() {
    // 1. Breaker endurecido (0.40): liquidación moderada (0.50) debe disparar veto
    let (_, _, is_veto_tight, _) = evaluate_ente_mercado_opinion(
        0.0,
        0.50,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0.0,
        0.40, // Breaker endurecido
        DEFAULT_WHALE_Z_THRESHOLD,
        DEFAULT_WHALE_MIN_FLOOR,
        DEFAULT_LIQ_DECAY_MULT,
        DEFAULT_LIQ_MIN_FLOOR,
        DEFAULT_CROWD_IMBALANCE_DECAY,
        DEFAULT_CROWD_MIN_FLOOR,
    );
    assert!(is_veto_tight, "Con breaker 0.40, liq=0.50 debe ser vetado");

    // 2. Mismo liq=0.50 con breaker canónico (0.85) NO debe vetar
    let (_, _, is_veto_canonical, _) = evaluate_ente_mercado_opinion(
        0.0,
        0.50,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0.0,
        DEFAULT_CASCADE_SEVERITY_BREAKER,
        DEFAULT_WHALE_Z_THRESHOLD,
        DEFAULT_WHALE_MIN_FLOOR,
        DEFAULT_LIQ_DECAY_MULT,
        DEFAULT_LIQ_MIN_FLOOR,
        DEFAULT_CROWD_IMBALANCE_DECAY,
        DEFAULT_CROWD_MIN_FLOOR,
    );
    assert!(!is_veto_canonical, "Con breaker 0.85, liq=0.50 NO debe ser vetado");

    // 3. Breaker relajado (0.95): liquidación severa (0.88) que normalmente vetaría, ahora se tolera
    let (_, _, is_veto_loose, _) = evaluate_ente_mercado_opinion(
        0.0,
        0.88,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0.0,
        0.95, // Breaker permisivo
        DEFAULT_WHALE_Z_THRESHOLD,
        DEFAULT_WHALE_MIN_FLOOR,
        DEFAULT_LIQ_DECAY_MULT,
        DEFAULT_LIQ_MIN_FLOOR,
        DEFAULT_CROWD_IMBALANCE_DECAY,
        DEFAULT_CROWD_MIN_FLOOR,
    );
    assert!(!is_veto_loose, "Con breaker 0.95, liq=0.88 se tolera sin veto");
}

#[test]
fn test_whale_and_crowd_damping_adaptation() {
    // 1. Calibración dinámica de ballena: umbral z=2.0 y piso más bajo 0.30
    // whale_z = 4.0 -> (4.0 / 2.0)^2 = 4.0 -> p_whale = 1 / (1 + 4) = 0.20 -> clampeado a floor=0.30
    let (_, conf_custom_whale, _, _) = evaluate_ente_mercado_opinion(
        4.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0.0,
        DEFAULT_CASCADE_SEVERITY_BREAKER,
        2.0,  // z_thresh más sensible
        0.30, // floor más profundo
        DEFAULT_LIQ_DECAY_MULT,
        DEFAULT_LIQ_MIN_FLOOR,
        DEFAULT_CROWD_IMBALANCE_DECAY,
        DEFAULT_CROWD_MIN_FLOOR,
    );
    assert!((conf_custom_whale - 0.30).abs() < 1e-9, "Debe amortiguar hasta el piso adaptado 0.30");

    // 2. Calibración dinámica de multitud: tasa de decaimiento 1.0 (más agresiva que 0.25)
    // LS ratio = e ≈ 2.7182818 -> ln(e) = 1.0 -> crowd_imbalance = 1.0
    // p_crowd = 1 / (1 + 1.0 * 1.0^2) = 0.50
    let (_, conf_custom_crowd, _, _) = evaluate_ente_mercado_opinion(
        0.0,
        0.0,
        0.0,
        0.0,
        std::f64::consts::E,
        1.0,
        1.0,
        0,
        0.0,
        DEFAULT_CASCADE_SEVERITY_BREAKER,
        DEFAULT_WHALE_Z_THRESHOLD,
        DEFAULT_WHALE_MIN_FLOOR,
        DEFAULT_LIQ_DECAY_MULT,
        DEFAULT_LIQ_MIN_FLOOR,
        1.0,  // decay más rápido
        0.40, // floor
    );
    assert!((conf_custom_crowd - 0.50).abs() < 1e-4, "Crowd imbalance con decay=1.0 debe producir convicción 0.50");
}

#[test]
fn test_ente_mercado_fail_closed_sanitization() {
    // Entradas no finitas o corruptas deben resolverse fail-closed sin pánicos
    let (dir_nan, conf_nan, is_veto_nan, _) = evaluate_ente_mercado_opinion(
        f64::NAN,
        f64::INFINITY,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        0,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
    );
    assert_eq!(dir_nan, 0.0);
    assert_eq!(conf_nan, 0.0);
    assert!(!is_veto_nan);

    // Dirección válida pero parámetros corruptos: debe usar los defaults seguros
    let (dir_valid, conf_valid, is_veto_valid, _) = evaluate_ente_mercado_opinion(
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0.0,
        f64::NAN,
        -5.0,
        2.0,
        -1.0,
        -0.5,
        -10.0,
        1.5,
    );
    assert_eq!(dir_valid, 1.0);
    assert!(conf_valid.is_finite() && conf_valid > 0.0);
    assert!(!is_veto_valid);
}

#[test]
fn test_council_params_builder_and_validation() {
    let base = CouncilParams::default();

    // 1. Configuración válida con el nuevo builder
    let custom = base
        .with_market_entity_bounds(0.70, 3.0, 0.50, 1.5, 0.35, 0.30, 0.65)
        .expect("Parámetros válidos deben construirse sin error");
    assert_eq!(custom.cascade_severity_breaker, 0.70);
    assert_eq!(custom.whale_z_threshold, 3.0);
    assert_eq!(custom.whale_min_floor, 0.50);
    assert_eq!(custom.liq_decay_mult, 1.5);
    assert_eq!(custom.liq_min_floor, 0.35);
    assert_eq!(custom.crowd_imbalance_decay, 0.30);
    assert_eq!(custom.crowd_min_floor, 0.65);

    // 2. Parámetros inválidos deben ser rechazados por validate()
    assert!(base
        .with_market_entity_bounds(-0.1, 3.0, 0.50, 1.5, 0.35, 0.30, 0.65)
        .is_err(), "cascade_breaker < 0 debe fallar");
    assert!(base
        .with_market_entity_bounds(0.70, 0.2, 0.50, 1.5, 0.35, 0.30, 0.65)
        .is_err(), "whale_z_threshold < 0.5 debe fallar");
    assert!(base
        .with_market_entity_bounds(0.70, 3.0, 0.05, 1.5, 0.35, 0.30, 0.65)
        .is_err(), "whale_min_floor < 0.10 debe fallar");
    assert!(base
        .with_market_entity_bounds(0.70, 3.0, 0.50, 0.05, 0.35, 0.30, 0.65)
        .is_err(), "liq_decay_mult < 0.1 debe fallar");
    assert!(base
        .with_market_entity_bounds(0.70, 3.0, 0.50, 1.5, 0.01, 0.30, 0.65)
        .is_err(), "liq_min_floor < 0.05 debe fallar");
}

#[test]
fn test_cow_propagation_in_deliberar_traced() {
    let mut params = CouncilParams::default();
    // Endurecemos el breaker de cascada a 0.50
    params = params
        .with_market_entity_bounds(0.50, 4.0, 0.60, 1.2, 0.40, 0.25, 0.70)
        .expect("Valid bounds");

    let consejo = ConsejoDeliberacion::with_params(params).expect("Consejo con params válidos");

    // Payload con los valores por defecto (breaker = 0.85), pero con liquidación 0.60
    let payload = MarketSnapshotPayload {
        intended_direction: 1.0,
        liquidation_severity: 0.60,
        ml_prob: 0.70,
        fused_score: 0.80,
        persistence: 0.50,
        book_imbalance: 0.40,
        ..Default::default()
    };

    let trace = consejo.deliberar_traced(&payload, 0.65, None);
    let ente_op = trace
        .opinions
        .iter()
        .find(|op| op.role == SeniorRole::EnteMercado)
        .expect("EnteMercado opinion must exist");

    // Debido a que CouncilParams tiene breaker=0.50, y liq=0.60 > 0.50, debe VETAR
    assert!(ente_op.is_veto, "EnteMercado debió vetar porque liq=0.60 > consejo.params.cascade_severity_breaker (0.50)");
    assert!(!trace.consensus.approved, "Deliberación debe rechazar entrada ante veto de EnteMercado");
}
