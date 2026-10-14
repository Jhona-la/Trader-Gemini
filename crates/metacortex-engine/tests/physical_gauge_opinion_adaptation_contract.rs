//! CONTRATO FORMAL DE CERTIFICACIÓN: ADAPTABILIDAD DINÁMICA DE FÍSICA DE ROTACIONAL HELMHOLTZ-HODGE
//! Y CORRIENTE GAUGE DE YANG-MILLS (Ola Ω116).
//!
//! # Foco Matemático y Físico
//! 1. Desacoplamiento dinámico de coeficientes de atenuación de rotacional y piso laminar en `SeniorMicroestructura`:
//!    `laminar_factor = (1.0 - curl_att * curl).clamp(lam_floor, 1.0)`
//! 2. Desacoplamiento dinámico de corriente de referencia y peso máximo asintótico en `SeniorSeriesTemporales`:
//!    `w_ym = ym_max_w * (ym_sq / (ym_sq + ym_ref^2))`
//! 3. Sanitización fail-closed ante NaN/Inf o valores fuera de cota.
//! 4. Propagación lock-free `Cow` en `deliberar_traced`.
//! 5. Sincronización atómica en `QuantumConfig` y resolución por activo en `council_snapshot`.

use metacortex_engine::consejo_seniors::{
    CouncilParams, SeniorAgent, SeniorMicroestructura, SeniorSeriesTemporales,
};
use metacortex_engine::{
    evaluate_microstructure_flow_confidence, evaluate_series_temporales_ym, ConsejoDeliberacion,
    MarketSnapshotPayload, SeniorRole, DEFAULT_HODGE_CURL_ATTENUATION, DEFAULT_HODGE_LAMINAR_FLOOR,
    DEFAULT_YM_MAX_WEIGHT, DEFAULT_YM_REF_CURRENT,
};

fn base_physical_payload() -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        intended_direction: 1.0,
        book_imbalance: 0.60,
        fused_score: 0.50,
        persistence: 0.40,
        hodge_curl_share: 0.25,
        yang_mills_current: 0.10,
        navier_laminar_share: 0.80,
        ..Default::default()
    }
}

#[test]
fn test_physical_gauge_canonical_defaults_exact_parity() {
    let payload = base_physical_payload();

    // 1. Verificación de SeniorMicroestructura contra fórmula canónica original
    let op_micro = SeniorMicroestructura.evaluate(&payload, 0.55);
    assert_eq!(op_micro.role, SeniorRole::Microestructura);
    assert_eq!(op_micro.signal_direction, 1.0);

    let (conf_pure, lam_pure, hydro_pure, imb_pure) = evaluate_microstructure_flow_confidence(
        payload.book_imbalance,
        payload.hodge_curl_share,
        payload.navier_laminar_share,
        DEFAULT_HODGE_CURL_ATTENUATION,
        DEFAULT_HODGE_LAMINAR_FLOOR,
    );

    let expected_lam = (1.0 - 0.70 * 0.25_f64).clamp(0.30, 1.0);
    let expected_hydro = (0.40 + 0.60 * 0.80_f64).clamp(0.40, 1.0);
    let expected_conf = (0.60_f64 * expected_lam * expected_hydro).clamp(0.0, 1.0);

    assert_eq!(imb_pure, 0.60);
    assert!((lam_pure - expected_lam).abs() < 1e-12);
    assert!((hydro_pure - expected_hydro).abs() < 1e-12);
    assert!((conf_pure - expected_conf).abs() < 1e-12);
    assert!((op_micro.confidence - expected_conf).abs() < 1e-12);

    // 2. Verificación de SeniorSeriesTemporales contra fórmula canónica original
    let op_st = SeniorSeriesTemporales.evaluate(&payload, 0.55);
    assert_eq!(op_st.role, SeniorRole::SeriesTemporales);

    let (sig_pure, conf_pure_st, w_ym_pure, ym_pure) = evaluate_series_temporales_ym(
        payload.fused_score,
        payload.persistence,
        payload.yang_mills_current,
        DEFAULT_YM_REF_CURRENT,
        DEFAULT_YM_MAX_WEIGHT,
    );

    let ym_sq = 0.10 * 0.10;
    let thresh_sq = 0.05 * 0.05;
    let expected_w_ym = 0.20 * (ym_sq / (ym_sq + thresh_sq));
    assert_eq!(ym_pure, 0.10);
    assert!((w_ym_pure - expected_w_ym).abs() < 1e-12);
    assert!((sig_pure - op_st.signal_direction).abs() < 1e-12);
    assert!((conf_pure_st - op_st.confidence).abs() < 1e-12);
}

#[test]
fn test_physical_gauge_dynamic_payload_override() {
    let mut payload = base_physical_payload();

    // Modificar parámetros de rotacional: atenuación más agresiva (0.95) y piso más bajo (0.05)
    payload.hodge_curl_attenuation = 0.95;
    payload.hodge_laminar_floor = 0.05;

    let op_micro_custom = SeniorMicroestructura.evaluate(&payload, 0.55);
    let expected_custom_lam = (1.0_f64 - 0.95 * 0.25).clamp(0.05, 1.0);
    let expected_custom_conf = (0.60_f64 * expected_custom_lam * (0.40 + 0.60 * 0.80)).clamp(0.0, 1.0);

    assert!((op_micro_custom.confidence - expected_custom_conf).abs() < 1e-12);

    // Modificar parámetros de Yang-Mills: corriente de referencia más sensible (0.02) y peso mayor (0.40)
    payload.ym_ref_current = 0.02;
    payload.ym_max_weight = 0.40;

    let op_st_custom = SeniorSeriesTemporales.evaluate(&payload, 0.55);
    let (_, _, w_ym_custom, _) = evaluate_series_temporales_ym(
        payload.fused_score,
        payload.persistence,
        payload.yang_mills_current,
        0.02,
        0.40,
    );

    let ym_sq = 0.10 * 0.10;
    let thresh_sq = 0.02 * 0.02;
    let expected_w_ym_custom = 0.40 * (ym_sq / (ym_sq + thresh_sq));
    assert!((w_ym_custom - expected_w_ym_custom).abs() < 1e-12);
    assert!((op_st_custom.confidence - op_st_custom.signal_direction.abs()).abs() < 1e-12);
}

#[test]
fn test_physical_gauge_council_params_propagation() {
    let mut consejo = ConsejoDeliberacion::new();
    // Configurar el consejo con bounds físicos personalizados
    consejo.params = consejo
        .params
        .with_physical_gauge_bounds(0.85, 0.10, 0.03, 0.35)
        .expect("Valid custom physical gauge bounds");

    // Payload con defaults canónicos
    let payload = base_physical_payload();
    assert_eq!(payload.hodge_curl_attenuation, DEFAULT_HODGE_CURL_ATTENUATION);
    assert_eq!(payload.hodge_laminar_floor, DEFAULT_HODGE_LAMINAR_FLOOR);
    assert_eq!(payload.ym_ref_current, DEFAULT_YM_REF_CURRENT);
    assert_eq!(payload.ym_max_weight, DEFAULT_YM_MAX_WEIGHT);

    let trace = consejo.deliberar_traced(&payload, 0.65, None);
    assert!(trace.consensus.approved);

    let micro_op = trace
        .opinions
        .iter()
        .find(|o| o.role == SeniorRole::Microestructura)
        .expect("SeniorMicroestructura opinion present");

    let expected_lam = (1.0_f64 - 0.85 * 0.25).clamp(0.10, 1.0);
    let expected_conf = (0.60_f64 * expected_lam * (0.40 + 0.60 * 0.80)).clamp(0.0, 1.0);
    assert!((micro_op.confidence - expected_conf).abs() < 1e-12);
}

#[test]
fn test_physical_gauge_bounds_validation_and_fail_closed() {
    // 1. Validar que la función pura sanitiza parámetros corruptos cayendo a defaults
    let (conf, lam, _, _) = evaluate_microstructure_flow_confidence(
        0.50,
        0.40,
        0.70,
        f64::NAN, // curl_attenuation corrupto -> fallback a 0.70
        f64::INFINITY, // laminar_floor corrupto -> fallback a 0.30
    );
    let expected_lam = (1.0_f64 - 0.70 * 0.40).clamp(0.30, 1.0);
    assert!((lam - expected_lam).abs() < 1e-12);
    assert!(conf.is_finite());

    // 2. Validar que evaluate_series_temporales_ym sanitiza ante no-finitos
    let (sig, conf_ym, w_ym, _) = evaluate_series_temporales_ym(
        0.30,
        0.20,
        f64::NAN, // ym_current corrupto -> 0.0
        -0.05,    // ym_ref corrupto -> fallback a 0.05
        1.50,     // ym_max_w corrupto -> fallback a 0.20
    );
    assert!(sig.is_finite());
    assert!(conf_ym.is_finite());
    assert_eq!(w_ym, 0.0); // En ym_curr = 0.0, w_ym es 0.0

    // 3. CouncilParams::with_physical_gauge_bounds rechaza valores fuera de rango
    let params = CouncilParams::default();
    assert!(params.with_physical_gauge_bounds(-0.1, 0.30, 0.05, 0.20).is_err());
    assert!(params.with_physical_gauge_bounds(1.1, 0.30, 0.05, 0.20).is_err());
    assert!(params.with_physical_gauge_bounds(0.70, -0.05, 0.05, 0.20).is_err());
    assert!(params.with_physical_gauge_bounds(0.70, 0.30, 0.0, 0.20).is_err());
    assert!(params.with_physical_gauge_bounds(0.70, 0.30, -0.05, 0.20).is_err());
    assert!(params.with_physical_gauge_bounds(0.70, 0.30, 0.05, 1.25).is_err());
}

#[test]
fn test_physical_gauge_lorentz_smoothness_and_helmholtz_decay() {
    // 1. Monotonicidad y decaimiento suave en Helmholtz-Hodge:
    // A mayor vorticidad rotacional (curl -> 1.0), menor es la confianza laminar.
    let mut prev_conf = 1.0;
    for step in 0..=10 {
        let curl = step as f64 / 10.0;
        let (conf, _, _, _) = evaluate_microstructure_flow_confidence(
            0.80, curl, 1.0, 0.70, 0.30,
        );
        assert!(conf <= prev_conf + 1e-12, "Confianza debe ser monótonamente decreciente con curl");
        prev_conf = conf;
    }

    // 2. Suavidad C^∞ y saturación en Yang-Mills:
    // A mayor corriente restauradora |ym_curr|, mayor es el peso gauge w_ym convergiendo a ym_max_w (0.20)
    let mut prev_w = 0.0;
    for step in 1..=10 {
        let ym_curr = step as f64 * 0.05;
        let (_, _, w_ym, _) = evaluate_series_temporales_ym(
            0.50, 0.50, ym_curr, 0.05, 0.20,
        );
        assert!(w_ym >= prev_w - 1e-12, "w_ym debe ser monótonamente creciente con |ym_curr|");
        assert!(w_ym <= 0.20 + 1e-12, "w_ym no debe exceder el peso máximo");
        prev_w = w_ym;
    }
}
