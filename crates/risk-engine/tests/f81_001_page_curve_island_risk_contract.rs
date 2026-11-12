use risk_engine::{PageCurveIslandConfig, PageCurveIslandRiskEngine, MAX_PORTFOLIO_ASSETS};

#[test]
fn test_page_curve_island_risk_pre_page_regime() {
    let config = PageCurveIslandConfig::default();
    let engine = PageCurveIslandRiskEngine::new(config);

    // Capital saludable de $1000, piso de $5, riesgo moderado
    let weights = [0.25, 0.25, 0.25, 0.25];
    let report = engine.evaluate(1000.0, &weights, 0.05, 0.20);

    assert!(!report.post_page_transition);
    assert!(!report.island_veto_induced);
    assert!(report.quantum_island_present);
    assert!(report.page_entropy < report.bekenstein_hawking_entropy);
    assert!(report.slot_scale_factor >= 0.50 && report.slot_scale_factor <= 1.25);
}

#[test]
fn test_page_curve_island_risk_post_page_with_active_island_recovery() {
    let config = PageCurveIslandConfig {
        emergency_floor_usd: 50.0,
        ..Default::default()
    };
    let engine = PageCurveIslandRiskEngine::new(config);

    // Capital bajo de $60 (cerca del piso de $50), alta volatilidad, pero excelente diversificación (baja correlación)
    let weights = [0.25, 0.25, 0.25, 0.25];
    let report = engine.evaluate(60.0, &weights, 0.50, 0.10);

    assert!(
        report.post_page_transition,
        "Debe superar el tiempo de Page ante alta radiación de volatilidad"
    );
    assert!(
        report.quantum_island_present,
        "La baja correlación debe permitir la formación de la isla cuántica"
    );
    assert!(
        !report.island_veto_induced,
        "La isla cuántica debe evitar el veto al purificar la radiación unitariamente"
    );
    assert!(report.slot_scale_factor > 0.0);
}

#[test]
fn test_page_curve_island_risk_hard_veto_on_information_collapse() {
    let config = PageCurveIslandConfig {
        emergency_floor_usd: 50.0,
        ..Default::default()
    };
    let engine = PageCurveIslandRiskEngine::new(config);

    // Capital colapsado ($51, piso $50), altísima volatilidad, y correlación extrema (isla colapsada)
    let weights = [1.0, 0.0, 0.0, 0.0];
    let report = engine.evaluate(51.0, &weights, 1.50, 0.99);

    assert!(report.post_page_transition);
    assert!(
        !report.quantum_island_present,
        "Correlación de 0.99 debe destruir el área de la isla cuántica"
    );
    assert!(
        report.island_veto_induced,
        "Debe inducir veto absoluto por colapso de información no unitario"
    );
    assert_eq!(report.slot_scale_factor, 0.0);
}

#[test]
fn test_page_curve_island_risk_hard_veto_with_present_but_insufficient_island() {
    let engine = PageCurveIslandRiskEngine::new(PageCurveIslandConfig::default());

    let weights = [1.0, 0.0, 0.0, 0.0];
    let report = engine.evaluate(1.0, &weights, 1.50, 0.85);

    let s_bh = 1.2_f64.ln();
    let s_rad = 76.0_f64.ln();
    let s_island = 0.15 * s_bh + 0.85 * s_rad * 0.10;
    assert!((s_island - 0.39546056744343134).abs() < 1e-15);
    assert!(s_island > 1.02 * s_bh);
    assert!((1.02 * s_bh - 0.1859679876898337).abs() < 1e-9);

    assert!(report.post_page_transition);
    assert!(report.quantum_island_present);
    assert!(report.island_veto_induced);
    assert_eq!(report.page_entropy, report.bekenstein_hawking_entropy);
    assert_eq!(report.slot_scale_factor, 0.0);

    let absent = engine.evaluate(1.0, &weights, 1.50, 0.99);
    assert!(absent.post_page_transition);
    assert!(!absent.quantum_island_present);
    assert!(absent.island_veto_induced);
    assert_eq!(absent.page_entropy, absent.radiation_entropy);
    assert!(absent.page_entropy > absent.bekenstein_hawking_entropy);
    assert_eq!(absent.slot_scale_factor, 0.0);
}

#[test]
fn test_page_curve_island_risk_anticorrelation_preserves_island() {
    let engine = PageCurveIslandRiskEngine::new(PageCurveIslandConfig::default());

    let weights = [1.0, 1.0, 0.2, 0.1];
    let report = engine.evaluate(13.0, &weights, 0.01, -1.0);

    assert!(report.post_page_transition);
    assert!(
        report.quantum_island_present,
        "La anticorrelación perfecta diversifica; no debe colapsar el área de la isla"
    );
    assert!(
        !report.island_veto_induced,
        "Una isla presente con S_island = S_BH no debe inducir veto"
    );
    assert!(report.page_entropy <= report.bekenstein_hawking_entropy);
}

#[test]
fn test_page_curve_island_risk_stability_under_nan_and_extremes() {
    let config = PageCurveIslandConfig::default();
    let engine = PageCurveIslandRiskEngine::new(config);

    let weights_nan = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let rep_nan = engine.evaluate(f64::NAN, &weights_nan, f64::NAN, f64::NAN);

    assert!(rep_nan.bekenstein_hawking_entropy.is_finite());
    assert!(rep_nan.radiation_entropy.is_finite());
    assert!(rep_nan.page_entropy.is_finite());
    assert!(rep_nan.slot_scale_factor >= 0.0 && rep_nan.slot_scale_factor <= 1.25);

    let weights_zero = [0.0; MAX_PORTFOLIO_ASSETS];
    let rep_zero = engine.evaluate(0.0, &weights_zero, 0.0, 0.0);
    assert!(rep_zero.slot_scale_factor >= 0.0 && rep_zero.slot_scale_factor <= 1.25);
}

#[test]
fn test_page_curve_island_risk_deterministic_replay() {
    let config = PageCurveIslandConfig::default();
    let engine_a = PageCurveIslandRiskEngine::new(config);
    let engine_b = PageCurveIslandRiskEngine::new(config);

    let weights = [0.4, 0.3, 0.2, 0.1];
    let rep_a = engine_a.evaluate(250.0, &weights, 0.15, 0.35);
    let rep_b = engine_b.evaluate(250.0, &weights, 0.15, 0.35);

    assert_eq!(
        rep_a.bekenstein_hawking_entropy,
        rep_b.bekenstein_hawking_entropy
    );
    assert_eq!(rep_a.radiation_entropy, rep_b.radiation_entropy);
    assert_eq!(rep_a.page_entropy, rep_b.page_entropy);
    assert_eq!(rep_a.quantum_island_present, rep_b.quantum_island_present);
    assert_eq!(rep_a.island_veto_induced, rep_b.island_veto_induced);
    assert_eq!(rep_a.slot_scale_factor, rep_b.slot_scale_factor);
}
