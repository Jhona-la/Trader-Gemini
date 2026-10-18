//! Contrato de adaptación y comportamiento del Tribunal de Autoengaño y Auditor Interno (Ola Ω120)
//!
//! Verifica matemáticamente:
//! 1. Paridad canónica de `evaluate_tribunal_verdict` bajo umbrales estándar (0.20, 0.10, 0.05).
//! 2. Adaptación dinámica de `TribunalThresholds` y validación de cotas de seguridad.
//! 3. Robustez fail-closed ante entradas no finitas (NaN, Inf, negativos).
//! 4. Cálculo exacto de discrepancia máxima (`calculate_max_discrepancy` / `compute_internal_discrepancy`) sobre fills reales.
//! 6. Integración de extremo a extremo: registro de fills -> discrepancia -> OmniscientRegistry.

use audit_engine::{
    calculate_max_discrepancy, evaluate_tribunal_verdict, AuditorInterno, ExchangeFillRecord,
    TribunalThresholds, TribunalVerdict, DEFAULT_AUTOENGANO_THRESHOLD,
    DEFAULT_CULPABLE_PELIGROSO_THRESHOLD, DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD,
};
use omniscient_registry::OmniscientRegistry;

#[test]
fn test_tribunal_verdict_canonical_parity() {
    let thresholds = TribunalThresholds::default();
    assert_eq!(thresholds.autoengano_threshold, DEFAULT_AUTOENGANO_THRESHOLD);
    assert_eq!(
        thresholds.culpable_peligroso_threshold,
        DEFAULT_CULPABLE_PELIGROSO_THRESHOLD
    );
    assert_eq!(
        thresholds.culpable_recuperable_threshold,
        DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD
    );

    // Discrepancia <= 0.05 -> Inocente
    assert_eq!(
        evaluate_tribunal_verdict(0.00, &thresholds),
        TribunalVerdict::Inocente
    );
    assert_eq!(
        evaluate_tribunal_verdict(0.049, &thresholds),
        TribunalVerdict::Inocente
    );
    assert_eq!(
        evaluate_tribunal_verdict(0.050, &thresholds),
        TribunalVerdict::Inocente
    );

    // Discrepancia > 0.05 y <= 0.10 -> CulpableRecuperable
    assert_eq!(
        evaluate_tribunal_verdict(0.051, &thresholds),
        TribunalVerdict::CulpableRecuperable
    );
    assert_eq!(
        evaluate_tribunal_verdict(0.099, &thresholds),
        TribunalVerdict::CulpableRecuperable
    );
    assert_eq!(
        evaluate_tribunal_verdict(0.100, &thresholds),
        TribunalVerdict::CulpableRecuperable
    );

    // Discrepancia > 0.10 y <= 0.20 -> CulpablePeligroso
    assert_eq!(
        evaluate_tribunal_verdict(0.101, &thresholds),
        TribunalVerdict::CulpablePeligroso
    );
    assert_eq!(
        evaluate_tribunal_verdict(0.199, &thresholds),
        TribunalVerdict::CulpablePeligroso
    );
    assert_eq!(
        evaluate_tribunal_verdict(0.200, &thresholds),
        TribunalVerdict::CulpablePeligroso
    );

    // Discrepancia > 0.20 -> AutoengañoDetectado (Epistemic Crisis)
    assert_eq!(
        evaluate_tribunal_verdict(0.201, &thresholds),
        TribunalVerdict::AutoengañoDetectado
    );
    assert_eq!(
        evaluate_tribunal_verdict(0.500, &thresholds),
        TribunalVerdict::AutoengañoDetectado
    );
}

#[test]
fn test_tribunal_thresholds_dynamic_adaptation_and_validation() {
    let mut thresholds = TribunalThresholds::default();

    // Endurecimiento estricto: autoengaño en 0.12, peligroso en 0.06, recuperable en 0.02
    thresholds.autoengano_threshold = 0.12;
    thresholds.culpable_peligroso_threshold = 0.06;
    thresholds.culpable_recuperable_threshold = 0.02;
    assert!(thresholds.validate().is_ok());

    // Discrepancia 0.15 que antes era CulpablePeligroso ahora se convierte en AutoengañoDetectado
    assert_eq!(
        evaluate_tribunal_verdict(0.15, &thresholds),
        TribunalVerdict::AutoengañoDetectado
    );
    // Discrepancia 0.08 que antes era CulpableRecuperable ahora se convierte en CulpablePeligroso
    assert_eq!(
        evaluate_tribunal_verdict(0.08, &thresholds),
        TribunalVerdict::CulpablePeligroso
    );

    // Verificación de validación de orden e integridad de límites
    let mut invalid_order = thresholds;
    invalid_order.culpable_recuperable_threshold = 0.10;
    invalid_order.culpable_peligroso_threshold = 0.08; // Inversión inválida
    assert!(invalid_order.validate().is_err());

    let mut invalid_auto = thresholds;
    invalid_auto.autoengano_threshold = 0.04;
    invalid_auto.culpable_peligroso_threshold = 0.05; // Peligroso >= autoengaño
    assert!(invalid_auto.validate().is_err());

    let mut invalid_nan = thresholds;
    invalid_nan.autoengano_threshold = f64::NAN;
    assert!(invalid_nan.validate().is_err());
}

#[test]
fn test_tribunal_fail_closed_robustness() {
    let thresholds = TribunalThresholds::default();

    // Valores no finitos deben disparar AutoengañoDetectado para proteger el capital
    assert_eq!(
        evaluate_tribunal_verdict(f64::NAN, &thresholds),
        TribunalVerdict::AutoengañoDetectado
    );
    assert_eq!(
        evaluate_tribunal_verdict(f64::INFINITY, &thresholds),
        TribunalVerdict::AutoengañoDetectado
    );
    assert_eq!(
        evaluate_tribunal_verdict(f64::NEG_INFINITY, &thresholds),
        TribunalVerdict::AutoengañoDetectado
    );
    assert_eq!(
        evaluate_tribunal_verdict(-0.01, &thresholds),
        TribunalVerdict::AutoengañoDetectado
    );

    // Umbrales corruptos o no finitos dentro del struct caen en los canónicos sin pánico
    let corrupt_thresholds = TribunalThresholds {
        autoengano_threshold: f64::NAN,
        culpable_peligroso_threshold: -1.0,
        culpable_recuperable_threshold: 0.0,
        overfitting_recent_min: 0.05,
        overfitting_oos_max: -0.01,
        max_features_nominal: 64,
    };
    assert_eq!(
        evaluate_tribunal_verdict(0.25, &corrupt_thresholds),
        TribunalVerdict::AutoengañoDetectado
    );
    assert_eq!(
        evaluate_tribunal_verdict(0.02, &corrupt_thresholds),
        TribunalVerdict::Inocente
    );
}

#[test]
fn test_max_discrepancy_and_auditor_fill_computation() {
    let mut auditor = AuditorInterno::with_capital(13.0);

    // Sin fills registrados:
    // Reclamar métricas positivas sin evidencia produce 100% de discrepancia (1.0)
    assert_eq!(auditor.compute_internal_discrepancy(1.5, 0.60, 0.05), 1.0);
    // Afirmar honestamente 0.0 con 0 fills produce 0.0 de discrepancia
    assert_eq!(auditor.compute_internal_discrepancy(0.0, 0.0, 0.0), 0.0);

    // Registrar 10 fills cerrados: 6 victorias y 4 derrotas (WR = 0.60)
    for i in 0..10 {
        let pnl = if i < 6 { 0.50 } else { -0.30 };
        auditor.record_fill(ExchangeFillRecord {
            timestamp_ns: 1_700_000_000_000_000_000 + i * 1_000_000_000,
            symbol_id: 0,
            is_buyer: true,
            price: 50_000.0,
            qty: 0.001,
            fee_usd: 0.02,
            realized_pnl: pnl,
            is_closed_position: true,
        });
    }

    let independent = auditor.compute_independent_metrics();
    assert_eq!(independent.calculated_win_rate, 0.60);

    // Caso 1: Afirmación interna coincide con la realidad de ejecución
    let (max_disc, sharpe_disc, wr_disc, dd_disc) = calculate_max_discrepancy(
        independent.calculated_sharpe,
        0.60,
        independent.calculated_max_drawdown,
        &independent,
    );
    assert!(max_disc < 1e-9);
    assert!(wr_disc < 1e-9);
    assert!(sharpe_disc < 1e-9);
    assert!(dd_disc < 1e-9);

    // Caso 2: Afirmación interna de Win Rate inflada (afirma 0.90 cuando la realidad es 0.60)
    let disc_wr = auditor.compute_internal_discrepancy(
        independent.calculated_sharpe,
        0.90,
        independent.calculated_max_drawdown,
    );
    assert!((disc_wr - 0.30).abs() < 1e-9);

    // Caso 3: Afirmación interna de Drawdown subestimada (afirma 0.01 cuando la realidad es mayor)
    let disc_dd = auditor.compute_internal_discrepancy(independent.calculated_sharpe, 0.60, 0.0);
    assert_eq!(disc_dd, independent.calculated_max_drawdown);
}

#[test]
fn test_end_to_end_registry_discrepancy_pipeline() {
    let mut auditor = AuditorInterno::with_capital(13.0);
    let registry = OmniscientRegistry::new();

    // Registro de un trade perdedor real
    auditor.record_fill(ExchangeFillRecord {
        timestamp_ns: 1_700_000_000_000_000_000,
        symbol_id: 0,
        is_buyer: false,
        price: 2_500.0,
        qty: 0.01,
        fee_usd: 0.01,
        realized_pnl: -0.50,
        is_closed_position: true,
    });

    // Realidad: 0 victorias / 1 trade -> WR real = 0.0
    // Afirmación interna falsa: afirma WR = 0.75 aislando el sesgo de win rate
    let independent = auditor.compute_independent_metrics();
    let internal_claimed_wr = 0.75;
    let disc = auditor.compute_internal_discrepancy(
        independent.calculated_sharpe,
        internal_claimed_wr,
        independent.calculated_max_drawdown,
    );
    assert!((disc - 0.75).abs() < 1e-9);

    // Publicación al registro omnisciente
    registry.set("internal_discrepancy", disc);

    let retrieved_disc = registry.get_value_or("internal_discrepancy", 0.0);
    assert!((retrieved_disc - 0.75).abs() < 1e-9);

    // Evaluación por el tribunal con umbrales canónicos
    let verdict = evaluate_tribunal_verdict(retrieved_disc, &auditor.thresholds);
    assert_eq!(verdict, TribunalVerdict::AutoengañoDetectado);
}
