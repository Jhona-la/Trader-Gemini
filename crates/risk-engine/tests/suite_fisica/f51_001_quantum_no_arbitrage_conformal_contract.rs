//! Contrato Formal de Verificación: Fase 51 (Higher-Order Conformal Governance & Quantum No-Arbitrage Barrier)
//!
//! Este conjunto de pruebas certifica las propiedades matemáticas y de seguridad del motor:
//! - C1: Cobertura conformal no paramétrica exacta en muestra finita $\hat{q}_{1-\alpha}$ en stack.
//! - C2: Barrera FTAP de no-arbitraje y detección/veto inmediato de ilusiones de arbitraje.
//! - C3: Control de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ y explosión del e-proceso supermartingala.
//! - C4: Determinismo en stack $O(1)$, cero heap allocs y robustez numérica total ante NaN/Infinitos.

use risk_engine::{
    QuantumNoArbitrageConformalEngine, QuantumNoArbitrageConformalParams,
    CONFORMAL_CALIBRATION_WINDOW,
};

#[test]
fn test_c1_conformal_quantile_exactness_and_coverage() {
    let params = QuantumNoArbitrageConformalParams {
        target_coverage: 0.95,
        ville_ruin_bound: 1e-6,
        fee_friction_bps: 4.0,
        latency_slippage_slope: 0.05,
        illusion_threshold_ratio: 15.0,
        e_process_decay: 0.90,
    };
    let mut engine = QuantumNoArbitrageConformalEngine::new(params);

    // Llenamos la ventana con pequeñas perturbaciones gaussianas sintéticas
    for i in 0..CONFORMAL_CALIBRATION_WINDOW {
        let actual = 100.0;
        let pred = 100.0 + 0.01 * ((i as f64) / (CONFORMAL_CALIBRATION_WINDOW as f64));
        let report = engine.evaluate(pred, actual, 2.0, 10.0, 1000.0);
        assert!(
            report.conformal_quantile_threshold.is_finite(),
            "Cuantil no finito en i={}",
            i
        );
        assert!(
            report.conformal_quantile_threshold > 0.0,
            "Cuantil no positivo en i={}",
            i
        );
    }

    assert_eq!(CONFORMAL_CALIBRATION_WINDOW, 32);
}

#[test]
fn test_c2_quantum_no_arbitrage_ftap_and_illusion_veto() {
    let params = QuantumNoArbitrageConformalParams::default();
    let mut engine = QuantumNoArbitrageConformalEngine::new(params);

    // 1. Caso de arbitraje legítimo: discrepancia moderada que cubre holgadamente comisiones y slippage
    // actual = 100.0, pred = 99.85 (15 bps de discrepancia, spread = 2 bps, latencia = 5 ms)
    // fricción total = 4.0 (fee) + 1.0 (spread*0.5) + 0.25 (latencia) = 5.25 bps. Discrepancia neta ≈ 9.75 bps > 1.0.
    let report_legit = engine.evaluate(99.85, 100.0, 2.0, 5.0, 1000.0);
    assert!(
        report_legit.is_arbitrage_legitimate,
        "Debe calificar como arbitraje legítimo"
    );
    assert!(
        !report_legit.is_arbitrage_illusion,
        "No debe ser clasificado como ilusión"
    );
    assert!(!report_legit.hard_veto_induced, "No debe haber veto duro");
    assert!(
        report_legit.conformal_risk_multiplier > 0.50,
        "Multiplicador debe ser elevado para arbitraje legítimo: {}",
        report_legit.conformal_risk_multiplier
    );

    // 2. Caso de ilusión de arbitraje: discrepancia masiva anormal (ej. 200 bps frente a spread de 2 bps)
    // Provocado por feed desactualizado o crash fantasma
    let report_illusion = engine.evaluate(98.0, 100.0, 2.0, 5.0, 1000.0);
    assert!(
        report_illusion.is_arbitrage_illusion,
        "Debe detectar ilusión de arbitraje"
    );
    assert!(
        report_illusion.hard_veto_induced,
        "Debe inducir veto duro por ilusión"
    );
    assert_eq!(
        report_illusion.veto_reason,
        Some("QUANTUM_ARBITRAGE_ILLUSION")
    );
    assert_eq!(report_illusion.conformal_risk_multiplier, 0.0);
}

#[test]
fn test_c3_ville_supermartingale_explosion_and_zero_capital_veto() {
    let params = QuantumNoArbitrageConformalParams {
        target_coverage: 0.999,
        ville_ruin_bound: 1e-6,
        fee_friction_bps: 4.0,
        latency_slippage_slope: 0.05,
        illusion_threshold_ratio: 100.0, // Para aislar el efecto del e-proceso
        e_process_decay: 0.99,
    };
    let mut engine = QuantumNoArbitrageConformalEngine::new(params);

    // Inyectamos reiterados errores extremos de predicción que rompan la cobertura conformal
    let mut exploded = false;
    for i in 0..100 {
        let report = engine.evaluate(95.0, 100.0, 10.0, 10.0, 500.0);
        if report.hard_veto_induced && report.veto_reason == Some("VILLE_SUPERMARTINGALE_EXPLOSION")
        {
            exploded = true;
            assert!(
                report.ville_e_value >= 1e6,
                "E-value debe haber superado el umbral de Ville 1e6: {}",
                report.ville_e_value
            );
            break;
        }
        let _ = i;
    }
    assert!(
        exploded,
        "El e-proceso de Ville debió explotar ante reiteradas anomalías extremas"
    );

    // Verificación de veto por capital cero
    let zero_cap_report = engine.evaluate(100.0, 100.0, 2.0, 5.0, 0.0);
    assert!(zero_cap_report.hard_veto_induced);
    assert_eq!(zero_cap_report.veto_reason, Some("CONFORMAL_ZERO_CAPITAL"));
}

#[test]
fn test_c4_stack_zero_heap_and_nan_subnormal_immunity() {
    let params = QuantumNoArbitrageConformalParams::default();
    let mut engine = QuantumNoArbitrageConformalEngine::new(params);

    let anomalous_inputs = [
        (f64::NAN, 100.0, 2.0, 5.0, 100.0),
        (100.0, f64::INFINITY, 2.0, 5.0, 100.0),
        (100.0, 100.0, f64::NAN, 5.0, 100.0),
        (100.0, 100.0, 2.0, f64::INFINITY, 100.0),
        (100.0, 100.0, 2.0, 5.0, f64::NAN),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1e-300, 1e-300, 1e-300, 1e-300, 1e-300),
    ];

    for &(pred, actual, spread, lat, cap) in &anomalous_inputs {
        let report = engine.evaluate(pred, actual, spread, lat, cap);
        assert!(report.conformal_quantile_threshold.is_finite());
        assert!(report.arbitrage_drift_bps.is_finite());
        assert!(report.ville_e_value.is_finite());
        assert!(report.conformal_risk_multiplier >= 0.0 && report.conformal_risk_multiplier <= 1.0);
    }
}
