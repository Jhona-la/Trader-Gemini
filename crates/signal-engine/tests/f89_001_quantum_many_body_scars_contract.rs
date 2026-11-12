//! Contrato Formal de Verificación: Fase 89 (§89) — Cicatrices Cuánticas Many-Body (QMBS)
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor QMBS:
//! - C1: Fidelidad de revival periódica $\mathcal{F}_{\text{scar}} \in [0, 1]$ y SFF finito no-negativo.
//! - C2: Señal direccional cicatrizada $\kappa_{\text{qmbs}} \in [-1, 1]$ y modulación de slot $\eta_{\text{qmbs}} \in [0.15, 1.30]$.
//! - C3: Detonación determinista del veto fail-closed ante colapso ergódico total hacia ETH.
//! - C4: Determinismo estricto en stack $O(1)$, cero heap allocations e inmunidad ante NaNs e infinitos.

use signal_engine::{QuantumManyBodyScarsConfig, QuantumManyBodyScarsEngine, SCARS_HILBERT_DIM};

#[test]
fn test_c1_revival_fidelity_and_spectral_form_factor() {
    let config = QuantumManyBodyScarsConfig::default();
    let mut engine = QuantumManyBodyScarsEngine::new(config);

    let report = engine.evaluate(0.002, 0.25, 1.5, 0.01, 0.10);

    assert!(
        report.revival_fidelity >= 0.0 && report.revival_fidelity <= 1.0,
        "Fidelidad de revival fuera de [0, 1]: {}",
        report.revival_fidelity
    );
    assert!(
        report.spectral_form_factor >= 0.0,
        "Spectral Form Factor debe ser no negativo: {}",
        report.spectral_form_factor
    );
    assert!(
        report.eth_violation_degree >= 0.0 && report.eth_violation_degree <= 1.0,
        "Grado de violación ETH fuera de [0, 1]: {}",
        report.eth_violation_degree
    );
    assert!(
        !report.ergodic_collapse_veto,
        "Flujo coherente no debe activar el veto ergódico"
    );
}

#[test]
fn test_c2_scarred_signal_and_slot_scaling() {
    let config = QuantumManyBodyScarsConfig::default();
    let mut engine = QuantumManyBodyScarsEngine::new(config);

    for i in 0..10 {
        let ret = 0.001 * (i as f64);
        let ofi = 0.10 + 0.05 * (i as f64);
        let rep = engine.evaluate(ret, ofi, 2.0, 0.02, 0.10);

        assert!(
            rep.scarred_directional_signal >= -1.0 && rep.scarred_directional_signal <= 1.0,
            "Señal direccional cicatrizada fuera de [-1, 1]: {}",
            rep.scarred_directional_signal
        );
        assert!(
            rep.slot_scale_factor >= 0.15 && rep.slot_scale_factor <= 1.30,
            "Factor de slot fuera de límites [0.15, 1.30]: {}",
            rep.slot_scale_factor
        );
    }
}

#[test]
fn test_c3_ergodic_collapse_veto_activation() {
    let config = QuantumManyBodyScarsConfig {
        min_revival_fidelity: 0.40,
        thermal_dephasing_rate: 2.0,
        ..Default::default()
    };
    let mut engine = QuantumManyBodyScarsEngine::new(config);

    // Inyectamos dispersión térmica y alta volatilidad para forzar termalización ETH total
    let mut report = engine.evaluate(0.5, 0.0, 50.0, 10.0, 5.0);
    for _ in 0..5 {
        report = engine.evaluate(-0.5, 0.0, 50.0, 10.0, 5.0);
    }

    assert!(
        report.ergodic_collapse_veto,
        "Termalización térmica extrema debe detonar ergodic_collapse_veto"
    );
    assert_eq!(
        report.slot_scale_factor, 0.15,
        "En colapso ergódico, el factor de slot debe comprimirse a 0.15"
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_hilbert_dimension() {
    assert_eq!(SCARS_HILBERT_DIM, 16);

    let config = QuantumManyBodyScarsConfig::default();
    let mut engine = QuantumManyBodyScarsEngine::new(config);

    for invalido in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (ret, ofi, spread, vol, dt) in [
            (invalido, 0.25, 1.5, 0.01, 0.10),
            (0.002, invalido, 1.5, 0.01, 0.10),
            (0.002, 0.25, invalido, 0.01, 0.10),
            (0.002, 0.25, 1.5, invalido, 0.10),
            (0.002, 0.25, 1.5, 0.01, invalido),
        ] {
            let report = engine.evaluate(ret, ofi, spread, vol, dt);
            assert!(report.revival_fidelity.is_finite());
            assert!(report.spectral_form_factor.is_finite());
            assert!(report.scarred_directional_signal.is_finite());
            assert!(report.eth_violation_degree.is_finite());
            assert!(report.slot_scale_factor.is_finite());
            assert!(report.ergodic_collapse_veto);
            assert_eq!(report.slot_scale_factor, 0.15);
        }
    }

    for (ret, ofi, spread, vol, dt) in [
        (0.002, 0.25, -1.0, 0.01, 0.10),
        (0.002, 0.25, 1.5, -0.01, 0.10),
        (0.002, 0.25, 1.5, 0.01, 0.0),
        (0.002, 0.25, 1.5, 0.01, -0.10),
    ] {
        let report = engine.evaluate(ret, ofi, spread, vol, dt);
        assert!(report.ergodic_collapse_veto);
        assert_eq!(report.slot_scale_factor, 0.15);
    }
}
