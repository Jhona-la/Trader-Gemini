//! Suite de Contratos Formales F19-001 — Causalidad Dirigida Spot $\to$ Futuros (`SpotFuturesLeadLagEngine`),
//! Cazador Bayesiano de Liquidez Oculta L3 (`IcebergHunterL3Engine`) y Formalismo de Dos Vectores de Estado ABL / Weak Values.
//!
//! Verifica las 6 propiedades físico-matemáticas de la Fase 19:
//! - C1: Detección exacta de liderazgo causal cuando Spot antecede en 1 paso a Futuros ($\text{Corr}(S_{t-1}, F_t) \approx 1.0$) y viceversa.
//! - C2: Inferencia Bayesiana monótona en `IcebergHunterL3Engine` ante recargas ocultas sucesivas ($Q_{\text{exec}} > V_{\text{vis}}$).
//! - C3: Coincidencia del Valor Débil $A_w$ con el valor esperado cuántico estándar cuando $|\Phi_{\text{post}}\rangle = |\Psi_{\text{pre}}\rangle$.
//! - C4: Amplificación Cuántica Anómala de Aharonov-Albert-Vaidman ($|A_w| > \max_k |a_k|$) cuando $|\Psi_{\text{pre}}\rangle$ y $|\Phi_{\text{post}}\rangle$ son casi ortogonales.
//! - C5: Conservación unitaria y simetría temporal de la distribución de probabilidad de Aharonov-Bergmann-Lebowitz ($\sum_k P_{\text{ABL}}(a_k) \equiv 1.0$).
//! - C6: Inmunidad total frente a entradas `NaN` / `±Inf` en `SpotFuturesAblIcebergMasterEngine`.

use feature_engine::{
    AharonovBergmannLebowitzEngine, IcebergHunterL3Engine, SpotFuturesAblIcebergMasterEngine,
    SpotFuturesAblRegime, SpotFuturesLeadLagEngine, ABL_STATE_DIM, LEAD_LAG_WINDOW,
};

#[test]
fn c1_spot_futures_directed_lead_lag_causality() {
    let engine = SpotFuturesLeadLagEngine;
    // Spot genera la señal en t = 0..6, y Futuros la replica con 1 paso de retardo en t = 1..7
    let spot = [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, -0.01, 0.0];
    let futures = [0.0, 0.02, -0.01, 0.03, 0.01, -0.02, 0.04, -0.01];

    let rep = engine.evaluate_lead_lag(&spot, &futures, 100.0, 100.15);
    assert!(
        (rep.spot_leads_futures_corr - 1.0).abs() < 1e-12,
        "Spot leading futures by 1 lag must yield correlation 1.0, got {}",
        rep.spot_leads_futures_corr
    );
    assert!(rep.net_spot_leadership > 0.80);
    assert!((rep.relative_basis_bps - 15.0).abs() < 1e-10);
}

#[test]
fn c2_iceberg_hunter_l3_monotonic_bayesian_posterior() {
    let engine = IcebergHunterL3Engine::default();
    let v_init = 10.0;

    // Sin ejecución oculta (ejecutado 4, quedan 6 visibles): exceso == 0
    let rep_normal = engine.evaluate_level(v_init, 4.0, 6.0);
    assert!((rep_normal.replenishment_excess_ratio - 0.0).abs() < 1e-14);
    assert!(rep_normal.iceberg_posterior_prob < engine.prior_probability);

    // Con 1 recarga completa (ejecutado 15, aún quedan 8 visibles => total suministrado 23 vs 10 iniciales)
    let rep_refill1 = engine.evaluate_level(v_init, 15.0, 8.0);
    assert!(rep_refill1.replenishment_excess_ratio > 1.0);
    assert!(rep_refill1.iceberg_posterior_prob > 0.70);
    assert_eq!(rep_refill1.detected_refill_count, 1);

    // Con 4 recargas ocultas (ejecutado 45, aún quedan 9 visibles)
    let rep_refill4 = engine.evaluate_level(v_init, 45.0, 9.0);
    assert!(rep_refill4.iceberg_posterior_prob > rep_refill1.iceberg_posterior_prob);
    assert!(rep_refill4.iceberg_posterior_prob > 0.99);
    assert!(rep_refill4.estimated_hidden_reserve > 40.0);
}

#[test]
fn c3_abl_weak_value_matches_standard_expectation_when_pre_equals_post() {
    let engine = AharonovBergmannLebowitzEngine {
        overlap_regularization: 1e-4,
    };
    let state = [0.6, 0.8, 0.0, 0.0];
    let obs = [-1.0, 1.0, 2.0, -2.0];

    let rep = engine.evaluate_weak_measurement(&state, &state, &obs);
    // <Psi | A | Psi> = 0.36 * (-1) + 0.64 * (1) = 0.28
    assert!((rep.standard_expectation_value - 0.28).abs() < 1e-12);
    assert!((rep.weak_value_aw - 0.28).abs() < 1e-6);
    assert!((rep.pre_post_overlap - 1.0).abs() < 1e-12);
    assert!(!rep.is_anomalous_amplification);
}

#[test]
fn c4_aav_anomalous_weak_value_amplification_outside_eigenvalue_spectrum() {
    let engine = AharonovBergmannLebowitzEngine {
        overlap_regularization: 0.01,
    };
    // Observable con autovalores acotados estrictamente en [-1.0, 1.0]
    let obs = [1.0, -1.0, 0.0, 0.0];
    // Estados casi ortogonales: Psi = (1/sqrt(2), 1/sqrt(2)), Phi = (1/sqrt(2) + delta, -1/sqrt(2) + delta)
    // <Phi | Psi> = sqrt(2)*delta (pequeño), mientras <Phi | A | Psi> = 1.0 => A_w >> 1.0
    let psi = [1.0, 1.0, 0.0, 0.0];
    let phi = [1.10, -0.90, 0.0, 0.0];

    let rep = engine.evaluate_weak_measurement(&psi, &phi, &obs);
    assert!(
        rep.weak_value_aw > 1.0,
        "AAV weak value must amplify beyond max eigenvalue 1.0, got {}",
        rep.weak_value_aw
    );
    assert!(rep.is_anomalous_amplification);
}

#[test]
fn c5_abl_time_symmetric_probability_conservation() {
    let engine = AharonovBergmannLebowitzEngine::default();
    let psi = [0.5, -0.3, 0.7, 0.2];
    let phi = [0.2, 0.8, -0.4, 0.5];
    let obs = [1.0, 2.0, 3.0, 4.0];

    let rep_forward = engine.evaluate_weak_measurement(&psi, &phi, &obs);
    let rep_time_reversed = engine.evaluate_weak_measurement(&phi, &psi, &obs);

    let sum_p: f64 = rep_forward.abl_probabilities.iter().sum();
    assert!((sum_p - 1.0).abs() < 1e-14);

    // Simetría temporal exacta de Aharonov-Bergmann-Lebowitz: intercambiar Psi_pre <-> Phi_post deja P_ABL invariante
    for i in 0..ABL_STATE_DIM {
        assert!(
            (rep_forward.abl_probabilities[i] - rep_time_reversed.abl_probabilities[i]).abs()
                < 1e-14
        );
    }
}

#[test]
fn c6_nan_inf_adversarial_immunity_in_master_engine() {
    let master = SpotFuturesAblIcebergMasterEngine::default();
    let bad_ret = [f64::NAN; LEAD_LAG_WINDOW];
    let bad_state = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0];

    let rep = master.evaluate_cycle(
        &bad_ret,
        &bad_ret,
        f64::NAN,
        f64::INFINITY,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        &bad_state,
        &bad_state,
        &bad_state,
    );

    assert!(rep.lead_lag.net_spot_leadership.is_finite());
    assert!(rep.iceberg.iceberg_posterior_prob.is_finite());
    assert!(rep.abl.weak_value_aw.is_finite());
    assert!(rep.unified_conviction.is_finite());
    assert!(matches!(
        rep.regime,
        SpotFuturesAblRegime::SpotDrivenOrganicImpulse
            | SpotFuturesAblRegime::HiddenIcebergAbsorptionWall
            | SpotFuturesAblRegime::PerpFrothSqueezeFragility
    ));
}
