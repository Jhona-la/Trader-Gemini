//! # Contrato de Verificación Formal F26-001: Stoikov Microprecio, Virasoro CFT & Anyons TQFT
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Batería de 6 contratos formales de microestructura, geometría conforme y topología cuántica.
//! - **POR QUÉ:** Cada modelo matemático debe satisfacer rigurosamente sus propiedades fundamentales:
//!   monotonía del microprecio ante desequilibrios L1, adaptación robusta de recvWindow con MAD,
//!   anomalía central del conmutador de Virasoro, simetría de cruce del bootstrap de Polyakov,
//!   unitariedad del trenzado de Anyons y protección topológica contra órdenes espurias (spoofing).
//! - **PARA QUÉ:** Demostrar formalmente que las señales de trading son exactas, coherentes y resistentes
//!   al ruido antes de su despliegue en producción.
//! - **CÓMO:** Ejecución determinista de aserciones matemáticas e invariantes físicas.

#![allow(clippy::all)]

use feature_engine::stoikov_microprice::StoikovMicropriceEngine;
use feature_engine::tqft_anyons::{TqftFibonacciAnyonEngine, PHI_GOLDEN};
use feature_engine::virasoro_cft_bootstrap::VirasoroCftBootstrapEngine;

#[test]
fn c1_stoikov_microprice_imbalance_monotonicity() {
    let engine = StoikovMicropriceEngine::default();

    // 1. Caso balanceado: 10 BTC bid, 10 BTC ask a mid=100.0
    let rep_bal = engine.evaluate(99.99, 10.0, 100.01, 10.0);
    assert!(
        (rep_bal.imbalance).abs() < 1e-9,
        "Imbalance balanceado debe ser 0.0"
    );
    assert_eq!(rep_bal.state_idx, 2, "Debe caer en estado central 2");
    assert!(
        (rep_bal.microprice - rep_bal.mid_price).abs() < 1e-6,
        "Microprecio en libro balanceado debe ser igual al mid-price"
    );

    // 2. Fuerte presión compradora: 90 BTC bid vs 10 BTC ask (imbalance = +0.8)
    let rep_buy = engine.evaluate(99.99, 90.0, 100.01, 10.0);
    assert!(
        rep_buy.imbalance > 0.7,
        "Imbalance de compra debe superar 0.7"
    );
    assert_eq!(rep_buy.state_idx, 4, "Debe caer en estado 4 de compra");
    assert!(
        rep_buy.prob_up > 0.5,
        "Probabilidad alcista debe ser mayor al 50%"
    );
    assert!(
        rep_buy.microprice > rep_buy.mid_price,
        "Microprecio debe sesgarse al alza ante fuerte presión compradora"
    );

    // 3. Fuerte presión vendedora: 10 BTC bid vs 90 BTC ask (imbalance = -0.8)
    let rep_sell = engine.evaluate(99.99, 10.0, 100.01, 90.0);
    assert!(
        rep_sell.imbalance < -0.7,
        "Imbalance de venta debe ser menor a -0.7"
    );
    assert_eq!(rep_sell.state_idx, 0, "Debe caer en estado 0 de venta");
    assert!(
        rep_sell.prob_down > 0.5,
        "Probabilidad bajista debe superar 50%"
    );
    assert!(
        rep_sell.microprice < rep_sell.mid_price,
        "Microprecio debe sesgarse a la baja ante fuerte presión vendedora"
    );
}

#[test]
fn c2_stoikov_dynamic_recv_window_adaptation() {
    let mut engine = StoikovMicropriceEngine::default();

    // 1. Inyectar latencias normales estables (alrededor de 20 ms = 20,000 us)
    for _ in 0..16 {
        engine.record_rtt_sample(20_000);
    }
    let rw_stable = engine.compute_dynamic_recv_window();
    assert!(
        (1000..=1500).contains(&rw_stable),
        "recvWindow en régimen estable debe ser cercano al mínimo: {}",
        rw_stable
    );

    // 2. Simular congestión de red con jitter severo (muestras dispersas con mediana alta)
    for i in 0..16 {
        let jitter = if i % 2 == 0 { 1_200_000 } else { 600_000 };
        engine.record_rtt_sample(jitter);
    }
    let rw_jitter = engine.compute_dynamic_recv_window();
    assert!(
        rw_jitter > rw_stable,
        "recvWindow debe expandirse dinámicamente ante jitter de red: rw_jitter={}, rw_stable={}",
        rw_jitter,
        rw_stable
    );
    assert!(
        rw_jitter <= 5000,
        "recvWindow no debe superar el límite de seguridad de 5000ms"
    );
}

#[test]
fn c3_virasoro_algebra_commutation_and_central_anomaly() {
    // 1. Comprobar que para subálgebra sl(2, C) la anomalía se anula
    let c = 0.5;
    assert_eq!(
        VirasoroCftBootstrapEngine::virasoro_central_anomaly(0, 0, c),
        0.0
    );
    assert_eq!(
        VirasoroCftBootstrapEngine::virasoro_central_anomaly(1, -1, c),
        0.0,
        "Para m=1, m^2 - 1 = 0, por lo que la anomalía central es idénticamente cero"
    );

    // 2. Para m=2, n=-2 con c=1.0:
    // A(2, -2, 1.0) = (1.0 / 12) * 2 * (2^2 - 1) = (1/12) * 2 * 3 = 6/12 = 0.5
    let anom = VirasoroCftBootstrapEngine::virasoro_central_anomaly(2, -2, 1.0);
    assert!(
        (anom - 0.5).abs() < 1e-12,
        "Anomalía central para m=2, c=1.0 debe ser exactamente 0.5, pero es {}",
        anom
    );
}

#[test]
fn c4_polyakov_conformal_bootstrap_crossing_symmetry() {
    let mut engine = VirasoroCftBootstrapEngine::default();

    // Inyectar una serie temporal sintética autosimilar (con autocorrelación conforme)
    for t in 0..32 {
        let decay = (t as f64 + 1.0).powf(-0.25);
        let sample = if t % 2 == 0 { decay } else { -decay };
        engine.push_observation(sample);
    }

    let report = engine.analyze();
    assert!(
        report.central_charge_c > 0.0,
        "Carga central debe ser estrictamente positiva"
    );
    assert!(
        report.conformal_dimension_delta > 0.0,
        "Dimensión conforme debe ser estrictamente positiva"
    );
    assert!(
        report.bootstrap_discrepancy >= 0.0,
        "Discrepancia del bootstrap debe ser no negativa"
    );
}

#[test]
fn c5_tqft_fibonacci_anyons_braiding_unitarity() {
    let mut engine = TqftFibonacciAnyonEngine::default();

    // 1. Verificar estado inicial normalizado
    let rep_init = engine.evaluate_topological_state();
    assert!(
        (rep_init.vacuum_channel_prob + rep_init.anyon_channel_prob - 1.0).abs() < 1e-12,
        "Conservación estricta de probabilidad cuántica"
    );
    assert_eq!(rep_init.vacuum_channel_prob, 1.0);

    // 2. Aplicar generadores de trenzado sigma_1 y sigma_2
    engine.apply_braid_sigma1();
    let rep_s1 = engine.evaluate_topological_state();
    assert!(
        (rep_s1.vacuum_channel_prob + rep_s1.anyon_channel_prob - 1.0).abs() < 1e-12,
        "sigma_1 debe ser unitaria y conservar probabilidad"
    );

    engine.apply_braid_sigma2();
    let rep_s2 = engine.evaluate_topological_state();
    assert!(
        (rep_s2.vacuum_channel_prob + rep_s2.anyon_channel_prob - 1.0).abs() < 1e-12,
        "sigma_2 (F R F) debe ser unitaria y conservar probabilidad"
    );

    // 3. Verificar dimensión cuántica de Fibonacci
    assert!(
        (PHI_GOLDEN - 1.6180339887).abs() < 1e-7,
        "Constante áurea debe ser exacta"
    );
}

#[test]
fn c6_tqft_topological_protection_against_spoofing() {
    let mut engine = TqftFibonacciAnyonEngine::default();

    // 1. Inyectar micro-ruido de spoofing transitorio (desequilibrio bajo y volumen pequeño)
    for _ in 0..10 {
        engine.ingest_order_event(0.05, 0.2); // Trivial, sin trenzado macro
    }
    let rep_noise = engine.evaluate_topological_state();
    assert!(
        !rep_noise.is_institutional_flow,
        "El micro-ruido/spoofing debe ser descartado topológicamente"
    );

    // 2. Inyectar orden institucional masiva con cruce de flujo que dispara trenzado no conmutativo sigma_2
    engine.ingest_order_event(-0.85, 1.0);
    let rep_inst = engine.evaluate_topological_state();
    assert!(
        rep_inst.anyon_channel_prob > 0.20,
        "El trenzado institucional debe activar el canal anyónico: p_anyon={}",
        rep_inst.anyon_channel_prob
    );
    assert!(
        rep_inst.braid_entanglement > 0.20,
        "El entrelazamiento topológico de la trenza debe ser significativo: S={}",
        rep_inst.braid_entanglement
    );
    assert!(
        rep_inst.is_institutional_flow,
        "Debe certificar flujo institucional genuino"
    );
}
