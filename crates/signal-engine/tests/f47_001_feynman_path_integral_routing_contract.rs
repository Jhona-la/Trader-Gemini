//! ⚛️ CONTRATO FORMAL F47-001: ENRUTAMIENTO CUÁNTICO POR INTEGRAL DE CAMINOS DE FEYNMAN
//!
//! # ESPECIFICACIÓN DEL CONTRATO:
//! - c1: Principio de Mínima Acción Clásica y consistencia de trayectorias.
//! - c2: Conservación de probabilidad cuántica $\sum P_m \approx 1.0$ y coherencia $C_{\text{coh}} \in [0.0, 1.0]$.
//! - c3: Inmunidad total fail-closed ante inputs no finitos (NaN, ±Inf) o cero liquidez.
//! - c4: Determinismo estricto bit a bit y modulación acotada $\chi_{\text{routing}} \in [0.10, 1.50]$.

use signal_engine::feynman_path_integral_routing::{
    FeynmanPathIntegralRoutingEngine, FeynmanRoutingParams, ROUTING_PATH_CANDIDATES,
};

#[test]
fn test_c1_hamiltonian_minimal_action_consistency() {
    let engine = FeynmanPathIntegralRoutingEngine::default();

    // 1. Caso libro profundo con OFI positivo (favorable a ejecución temprana)
    let rep_favorable = engine.evaluate(10.0, 1.0, 0.02);
    assert!(rep_favorable.action_min.is_finite());
    assert!(rep_favorable.action_mean >= rep_favorable.action_min);
    assert!(rep_favorable.best_path_index < ROUTING_PATH_CANDIDATES);

    // 2. Caso libro ilíquido con OFI negativo adverso
    let rep_adverso = engine.evaluate(0.1, -1.0, 0.05);
    assert!(rep_adverso.action_min.is_finite());
    assert!(rep_adverso.action_mean >= rep_adverso.action_min);
    assert!(rep_adverso.best_path_index < ROUTING_PATH_CANDIDATES);
}

#[test]
fn test_c2_quantum_probability_conservation_and_coherence() {
    let engine = FeynmanPathIntegralRoutingEngine::default();

    let scenarios = [
        (1.0, 0.0, 0.01),
        (5.0, 0.8, 0.02),
        (0.5, -0.9, 0.04),
        (20.0, 0.1, 0.005),
    ];

    for &(depth, ofi, vol) in &scenarios {
        let rep = engine.evaluate(depth, ofi, vol);

        // Conservación de probabilidad cuántica
        let sum_p: f64 = rep.path_probabilities.iter().sum();
        assert!(
            (sum_p - 1.0).abs() < 1e-6,
            "Suma de probabilidades debe ser 1.0: {}",
            sum_p
        );

        // Coherencia cuántica acotada
        assert!(
            rep.constructive_coherence >= 0.0 && rep.constructive_coherence <= 1.0,
            "Coherencia fuera de rango [0, 1]: {}",
            rep.constructive_coherence
        );

        // Fracción óptima inmediata finita y acotada
        assert!(
            rep.optimal_slice_fraction >= 0.0 && rep.optimal_slice_fraction <= 1.0,
            "Fracción óptima fuera de [0, 1]: {}",
            rep.optimal_slice_fraction
        );
    }
}

#[test]
fn test_c3_immunity_to_nan_inf_and_singularities() {
    let engine = FeynmanPathIntegralRoutingEngine::default();

    let degenerates = [
        (f64::NAN, 0.0, 0.01),
        (1.0, f64::INFINITY, 0.01),
        (0.0, 0.0, f64::NAN),
        (-5.0, -100.0, -0.01),
        (f64::NEG_INFINITY, f64::NAN, f64::INFINITY),
    ];

    for &(depth, ofi, vol) in &degenerates {
        let rep = engine.evaluate(depth, ofi, vol);
        assert!(rep.optimal_slice_fraction.is_finite());
        assert!(rep.constructive_coherence.is_finite());
        assert!(rep.routing_urgency_modulator >= 0.10 && rep.routing_urgency_modulator <= 1.50);
        assert!(rep.action_min.is_finite());
        assert!(rep.action_mean.is_finite());
    }
}

#[test]
fn test_c4_determinism_and_modulator_bounds() {
    let params = FeynmanRoutingParams {
        hbar_routing: 1.5,
        mass: 0.8,
        slip_penalty: 0.4,
        ofi_coupling: 0.25,
        coherence_threshold: 0.25,
    };
    let engine = FeynmanPathIntegralRoutingEngine::new(params);

    let depth = 2.5;
    let ofi = 0.45;
    let vol = 0.015;

    let rep1 = engine.evaluate(depth, ofi, vol);
    let rep2 = engine.evaluate(depth, ofi, vol);

    assert_eq!(
        rep1, rep2,
        "Evaluaciones idénticas deben ser exactamente iguales bit a bit"
    );
    assert!(rep1.routing_urgency_modulator >= 0.10 && rep1.routing_urgency_modulator <= 1.50);
}
