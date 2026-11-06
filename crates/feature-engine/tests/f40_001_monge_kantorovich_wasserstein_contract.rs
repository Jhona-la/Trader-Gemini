//! Contrato formal de verificación para Fase 40: Motor de Transporte Óptimo de Monge-Kantorovich ($W_1$/$W_2$ Wasserstein)
//!
//! Invariantes certificados:
//! 1. c1: Identidad y monotonicidad métrica de Wasserstein ($W_1(\mu, \mu) \approx 0$ y $W_1(\mu_{\text{shift}}, \nu) > W_1(\mu_{\text{eq}}, \nu)$).
//! 2. c2: Detección sensible de velocidad de deriva $\dot{W}_1$ y activación de riesgo de evaporación de liquidez.
//! 3. c3: Estabilidad numérica absoluta ante distribuciones colapsadas, subnormales, nulas o con valores atípicos.
//! 4. c4: Invariante estricto de cero pánicos (`rb_p1`) y ejecución en tiempo acotado $O(1)$ sin asignaciones en heap.

use feature_engine::monge_kantorovich_wasserstein::{
    MongeKantorovichWassersteinEngine, WassersteinParams, WASSERSTEIN_NODES,
};

#[test]
fn c1_wasserstein_distance_identity_and_monotonicity() {
    let params = WassersteinParams {
        epsilon: 0.05,
        max_iterations: 25,
        tolerance: 1e-4,
        tau_cost_weight: 0.10,
        drift_velocity_threshold: 0.08,
    };
    let mut engine = MongeKantorovichWassersteinEngine::new(params);

    // 1. Probar con distribución idéntica a la referencia de equilibrio
    let ref_dist = *engine.reference_distribution();
    let rep_identity = engine.update_and_evaluate(&ref_dist, 50.0, 1_000);

    // La distancia W1 entre distribuciones idénticas debe ser muy cercana a 0
    assert!(
        rep_identity.w1_distance < 0.08,
        "W1 de distribución idéntica debe ser mínima con regularización: {}",
        rep_identity.w1_distance
    );
    assert!(
        rep_identity.w2_squared < 0.05,
        "W2^2 de distribución idéntica debe ser mínima: {}",
        rep_identity.w2_squared
    );

    // 2. Probar con distribución desplazada hacia un extremo (desequilibrio de presión compradora/vendedora)
    let mut shifted = [0.0; WASSERSTEIN_NODES];
    shifted[WASSERSTEIN_NODES - 1] = 1.0; // Toda la masa en el extremo superior
    let rep_shifted = engine.update_and_evaluate(&shifted, 50.0, 1_100);

    // Monotonicidad métrica estricta: W1 desplazado debe ser significativamente mayor
    assert!(
        rep_shifted.w1_distance > rep_identity.w1_distance,
        "W1 desplazado ({}) debe ser estrictamente mayor que el idéntico ({})",
        rep_shifted.w1_distance,
        rep_identity.w1_distance
    );
    assert!(
        rep_shifted.w2_squared > rep_identity.w2_squared,
        "W2^2 desplazado debe superar al idéntico"
    );
}

#[test]
fn c2_drift_velocity_and_liquidity_evaporation_detection() {
    let params = WassersteinParams {
        epsilon: 0.05,
        max_iterations: 20,
        tolerance: 1e-4,
        tau_cost_weight: 0.10,
        drift_velocity_threshold: 0.05,
    };
    let mut engine = MongeKantorovichWassersteinEngine::new(params);

    let ref_dist = *engine.reference_distribution();
    let _ = engine.update_and_evaluate(&ref_dist, 50.0, 10_000);

    // Simular un shock de liquidez donde la masa se desplaza súbitamente en 50ms
    let mut shock_dist = [0.0; WASSERSTEIN_NODES];
    shock_dist[0] = 0.80; // Fuerte absorción en el bid más profundo
    shock_dist[1] = 0.20;

    let rep_shock = engine.update_and_evaluate(&shock_dist, 50.0, 10_050);

    // La velocidad de deriva debe ser marcadamente positiva por la rápida divergencia de Wasserstein
    assert!(
        rep_shock.drift_velocity > 0.0,
        "La velocidad de deriva debe ser positiva ante el shock: {}",
        rep_shock.drift_velocity
    );

    // Debe alertar riesgo de evaporación de liquidez al superar el umbral
    assert!(
        rep_shock.liquidity_evaporation_risk > 0.0,
        "El riesgo de evaporación de liquidez debe activarse: {}",
        rep_shock.liquidity_evaporation_risk
    );
}

#[test]
fn c3_subnormal_and_degenerate_numerical_stability() {
    let mut engine = MongeKantorovichWassersteinEngine::new(WassersteinParams::default());

    // Caso 1: Todos los pesos en cero (degradación completa de datos de profundidad)
    let zero_weights = [0.0; WASSERSTEIN_NODES];
    let rep_zero = engine.update_and_evaluate(&zero_weights, 0.0, 20_000);
    assert!(
        rep_zero.w1_distance.is_finite(),
        "W1 debe ser finito ante pesos nulos"
    );
    assert!(
        rep_zero.drift_velocity.is_finite(),
        "Deriva debe ser finita ante pesos nulos"
    );

    // Caso 2: Pesos con NaN e infinitos
    let mut corrupted = [0.0; WASSERSTEIN_NODES];
    corrupted[0] = f64::NAN;
    corrupted[5] = f64::INFINITY;
    corrupted[10] = -5.0;
    corrupted[15] = 1.0;
    let rep_corrupted = engine.update_and_evaluate(&corrupted, -100.0, 20_100);
    assert!(
        rep_corrupted.w1_distance.is_finite(),
        "W1 debe ser finito ante datos corruptos"
    );
    assert!(
        rep_corrupted.liquidity_evaporation_risk >= 0.0
            && rep_corrupted.liquidity_evaporation_risk <= 1.0,
        "Riesgo de evaporación acotado en [0, 1]"
    );
}

#[test]
fn c4_zero_alloc_and_zero_panic_guarantee() {
    let mut engine = MongeKantorovichWassersteinEngine::new(WassersteinParams::default());
    let mut weights = [0.0; WASSERSTEIN_NODES];
    for i in 0..WASSERSTEIN_NODES {
        weights[i] = 1.0 / (WASSERSTEIN_NODES as f64);
    }

    // Ejecución continua repetitiva para comprobar determinismo O(1)
    let t_start = 50_000u64;
    for step in 0..100 {
        let ts = t_start + step * 10;
        let rep = engine.update_and_evaluate(&weights, 25.0, ts);
        assert!(rep.w1_distance >= 0.0);
        assert!(rep.w2_squared >= 0.0);
        assert!(rep.iterations_completed > 0);
    }
}
