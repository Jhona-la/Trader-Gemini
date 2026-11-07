//! ⚛️ CONTRATO FORMAL F48-001: METAMORFOSIS TOPOLÓGICA DEL GRAFO VIVO AUTÓNOMO
//!
//! # ESPECIFICACIÓN DEL CONTRATO:
//! - c1: Estabilidad asintótica de Lyapunov y cota de Gershgorin ($\|A\|_{\infty} \le 0.95$).
//! - c2: Poda de ruido dura (Noise Pruning) bajo umbral $\theta_{\text{prune}}$.
//! - c3: Inmunidad fail-closed ante activaciones corruptas (NaN, $\pm\infty$) o desempeño degenerado.
//! - c4: Determinismo estricto bit a bit y modulación acotada $\mu_{\text{plasticity}} \in [0.10, 1.50]$.

use god_engine_core::topological_metamorphosis::{
    METAMORPHOSIS_NODES, TopologicalMetamorphosisEngine, TopologicalMetamorphosisParams,
};

#[test]
fn test_c1_lyapunov_stability_and_gershgorin_bound() {
    let params = TopologicalMetamorphosisParams {
        learning_rate: 0.10,
        decay_rate: 0.005,
        pruning_threshold: 0.01,
        stability_margin: 0.90,
    };
    let mut engine = TopologicalMetamorphosisEngine::new(params);

    // Estimular agresivamente con co-activación completa durante 100 pasos
    let high_acts = [1.0; METAMORPHOSIS_NODES];
    let perf = 5.0; // Desempeño excelente que amplifica el aprendizaje

    for _ in 0..100 {
        let rep = engine.update(&high_acts, perf);
        assert!(
            rep.spectral_radius_bound <= 0.90 + 1e-9,
            "Cota de Gershgorin excedida: {} > 0.90",
            rep.spectral_radius_bound
        );
        assert!(rep.max_edge_weight <= 0.90);
    }
}

#[test]
fn test_c2_noise_pruning_under_threshold() {
    let thresh = 0.05;
    let params = TopologicalMetamorphosisParams {
        learning_rate: 0.01,
        decay_rate: 0.05,
        pruning_threshold: thresh,
        stability_margin: 0.95,
    };
    let mut engine = TopologicalMetamorphosisEngine::new(params);

    // Activaciones nulas para forzar decaimiento homeostático hacia el cero
    let zero_acts = [0.0; METAMORPHOSIS_NODES];
    for _ in 0..50 {
        engine.update(&zero_acts, 0.0);
    }

    let rep = engine.update(&zero_acts, 0.0);
    // Verificar que todas las aristas no nulas superan estrictamente el umbral
    for i in 0..METAMORPHOSIS_NODES {
        for j in 0..METAMORPHOSIS_NODES {
            let w = rep.adjacency_matrix[i][j].abs();
            if w > 0.0 {
                assert!(
                    w >= thresh - 1e-9,
                    "Arista sin podar por debajo del umbral: {} < {}",
                    w,
                    thresh
                );
            }
        }
    }
}

#[test]
fn test_c3_immunity_to_nan_inf_and_corrupt_inputs() {
    let mut engine = TopologicalMetamorphosisEngine::default();

    let mut corrupt_acts = [0.0; METAMORPHOSIS_NODES];
    corrupt_acts[0] = f64::NAN;
    corrupt_acts[1] = f64::INFINITY;
    corrupt_acts[2] = f64::NEG_INFINITY;
    corrupt_acts[5] = 100.0; // Fuera del rango [-1, 1]

    let rep = engine.update(&corrupt_acts, f64::NAN);
    assert!(rep.spectral_radius_bound.is_finite());
    assert!(rep.mean_connectivity.is_finite());
    assert!(rep.max_edge_weight.is_finite());
    assert!(rep.plasticity_modulator >= 0.10 && rep.plasticity_modulator <= 1.50);

    for i in 0..METAMORPHOSIS_NODES {
        for j in 0..METAMORPHOSIS_NODES {
            assert!(
                rep.adjacency_matrix[i][j].is_finite(),
                "Valor no finito en celda ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_c4_determinism_and_modulator_bounds() {
    let mut engine1 = TopologicalMetamorphosisEngine::default();
    let mut engine2 = TopologicalMetamorphosisEngine::default();

    let mut acts = [0.0; METAMORPHOSIS_NODES];
    for (i, val) in acts.iter_mut().enumerate() {
        *val = if i % 2 == 0 { 0.50 } else { -0.30 };
    }

    let rep1 = engine1.update(&acts, 1.5);
    let rep2 = engine2.update(&acts, 1.5);

    assert_eq!(
        rep1, rep2,
        "Dos ejecuciones idénticas deben producir exactamente el mismo reporte"
    );
    assert!(rep1.plasticity_modulator >= 0.10 && rep1.plasticity_modulator <= 1.50);
}
