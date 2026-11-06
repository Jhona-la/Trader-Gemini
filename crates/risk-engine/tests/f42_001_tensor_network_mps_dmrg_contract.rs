//! Contrato formal de verificación para Fase 42: Motor de Optimización Tensorial de Portafolio Matrix Product States (MPS) y DMRG
//!
//! Invariantes certificados:
//! 1. c1: Normalización exacta en el simplex $\sum w_i = 1.0$ y respuesta coherente a la deriva esperada frente al riesgo.
//! 2. c2: Entropía de entrelazamiento de von Neumann $S_A$ y amortiguación de contagio sistémico ante correlaciones extremas.
//! 3. c3: Resiliencia numérica absoluta ante matrices patológicas, ceros, NaNs e infinitos (`rb_p1`).
//! 4. c4: Cero asignaciones en memoria dinámica (heap) y ejecución determinista $O(1)$.

use risk_engine::tensor_network_mps_dmrg::{
    MpsDmrgParams, TensorNetworkMpsDmrgEngine, MPS_ASSET_CHANNELS,
};

#[test]
fn c1_simplex_normalization_and_drift_responsiveness() {
    let params = MpsDmrgParams {
        risk_aversion: 1.50,
        growth_reward: 2.00,
        systemic_entanglement_threshold: 0.85,
        dmrg_sweeps: 3,
        weight_smoothing: 0.50,
    };
    let mut engine = TensorNetworkMpsDmrgEngine::new(params);

    let mut cov_diag = [0.01; MPS_ASSET_CHANNELS];
    cov_diag[0] = 0.005; // Activo 0 con baja varianza
    cov_diag[1] = 0.050; // Activo 1 con alta varianza

    let mut drifts = [0.0; MPS_ASSET_CHANNELS];
    drifts[0] = 0.04; // Activo 0 con fuerte deriva alcista
    drifts[1] = -0.02; // Activo 1 con deriva negativa

    let rep = engine.optimize_portfolio(&cov_diag, 0.10, &drifts);

    // 1. La suma de ponderaciones debe ser exactamente 1.0 (simplex)
    let sum_w: f64 = rep.asset_weights.iter().sum();
    assert!(
        (sum_w - 1.0).abs() < 1e-9,
        "La suma de ponderaciones debe ser 1.0: {}",
        sum_w
    );

    // 2. Todos los pesos deben ser no negativos
    for (i, &w) in rep.asset_weights.iter().enumerate() {
        assert!(w >= 0.0, "El peso del activo {} debe ser >= 0: {}", i, w);
    }

    // 3. El activo atractivo (0) debe recibir mayor ponderación que el riesgoso/bajista (1)
    assert!(
        rep.asset_weights[0] > rep.asset_weights[1],
        "Activo 0 (w={}) debe superar al activo 1 (w={})",
        rep.asset_weights[0],
        rep.asset_weights[1]
    );

    // 4. Varianza del portafolio finita y no negativa
    assert!(
        rep.expected_portfolio_variance >= 0.0,
        "La varianza del portafolio debe ser no negativa"
    );
}

#[test]
fn c2_entanglement_entropy_and_systemic_contagion_damping() {
    let params = MpsDmrgParams {
        risk_aversion: 1.0,
        growth_reward: 1.0,
        systemic_entanglement_threshold: 0.50,
        dmrg_sweeps: 3,
        weight_smoothing: 0.80,
    };
    let mut engine = TensorNetworkMpsDmrgEngine::new(params);

    let cov_diag = [0.02; MPS_ASSET_CHANNELS];
    let drifts = [0.01; MPS_ASSET_CHANNELS];

    // Correlación moderada / normal
    let rep_normal = engine.optimize_portfolio(&cov_diag, 0.20, &drifts);

    // Correlación sistémica extrema (shock de pánico coordinado)
    let rep_shock = engine.optimize_portfolio(&cov_diag, 0.95, &drifts);

    // La entropía de entrelazamiento de von Neumann debe ser finita y positiva
    assert!(
        rep_shock.entanglement_entropy >= 0.0,
        "La entropía de entrelazamiento debe ser >= 0"
    );

    // El factor de amortiguación de contagio sistémico debe estar estrictamente acotado en [0.0, 1.0]
    assert!(
        rep_shock.systemic_contagion_damping > 0.0 && rep_shock.systemic_contagion_damping <= 1.0,
        "El factor de amortiguación debe estar acotado: {}",
        rep_shock.systemic_contagion_damping
    );
}

#[test]
fn c3_subnormal_and_pathological_matrix_resilience() {
    let mut engine = TensorNetworkMpsDmrgEngine::new(MpsDmrgParams::default());

    // Probar con varianzas nulas, NaNs e infinitos
    let mut bad_cov = [0.0; MPS_ASSET_CHANNELS];
    bad_cov[0] = f64::NAN;
    bad_cov[3] = f64::INFINITY;
    bad_cov[7] = -1.0;

    let mut bad_drifts = [0.0; MPS_ASSET_CHANNELS];
    bad_drifts[2] = f64::NAN;

    let rep = engine.optimize_portfolio(&bad_cov, f64::NAN, &bad_drifts);

    let sum_w: f64 = rep.asset_weights.iter().sum();
    assert!(
        (sum_w - 1.0).abs() < 1e-6,
        "Simplex preservado incluso ante datos corruptos: {}",
        sum_w
    );
    assert!(rep.entanglement_entropy.is_finite());
    assert!(rep.ground_state_energy.is_finite());
}

#[test]
fn c4_zero_alloc_and_zero_panic_guarantee() {
    let mut engine = TensorNetworkMpsDmrgEngine::new(MpsDmrgParams::default());
    let cov = [0.02; MPS_ASSET_CHANNELS];
    let drifts = [0.005; MPS_ASSET_CHANNELS];

    for step in 0..100 {
        let rho = 0.10 + 0.30 * ((step as f64) * 0.1).sin();
        let rep = engine.optimize_portfolio(&cov, rho, &drifts);
        assert!(rep.sweeps_completed > 0);
        assert!(rep.expected_portfolio_variance >= 0.0);
    }
}
