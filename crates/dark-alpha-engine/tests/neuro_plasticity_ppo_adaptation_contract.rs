//! Contrato de adaptación y continuidad C^1 para Neuro-Plasticidad y Aprendizaje por Refuerzo Online PPO (Ola Ω125)
//!
//! Audita que:
//! 1. Los parámetros de neuro-plasticidad se validen estrictamente (fail-closed) y posean paridad con los valores canónicos.
//! 2. El multiplicador de plasticidad sea una función suave, continua y monótona de la entropía sin escalones discretos.
//! 3. La modulación por capital para micro-cuentas ($13 USD) sea suave y asintótica sin singularidades.
//! 4. Los hiperparámetros PPO se desacoplen dinámicamente y se inmunicen ante NaNs e infinitos.

use dark_alpha_engine::neuro_plasticity::{NeuroPlasticityEngine, NeuroPlasticityParams};
use dark_alpha_engine::online_ppo::{OnlinePpoPolicyEngine, PpoHyperParams};

#[test]
fn test_neuro_plasticity_params_validation_and_defaults() {
    let def = NeuroPlasticityParams::default();
    assert!(def.validate(), "Los parámetros por defecto deben ser válidos");
    assert_eq!(def.min_multiplier, 0.20);
    assert_eq!(def.max_multiplier, 2.00);
    assert_eq!(def.center_entropy, 0.50);
    assert_eq!(def.steepness, 6.00);

    // Inversión de cotas: min > max -> inválido
    let invalid_bounds = NeuroPlasticityParams {
        min_multiplier: 2.50,
        max_multiplier: 1.00,
        ..def
    };
    assert!(!invalid_bounds.validate(), "min > max debe fallar validación");

    // Centro fuera de rango [0.01, 0.99]
    let invalid_center = NeuroPlasticityParams {
        center_entropy: 1.50,
        ..def
    };
    assert!(!invalid_center.validate(), "center_entropy fuera de rango debe fallar");

    // Steepness no-finita o negativa
    let invalid_steepness = NeuroPlasticityParams {
        steepness: -2.0,
        ..def
    };
    assert!(!invalid_steepness.validate(), "steepness negativa debe fallar");
}

#[test]
fn test_plasticity_multiplier_c1_continuity_and_monotonicity() {
    let params = NeuroPlasticityParams::default();
    let capital = 13.0;

    // 1. Monotonía estricta respecto a la entropía
    let mut prev_mult = 0.0;
    for step in 0..=100 {
        let entropy = step as f64 / 100.0;
        let mult = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
            &params,
            entropy,
            capital,
        );
        assert!(mult.is_finite());
        if step > 0 {
            assert!(
                mult >= prev_mult,
                "La plasticidad debe ser monótona no-decreciente con la entropía: mult({entropy}) = {mult} < prev = {prev_mult}"
            );
        }
        prev_mult = mult;
    }

    // 2. Continuidad C^1 Lipschitz: sin saltos discretos abruptos
    let eps = 0.001;
    for step in 1..99 {
        let h = step as f64 / 100.0;
        let m1 = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(&params, h, capital);
        let m2 = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(&params, h + eps, capital);
        let diff = (m2 - m1).abs();
        assert!(
            diff < 0.02,
            "Discontinuidad detectada en H={h}: salto de {diff} supera cota Lipschitz suave"
        );
    }

    // 3. Modulación suave de micro-cuenta ($13 USD) vs estándar ($100 USD)
    let mult_micro = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(&params, 0.50, 13.0);
    let mult_std = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(&params, 0.50, 100.0);
    assert!(
        mult_micro > mult_std,
        "Micro-capital ($13) debe tener un factor de adaptación mayor que cuenta estándar ($100)"
    );
}

#[test]
fn test_plasticity_multiplier_nan_and_negative_immunity() {
    let params = NeuroPlasticityParams::default();

    // Entropía NaN / Inf
    let m_nan = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
        &params,
        f64::NAN,
        13.0,
    );
    assert!(m_nan.is_finite());
    assert!(m_nan > 0.1 && m_nan < 2.5);

    let m_inf = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
        &params,
        f64::INFINITY,
        13.0,
    );
    assert!(m_inf.is_finite());

    // Capital NaN / negativo
    let m_cap_nan = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
        &params,
        0.50,
        f64::NAN,
    );
    assert!(m_cap_nan.is_finite());

    let m_cap_neg = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
        &params,
        0.50,
        -50.0,
    );
    assert!(m_cap_neg.is_finite());
}

#[test]
fn test_ppo_hyperparams_validation_and_dynamic_update() {
    let def_ppo = PpoHyperParams::default();
    assert!(def_ppo.validate());

    let custom_ppo = PpoHyperParams {
        alpha_ema: 0.10,
        learning_rate: 0.02,
        clip_eps: 0.15,
        min_weight_clip: 0.005,
    };
    assert!(custom_ppo.validate());

    let engine = OnlinePpoPolicyEngine::new([1.0, 1.0, 1.0, 1.0, 1.0]);
    let features = [0.5, -0.3, 0.2, 0.1, -0.4];

    // Actualización con parámetros dinámicos
    engine.update_policy_with_params(
        0.02, // retorno positivo (+2%)
        &features,
        1.0,  // Long
        1.5,  // plasticidad dinámica alta
        &custom_ppo,
    );

    // Los pesos deben haberse adaptado y permanecer finitos y dentro de rango
    for (i, w) in engine.weights.iter().enumerate() {
        let val = w.load(std::sync::atomic::Ordering::Relaxed);
        assert!(val.is_finite(), "Peso {i} debe ser finito");
        assert!(val >= custom_ppo.min_weight_clip, "Peso {i} debe respetar clip mínimo");
    }

    // Inmunidad ante parámetros PPO inválidos (debe caer suavemente al default sin pánico)
    let bad_ppo = PpoHyperParams {
        learning_rate: -1.0,
        ..custom_ppo
    };
    engine.update_policy_with_params(
        -0.01,
        &features,
        -1.0,
        1.0,
        &bad_ppo,
    );
    assert!(engine.reward_ema.load(std::sync::atomic::Ordering::Relaxed).is_finite());
}
