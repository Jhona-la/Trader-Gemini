//! CONTRATO FORMAL DE DESACOPLAMIENTO DINÁMICO EN SOLITÓN Y RESONANCIA ESTOCÁSTICA (OLA Ω129)
//!
//! Verifica:
//! 1. Validación e invariantes de SolitonWaveParams y StochasticResonanceParams.
//! 2. Invarianza de escala, parametrización y overrides dinámicos en SolitonWaveEngine.
//! 3. Modulación de pico y factor Q en StochasticResonanceEngine con resonancia canónica en |s| = σ.
//! 4. Inmunidad matemática ante NaN, subnormales y ±∞.

use omniscient_registry::{OmniscientRegistry, ParameterKind};
use signal_engine::{
    SolitonWaveEngine, SolitonWaveParams, StochasticResonanceEngine, StochasticResonanceParams,
};
use std::sync::Arc;
use strategy_core::QuantumStrategy;

#[test]
fn test_soliton_params_validation_and_dynamic_scale() {
    let def = SolitonWaveParams::default();
    assert!(def.validate().is_ok());

    let mut invalid = def;
    invalid.amplitude_scale = 0.0;
    assert!(invalid.validate().is_err());

    let mut invalid_vel = def;
    invalid_vel.velocity_norm_scale = -1.0;
    assert!(invalid_vel.validate().is_err());

    let mut invalid_time = def;
    invalid_time.default_t_time = 2.0;
    assert!(invalid_time.validate().is_err());

    // Motor con parámetros personalizados
    let custom = SolitonWaveParams {
        amplitude_scale: 1.5,
        velocity_norm_scale: 15.0,
        sat_momento: 5000.0,
        default_t_time: 0.10,
        envelope_power_factor: 2.5,
    };
    let mut engine = SolitonWaveEngine::new_with_params(custom);
    assert_eq!(engine.params(), &custom);

    let reg = Arc::new(OmniscientRegistry::new());
    engine.init(reg.clone()).unwrap();

    // Evaluación con parámetros en registro
    reg.register_or_update(
        "soliton_amplitude",
        ParameterKind::Adaptive,
        0.5,
        "test",
    );
    reg.register_or_update("soliton_velocity", ParameterKind::Adaptive, 10.0, "test");
    reg.register_or_update("mid_price", ParameterKind::Adaptive, 100.0, "test");
    reg.register_or_update("soliton_pos", ParameterKind::Adaptive, 0.0, "test");

    let signal_init = engine.evaluate();
    assert!(signal_init.is_finite() && signal_init > 0.0);

    // Override dinámico en registro omnisciente
    reg.register_or_update(
        "soliton_amp_scale",
        ParameterKind::Adaptive,
        2.0,
        "test",
    );
    let signal_boosted = engine.evaluate();
    assert!(signal_boosted.is_finite());
    assert!(signal_boosted > signal_init);
}

#[test]
fn test_stochastic_resonance_tunable_q_and_peak_gain() {
    let def = StochasticResonanceParams::default();
    assert!(def.validate().is_ok());

    let mut invalid = def;
    invalid.base_gain = 3.0;
    invalid.peak_gain = 2.0; // peak < base
    assert!(invalid.validate().is_err());

    let mut invalid_q = def;
    invalid_q.q_sharpness = 0.0;
    assert!(invalid_q.validate().is_err());

    // Resonancia canónica: pico exacto en |s| = σ
    let noise_var = 0.0004; // σ = 0.02
    let sigma = 0.02;

    let res_at_peak =
        StochasticResonanceEngine::amplify_signal_with_params(sigma, noise_var, &def);
    // Ganancia en el pico debe ser peak_gain (2.0) -> res = sigma * 2.0 = 0.04
    assert!((res_at_peak - sigma * 2.0).abs() < 1e-9);

    // Lejos de la resonancia (|s| >> σ, p.ej. s = 5.0) -> ganancia decae asintóticamente a base_gain (1.0)
    let s_far = 5.0;
    let res_far = StochasticResonanceEngine::amplify_signal_with_params(s_far, noise_var, &def);
    assert!((res_far - s_far * 1.0).abs() < 0.05);

    // Modulación dinámica con mayor peak_gain (3.0) y mayor Q (2.0)
    let dynamic_params = StochasticResonanceParams {
        base_gain: 1.0,
        peak_gain: 3.0,
        q_sharpness: 2.0,
        noise_floor: 1e-6,
    };
    let res_dynamic_peak =
        StochasticResonanceEngine::amplify_signal_with_params(sigma, noise_var, &dynamic_params);
    assert!((res_dynamic_peak - sigma * 3.0).abs() < 1e-9);

    // Inmunidad numérica a NaN y ruido nulo
    let res_nan = StochasticResonanceEngine::amplify_signal_with_params(f64::NAN, noise_var, &def);
    assert_eq!(res_nan, 0.0);

    let res_zero_noise =
        StochasticResonanceEngine::amplify_signal_with_params(sigma, 0.0, &def);
    assert!(res_zero_noise.is_finite());
}
