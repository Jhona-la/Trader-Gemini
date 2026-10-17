//! Parameter boundaries and representable scalar outputs, not a profitability
//! or physical stochastic-resonance oracle. Real dependency integration is Cargo.
use omniscient_registry::OmniscientRegistry;
use signal_engine::{
    SolitonWaveEngine, SolitonWaveParams, StochasticResonanceEngine, StochasticResonanceParams,
};
use std::sync::Arc;
use strategy_core::QuantumStrategy;

fn resonance_registry() -> Arc<OmniscientRegistry> {
    let registry = Arc::new(OmniscientRegistry::new());
    registry.set("weak_alpha_signal", 0.1);
    registry.set("microstructure_noise_variance", 0.01);
    registry
}

fn soliton_registry() -> Arc<OmniscientRegistry> {
    let registry = Arc::new(OmniscientRegistry::new());
    registry.set("soliton_amplitude", 0.4);
    registry.set("soliton_velocity", 0.001);
    registry.set("soliton_pos", 0.2);
    registry.set("mid_price", 100.0);
    registry
}

#[test]
fn stochastic_constructor_uses_the_declared_parameter_domain() {
    let defaults = StochasticResonanceParams::default();
    for invalid in [
        StochasticResonanceParams {
            base_gain: -2.0,
            peak_gain: -1.0,
            ..defaults
        },
        StochasticResonanceParams {
            peak_gain: 100.0,
            ..defaults
        },
        StochasticResonanceParams {
            q_sharpness: 0.06,
            ..defaults
        },
        StochasticResonanceParams {
            noise_floor: 0.0,
            ..defaults
        },
    ] {
        assert!(invalid.validate().is_err());
        assert_eq!(
            *StochasticResonanceEngine::new_with_params(invalid).params(),
            defaults
        );
    }
}

#[test]
fn soliton_constructor_uses_the_declared_parameter_domain() {
    let defaults = SolitonWaveParams::default();
    for invalid in [
        SolitonWaveParams {
            amplitude_scale: -1.0,
            ..defaults
        },
        SolitonWaveParams {
            velocity_norm_scale: 101.0,
            ..defaults
        },
        SolitonWaveParams {
            sat_momento: 1.0,
            ..defaults
        },
        SolitonWaveParams {
            default_t_time: 2.0,
            ..defaults
        },
        SolitonWaveParams {
            envelope_power_factor: 0.0,
            ..defaults
        },
    ] {
        assert!(invalid.validate().is_err());
        assert_eq!(
            *SolitonWaveEngine::new_with_params(invalid).params(),
            defaults
        );
    }
}

#[test]
fn resonance_scalar_rejects_invalid_parameters_before_arithmetic() {
    let invalid = StochasticResonanceParams {
        base_gain: -2.0,
        peak_gain: -1.0,
        ..StochasticResonanceParams::default()
    };
    assert_eq!(
        StochasticResonanceEngine::amplify_signal_with_params(0.1, 0.01, &invalid),
        0.0
    );
}

#[test]
fn declared_valid_parameter_endpoints_remain_accepted() {
    for params in [
        StochasticResonanceParams {
            base_gain: 0.1,
            peak_gain: 1.0,
            q_sharpness: 0.1,
            noise_floor: 1e-12,
        },
        StochasticResonanceParams {
            base_gain: 5.0,
            peak_gain: 10.0,
            q_sharpness: 10.0,
            noise_floor: 1e-2,
        },
    ] {
        assert!(params.validate().is_ok());
        assert_eq!(
            *StochasticResonanceEngine::new_with_params(params).params(),
            params
        );
        let value = StochasticResonanceEngine::amplify_signal_with_params(0.1, 0.01, &params);
        assert!(value.is_finite() && value >= 0.1 * params.base_gain);
        assert!(value <= 0.1 * params.peak_gain);
    }
}

#[test]
fn finite_extreme_signal_does_not_become_zero_through_ratio_overflow() {
    // With the default base gain 1, sech(log(|s|/sigma)) tends to zero.
    // The represented result is +/-s, although |s|/sigma overflows f64.
    for signal in [1e308, -1e308] {
        assert_eq!(
            StochasticResonanceEngine::amplify_signal_with_noise(signal, 1e-6),
            signal
        );
    }
}

#[test]
fn default_gain_matches_an_independent_rational_identity() {
    // sech(log(r)) = 2r/(1+r^2) for finite positive r. No logarithm/cosh
    // in this reference; test below the existing signal/noise floors.
    let sigma = 0.1_f64;
    for signal in [1e-8_f64, 0.001, 0.03, 0.1, 0.3, 2.0, 10.0] {
        let r = signal / sigma;
        let expected = signal * (1.0 + 2.0 * r / (1.0 + r * r));
        let actual = StochasticResonanceEngine::amplify_signal_with_noise(signal, sigma * sigma);
        assert!((actual - expected).abs() <= 2e-14 * expected.abs().max(1e-8));
        assert_eq!(
            StochasticResonanceEngine::amplify_signal_with_noise(-signal, sigma * sigma),
            -actual
        );
    }
}

#[test]
fn invalid_public_parameters_produce_no_strategy_vote() {
    let mut sr = StochasticResonanceEngine::new();
    sr.init(resonance_registry()).unwrap();
    sr.params.peak_gain = 100.0;
    assert_eq!(sr.evaluate(), 0.0);
    assert_eq!(sr.params.peak_gain, 100.0);
    let mut soliton = SolitonWaveEngine::new();
    soliton.init(soliton_registry()).unwrap();
    soliton.params.velocity_norm_scale = -1.0;
    assert_eq!(soliton.evaluate(), 0.0);
    assert_eq!(soliton.params.velocity_norm_scale, -1.0);
}

#[test]
fn invalid_resonance_override_preserves_the_configured_peak() {
    let params = StochasticResonanceParams {
        peak_gain: 3.0,
        ..StochasticResonanceParams::default()
    };
    let registry = resonance_registry();
    let mut engine = StochasticResonanceEngine::new_with_params(params);
    engine.init(registry.clone()).unwrap();
    let configured = engine.evaluate();
    for value in [0.5, 100.0, f64::INFINITY, f64::NAN] {
        registry.set("stochastic_res_peak_gain", value);
        assert_eq!(engine.evaluate(), configured);
    }
}

#[test]
fn invalid_resonance_q_preserves_the_configured_q() {
    let params = StochasticResonanceParams {
        q_sharpness: 2.0,
        ..StochasticResonanceParams::default()
    };
    let registry = resonance_registry();
    registry.set("weak_alpha_signal", 0.03);
    let mut engine = StochasticResonanceEngine::new_with_params(params);
    engine.init(registry.clone()).unwrap();
    let configured = engine.evaluate();
    for value in [0.06, 11.0, f64::INFINITY, f64::NAN] {
        registry.set("stochastic_res_q", value);
        assert_eq!(engine.evaluate(), configured);
    }
}

#[test]
fn independent_valid_resonance_override_survives_an_invalid_one() {
    let registry = resonance_registry();
    registry.set("weak_alpha_signal", 0.03);
    registry.set("stochastic_res_peak_gain", 100.0);
    registry.set("stochastic_res_q", 3.0);
    let mut engine = StochasticResonanceEngine::new();
    engine.init(registry).unwrap();
    let params = StochasticResonanceParams {
        q_sharpness: 3.0,
        ..StochasticResonanceParams::default()
    };
    assert_eq!(
        engine.evaluate(),
        StochasticResonanceEngine::amplify_signal_with_params(0.03, 0.01, &params)
    );
}

#[test]
fn valid_resonance_overrides_still_reach_the_kernel() {
    let registry = resonance_registry();
    registry.set("weak_alpha_signal", 0.03);
    registry.set("stochastic_res_peak_gain", 4.0);
    registry.set("stochastic_res_q", 3.0);
    let mut engine = StochasticResonanceEngine::new();
    engine.init(registry).unwrap();
    let params = StochasticResonanceParams {
        peak_gain: 4.0,
        q_sharpness: 3.0,
        ..StochasticResonanceParams::default()
    };
    assert_eq!(
        engine.evaluate(),
        StochasticResonanceEngine::amplify_signal_with_params(0.03, 0.01, &params)
    );
}

#[test]
fn invalid_soliton_overrides_preserve_the_configured_fields() {
    let params = SolitonWaveParams {
        amplitude_scale: 1.5,
        velocity_norm_scale: 15.0,
        sat_momento: 5000.0,
        default_t_time: 0.10,
        ..SolitonWaveParams::default()
    };
    for (key, values) in [
        ("soliton_amp_scale", [0.0, 11.0, f64::NAN]),
        ("soliton_vel_norm_scale", [0.0, 101.0, f64::NAN]),
        ("soliton_sat_momento", [1.0, 1e7, f64::NAN]),
        ("soliton_time", [0.0, 2.0, f64::NAN]),
    ] {
        let registry = soliton_registry();
        let mut engine = SolitonWaveEngine::new_with_params(params);
        engine.init(registry.clone()).unwrap();
        let configured = engine.evaluate();
        for value in values {
            registry.set(key, value);
            assert_eq!(engine.evaluate(), configured, "invalid {key}={value}");
        }
    }
}
