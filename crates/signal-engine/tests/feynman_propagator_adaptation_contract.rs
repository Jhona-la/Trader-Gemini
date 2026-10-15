use signal_engine::feynman_propagator::{FeynmanPropagatorEngine, FeynmanPropagatorParams};
use signal_engine::voto_espectral::ESCALAS_VOTO;
use strategy_core::QuantumStrategy;
use omniscient_registry::{OmniscientRegistry, ParameterKind};
use std::sync::Arc;

#[test]
fn test_feynman_params_validation_and_defaults() {
    let def = FeynmanPropagatorParams::default();
    assert!(def.validate().is_ok());
    assert!((def.hbar_effective - 1.0).abs() < 1e-9);
    assert!((def.mass - 1.0).abs() < 1e-9);
    assert!((def.k_spring - 1.0).abs() < 1e-9);
    assert!((def.lambda_quartic - 0.10).abs() < 1e-9);
    assert!((def.action_coupling - 0.10).abs() < 1e-9);
    assert!((def.amplitude_scale - 0.50).abs() < 1e-9);
    assert!((def.coherence_threshold - 0.20).abs() < 1e-9);

    let mut invalid = def;
    invalid.hbar_effective = 0.001; // < 0.01
    assert!(invalid.validate().is_err());

    let mut invalid_mass = def;
    invalid_mass.mass = -1.0;
    assert!(invalid_mass.validate().is_err());

    let mut invalid_coh = def;
    invalid_coh.coherence_threshold = 0.95; // > 0.80
    assert!(invalid_coh.validate().is_err());
}

#[test]
fn test_feynman_engine_with_params_and_update_params() {
    let custom = FeynmanPropagatorParams {
        hbar_effective: 2.0,
        mass: 0.5,
        k_spring: 3.0,
        lambda_quartic: 0.25,
        action_coupling: 0.15,
        amplitude_scale: 0.80,
        coherence_threshold: 0.35,
    };
    let mut engine = FeynmanPropagatorEngine::new().with_params(custom);
    assert_eq!(engine.params, custom);

    let updated = FeynmanPropagatorParams {
        hbar_effective: 1.5,
        mass: 1.2,
        k_spring: 0.8,
        lambda_quartic: 0.05,
        action_coupling: 0.05,
        amplitude_scale: 0.40,
        coherence_threshold: 0.15,
    };
    engine.update_params(updated);
    assert_eq!(engine.params, updated);
}

#[test]
fn test_feynman_voto_espectral_with_params_continuous_response() {
    let mut desplazamientos = [0.0f64; ESCALAS_VOTO];
    for k in 0..ESCALAS_VOTO {
        desplazamientos[k] = (k as f64 + 1.0) * 0.1;
    }

    let params_low_hbar = FeynmanPropagatorParams {
        hbar_effective: 0.5,
        ..Default::default()
    };
    let params_high_hbar = FeynmanPropagatorParams {
        hbar_effective: 5.0,
        ..Default::default()
    };

    let (voto_low, coh_low) = FeynmanPropagatorEngine::voto_espectral_con_coherencia_with_params(
        &desplazamientos,
        &params_low_hbar,
    );
    let (voto_high, coh_high) = FeynmanPropagatorEngine::voto_espectral_con_coherencia_with_params(
        &desplazamientos,
        &params_high_hbar,
    );

    assert!(coh_low.is_finite() && coh_low >= 0.0 && coh_low <= 1.0);
    assert!(coh_high.is_finite() && coh_high >= 0.0 && coh_high <= 1.0);
    assert!(voto_low.media_banda(0, 31).unwrap().is_finite());
    assert!(voto_high.media_banda(0, 31).unwrap().is_finite());
}

#[test]
fn test_feynman_hermite_transmission_gate_adaptive_cutoff() {
    let registry = Arc::new(OmniscientRegistry::new());
    registry.register_or_update(
        "consenso_espectral_dominante",
        ParameterKind::Adaptive,
        1.0,
        "Test",
    );
    // Escala dominante en banda operable (> TAU_ANCHOR_FAST_MS = 30_000.0 ms)
    registry.register_or_update(
        "consenso_espectral_tau",
        ParameterKind::Adaptive,
        60_000.0,
        "Test",
    );

    // Caso A: coherencia = 0.25 con threshold por defecto (0.20)
    registry.register_or_update(
        "feynman_coherence",
        ParameterKind::Adaptive,
        0.25,
        "Test",
    );

    let mut engine_low_thresh = FeynmanPropagatorEngine::new().with_params(FeynmanPropagatorParams {
        coherence_threshold: 0.20,
        ..Default::default()
    });
    engine_low_thresh.init(Arc::clone(&registry)).unwrap();
    let eval_low = engine_low_thresh.evaluate();
    assert!(eval_low > 0.0, "Coherence 0.25 > 0.20 threshold should transmit signal");

    // Caso B: mismo estado pero con threshold elevado a 0.30 -> debe cortar a 0.0 (decoherencia)
    let mut engine_high_thresh = FeynmanPropagatorEngine::new().with_params(FeynmanPropagatorParams {
        coherence_threshold: 0.30,
        ..Default::default()
    });
    engine_high_thresh.init(Arc::clone(&registry)).unwrap();
    let eval_high = engine_high_thresh.evaluate();
    assert_eq!(eval_high, 0.0, "Coherence 0.25 <= 0.30 threshold must strictly collapse signal to 0.0");
}

#[test]
fn test_feynman_fail_closed_under_non_finite_params() {
    let nan_params = FeynmanPropagatorParams {
        hbar_effective: f64::NAN,
        mass: f64::INFINITY,
        k_spring: f64::NEG_INFINITY,
        lambda_quartic: f64::NAN,
        action_coupling: f64::NAN,
        amplitude_scale: f64::NAN,
        coherence_threshold: f64::NAN,
    };
    assert!(nan_params.validate().is_err());

    let desplazamientos = [f64::NAN; ESCALAS_VOTO];
    let (voto, coh) = FeynmanPropagatorEngine::voto_espectral_con_coherencia_with_params(
        &desplazamientos,
        &nan_params,
    );
    assert_eq!(coh, 0.0);
    for k in 0..ESCALAS_VOTO {
        assert_eq!(voto.en_escala(k), 0.0);
    }
}
