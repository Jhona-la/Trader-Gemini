use signal_engine::feynman_propagator::FeynmanPropagatorEngine;
use signal_engine::voto_espectral::ESCALAS_VOTO;
use strategy_core::{QuantumStrategy, TradeHorizon};
use omniscient_registry::{OmniscientRegistry, ParameterKind};
use std::sync::Arc;

#[test]
fn test_lagrangian_kinetic_potential_consistency() {
    let z = 2.0;
    let v = 1.5;
    let mass = 1.0;
    let k_spring = 2.0;
    let lambda = 0.10;

    let l = FeynmanPropagatorEngine::compute_lagrangian(z, v, mass, k_spring, lambda);

    // T = 0.5 * 1.0 * (1.5)^2 = 1.125
    // V = 0.5 * 2.0 * 4.0 + 0.25 * 0.10 * 16.0 = 4.0 + 0.4 = 4.4
    // L = 1.125 - 4.4 = -3.275
    let expected = 1.125 - 4.4;
    assert!((l - expected).abs() < 1e-9, "Lagrangian mismatch: {} vs {}", l, expected);
}

#[test]
fn test_quantum_phase_modulus_is_strictly_unity() {
    let test_actions = [-100.0, -10.5, -0.001, 0.0, 0.5, 3.14159, 42.0, 999.0];
    for &action in &test_actions {
        let (cos_th, sin_th) = FeynmanPropagatorEngine::compute_quantum_phase(action, 1.0);
        let modulus_sq = cos_th * cos_th + sin_th * sin_th;
        assert!(
            (modulus_sq - 1.0).abs() < 1e-12,
            "Quantum phase modulus squared must be 1.0, got {}",
            modulus_sq
        );
    }
}

#[test]
fn test_pure_constructive_interference_reaches_maximum_coherence() {
    // Si todas las 32 escalas tienen exactamente la misma fase (e.g. theta = 0) y amplitud 1.0,
    // |sum psi_k|^2 = (32 * 1.0)^2 = 1024.
    // incoh_sum = 32 * 1.0^2 = 32.
    // C_coh = 1024 / 32 = 32.0 (el máximo absoluto en 32 escalas).
    let phases = [(1.0f64, 0.0f64); ESCALAS_VOTO];
    let amplitudes = [1.0f64; ESCALAS_VOTO];

    let c_coh = FeynmanPropagatorEngine::compute_coherence_factor(&phases, &amplitudes);
    assert!(
        (c_coh - 32.0).abs() < 1e-6,
        "Coherence factor should reach 32.0 under full constructive interference, got {}",
        c_coh
    );
}

#[test]
fn test_pure_destructive_interference_cancels_coherence() {
    // Si las fases alternan exactamente en pi (cos = 1 vs cos = -1),
    // la suma coherente es 16*(+1) + 16*(-1) = 0.
    // C_coh debe colapsar a 0.0 exacto.
    let mut phases = [(1.0f64, 0.0f64); ESCALAS_VOTO];
    for k in 0..ESCALAS_VOTO {
        if k % 2 == 1 {
            phases[k] = (-1.0f64, 0.0f64);
        }
    }
    let amplitudes = [1.0f64; ESCALAS_VOTO];

    let c_coh = FeynmanPropagatorEngine::compute_coherence_factor(&phases, &amplitudes);
    assert!(
        c_coh.abs() < 1e-9,
        "Coherence factor should collapse to 0.0 under destructive cancellation, got {}",
        c_coh
    );
}

#[test]
fn test_voto_espectral_coherent_momentum_amplification() {
    // Un vector coherente de momentum positivo a lo largo de todas las escalas
    let desplazamientos = [2.5f64; ESCALAS_VOTO];
    let voto = FeynmanPropagatorEngine::voto_espectral(&desplazamientos, 1.0, 1.0, 1.0);

    // Debe tener dominante positivo de alta convicción
    let (dom_scale, dom_vote) = voto.dominante().expect("Coherent momentum must produce dominant scale");
    assert!(dom_vote > 0.50, "Dominant vote should be strong and positive, got {}", dom_vote);
    assert!(dom_scale < ESCALAS_VOTO, "Scale index must be valid");
}

#[test]
fn test_fail_closed_nan_and_infinite_immunity() {
    let bad_values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    for &bad in &bad_values {
        let l = FeynmanPropagatorEngine::compute_lagrangian(bad, bad, bad, bad, bad);
        assert!(l.is_finite(), "Lagrangian must be finite on non-finite inputs");

        let (c, s) = FeynmanPropagatorEngine::compute_quantum_phase(bad, bad);
        assert!(c.is_finite() && s.is_finite(), "Phase must be finite on non-finite inputs");

        let bad_displacements = [bad; ESCALAS_VOTO];
        let voto = FeynmanPropagatorEngine::voto_espectral(&bad_displacements, bad, bad, bad);
        for k in 0..ESCALAS_VOTO {
            assert!(voto.en_escala(k).is_finite(), "Scale vote must be finite");
            assert_eq!(voto.en_escala(k), 0.0, "Scale vote must fail-closed to 0.0");
        }
    }
}

#[test]
fn test_quantum_strategy_contract_and_registry_evaluation() {
    let mut engine = FeynmanPropagatorEngine::new();
    assert_eq!(engine.name(), "FeynmanPropagatorEngine");
    assert_eq!(engine.horizon(), TradeHorizon::Continuous);

    let registry = Arc::new(OmniscientRegistry::new());
    engine.init(registry.clone()).expect("Initialization must succeed");

    // En arranque en frío con defaults: evaluate retorna valor finito y acotado
    let val_cold = engine.evaluate();
    assert!(val_cold.is_finite());
    assert!((-1.0..=1.0).contains(&val_cold));

    // Con coherencia alta y dominante alcista en el registro:
    registry.register_or_update("feynman_coherence", ParameterKind::Adaptive, 0.90, "Test");
    registry.register_or_update("consenso_espectral_dominante", ParameterKind::Adaptive, 0.80, "Test");

    let val_active = engine.evaluate();
    assert!(val_active > 0.50, "Active coherence must transmit high conviction, got {}", val_active);
}
