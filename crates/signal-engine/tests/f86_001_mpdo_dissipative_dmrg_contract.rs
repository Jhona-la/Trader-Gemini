use signal_engine::{MpdoDissipativeConfig, MpdoDissipativeDmrgEngine};

#[test]
fn test_mpdo_dissipative_coherent_evolution() {
    let config = MpdoDissipativeConfig::default();
    let mut engine = MpdoDissipativeDmrgEngine::new(config);

    // Flujo positivo ordenado con volatilidad normal
    let report = engine.evaluate(0.80, 10.0, 0.05, 0.10);

    assert!(report.quantum_purity >= 0.50 && report.quantum_purity <= 1.0);
    assert!(report.von_neumann_entropy >= 0.0);
    assert!(report.coherent_fidelity >= 0.50);
    assert!(report.directional_signal > 0.0);
    assert!(!report.dissipative_decoherence_veto);
    assert!(report.slot_scale_factor >= 0.15 && report.slot_scale_factor <= 1.30);
}

#[test]
fn test_mpdo_dissipative_decoherence_veto() {
    // Configuración con piso de pureza elevado para inducir veto determinista
    let config = MpdoDissipativeConfig {
        coherent_omega: 1.0,
        dissipation_rate_gamma: 5.0,
        dephasing_rate_gamma: 5.0,
        critical_purity_floor: 0.99, // Exige pureza casi absoluta
    };
    let mut engine = MpdoDissipativeDmrgEngine::new(config);

    // Ruido térmico elevado
    let report = engine.evaluate(0.0, 0.10, 2.5, 0.50);

    assert!(report.dissipative_decoherence_veto);
    assert_eq!(report.slot_scale_factor, 0.0);
}

#[test]
fn test_mpdo_dissipative_reversal_polarization() {
    let config = MpdoDissipativeConfig::default();
    let mut engine = MpdoDissipativeDmrgEngine::new(config);

    // Orientación negativa sostenida
    let mut rep = engine.evaluate(-0.90, 5.0, 0.02, 0.10);
    for _ in 0..10 {
        rep = engine.evaluate(-0.90, 5.0, 0.02, 0.10);
    }

    assert!(rep.directional_signal < 0.0);
    assert!(rep.coherent_fidelity > 0.50);
    assert!(!rep.dissipative_decoherence_veto);
}

#[test]
fn test_mpdo_dissipative_nan_stability_and_determinism() {
    let config = MpdoDissipativeConfig::default();
    let mut engine_a = MpdoDissipativeDmrgEngine::new(config);
    let mut engine_b = MpdoDissipativeDmrgEngine::new(config);

    // Evaluación con NaNs y valores patológicos
    let rep_nan = engine_a.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    assert!(rep_nan.quantum_purity.is_finite());
    assert!(rep_nan.von_neumann_entropy.is_finite());
    assert!(rep_nan.coherent_fidelity.is_finite());
    assert!(rep_nan.directional_signal.is_finite());
    assert!(rep_nan.slot_scale_factor.is_finite());

    // Determinismo idéntico
    let rep_a = engine_a.evaluate(0.55, 2.0, 0.12, 0.05);
    let _ = engine_b.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    let rep_b = engine_b.evaluate(0.55, 2.0, 0.12, 0.05);

    assert_eq!(rep_a.quantum_purity, rep_b.quantum_purity);
    assert_eq!(rep_a.von_neumann_entropy, rep_b.von_neumann_entropy);
    assert_eq!(rep_a.coherent_fidelity, rep_b.coherent_fidelity);
    assert_eq!(rep_a.directional_signal, rep_b.directional_signal);
    assert_eq!(
        rep_a.dissipative_decoherence_veto,
        rep_b.dissipative_decoherence_veto
    );
    assert_eq!(rep_a.slot_scale_factor, rep_b.slot_scale_factor);
}
