use signal_engine::{NonAbelianBraidingConfig, NonAbelianBraidingJonesEngine, BRAID_STRANDS};

#[test]
fn test_non_abelian_braiding_initial_state_unbraided() {
    let config = NonAbelianBraidingConfig::default();
    let mut engine = NonAbelianBraidingJonesEngine::new(config);

    let strands = [1.0, 2.0, 3.0, 4.0];
    let report = engine.evaluate(&strands);

    assert_eq!(report.cumulative_writhe, 0);
    assert_eq!(report.jones_polynomial_norm, 1.0);
    assert_eq!(report.braid_holonomy_phase, 0.0);
    assert!(!report.braid_decoherence_detected);
    assert_eq!(report.holographic_conviction_modulator, 1.0);
}

#[test]
fn test_non_abelian_braiding_crossing_and_writhe_accumulation() {
    let config = NonAbelianBraidingConfig::default();
    let mut engine = NonAbelianBraidingJonesEngine::new(config);

    // Paso 1: Orden inicial [0, 1, 2, 3]
    let s1 = [1.0, 2.0, 3.0, 4.0];
    engine.evaluate(&s1);

    // Paso 2: Intercambio entre strand 0 y 1 (cruzar hebra 0 sobre 1)
    let s2 = [2.5, 0.8, 3.0, 4.0];
    let rep2 = engine.evaluate(&s2);

    assert_ne!(rep2.cumulative_writhe, 0);
    assert!(rep2.jones_polynomial_norm > 1.0);
    assert!(rep2.braid_holonomy_phase.is_finite());
    assert!(
        rep2.holographic_conviction_modulator >= 0.40
            && rep2.holographic_conviction_modulator <= 1.45
    );

    // Paso 3: Segundo cruce entre strand 1 y 2
    let s3 = [2.5, 3.5, 0.5, 4.0];
    let rep3 = engine.evaluate(&s3);

    assert!(rep3.jones_polynomial_norm.is_finite());
    assert!(
        rep3.holographic_conviction_modulator >= 0.40
            && rep3.holographic_conviction_modulator <= 1.45
    );
}

#[test]
fn test_non_abelian_braiding_stability_under_nan_and_extremes() {
    let config = NonAbelianBraidingConfig::default();
    let mut engine = NonAbelianBraidingJonesEngine::new(config);

    let strands_nan = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let rep_nan = engine.evaluate(&strands_nan);

    assert!(rep_nan.jones_polynomial_norm.is_finite());
    assert!(rep_nan.braid_holonomy_phase.is_finite());
    assert!(
        rep_nan.holographic_conviction_modulator >= 0.40
            && rep_nan.holographic_conviction_modulator <= 1.45
    );

    let strands_zeros = [0.0; BRAID_STRANDS];
    let rep_zeros = engine.evaluate(&strands_zeros);
    assert!(rep_zeros.jones_polynomial_norm.is_finite());
    assert!(
        rep_zeros.holographic_conviction_modulator >= 0.40
            && rep_zeros.holographic_conviction_modulator <= 1.45
    );
}

#[test]
fn test_non_abelian_braiding_deterministic_replay() {
    let config = NonAbelianBraidingConfig::default();
    let mut engine_a = NonAbelianBraidingJonesEngine::new(config);
    let mut engine_b = NonAbelianBraidingJonesEngine::new(config);

    let s1 = [1.2, 0.8, 2.4, 3.1];
    let s2 = [0.5, 1.5, 2.0, 3.5];
    let s3 = [2.1, 0.9, 3.2, 1.8];

    let _ = engine_a.evaluate(&s1);
    let rep_a2 = engine_a.evaluate(&s2);
    let rep_a3 = engine_a.evaluate(&s3);

    let _ = engine_b.evaluate(&s1);
    let rep_b2 = engine_b.evaluate(&s2);
    let rep_b3 = engine_b.evaluate(&s3);

    assert_eq!(rep_a2.cumulative_writhe, rep_b2.cumulative_writhe);
    assert_eq!(rep_a2.jones_polynomial_norm, rep_b2.jones_polynomial_norm);
    assert_eq!(rep_a3.cumulative_writhe, rep_b3.cumulative_writhe);
    assert_eq!(rep_a3.jones_polynomial_norm, rep_b3.jones_polynomial_norm);
    assert_eq!(
        rep_a3.holographic_conviction_modulator,
        rep_b3.holographic_conviction_modulator
    );
}
