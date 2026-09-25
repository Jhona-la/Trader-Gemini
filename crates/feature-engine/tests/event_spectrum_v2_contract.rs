use feature_engine::spectral::{SpectralCycleEngine, SpectrumError};

fn signal(amplitude: f64, offset: f64, bin: f64) -> SpectralCycleEngine {
    let mut engine = SpectralCycleEngine::new();
    for i in 0..64 {
        engine.push(offset + amplitude * (std::f64::consts::TAU * bin * i as f64 / 64.0).sin());
    }
    engine
}

#[test]
fn constant_nonzero_input_has_no_oscillatory_power_in_v2() {
    for value in [0.0, 1.0, -42.0, f64::MAX] {
        let mut engine = SpectralCycleEngine::new();
        for _ in 0..64 {
            engine.push(value);
        }
        let spectrum = engine.analyze_event_spectrum_v2().unwrap();
        assert_eq!(spectrum.dominant_bin, 0);
        assert_eq!(spectrum.centroid_bin, 0.0);
        assert!(spectrum.power_by_bin.iter().all(|p| *p == 0.0));
    }
}

#[test]
fn v2_shape_is_amplitude_invariant_and_power_scales_quadratically() {
    let reference = signal(1.0, 0.0, 4.0).analyze_event_spectrum_v2().unwrap();
    for amplitude in [1e-20, 0.01, 10.0, 1e100] {
        let scaled = signal(amplitude, 0.0, 4.0)
            .analyze_event_spectrum_v2()
            .unwrap();
        assert_eq!(scaled.dominant_bin, 4);
        assert!((scaled.centroid_bin - reference.centroid_bin).abs() < 1e-10);
        assert!(
            (scaled.power_by_bin[4] / amplitude / amplitude - reference.power_by_bin[4]).abs()
                < 1e-10
        );
    }
}

#[test]
fn offset_is_removed_before_tapering() {
    let reference = signal(1.0, 0.0, 7.3).analyze_event_spectrum_v2().unwrap();
    let shifted = signal(1.0, 100.0, 7.3).analyze_event_spectrum_v2().unwrap();
    assert_eq!(reference.dominant_bin, shifted.dominant_bin);
    for (a, b) in reference
        .power_by_bin
        .iter()
        .zip(shifted.power_by_bin.iter())
    {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn unilateral_power_matches_parseval_including_nyquist() {
    let values: Vec<f64> = (0..64)
        .map(|i| {
            (if i % 2 == 0 { 2.0 } else { -2.0 })
                + (std::f64::consts::TAU * 5.0 * i as f64 / 64.0).sin()
        })
        .collect();
    let mean = values.iter().sum::<f64>() / 64.0;
    let mut weighted_energy = 0.0;
    let mut window_energy = 0.0;
    let mut engine = SpectralCycleEngine::new();
    for (i, value) in values.iter().enumerate() {
        engine.push(*value);
        let w = if i < 16 {
            0.5 * (1.0 - (std::f64::consts::PI * i as f64 / 16.0).cos())
        } else {
            1.0
        };
        weighted_energy += ((value - mean) * w).powi(2);
        window_energy += w * w;
    }
    let spectrum = engine.analyze_event_spectrum_v2().unwrap();
    assert_eq!(spectrum.dominant_bin, 32);
    assert!(
        (spectrum.power_by_bin.iter().sum::<f64>() - weighted_energy / window_energy).abs() < 1e-12
    );
}

#[test]
fn quality_distinguishes_warmup_and_imputation_until_the_bad_event_expires() {
    let mut engine = SpectralCycleEngine::new();
    assert_eq!(
        engine.analyze_event_spectrum_v2(),
        Err(SpectrumError::InsufficientSamples { received: 0 })
    );
    for _ in 0..63 {
        engine.push(1.0);
    }
    engine.push(f64::NAN);
    assert_eq!(
        engine.analyze_event_spectrum_v2(),
        Err(SpectrumError::InvalidSamples { count: 1 })
    );
    for _ in 0..63 {
        engine.push(1.0);
    }
    assert!(engine.analyze_event_spectrum_v2().is_err());
    engine.push(1.0);
    assert_eq!(engine.analyze_event_spectrum_v2().unwrap().dominant_bin, 0);
}

#[test]
fn nonrepresentable_power_is_not_reported_as_zero() {
    let engine = signal(1e200, 0.0, 4.0);
    assert_eq!(
        engine.analyze_event_spectrum_v2(),
        Err(SpectrumError::UnrepresentablePower)
    );
    assert_eq!(
        signal(1e-200, 0.0, 4.0).analyze_event_spectrum_v2(),
        Err(SpectrumError::UnrepresentablePower)
    );
}

#[test]
fn wrapped_ring_matches_the_same_last_window_in_chronological_order() {
    let mut wrapped = SpectralCycleEngine::new();
    let mut last_window = SpectralCycleEngine::new();
    for i in 0..137 {
        let value = (0.37 * i as f64).sin() + (0.71 * i as f64).cos();
        wrapped.push(value);
        if i >= 73 {
            last_window.push(value);
        }
    }
    let a = wrapped.analyze_event_spectrum_v2().unwrap();
    let b = last_window.analyze_event_spectrum_v2().unwrap();
    assert_eq!(a.dominant_bin, b.dominant_bin);
    for (x, y) in a.power_by_bin.iter().zip(b.power_by_bin.iter()) {
        assert!((x - y).abs() < 1e-12);
    }
}

#[test]
fn legacy_ml_path_still_exhibits_dc_leakage_and_amplitude_floor() {
    // Diagnostic of OPEN FMT-004: legacy is retained until schema migration.
    let mut flat = SpectralCycleEngine::new();
    for _ in 0..64 {
        flat.push(1.0);
    }
    assert!(flat.analyze_spectrum().1 > 0.0);
    let weak = signal(1e-20, 0.0, 4.0).analyze_spectrum();
    assert_eq!(weak.0, 4);
    assert_eq!(weak.2, 0.0);
}
