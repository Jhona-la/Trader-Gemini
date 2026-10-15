//! Contrato de Continuidad C^1 en Confluencia Multifractal y Microestructura (Ola Ω128)
//!
//! Verifica la erradicación analítica de acantilados discretos en la confluencia de Hurst,
//! la activación suave Hermite cúbica de persistencia fractal, la intensidad continua de
//! ráfagas de volumen institucional (Welford Z-Score) y la activación suave del detector de spoofing.

use feature_engine::{
    smooth_persistence_degree, InstitutionalVolumeTracker, MultiScaleHurstConfluence,
    MultifractalParams, SpoofingDetector,
};

#[test]
fn contrato_multifractal_params_validacion_y_confluencia_continua() {
    // 1. Validación de hiperparámetros
    let default_params = MultifractalParams::default();
    assert!(default_params.validate().is_ok());

    let invalid_weights = MultifractalParams {
        w_micro: -0.1,
        ..default_params
    };
    assert!(invalid_weights.validate().is_err());

    let zero_sum_weights = MultifractalParams {
        w_micro: 0.0,
        w_meso: 0.0,
        w_macro: 0.0,
        ..default_params
    };
    assert!(zero_sum_weights.validate().is_err());

    let invalid_tanh = MultifractalParams {
        tanh_scale: 0.0,
        ..default_params
    };
    assert!(invalid_tanh.validate().is_err());

    // 2. Confluencia continua con parámetros dinámicos
    let mut confluence = MultiScaleHurstConfluence::new();
    let mut price = 60000.0;

    // Generar serie sintética
    for i in 0..50 {
        price += (i as f64 * 0.15).sin() * 12.0;
        let (h_mic, h_mes, h_mac, score, p_mic, p_mac) =
            confluence.update_continuous_with_params(price, &default_params);

        assert!(h_mic >= 0.05 && h_mic <= 0.95);
        assert!(h_mes >= 0.05 && h_mes <= 0.95);
        assert!(h_mac >= 0.05 && h_mac <= 0.95);
        assert!(score >= -1.0 && score <= 1.0);
        assert!(p_mic >= 0.0 && p_mic <= 1.0);
        assert!(p_mac >= 0.0 && p_mac <= 1.0);
    }
}

#[test]
fn contrato_smooth_persistence_degree_c1_hermite() {
    let threshold = 0.10;
    let width = 0.03;

    // En el centro nulo browniano H=0.50 (dev=0.0): persistencia = 0.0
    assert_eq!(smooth_persistence_degree(0.50, threshold, width), 0.0);

    // Bien por debajo del umbral (dev <= 0.07): persistencia = 0.0
    assert_eq!(smooth_persistence_degree(0.55, threshold, width), 0.0);
    assert_eq!(smooth_persistence_degree(0.45, threshold, width), 0.0);
    assert_eq!(smooth_persistence_degree(0.57, threshold, width), 0.0);

    // Exactamente en el umbral (dev = 0.10): u = 0.5, 3u^2 - 2u^3 = 0.50
    let at_th_up = smooth_persistence_degree(0.60, threshold, width);
    let at_th_dn = smooth_persistence_degree(0.40, threshold, width);
    assert!((at_th_up - 0.50).abs() < 1e-12);
    assert!((at_th_dn - 0.50).abs() < 1e-12);

    // Bien por encima del umbral (dev >= 0.13): persistencia = 1.0
    assert_eq!(smooth_persistence_degree(0.65, threshold, width), 1.0);
    assert_eq!(smooth_persistence_degree(0.35, threshold, width), 1.0);
    assert_eq!(smooth_persistence_degree(0.80, threshold, width), 1.0);

    // Monotonía estricta en el intervalo de transición
    let mut prev_p = 0.0;
    for step in 0..=30 {
        let h = 0.57 + (step as f64) * 0.002; // de 0.57 a 0.63 (dev de 0.07 a 0.13)
        let p = smooth_persistence_degree(h, threshold, width);
        assert!(p >= prev_p - 1e-14, "Debe ser monótona no decreciente");
        assert!(p >= 0.0 && p <= 1.0);
        prev_p = p;
    }

    // Inmunidad matemática ante NaN / Infinitos
    assert_eq!(smooth_persistence_degree(f64::NAN, threshold, width), 0.0);
    assert_eq!(smooth_persistence_degree(f64::INFINITY, threshold, width), 0.0);
    assert_eq!(smooth_persistence_degree(0.60, f64::NAN, width), 0.0);
}

#[test]
fn contrato_institutional_volume_burst_intensity_c1() {
    let mut tracker = InstitutionalVolumeTracker::new(2.50).with_min_samples(20.0);

    // Llenar muestras normales
    for _ in 0..40 {
        let (z, accel, intensity) = tracker.update_continuous(10.0);
        assert!(z.is_finite());
        assert!(accel.is_finite());
        assert_eq!(intensity, 0.0);
    }

    // Ráfaga masiva repentina
    let (z_spike, accel_spike, intensity_spike) = tracker.update_continuous(120.0);
    assert!(z_spike > 2.50);
    assert!(accel_spike > 50.0);
    assert!(intensity_spike > 0.50, "Ráfaga institucional debe activar intensidad > 0.50");

    // Verificar paridad con método booleano canónico
    let is_burst = tracker.update(120.0).2;
    assert!(is_burst);
}

#[test]
fn contrato_spoofing_detector_continuous_c1_decay_and_activation() {
    let mut sp = SpoofingDetector::new(2.0);

    // Estado inicial en reposo
    let r0 = sp.evaluate_wall_decay(100.0, 10.0, 1000);
    assert_eq!(r0, 0.0);

    // Pequeña oscilación (< 20% caída) en 100ms: no debe activar riesgo significativo
    let r_small = sp.evaluate_wall_decay(85.0, 10.0, 1100);
    assert!(r_small < 0.10);

    // Cancelación abrupta (80% caída) en 100ms: activa alerta suave C1
    let mut sp2 = SpoofingDetector::new(2.0);
    sp2.evaluate_wall_decay(100.0, 10.0, 1000);
    let r_abrupt = sp2.evaluate_wall_decay(20.0, 10.0, 1100);
    assert!(r_abrupt >= 0.50, "Caída del 80% en 100ms debe disparar alerta de spoofing >= 0.50: dio {}", r_abrupt);
}
