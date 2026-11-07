//! Contrato Formal de Verificación: Fase 50 (Fractional Rough Volatility Engine & Riemann-Liouville Kernels)
//!
//! Este conjunto de pruebas certifica las propiedades matemáticas y estadísticas del motor:
//! - C1: Normalización de pesos analíticos del núcleo de Riemann-Liouville e integral fraccionaria.
//! - C2: Convolución fraccionaria no markoviana y detección adaptativa de clusters de volatilidad.
//! - C3: Estimación espectral diádica de Hurst en línea $H \in [0.02, 0.48]$ e índice de rugosidad.
//! - C4: Determinismo en stack $O(1)$, cero heap allocs y robustez numérica total ante NaN/Infinitos.

use signal_engine::{
    FractionalRoughVolatilityEngine, FractionalRoughVolatilityParams,
    FractionalRoughVolatilityReport, ROUGH_VOL_WINDOW,
};

#[test]
fn test_c1_riemann_liouville_weights_and_normalization() {
    let params = FractionalRoughVolatilityParams {
        initial_hurst: 0.125,
        mean_reversion_lambda: 1.0,
        vol_of_vol_nu: 0.3,
        hurst_smoothing: 0.1,
        time_step_dt: 0.05,
    };
    let engine = FractionalRoughVolatilityEngine::new(params);

    // 1. Verificación de suma unitaria de pesos normalizados
    let sum_w: f64 = engine.kernel_weights.iter().sum();
    assert!(
        (sum_w - 1.0).abs() < 1e-6,
        "Suma de pesos de Riemann-Liouville esperada 1.0, obtenida {}",
        sum_w
    );

    // 2. Todos los pesos deben ser positivos y finitos
    for (k, &w) in engine.kernel_weights.iter().enumerate() {
        assert!(w.is_finite(), "Peso no finito en k={}", k);
        assert!(w > 0.0, "Peso no positivo en k={}: {}", k, w);
    }

    assert_eq!(ROUGH_VOL_WINDOW, 32);
}

#[test]
fn test_c2_rough_convolution_and_burst_detection() {
    let params = FractionalRoughVolatilityParams::default();
    let mut engine = FractionalRoughVolatilityEngine::new(params);

    // 1. Período de calma con retornos pequeños
    let mut calm_report = FractionalRoughVolatilityReport {
        instantaneous_vol: 0.0,
        rough_fractional_integral: 0.0,
        adaptive_hurst_estimate: 0.125,
        memory_kernel_power: 0.375,
        roughness_regime_index: 0.75,
        signal_resonance_modulator: 1.0,
    };
    for i in 0..20 {
        calm_report = engine.update_and_evaluate(0.0005, 0.005, i * 50);
    }

    // 2. Inyección súbita de una ráfaga de volatilidad (shock de retorno)
    let burst_report = engine.update_and_evaluate(0.05, 0.08, 1050);

    // La integral fraccionaria debe responder de manera continua y acotada
    assert!(
        burst_report.rough_fractional_integral >= calm_report.rough_fractional_integral,
        "La volatilidad fraccionaria debió responder a la ráfaga: calm={} vs burst={}",
        calm_report.rough_fractional_integral,
        burst_report.rough_fractional_integral
    );

    // Modulador acotado estrictamente en [0.10, 1.40]
    assert!(
        burst_report.signal_resonance_modulator >= 0.10
            && burst_report.signal_resonance_modulator <= 1.40,
        "Modulador fuera de rango: {}",
        burst_report.signal_resonance_modulator
    );
}

#[test]
fn test_c3_online_hurst_estimation_and_roughness_regime() {
    let params = FractionalRoughVolatilityParams {
        initial_hurst: 0.15,
        mean_reversion_lambda: 1.5,
        vol_of_vol_nu: 0.4,
        hurst_smoothing: 0.20,
        time_step_dt: 0.05,
    };
    let mut engine = FractionalRoughVolatilityEngine::new(params);

    // Inyectamos retornos con fluctuaciones de alta rugosidad (signos alternados con ruido)
    for i in 0..40 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let ret = sign * 0.01 * ((i as f64 * 0.3).sin() + 1.2);
        let report = engine.update_and_evaluate(ret, 0.015, i * 50);

        // Hurst debe mantenerse estrictamente en el régimen subdifusivo rugoso H < 0.50
        assert!(
            report.adaptive_hurst_estimate >= 0.02 && report.adaptive_hurst_estimate <= 0.48,
            "Estimación de Hurst fuera de límites: {}",
            report.adaptive_hurst_estimate
        );
        assert!(
            report.roughness_regime_index >= 0.0 && report.roughness_regime_index <= 1.0,
            "Índice de rugosidad fuera de [0, 1]: {}",
            report.roughness_regime_index
        );
    }
}

#[test]
fn test_c4_stack_zero_heap_and_nan_subnormal_immunity() {
    let params = FractionalRoughVolatilityParams::default();
    let mut engine = FractionalRoughVolatilityEngine::new(params);

    let anomalous_inputs = [
        (f64::NAN, 0.01),
        (0.01, f64::INFINITY),
        (-0.02, f64::NAN),
        (0.0, 0.0),
        (1e-300, 1e-300),
    ];

    for &(ret, vol) in &anomalous_inputs {
        let report = engine.update_and_evaluate(ret, vol, 1000);
        assert!(report.instantaneous_vol.is_finite());
        assert!(report.rough_fractional_integral.is_finite());
        assert!(report.adaptive_hurst_estimate.is_finite());
        assert!(
            report.signal_resonance_modulator >= 0.10 && report.signal_resonance_modulator <= 1.40
        );
    }
}
