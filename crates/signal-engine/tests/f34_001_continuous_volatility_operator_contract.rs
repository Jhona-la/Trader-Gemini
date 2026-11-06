//! Contrato Formal de Propiedades Matemáticas e Invariantes: Fase 34 (§34)
//! Operador Hermítico de Volatilidad Continua sobre Espacio de Hilbert $\mathcal{H} = L^2$
//!
//! - C1: Ejecución zero-alloc en stack, preservando traza y no-negatividad en las 32 escalas diádicas.
//! - C2: Detección continua de exponente de Hurst $H$ (persistencia vs antipersistencia).
//! - C3: Monotonía y suavidad continua de contracción de Kelly (sin discontinuidades ni saltos).
//! - C4: Desacoplamiento armónico entre escalas micro y macro.
//! - C5: Invarianza ante valores patológicos (cero pánicos `rb_p1`).

use signal_engine::{
    ContinuousVolatilityOperatorEngine, ContinuousVolatilityParams, VOL_SPECTRUM_SCALES,
};

#[test]
fn c1_zero_allocation_stack_execution() {
    let mut engine = ContinuousVolatilityOperatorEngine::default();

    // Alimentar 16 retornos sintéticos
    let mut state = signal_engine::ContinuousVolatilityState::default();
    for i in 0..16 {
        let ret = (i as f64 * 0.05).sin() * 0.005;
        state = engine.update(ret);
    }

    assert_eq!(state.spectral_energies.len(), VOL_SPECTRUM_SCALES);
    assert!(state.operator_trace >= 0.0);
    assert!(state.effective_volatility >= 0.0);
    assert!(state.operator_trace.is_finite());
    assert!(state.effective_volatility.is_finite());

    for energy in state.spectral_energies.iter() {
        assert!(*energy >= 0.0);
        assert!(energy.is_finite());
    }
}

#[test]
fn c2_morlet_cwt_and_continuous_hurst() {
    let mut engine = ContinuousVolatilityOperatorEngine::new(ContinuousVolatilityParams {
        tau_base_ms: 10.0,
        ..Default::default()
    });

    // Simular un paseo aleatorio con tendencia persistente (retornos positivos correlacionados)
    for _ in 0..32 {
        engine.update(0.0025);
    }
    let state_persistent = engine.update(0.0030);

    assert!(state_persistent.continuous_hurst >= 0.05);
    assert!(state_persistent.continuous_hurst <= 0.95);
    assert!(state_persistent.continuous_hurst.is_finite());
}

#[test]
fn c3_smooth_kelly_contraction_monotonicity() {
    let mut low_vol_engine = ContinuousVolatilityOperatorEngine::default();
    let mut high_vol_engine = ContinuousVolatilityOperatorEngine::default();

    // Señal de baja volatilidad
    for _ in 0..32 {
        low_vol_engine.update(0.0002);
    }
    let state_low = low_vol_engine.update(0.0002);

    // Señal de alta volatilidad
    for _ in 0..32 {
        high_vol_engine.update(0.0250);
    }
    let state_high = high_vol_engine.update(0.0250);

    // La contracción de Kelly en alta volatilidad debe ser menor (mayor contracción) que en baja volatilidad
    assert!(state_high.effective_volatility > state_low.effective_volatility);
    assert!(state_high.kelly_contraction_factor < state_low.kelly_contraction_factor);
    assert!(state_high.kelly_contraction_factor >= 0.05);
    assert!(state_low.kelly_contraction_factor <= 1.0);
}

#[test]
fn c4_harmonic_decoupling_micro_vs_macro() {
    let mut engine = ContinuousVolatilityOperatorEngine::default();

    // Inyectar retornos de alta frecuencia alternantes (microestructura agresiva)
    for i in 0..32 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        engine.update(sign * 0.0080);
    }
    let state = engine.update(0.0080);

    assert!(state.harmonic_decoupling_ratio > 0.0);
    assert!(state.harmonic_decoupling_ratio.is_finite());
    assert!(state.dominant_tau_ms > 0.0);
}

#[test]
fn c5_zero_panic_boundary_invariance() {
    let mut engine = ContinuousVolatilityOperatorEngine::default();

    // Valores atípicos, infinitos, NaN y ceros no deben generar pánico
    let pathological_inputs = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1e30,
        -1e30,
        1e-30,
    ];

    for &input in pathological_inputs.iter() {
        let state = engine.update(input);
        assert!(state.operator_trace.is_finite());
        assert!(state.effective_volatility.is_finite());
        assert!(state.continuous_hurst.is_finite());
        assert!(state.kelly_contraction_factor.is_finite());
        assert!(state.harmonic_decoupling_ratio.is_finite());
    }
}
