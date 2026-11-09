//! Test de contrato F71-001: Exponentes de Hurst Multifractales y MF-DFA.
//!
//! Verifica las invariantes fundamentales de la Fase 71:
//! 1. Exponentes generalizados de Hurst $h(q)$ acotados en $[0.01, 1.50]$ con respuesta multifractal.
//! 2. Espectro de singularidad con ancho $\Delta \alpha \ge 0$ y asimetría $A \in [-1.0, 1.0]$.
//! 3. Modulación continua de convicción de señal $\in [0.50, 1.35]$ y detección de régimen turbulento.
//! 4. Invariante `rb_p1`: cero pánicos ante valores patológicos (NaN, Inf, series constantes).

use signal_engine::mfdfa_multifractal_hurst::{
    MfDfaHurstReport, MfDfaMultifractalHurstEngine, MfDfaParams, MFDFA_WINDOW_CAPACITY,
};

#[test]
fn test_c1_mfdfa_generalized_hurst_exponents_bounded() {
    let params = MfDfaParams::default();
    let mut engine = MfDfaMultifractalHurstEngine::new(params);

    // Serie sintética persistente con tendencia fractal
    for i in 0..MFDFA_WINDOW_CAPACITY {
        let ret = 0.005 * ((i as f64) * 0.2).sin() + 0.001;
        let rep = engine.update(ret);
        assert!(rep.generalized_hurst_h2 >= 0.01 && rep.generalized_hurst_h2 <= 1.50);
        assert!(rep.generalized_hurst_h_minus2 >= 0.01 && rep.generalized_hurst_h_minus2 <= 1.50);
        assert!(rep.generalized_hurst_h0 >= 0.01 && rep.generalized_hurst_h0 <= 1.50);
    }
}

#[test]
fn test_c2_singularity_spectrum_width_and_asymmetry() {
    let params = MfDfaParams::default();
    let mut engine = MfDfaMultifractalHurstEngine::new(params);

    // Serie de retornos con heterocedasticidad pronunciada (volatilidad intermitente)
    for i in 0..MFDFA_WINDOW_CAPACITY {
        let shock = if i % 5 == 0 { 0.04 } else { 0.002 * (i as f64).cos() };
        let rep = engine.update(shock);

        if i >= 16 {
            assert!(rep.multifractal_spectrum_width >= 0.0, "Ancho de espectro debe ser no negativo");
            assert!(
                rep.spectral_asymmetry >= -1.0 && rep.spectral_asymmetry <= 1.0,
                "Asimetría espectral debe estar en [-1, 1]"
            );
        }
    }
}

#[test]
fn test_c3_conviction_modulation_and_persistence_regimes() {
    let params = MfDfaParams::default();
    let mut engine = MfDfaMultifractalHurstEngine::new(params);

    // Serie de tendencia sostenida
    let mut last_rep = MfDfaHurstReport::default();
    for i in 0..MFDFA_WINDOW_CAPACITY {
        let ret = 0.003 + 0.0005 * (i as f64);
        last_rep = engine.update(ret);
    }

    assert!(
        last_rep.signal_conviction_modulator >= params.min_conviction
            && last_rep.signal_conviction_modulator <= params.max_conviction,
        "Modulador de convicción debe estar acotado"
    );
}

#[test]
fn test_c4_zero_panic_rb_p1_resilience() {
    let params = MfDfaParams::default();
    let mut engine = MfDfaMultifractalHurstEngine::new(params);

    let pathological = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, 1e20, -1e20];

    for &val in &pathological {
        let rep = engine.update(val);
        assert!(rep.generalized_hurst_h2.is_finite());
        assert!(rep.multifractal_spectrum_width.is_finite());
        assert!(rep.spectral_asymmetry.is_finite());
        assert!(rep.signal_conviction_modulator.is_finite());
    }
}
