//! CONTRATO FORMAL DE SÍMPLEX ESPECTRAL CONTINUO Y ENTROPÍA DE RÉNYI / SHANNON (OLA Ω58, FICHA #693)
//!
//! Verifica matemáticamente:
//!  1. Conservación estricta de la probabilidad en el símplex Δ³: Σ p_i = 1.0
//!  2. Inmunidad a entradas no finitas (NaN, ±inf, negativas)
//!  3. Cotas analíticas de entropía de Shannon: H ∈ [0, ln(4)]
//!  4. Generalización de Rényi H_α hacia Shannon cuando α → 1
//!  5. Polarización direccional continua Π_dir ∈ [-1, 1] e índice de turbulencia
//!  6. Proyección MAP discreta coherente con `MarketRegime` legado

use risk_engine::regime::{MarketRegime, RegimeDetector, SpectralMarketRegime};

#[test]
fn simplex_partition_of_unity_and_positivity() {
    let test_vectors = [
        (1.0, 0.0, 0.0, 0.0),
        (0.25, 0.25, 0.25, 0.25),
        (10.0, 20.0, 30.0, 40.0),
        (0.001, 0.002, 0.003, 0.004),
        (1e6, 2e6, 3e6, 4e6),
    ];

    for (r, b, c, ch) in test_vectors {
        let s = SpectralMarketRegime::new(r, b, c, ch);
        let sum = s.p_range + s.p_bull + s.p_crash + s.p_chaos;
        assert!((sum - 1.0).abs() < 1e-11, "La partición de la unidad debe ser 1.0 exacto");
        assert!(s.p_range >= 0.0 && s.p_range <= 1.0);
        assert!(s.p_bull >= 0.0 && s.p_bull <= 1.0);
        assert!(s.p_crash >= 0.0 && s.p_crash <= 1.0);
        assert!(s.p_chaos >= 0.0 && s.p_chaos <= 1.0);
    }
}

#[test]
fn simplex_nan_and_negative_immunity() {
    let s_nan = SpectralMarketRegime::new(f64::NAN, 1.0, f64::INFINITY, -5.0);
    let sum = s_nan.p_range + s_nan.p_bull + s_nan.p_crash + s_nan.p_chaos;
    assert!((sum - 1.0).abs() < 1e-11);
    assert_eq!(s_nan.p_bull, 1.0);
    assert_eq!(s_nan.p_range, 0.0);

    let s_all_nan = SpectralMarketRegime::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert_eq!(s_all_nan, SpectralMarketRegime::default());
    assert_eq!(s_all_nan.p_range, 1.0);
}

#[test]
fn shannon_and_renyi_entropy_bounds() {
    // 1. Estado puro (determinista): H = 0.0
    let pure = SpectralMarketRegime::new(1.0, 0.0, 0.0, 0.0);
    assert!(pure.shannon_entropy() < 1e-9, "Estado puro debe tener entropía nula");
    assert!(pure.renyi_entropy(2.0) < 1e-9);

    // 2. Estado uniforme (máxima incertidumbre en 4 dimensiones): H = ln(4) ≈ 1.386294
    let uniform = SpectralMarketRegime::new(1.0, 1.0, 1.0, 1.0);
    let expected_max_h = 4.0f64.ln();
    assert!(
        (uniform.shannon_entropy() - expected_max_h).abs() < 1e-4,
        "Estado uniforme debe alcanzar la cota máxima ln(4)"
    );

    // Rényi de orden 2.0 y 0.5 sobre uniforme debe coincidir con ln(4)
    assert!((uniform.renyi_entropy(2.0) - expected_max_h).abs() < 1e-4);
    assert!((uniform.renyi_entropy(0.5) - expected_max_h).abs() < 1e-4);

    // 3. Rényi de orden α → 1 debe converger suavemente a Shannon
    let non_uniform = SpectralMarketRegime::new(0.5, 0.3, 0.15, 0.05);
    let h_shannon = non_uniform.shannon_entropy();
    let h_renyi_near_1 = non_uniform.renyi_entropy(1.0001);
    assert!((h_shannon - h_renyi_near_1).abs() < 1e-3);
}

#[test]
fn directional_bias_and_turbulence_indices() {
    let bull_dominant = SpectralMarketRegime::new(0.05, 0.85, 0.05, 0.05);
    assert!(bull_dominant.directional_bias() > 0.70);
    assert_eq!(bull_dominant.map_discrete(), MarketRegime::BullRun);

    let crash_dominant = SpectralMarketRegime::new(0.05, 0.05, 0.85, 0.05);
    assert!(crash_dominant.directional_bias() < -0.70);
    assert!(crash_dominant.turbulence_index() > 0.80);
    assert_eq!(crash_dominant.map_discrete(), MarketRegime::Crash);

    let chaos_dominant = SpectralMarketRegime::new(0.10, 0.10, 0.10, 0.70);
    assert!(chaos_dominant.turbulence_index() > 0.75);
    assert_eq!(chaos_dominant.map_discrete(), MarketRegime::Chaotic);
}

#[test]
fn detector_continuous_spectral_smooth_updates() {
    let mut detector = RegimeDetector::new(0.6, 0.02);

    let s1 = detector.update_spectral(0.85, 0.04);
    assert_eq!(detector.current_regime, MarketRegime::BullRun);
    assert!(s1.p_bull > s1.p_crash);
    assert!(s1.directional_bias() > 0.0);

    let s2 = detector.update_spectral(0.85, -0.04);
    assert_eq!(detector.current_regime, MarketRegime::Crash);
    assert!(s2.p_crash > s2.p_bull);
    assert!(s2.directional_bias() < 0.0);

    // Entradas no finitas no mutan ni corrompen el estado anterior
    let s_nan = detector.update_spectral(f64::NAN, 0.0);
    assert_eq!(s_nan, s2);
    assert_eq!(detector.current_regime, MarketRegime::Crash);
}
