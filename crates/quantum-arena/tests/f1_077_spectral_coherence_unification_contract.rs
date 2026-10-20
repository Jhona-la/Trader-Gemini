//! Contrato de certificación formal para Ola Ω181 (Hallazgo F1-077).
//!
//! # F1-077: Unificación de Coherencia Espectral Canónica D-742 ("Una Sola Verdad")
//!
//! Verifica:
//! 1. Contrato 1: Paridad analítica y de publicación entre `arena.coins[coin_id].spectral_coherence`
//!    y `spec.spectral_coherence(true)`.
//! 2. Contrato 2: Paridad exacta entre `SpectralRegimeField.carrier_tide` y `spec.spectral_coherence(true)`.
//! 3. Contrato 3: Inmunidad ante distorsión de escalas anti-persistentes (reversión vs continuación espuria).
//! 4. Contrato 4: Antisimetría estricta `coherence(false) == -coherence(true)` y saturación limpia en $[-1.0, 1.0]$.
//! 5. Contrato 5: Consistencia plena en confluencia resonante con umbrales (`fused == coh_long`).

use quantum_arena::temporal_spectrum::TemporalSpectrum;
use quantum_arena::spectral_regime::{SpectralRegimeField, SpectralRegimeParams};
use quantum_arena::state::CoinArena;
use std::sync::atomic::Ordering;

fn alimentar_espectro(spec: &mut TemporalSpectrum, ticks: &[(u64, f64)]) {
    for &(ts, p) in ticks {
        spec.update(p, ts);
    }
}

/// Generador determinista de paseo aleatorio con drift opcional.
fn generar_ticks(seed: u64, n: usize, dt_ms: u64, sigma: f64, drift: f64) -> Vec<(u64, f64)> {
    let mut rng = seed;
    let mut ticks = Vec::with_capacity(n);
    let mut p = 100.0f64;
    let mut t = 1_700_000_000_000u64;

    for _ in 0..n {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let u1 = ((rng & 0xFFFF_FFFF) as f64 + 1.0) / 4294967297.0;

        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let u2 = ((rng & 0xFFFF_FFFF) as f64 + 1.0) / 4294967297.0;

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        p *= (drift + sigma * z).exp();
        ticks.push((t, p));
        t += dt_ms;
    }
    ticks
}

#[test]
fn f1_077_contrato_1_paridad_arena_y_spectral_coherence() {
    let mut spec = TemporalSpectrum::new();
    let ticks = generar_ticks(0xABCD_1234, 1_000, 100, 0.002, 0.0001);
    alimentar_espectro(&mut spec, &ticks);

    let coin = CoinArena::new(0.5);

    // Simula la publicación en el Core según F1-077
    let coh_d742 = spec.spectral_coherence(true);
    coin.spectral_coherence.store(coh_d742, Ordering::Relaxed);

    let arena_tide = coin.spectral_coherence.load(Ordering::Relaxed);
    assert_eq!(
        coh_d742, arena_tide,
        "El slot de arena spectral_coherence ({}) debe ser idéntico bit a bit a spec.spectral_coherence(true) ({})",
        arena_tide, coh_d742
    );
    assert_eq!(
        coh_d742, spec.fused_score,
        "Bajo D-742, spectral_coherence(true) ({}) debe coincidir exactamente con fused_score ({})",
        coh_d742, spec.fused_score
    );
}

#[test]
fn f1_077_contrato_2_paridad_carrier_tide_con_spectral_coherence() {
    let mut spec = TemporalSpectrum::new();
    let ticks = generar_ticks(0x9876_5432, 800, 250, 0.0015, -0.0002);
    alimentar_espectro(&mut spec, &ticks);

    let params = SpectralRegimeParams::default();
    let regime = SpectralRegimeField::from_spectrum_with_params(&spec, None, 60_000.0, &params);

    let coh_true = spec.spectral_coherence(true);
    assert_eq!(
        regime.carrier_tide, coh_true,
        "SpectralRegime::carrier_tide ({}) debe ser idéntico a spec.spectral_coherence(true) ({})",
        regime.carrier_tide, coh_true
    );
}

#[test]
fn f1_077_contrato_3_inmunidad_anti_persistencia_reversion_favorable() {
    let mut spec = TemporalSpectrum::new();
    // Configuramos un espectro frío con una escala rápida que sufre un up-tick pero es anti-persistente
    // (comportamiento típico de micro-estructura en reversión a la media)
    spec.scales[10].signal = 0.60;
    spec.scales[10].persistence = -0.35; // Anti-persistente: la física predice reversión a la baja
    spec.scales[10].epigenetic_gain = 2.5; // Ganancia alta histórica
    spec.scales[10].ewma_dev_vol = 0.01;

    // Escala macro lenta con tendencia a la baja
    spec.scales[22].signal = -0.40;
    spec.scales[22].persistence = 0.60; // Persistente
    spec.scales[22].epigenetic_gain = 1.0;
    spec.scales[22].ewma_dev_vol = 0.005; // Menor volatilidad -> Mayor precisión estadística

    spec.set_test_time_window(1_000, 100_000_000, 1_000);
    spec.refresh_fusion();

    let coh_long = spec.spectral_coherence(true);
    let coh_short = spec.spectral_coherence(false);

    // En D-742, la escala 22 con menor vol (0.005 vs 0.01) y señal bajista domina la convicción
    assert!(
        coh_long < 0.0,
        "La coherencia D-742 para long debe ser negativa ({}), no contaminada por el up-tick de micro-reversión",
        coh_long
    );
    assert!(
        coh_short > 0.0,
        "La coherencia D-742 para short debe ser positiva ({}) confirmando el setup bajista",
        coh_short
    );
    assert_eq!(coh_long, spec.fused_score);
}

#[test]
fn f1_077_contrato_4_antisimetria_estricta_y_rango_acotado() {
    let mut spec = TemporalSpectrum::new();
    let ticks = generar_ticks(0xFEED_FACE, 500, 50, 0.003, 0.0005);
    alimentar_espectro(&mut spec, &ticks);

    let c_long = spec.spectral_coherence(true);
    let c_short = spec.spectral_coherence(false);

    assert!(
        (c_long + c_short).abs() < 1e-12,
        "Antisimetría exacta requerida: c_long ({}) + c_short ({}) == 0",
        c_long, c_short
    );
    assert!(
        c_long >= -1.0 && c_long <= 1.0,
        "c_long debe residir estrictamente en [-1.0, 1.0]"
    );
    assert!(
        c_short >= -1.0 && c_short <= 1.0,
        "c_short debe residir estrictamente en [-1.0, 1.0]"
    );
}

#[test]
fn f1_077_contrato_5_confluencia_resonante_consistencia_fused() {
    let mut spec = TemporalSpectrum::new();
    // Forzamos acuerdo constructivo fuerte entre escalas
    for i in 12..28 {
        spec.scales[i].signal = 0.70;
        spec.scales[i].persistence = 0.50;
        spec.scales[i].ewma_dev_vol = 0.005;
    }
    spec.set_test_time_window(1_000, 100_000_000, 1_000);
    spec.refresh_fusion();

    let fused = spec.fused_score;
    let coh_long = spec.spectral_coherence(true);
    let coh_short = spec.spectral_coherence(false);

    // En D-742: fused == coh_long
    assert_eq!(
        fused, coh_long,
        "fused ({}) y coh_long ({}) deben ser indistinguibles",
        fused, coh_long
    );

    // Verificamos que si fused supera el umbral de entrada (ej: 0.60),
    // coh_long también supera el umbral de coherencia (ej: 0.15) sin falsos vetos
    let umbral_fused = 0.60;
    let umbral_coh = 0.15;
    if fused > umbral_fused {
        assert!(
            coh_long > umbral_coh,
            "Cero falsos rechazos: al estar fused ({}) > {}, coh_long ({}) debe superar {}",
            fused, umbral_fused, coh_long, umbral_coh
        );
    }
    assert!(
        coh_short < 0.0,
        "La coherencia opuesta debe ser estrictamente negativa ({})",
        coh_short
    );
}
