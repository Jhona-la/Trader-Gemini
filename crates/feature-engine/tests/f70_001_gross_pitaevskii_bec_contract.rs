//! Test de contrato F70-001: Condensados de Bose-Einstein (BEC) y Vórtices Cuantizados de Gross-Pitaevskii.
//!
//! Verifica las invariantes fundamentales de la Fase 70:
//! 1. Fracción de condensado en ground state $f_0 \in [0.0, 1.0]$, velocidad de Bogoliubov $c_s > 0$ y healing length $\xi > 0$.
//! 2. Cuantización de circulación de Onsager-Feynman: $\Gamma = \frac{2\pi \hbar}{m} n$ con $n \in \mathbb{Z}$.
//! 3. Detección de vórtice cuántico activo y modulación defensiva de slot ($0.15$).
//! 4. Invariante `rb_p1`: cero pánicos ante valores patológicos (NaN, Inf, desbalances extremos).

use feature_engine::gross_pitaevskii_bec::{
    GrossPitaevskiiBecEngine, GrossPitaevskiiParams, HBAR_EFF, PARTICLE_MASS_M,
};
use core::f64::consts::PI;

#[test]
fn test_c1_gross_pitaevskii_ground_state_and_bogoliubov_speed() {
    let params = GrossPitaevskiiParams::default();
    let mut engine = GrossPitaevskiiBecEngine::new(params);

    // Entorno pasivo calmo: bajo spread (1.0 bps), baja volatilidad (0.005), flujo balanceado
    let rep = engine.update(0.0001, 0.05, 0.02, 1.0, 0.005);

    assert!(rep.condensate_fraction >= 0.0 && rep.condensate_fraction <= 1.0);
    assert!(rep.condensate_fraction > 0.50, "En calma la fracción de condensado debe ser dominante");
    assert!(rep.bogoliubov_sound_speed > 0.0, "Velocidad del sonido de Bogoliubov debe ser estrictamente positiva");
    assert!(rep.healing_length > 0.0 && rep.healing_length <= 10.0, "Healing length debe ser finita");
    assert!(!rep.quantum_vortex_shedding_detected, "No debe haber desprendimiento de vórtice en calma");
    assert!(rep.slot_scale_factor >= 0.85, "Slot scale factor debe ser constructivo en condensado denso");
}

#[test]
fn test_c2_onsager_feynman_circulation_quantization() {
    let params = GrossPitaevskiiParams::default();
    let mut engine = GrossPitaevskiiBecEngine::new(params);

    // Múltiples ticks con rotación en el espacio de fase
    let mut charges = Vec::new();
    for i in 0..15 {
        let angle = (i as f64) * 0.4;
        let ret = 0.01 * angle.cos();
        let ofi = 0.8 * angle.sin();
        let depth = 0.5 * angle.cos();
        let rep = engine.update(ret, ofi, depth, 2.5, 0.02);
        charges.push(rep.quantized_vortex_charge);

        // Verificación de la ley física de circulación: \Gamma = (2\pi \hbar / m) * n
        let expected_gamma = (2.0 * PI * HBAR_EFF / PARTICLE_MASS_M) * (rep.quantized_vortex_charge as f64);
        let diff = (rep.superfluid_circulation - expected_gamma).abs();
        assert!(diff < 1e-10, "La circulación superfluida debe estar rígidamente cuantizada según Onsager-Feynman");
    }

    assert_eq!(charges.len(), 15);
}

#[test]
fn test_c3_vortex_shedding_and_defensive_slot_scaling() {
    let params = GrossPitaevskiiParams::default();
    let mut engine = GrossPitaevskiiBecEngine::new(params);

    // Provocamos cavitación térmica: spread alto (15.0 bps) y volatilidad extrema (0.45)
    let rep = engine.update(0.15, -0.90, -0.85, 15.0, 0.45);

    assert!(rep.quantum_vortex_shedding_detected, "Debe detectarse desprendimiento de vórtice / cavitación");
    assert_eq!(
        rep.slot_scale_factor, params.min_slot_scale,
        "El modulador de slot debe colapsar al mínimo defensivo (0.15)"
    );
}

#[test]
fn test_c4_zero_panic_rb_p1_resilience() {
    let params = GrossPitaevskiiParams::default();
    let mut engine = GrossPitaevskiiBecEngine::new(params);

    let pathological_inputs = [
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, 0.0, -10.0, -5.0),
        (1e20, -1e20, 1e20, 1e20, 1e20),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ];

    for &(ret, ofi, depth, spread, vol) in &pathological_inputs {
        let rep = engine.update(ret, ofi, depth, spread, vol);
        assert!(rep.condensate_fraction.is_finite());
        assert!(rep.central_density.is_finite());
        assert!(rep.bogoliubov_sound_speed.is_finite());
        assert!(rep.healing_length.is_finite());
        assert!(rep.superfluid_circulation.is_finite());
        assert!(rep.slot_scale_factor.is_finite());
    }
}
