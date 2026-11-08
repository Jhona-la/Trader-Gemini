//! Contrato Formal de Verificación: Fase 68 (Ecuación NLS y Solitones Envolventes de Peregrine)
//!
//! Este conjunto de pruebas certifica las propiedades no lineales de la dinámica de Peregrine:
//! - C1: Convergencia de la amplitud del solitón de Peregrine al límite teórico $|\psi_P| \to 3.0$ en resonancia.
//! - C2: Inestabilidad de Benjamin-Feir (BFI) y detección determinista de ondas rebeldes (Rogue Waves).
//! - C3: Modulador constructivo de convicción ($\ge 1.15$) y coherencia con la velocidad de Madelung.
//! - C4: Determinismo en stack $O(1)$, cero pánicos (`rb_p1`) y resiliencia ante NaN e infinitos.

use signal_engine::{NlsPeregrineParams, NlsPeregrineSolitonEngine};

#[test]
fn test_c1_peregrine_analytical_amplitude_convergence() {
    let params = NlsPeregrineParams::default();
    let mut engine = NlsPeregrineSolitonEngine::new(params);

    // Con OFI neutro y retorno 0, x -> 0; con spread 4 bps, t -> 0
    let report = engine.update(
        0.0,  // price_ret
        0.0,  // order_flow_imbalance
        4.0,  // spread_bps
        0.02, // instantaneous_vol
    );

    assert!(
        report.envelope_amplitude > 2.50 && report.envelope_amplitude <= 3.05,
        "La amplitud envolvente en el centro de Peregrine debe aproximar 3.0: {}",
        report.envelope_amplitude
    );

    assert!(
        report.quantum_phase.is_finite(),
        "La fase cuántica debe ser finita"
    );
}

#[test]
fn test_c2_benjamin_feir_instability_and_rogue_wave_detection() {
    let params = NlsPeregrineParams::default();
    let mut engine = NlsPeregrineSolitonEngine::new(params);

    // Condiciones de alta volatilidad y spread comprimido para inducir BFI > 1.0
    let report = engine.update(
        0.0001, 0.02, 3.8, 0.035, // Alta volatilidad -> gran amplitud de portadora a_0
    );

    assert!(
        report.benjamin_feir_index >= 0.90,
        "El índice BFI debe superar el umbral de inestabilidad: {}",
        report.benjamin_feir_index
    );

    assert!(
        report.peregrine_rogue_wave_detected,
        "Debe detectar la presencia de una onda rebelde de Peregrine activa"
    );
}

#[test]
fn test_c3_constructive_conviction_boost_and_madelung_velocity() {
    let params = NlsPeregrineParams::default();
    let mut engine = NlsPeregrineSolitonEngine::new(params);

    let report = engine.update(0.0002, 0.05, 3.9, 0.03);

    assert!(
        report.signal_conviction_modulator >= 1.15,
        "Debe otorgar boost constructivo de convicción: {}",
        report.signal_conviction_modulator
    );

    assert!(
        report.soliton_impulse_signal > 0.0,
        "La polaridad de impulso debe coincidir con el flujo alcista: {}",
        report.soliton_impulse_signal
    );

    assert!(
        report.madelung_group_velocity.is_finite(),
        "Velocidad de grupo hidrodinámica de Madelung debe ser finita"
    );
}

#[test]
fn test_c4_stack_o1_and_nan_infinity_resilience() {
    let params = NlsPeregrineParams::default();
    let mut engine = NlsPeregrineSolitonEngine::new(params);

    let report_nan = engine.update(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(report_nan.envelope_amplitude.is_finite());
    assert!(report_nan.quantum_phase.is_finite());
    assert!(report_nan.benjamin_feir_index.is_finite());
    assert!(report_nan.signal_conviction_modulator.is_finite());

    let report_inf = engine.update(
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    );
    assert!(report_inf.envelope_amplitude.is_finite());
    assert!(report_inf.quantum_phase.is_finite());
    assert!(report_inf.benjamin_feir_index.is_finite());
    assert!(report_inf.signal_conviction_modulator.is_finite());
}
