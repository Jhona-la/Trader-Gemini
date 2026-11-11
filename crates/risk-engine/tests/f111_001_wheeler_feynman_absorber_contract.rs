//! Contrato Formal de Certificación: Fase 111 (§111) — Teoría del Absorbedor de Wheeler-Feynman & Radiación de Dirac
//!
//! Verifica en silicio:
//! 1. Invariantes de simetría temporal del absorbedor en estado nominal.
//! 2. Cálculo analítico de derivadas de orden 3 (jerk) y fuerza de radiación de Dirac.
//! 3. Activación determinista fail-closed del veto ante déficit del absorbedor.
//! 4. Detección determinista fail-closed del veto de fuga de auto-aceleración (runaway mode).
//! 5. Robustez numérica absoluta e inmunidad a NaN/Inf (invariante `rb_p1`).

use risk_engine::wheeler_feynman_absorber::{
    WheelerFeynmanAbsorberConfig, WheelerFeynmanAbsorberEngine,
};

#[test]
fn test_c1_nominal_wheeler_feynman_invariants() {
    let mut engine = WheelerFeynmanAbsorberEngine::new(WheelerFeynmanAbsorberConfig::default());

    let rep = engine.update(100.0, 0.05, 0.50, 1.5);

    assert_eq!(rep.liquidity_jerk, 0.0);
    assert!(rep.absorber_absorption_capacity > 0.0);
    assert!(rep.absorber_deficit <= 1.0);
    assert!(!rep.pre_acceleration_runaway_detected);
    assert!(!rep.wheeler_feynman_absorber_veto);
    assert!(!rep.dirac_radiation_runaway_veto);
    assert!(rep.slot_scale >= 0.10 && rep.slot_scale <= 1.25);
}

#[test]
fn test_c2_dirac_radiation_reaction_jerk_scaling() {
    let mut engine = WheelerFeynmanAbsorberEngine::new(WheelerFeynmanAbsorberConfig::default());

    // Serie temporal con jerk pronunciado (derivada tercera no nula)
    let prices = [100.0, 100.05, 100.20, 100.60, 101.50];
    let mut last_rep = engine.update(prices[0], 0.10, 0.50, 1.0);
    for &p in &prices[1..] {
        last_rep = engine.update(p, 0.10, 0.50, 1.0);
    }

    assert!(last_rep.liquidity_jerk.abs() > 0.0);
    assert!(last_rep.dirac_radiation_reaction_force.abs() > 0.0);
    assert!(
        (last_rep.dirac_radiation_reaction_force - last_rep.liquidity_jerk * (2.0 / 3.0)).abs()
            < 1e-9
    );
}

#[test]
fn test_c3_absorber_deficit_veto_propagation() {
    let mut config = WheelerFeynmanAbsorberConfig::default();
    config.max_absorber_deficit = 0.20; // Umbral ultraestricto de déficit
    let mut engine = WheelerFeynmanAbsorberEngine::new(config);

    // Spread y volatilidad masivos que colapsan la densidad del absorbedor
    let rep = engine.update(100.0, 0.90, 8.0, 45.0);

    assert!(rep.wheeler_feynman_absorber_veto);
    assert_eq!(rep.slot_scale, 0.10);
}

#[test]
fn test_c4_dirac_runaway_preacceleration_veto_propagation() {
    let mut config = WheelerFeynmanAbsorberConfig::default();
    config.max_runaway_jerk_threshold = 5.0;
    let mut engine = WheelerFeynmanAbsorberEngine::new(config);

    // Pulsos explosivos para inducir un jerk gigantesco
    engine.update(100.0, 0.0, 1.0, 1.0);
    engine.update(100.0, 0.0, 1.0, 1.0);
    engine.update(100.0, 0.0, 1.0, 1.0);
    let rep = engine.update(110.0, 0.0, 1.0, 1.0); // Salto discreto del 10%

    assert!(rep.pre_acceleration_runaway_detected);
    assert!(rep.dirac_radiation_runaway_veto);
    assert_eq!(rep.slot_scale, 0.10);
}

#[test]
fn test_c5_zero_panic_rb_p1_and_numerical_guards() {
    let mut engine = WheelerFeynmanAbsorberEngine::new(WheelerFeynmanAbsorberConfig::default());

    let degenerate_inputs = [
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, 1e25, -100.0),
        (-50.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 50.0, 200.0),
    ];

    for &(p, ofi, vol, spr) in &degenerate_inputs {
        let rep = engine.update(p, ofi, vol, spr);
        assert!(rep.liquidity_jerk.is_finite());
        assert!(rep.dirac_radiation_reaction_force.is_finite());
        assert!(rep.absorber_absorption_capacity.is_finite());
        assert!(rep.absorber_deficit.is_finite());
        assert!(rep.slot_scale >= 0.10 && rep.slot_scale <= 1.25);
    }
}
