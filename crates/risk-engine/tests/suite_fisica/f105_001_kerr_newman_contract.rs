//! # Contrato de Pruebas: Fase 105 (§105) — Métrica Kerr-Newman & Termodinámica Bardeen-Carter-Hawking
//!
//! Verifica las invariantes geométricas del horizonte de eventos, censura cósmica de Penrose,
//! extracción de energía de ergosfera y resiliencia determinista RB-P1.

use core::f64::consts::PI;
use risk_engine::kerr_newman_black_hole::{KerrNewmanBlackHoleEngine, KerrNewmanConfig};

#[test]
fn test_c1_kerr_newman_initialization_and_purity() {
    let mut engine = KerrNewmanBlackHoleEngine::new(KerrNewmanConfig::default());
    let rep = engine.update(1.0, 0.20, 0.10, PI * 0.50);

    assert!(rep.event_horizon_r_plus > 1.0);
    assert!(rep.inner_cauchy_horizon_r_minus < 1.0);
    assert!(rep.inner_cauchy_horizon_r_minus > 0.0);
    assert!(rep.ergosphere_radius_equator >= rep.event_horizon_r_plus);
    assert!(rep.hawking_temperature >= 0.0);
    assert!(rep.bekenstein_hawking_entropy > 0.0);
    assert!(!rep.naked_singularity_veto);
    assert!(!rep.hawking_thermal_evaporation_veto);
    assert!(rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25);
}

#[test]
fn test_c2_event_horizon_and_ergosphere_geometry() {
    let mut engine = KerrNewmanBlackHoleEngine::default();
    let m = 2.0;
    let a = 0.50;
    let q = 0.30;
    let rep = engine.update(m, a, q, PI * 0.50);

    let expected_disc = (m * m - a * a - q * q).sqrt();
    let expected_r_plus = m + expected_disc;
    let expected_r_minus = m - expected_disc;

    assert!((rep.event_horizon_r_plus - expected_r_plus).abs() < 1e-10);
    assert!((rep.inner_cauchy_horizon_r_minus - expected_r_minus).abs() < 1e-10);

    let expected_area = 4.0 * PI * (expected_r_plus * expected_r_plus + a * a);
    let expected_entropy = expected_area * 0.25;
    assert!((rep.bekenstein_hawking_entropy - expected_entropy).abs() < 1e-10);
}

#[test]
fn test_c3_penrose_process_energy_efficiency() {
    let mut engine = KerrNewmanBlackHoleEngine::default();
    // Agujero negro en rotación rápida sub-extremal
    let rep = engine.update(1.5, 0.90, 0.20, PI * 0.50);

    assert!(rep.irreducible_mass < 1.5);
    assert!(rep.penrose_efficiency > 0.0);
    assert!(rep.penrose_efficiency <= 0.50);
    assert!(rep.slot_scaling_factor >= 0.10);
}

#[test]
fn test_c4_cosmic_censorship_naked_singularity_veto() {
    let mut engine = KerrNewmanBlackHoleEngine::default();
    // Parámetros super-extremales donde a^2 + Q^2 > M^2
    let rep = engine.update(1.0, 0.95, 0.95, PI * 0.50);

    assert!(rep.naked_singularity_veto);
    assert!((rep.slot_scaling_factor - 0.10).abs() < 1e-6);
}

#[test]
fn test_c5_kerr_newman_rb_p1_resilience() {
    let mut engine = KerrNewmanBlackHoleEngine::default();
    // Parámetros con NaN e infinitos
    let rep_nan = engine.update(f64::NAN, f64::INFINITY, -f64::INFINITY, f64::NAN);
    assert!(rep_nan.event_horizon_r_plus.is_finite());
    assert!(rep_nan.inner_cauchy_horizon_r_minus.is_finite());
    assert!(rep_nan.hawking_temperature.is_finite());
    assert!(rep_nan.bekenstein_hawking_entropy.is_finite());
    assert!(rep_nan.slot_scaling_factor >= 0.10);

    // Parámetro con masa cercana a cero
    let rep_zero = engine.update(0.0, 0.0, 0.0, 0.0);
    assert!(rep_zero.event_horizon_r_plus.is_finite());
    assert!(rep_zero.slot_scaling_factor >= 0.10);
}
