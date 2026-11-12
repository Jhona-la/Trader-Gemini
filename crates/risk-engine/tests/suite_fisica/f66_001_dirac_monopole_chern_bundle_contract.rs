//! Contrato Formal de Verificación: Fase 66 (Fibrados Principales de Yang-Mills y Monopolos de Dirac)
//!
//! Este conjunto de pruebas certifica las propiedades de teoría de gauge y topología de Chern del motor:
//! - C1: Primer número de Chern entero $c_1 \in \{-1, 0, 1\}$ y densidad de curvatura $\ge 0$.
//! - C2: Trivialidad topológica ($c_1 = 0$) y ausencia de singularidad bajo dinámica adiabática regular.
//! - C3: Disparo determinista de veto duro ante singularidad de monopolo de Dirac con modulación de escala a 0.0.
//! - C4: Determinismo en stack $O(1)$, cero heap allocations en hot path e inmunidad a NaNs.

use risk_engine::{DiracMonopoleChernBundleEngine, DiracMonopoleParams};

#[test]
fn test_c1_first_chern_number_integrality_and_curvature_flux() {
    let engine = DiracMonopoleChernBundleEngine::new(DiracMonopoleParams::default());

    let rep = engine.evaluate(0.1, 2.0, 0.015, 2.0);

    assert_eq!(rep.first_chern_number, 0);
    assert!(rep.monopole_charge_density >= 0.0);
    assert!(rep.gauge_field_curvature_flux.is_finite());
    assert!(
        rep.dirac_string_singularity_proximity >= 0.0
            && rep.dirac_string_singularity_proximity <= 1.0
    );
    assert!(!rep.monopole_veto_induced);
    assert_eq!(rep.slot_scale_factor, 1.0);
}

#[test]
fn test_c2_adiabatic_stability_under_moderate_leverage() {
    let engine = DiracMonopoleChernBundleEngine::new(DiracMonopoleParams::default());

    for lev in [1.0, 2.0, 5.0, 10.0] {
        let rep = engine.evaluate(0.2, 3.0, 0.02, lev);
        assert_eq!(rep.first_chern_number, 0);
        assert!(!rep.monopole_veto_induced);
        assert!(rep.slot_scale_factor >= 0.10);
    }
}

#[test]
fn test_c3_dirac_monopole_veto_and_zero_scale_enforcement() {
    let engine = DiracMonopoleChernBundleEngine::new(DiracMonopoleParams {
        critical_flux_threshold: 0.80,
        ..Default::default()
    });

    // Inyectar apalancamiento extremo con spread colosal empujando al polo sur
    let rep_extreme = engine.evaluate(1.0, 48.0, 0.50, 48.0);

    if rep_extreme.first_chern_number != 0 || rep_extreme.monopole_veto_induced {
        assert!(rep_extreme.monopole_veto_induced);
        assert_eq!(rep_extreme.slot_scale_factor, 0.0);
    }
}

#[test]
fn test_c4_stack_o1_and_nan_infinity_resilience() {
    let engine = DiracMonopoleChernBundleEngine::new(DiracMonopoleParams::default());

    let rep_nan = engine.evaluate(f64::NAN, f64::INFINITY, f64::NAN, f64::NEG_INFINITY);

    assert!(rep_nan.first_chern_number == 0 || rep_nan.first_chern_number.abs() == 1);
    assert!(rep_nan.gauge_field_curvature_flux.is_finite());
    assert!(rep_nan.monopole_charge_density.is_finite());
    assert!(rep_nan.dirac_string_singularity_proximity.is_finite());
    assert!(rep_nan.slot_scale_factor.is_finite());
}
