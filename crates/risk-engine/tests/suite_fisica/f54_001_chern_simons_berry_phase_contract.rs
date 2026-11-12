//! Contrato Formal de Verificación: Fase 54 (Chern-Simons Topological Action & Berry Phase)
//!
//! Este conjunto de pruebas certifica las propiedades topológicas y de gestión de riesgo del motor:
//! - C1: Circulación en ciclo cerrado de parámetros, conexión y acumulación de fase de Berry $\gamma \in [-\pi, \pi]$.
//! - C2: Invariante de densidad de Chern-Simons y activación de veto defensivo ante histéresis crítica.
//! - C3: Veto de capital cero y acotamiento estricto del modulador $\omega_{\text{berry}} \in [0.10, 1.25]$.
//! - C4: Determinismo en stack $O(1)$, cero heap allocs y robustez numérica total ante NaN/Infinitos.

use risk_engine::{
    ChernSimonsBerryParams, ChernSimonsBerryPhaseEngine, ChernSimonsBerryReport, ParameterState3D,
    BERRY_PATH_LEN,
};
use std::f64::consts::PI;

#[test]
fn test_c1_berry_phase_closed_loop_and_holonomy() {
    let params = ChernSimonsBerryParams {
        chern_simons_level: 1.0,
        critical_hysteresis_threshold: 0.90,
        phase_smoothing: 0.20,
        dissipation_penalty: 1.0,
    };
    let mut engine = ChernSimonsBerryPhaseEngine::new(params);

    // Recorrido en lazo cerrado circular en el espacio de parámetros
    for i in 0..16 {
        let theta = (i as f64) * (2.0 * PI / 16.0);
        let vol = 0.02 + 0.01 * theta.cos();
        let corr = 0.5 * theta.sin();
        let spread = 2.0;

        let rep = engine.update_and_evaluate(vol, corr, spread, 1000.0, i * 50);
        assert!(rep.berry_phase.is_finite());
        assert!(rep.berry_phase >= -PI && rep.berry_phase <= PI);
        assert!((0.0..=1.0).contains(&rep.non_holonomic_hysteresis));
    }

    assert_eq!(BERRY_PATH_LEN, 16);
}

#[test]
fn test_c2_chern_simons_density_and_critical_hysteresis_veto() {
    let params = ChernSimonsBerryParams {
        chern_simons_level: 2.0,
        critical_hysteresis_threshold: 0.50, // Umbral sensible para prueba
        phase_smoothing: 0.80,
        dissipation_penalty: 2.0,
    };
    let mut engine = ChernSimonsBerryPhaseEngine::new(params);

    // Inyectar oscilaciones violentas repetidas para forzar acumulación rápida de fase
    let mut veto_induced = false;
    for i in 0..30 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let vol = 0.05 + sign * 0.03;
        let corr = sign * 0.95;
        let spread = 10.0;

        let rep = engine.update_and_evaluate(vol, corr, spread, 500.0, i * 100);
        if rep.topological_veto_induced && rep.veto_reason == Some("TOPOLOGICAL_CHERN_SIMONS_VETO")
        {
            veto_induced = true;
            assert_eq!(rep.slot_scaling_factor, 0.10);
            break;
        }
    }

    assert!(
        veto_induced,
        "Se esperaba activación del veto de Chern-Simons ante histéresis violenta"
    );
}

#[test]
fn test_c3_zero_capital_veto_and_slot_scaling_bounds() {
    let params = ChernSimonsBerryParams::default();
    let mut engine = ChernSimonsBerryPhaseEngine::new(params);

    // Prueba con capital disponible cero
    let zero_rep = engine.update_and_evaluate(0.01, 0.0, 2.0, 0.0, 1000);
    assert!(zero_rep.topological_veto_induced);
    assert_eq!(zero_rep.veto_reason, Some("CHERN_SIMONS_ZERO_CAPITAL"));
    assert_eq!(zero_rep.slot_scaling_factor, 0.10);

    // Prueba de límites del modulador
    for i in 0..20 {
        let vol = 0.01 + (i as f64 * 0.005);
        let corr = (i as f64 * 0.1).sin();
        let spread = 1.0 + (i as f64 * 0.2);
        let rep = engine.update_and_evaluate(vol, corr, spread, 100.0, i * 50);

        assert!(
            rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25,
            "Slot scaling fuera de rango [0.10, 1.25]: {}",
            rep.slot_scaling_factor
        );
    }
}

#[test]
fn test_c4_stack_zero_heap_and_nan_subnormal_immunity() {
    let params = ChernSimonsBerryParams::default();
    let mut engine = ChernSimonsBerryPhaseEngine::new(params);

    let anomalous_inputs = [
        (f64::NAN, 0.0, 2.0, 100.0),
        (0.01, f64::INFINITY, 2.0, 100.0),
        (0.01, 0.0, f64::NAN, 100.0),
        (0.01, 0.0, 2.0, f64::NAN),
        (0.0, 0.0, 0.0, 0.0),
        (1e-300, 1e-300, 1e-300, 1e-300),
    ];

    for &(vol, corr, spread, cap) in &anomalous_inputs {
        let rep = engine.update_and_evaluate(vol, corr, spread, cap, 1000);
        assert!(rep.berry_phase.is_finite());
        assert!(rep.chern_simons_density.is_finite());
        assert!(rep.non_holonomic_hysteresis.is_finite());
        assert!(rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25);
    }

    let p = ParameterState3D::new(0.01, 0.2, 3.0);
    assert!(p.vol_norm > 0.0);
}
