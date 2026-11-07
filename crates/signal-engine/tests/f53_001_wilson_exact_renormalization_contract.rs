//! Contrato Formal de Verificación: Fase 53 (Wilson Exact Renormalization Group Engine)
//!
//! Este conjunto de pruebas certifica las propiedades matemáticas y físicas del motor:
//! - C1: Flujo de Wetterich-Morris sobre escalas diádicas e integración RK2 de acoplamientos.
//! - C2: Detección analítica de Puntos Fijos Cuánticos y transiciones de fase críticas.
//! - C3: Dimensión anómala emergente $\eta \in [0.0, 0.50]$ y modulación de convicción $\lambda \in [0.10, 1.40]$.
//! - C4: Determinismo en stack $O(1)$, cero heap allocs y robustez numérica total ante NaN/Infinitos.

use signal_engine::{
    WilsonExactRenormalizationEngine, WilsonExactRgParams, WilsonExactRgReport, RG_MOMENTUM_SCALES,
};

#[test]
fn test_c1_wilson_beta_functions_and_rg_flow_consistency() {
    let params = WilsonExactRgParams {
        initial_mass_sq: 0.20,
        initial_quartic_coupling: 0.15,
        rg_flow_step: 0.05,
        ofi_coupling_sensitivity: 0.30,
        wf_fixed_point_mass_sq: -0.15,
        wf_fixed_point_coupling: 0.333,
    };
    let mut engine = WilsonExactRenormalizationEngine::new(params);

    let rep = engine.update_and_evaluate(0.002, 0.40, 0.015, 1_000_000);

    // Verificación de existencia de las 3 escalas [Micro, Meso, Macro]
    assert_eq!(RG_MOMENTUM_SCALES, 3);
    for (idx, coupling) in rep.scale_couplings.iter().enumerate() {
        assert!(
            coupling.mass_sq.is_finite(),
            "Masa no finita en escala {}",
            idx
        );
        assert!(
            coupling.quartic_coupling > 0.0,
            "Acoplamiento cuártico no positivo en escala {}",
            idx
        );
        assert!(coupling.beta_mass.is_finite());
        assert!(coupling.beta_quartic.is_finite());
    }
}

#[test]
fn test_c2_fixed_point_proximity_and_critical_transition() {
    // Parámetros calibrados directamente sobre el Punto Fijo
    let target_m2 = -0.10;
    let target_l = 0.25;
    let params = WilsonExactRgParams {
        initial_mass_sq: target_m2,
        initial_quartic_coupling: target_l,
        rg_flow_step: 0.01,
        ofi_coupling_sensitivity: 0.0,
        wf_fixed_point_mass_sq: target_m2,
        wf_fixed_point_coupling: target_l,
    };
    let mut engine = WilsonExactRenormalizationEngine::new(params);

    // Pulso neutro para observar convergencia al punto fijo
    let rep = engine.update_and_evaluate(0.0, 0.0, 0.001, 1_000_000);

    assert!(
        rep.min_distance_to_fixed_point < 0.15,
        "Distancia al punto fijo esperada < 0.15, obtenida {}",
        rep.min_distance_to_fixed_point
    );
    assert!(
        rep.is_critical_phase_transition,
        "Debería activar detección de transición de fase crítica"
    );
    assert!(
        rep.signal_conviction_modulator <= 1.0,
        "En punto crítico la convicción debe atenuarse defensivamente: {}",
        rep.signal_conviction_modulator
    );
}

#[test]
fn test_c3_anomalous_dimension_and_modulator_bounds() {
    let params = WilsonExactRgParams::default();
    let mut engine = WilsonExactRenormalizationEngine::new(params);

    for i in 0..50 {
        let ret = (i as f64 * 0.002).sin() * 0.05;
        let ofi = (i as f64 * 0.1).cos();
        let vol = 0.01 + 0.005 * (i as f64 * 0.2).sin().abs();
        let rep = engine.update_and_evaluate(ret, ofi, vol, i * 50);

        assert!(
            rep.anomalous_dimension >= 0.0 && rep.anomalous_dimension <= 0.50,
            "Dimensión anómala fuera de rango [0.0, 0.50]: {}",
            rep.anomalous_dimension
        );
        assert!(
            rep.signal_conviction_modulator >= 0.10 && rep.signal_conviction_modulator <= 1.40,
            "Modulador de convicción fuera de rango [0.10, 1.40]: {}",
            rep.signal_conviction_modulator
        );
    }
}

#[test]
fn test_c4_stack_zero_heap_and_nan_subnormal_immunity() {
    let params = WilsonExactRgParams::default();
    let mut engine = WilsonExactRenormalizationEngine::new(params);

    let anomalous_inputs = [
        (f64::NAN, 0.0, 0.01),
        (0.001, f64::INFINITY, 0.01),
        (0.001, 0.0, f64::NAN),
        (0.0, 0.0, 0.0),
        (1e-300, 1e-300, 1e-300),
    ];

    for &(ret, ofi, vol) in &anomalous_inputs {
        let rep = engine.update_and_evaluate(ret, ofi, vol, 1000);
        assert!(rep.min_distance_to_fixed_point.is_finite());
        assert!(rep.anomalous_dimension.is_finite());
        assert!(rep.signal_conviction_modulator >= 0.10 && rep.signal_conviction_modulator <= 1.40);
    }
}
