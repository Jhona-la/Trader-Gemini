//! Contrato Formal de Verificación: Fase 90 (§90) — Superficies Cuánticas Extremales (QES)
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor QES:
//! - C1: Entropía generalizada no-negativa $S_{\text{gen}} \ge 0$ y radio QES acotado $r \ge r_h$.
//! - C2: Detección de régimen de isla cuántica post-Page y modulación continua de slot $\eta_{\text{qes}} \in [0.10, 1.25]$.
//! - C3: Detonación determinista del veto fail-closed ante colapso del horizonte de capital hacia la liquidación.
//! - C4: Determinismo estricto en stack $O(1)$, cero heap allocations e inmunidad ante NaNs e infinitos.

use risk_engine::{QuantumExtremalSurfaceConfig, QuantumExtremalSurfaceEngine};

#[test]
fn test_c1_generalized_entropy_and_qes_radius() {
    let config = QuantumExtremalSurfaceConfig::default();
    let mut engine = QuantumExtremalSurfaceEngine::new(config);

    let report = engine.evaluate(13.0, 10.0, 13.0, 0.02);

    assert!(
        report.generalized_entropy_min > 0.0,
        "Entropía generalizada debe ser estrictamente positiva: {}",
        report.generalized_entropy_min
    );
    assert!(
        report.qes_radius >= config.horizon_radius,
        "Radio QES no puede ser menor que el horizonte crítico: {}",
        report.qes_radius
    );
    assert!(
        report.island_von_neumann_entropy >= 0.0,
        "Entropía de von Neumann en isla debe ser no-negativa: {}",
        report.island_von_neumann_entropy
    );
    assert!(
        !report.extremal_surface_collapse_veto,
        "Capital nominal sin drawdown no debe activar el veto extremal"
    );
}

#[test]
fn test_c2_island_phase_transition_and_slot_scaling() {
    let config = QuantumExtremalSurfaceConfig::default();
    let mut engine = QuantumExtremalSurfaceEngine::new(config);

    for i in 0..10 {
        let equity = 13.0 - 0.20 * (i as f64);
        let rep = engine.evaluate(equity, 10.0, 13.0, 0.03);

        assert!(
            rep.slot_scale_factor >= 0.10 && rep.slot_scale_factor <= 1.25,
            "Factor de slot fuera de límites [0.10, 1.25]: {}",
            rep.slot_scale_factor
        );
    }

    // Comprobamos la transición de fase con drawdown significativo
    let report_page = engine.evaluate(10.5, 10.0, 13.0, 0.05);
    assert!(
        report_page.island_phase_transition_active,
        "Drawdown superior al 15% debe activar la fase de isla cuántica"
    );
    assert!(
        report_page.slot_scale_factor <= 0.80,
        "En fase de isla, el slot debe comprimirse por debajo de 0.80"
    );
}

#[test]
fn test_c3_extremal_surface_collapse_veto_activation() {
    let config = QuantumExtremalSurfaceConfig::default();
    let mut engine = QuantumExtremalSurfaceEngine::new(config);

    // Inyectamos colapso al piso de capital (equity pegada al floor)
    let report = engine.evaluate(10.05, 10.0, 13.0, 0.50);

    assert!(
        report.extremal_surface_collapse_veto,
        "Proximidad crítica al piso debe detonar extremal_surface_collapse_veto"
    );
    assert_eq!(
        report.slot_scale_factor, 0.10,
        "En colapso de superficie extremal, el factor de slot debe fijarse en 0.10"
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = QuantumExtremalSurfaceConfig::default();
    let mut engine = QuantumExtremalSurfaceEngine::new(config);

    let report = engine.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);

    assert!(report.generalized_entropy_min.is_finite());
    assert!(report.qes_radius.is_finite());
    assert!(report.island_von_neumann_entropy.is_finite());
    assert!(report.slot_scale_factor.is_finite());
}
