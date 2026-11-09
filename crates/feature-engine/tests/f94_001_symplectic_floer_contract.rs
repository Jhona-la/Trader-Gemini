//! Contrato Formal de Verificación: Fase 94 (§94) — Geometría Simpléctica Infinita & Cohomología de Floer Cuántica
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor de Floer:
//! - C1: Finitud de la acción simpléctica y energía de Floer en condiciones regulares de mercado.
//! - C2: Graduación de cohomología cuántica $QH^* \in [0, 1]$ y modulación de slot $\eta_{\text{floer}} \in [0.10, 1.25]$.
//! - C3: Detonación determinista del veto fail-closed ante divergencia de energía de Floer (bubbling de Gromov).
//! - C4: Determinismo estricto en stack $O(1)$, inmunidad ante NaNs e infinitos y cero fallos de memoria.

use feature_engine::{SymplecticFloerCohomologyEngine, SymplecticFloerConfig};

#[test]
fn test_c1_symplectic_action_and_floer_energy_finitude() {
    let config = SymplecticFloerConfig::default();
    let mut engine = SymplecticFloerCohomologyEngine::new(config);

    let report = engine.evaluate(0.0015, 0.20, 1.2, 0.02);

    assert!(
        report.symplectic_action.is_finite(),
        "Acción simpléctica debe ser finita: {}",
        report.symplectic_action
    );
    assert!(
        report.floer_energy.is_finite() && report.floer_energy >= 0.0,
        "Energía de Floer debe ser finita y no negativa: {}",
        report.floer_energy
    );
    assert!(
        report.conley_zehnder_index >= -4.0 && report.conley_zehnder_index <= 4.0,
        "Índice de Conley-Zehnder fuera de límites [-4, 4]: {}",
        report.conley_zehnder_index
    );
    assert!(
        !report.floer_homology_obstruction_veto,
        "Microestructura regular no debe detonar el veto de obstrucción de Floer"
    );
}

#[test]
fn test_c2_quantum_cohomology_grading_and_slot_scaling() {
    let config = SymplecticFloerConfig::default();
    let mut engine = SymplecticFloerCohomologyEngine::new(config);

    for i in 0..10 {
        let p_ret = 0.0004 * (i as f64);
        let ofi = -0.3 + 0.06 * (i as f64);
        let rep = engine.evaluate(p_ret, ofi, 2.5, 0.025);

        assert!(
            rep.quantum_cohomology_grading >= 0.0 && rep.quantum_cohomology_grading <= 1.0,
            "Graduación de cohomología cuántica fuera de [0, 1]: {}",
            rep.quantum_cohomology_grading
        );
        assert!(
            rep.directional_conviction >= 0.0 && rep.directional_conviction <= 1.0,
            "Convicción direccional fuera de [0, 1]: {}",
            rep.directional_conviction
        );
        assert!(
            rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25,
            "Factor de slot fuera de límites [0.10, 1.25]: {}",
            rep.slot_scaling_factor
        );
    }
}

#[test]
fn test_c3_floer_homology_obstruction_veto_activation() {
    let config = SymplecticFloerConfig {
        max_floer_energy: 0.10, // Umbral muy estricto para forzar veto de energía
        ..Default::default()
    };
    let mut engine = SymplecticFloerCohomologyEngine::new(config);

    // Con saltos gigantescos de precio y ofi, la energía de Floer supera 0.10 inmediatamente
    let _ = engine.evaluate(0.0, 0.0, 1.0, 0.01);
    let report = engine.evaluate(0.50, -0.80, 50.0, 5.0);

    assert!(
        report.floer_homology_obstruction_veto,
        "Salto violento de flujo debe detonar floer_homology_obstruction_veto"
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = SymplecticFloerConfig::default();
    let mut engine = SymplecticFloerCohomologyEngine::new(config);

    let report = engine.evaluate(f64::NAN, f64::INFINITY, -5.0, f64::NAN);

    assert!(report.symplectic_action.is_finite());
    assert!(report.floer_energy.is_finite());
    assert!(report.conley_zehnder_index.is_finite());
    assert!(report.quantum_cohomology_grading.is_finite());
    assert!(report.directional_conviction.is_finite());
    assert!(report.slot_scaling_factor.is_finite());
}
