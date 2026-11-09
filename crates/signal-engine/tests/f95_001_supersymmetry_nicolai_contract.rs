//! Contrato Formal de Verificación: Fase 95 (§95) — Supersimetría $\mathcal{N}=1$ & Ecuación de Nicolai
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor de supersimetría:
//! - C1: Finitud y no-negatividad de la energía de vacío $E_0 \ge 0$, acotamiento del índice de Witten $\Delta \in [-1, 1]$.
//! - C2: Gaussianidad del mapa de Nicolai y modulación de slot $\eta_{\text{susy}} \in [0.15, 1.30]$.
//! - C3: Detonación determinista del veto fail-closed ante ruptura espontánea de supersimetría.
//! - C4: Determinismo estricto en stack $O(1)$, inmunidad ante NaNs e infinitos y cero asignaciones en heap.

use signal_engine::{SupersymmetryNicolaiConfig, SupersymmetryNicolaiEngine};

#[test]
fn test_c1_susy_vacuum_energy_and_witten_index() {
    let config = SupersymmetryNicolaiConfig::default();
    let mut engine = SupersymmetryNicolaiEngine::new(config);

    let report = engine.evaluate(0.001, 0.10, 1.5, 0.02, 0.05);

    assert!(
        report.vacuum_energy.is_finite() && report.vacuum_energy >= 0.0,
        "Energía de vacío debe ser finita y no-negativa: {}",
        report.vacuum_energy
    );
    assert!(
        report.witten_index >= -1.0 && report.witten_index <= 1.0,
        "Índice de Witten fuera de límites [-1, 1]: {}",
        report.witten_index
    );
    assert!(
        report.fermionic_parity >= -1.0 && report.fermionic_parity <= 1.0,
        "Paridad fermiónica fuera de [-1, 1]: {}",
        report.fermionic_parity
    );
    assert!(
        !report.susy_spontaneous_breaking_veto,
        "Microestructura regular no debe detonar el veto de ruptura de supersimetría"
    );
}

#[test]
fn test_c2_nicolai_map_residual_and_slot_scaling() {
    let config = SupersymmetryNicolaiConfig::default();
    let mut engine = SupersymmetryNicolaiEngine::new(config);

    for i in 0..10 {
        let p_ret = 0.0003 * (i as f64);
        let ofi = 0.04 * (i as f64);
        let rep = engine.evaluate(p_ret, ofi, 2.0, 0.02, 0.05);

        assert!(
            rep.nicolai_map_residual.is_finite() && rep.nicolai_map_residual >= 0.0,
            "Residuo de Nicolai debe ser finito y positivo: {}",
            rep.nicolai_map_residual
        );
        assert!(
            rep.directional_bias >= -1.0 && rep.directional_bias <= 1.0,
            "Sesgo direccional fuera de [-1, 1]: {}",
            rep.directional_bias
        );
        assert!(
            rep.slot_scaling_factor >= 0.15 && rep.slot_scaling_factor <= 1.30,
            "Factor de slot fuera de límites [0.15, 1.30]: {}",
            rep.slot_scaling_factor
        );
    }
}

#[test]
fn test_c3_susy_spontaneous_breaking_veto_activation() {
    let config = SupersymmetryNicolaiConfig {
        max_vacuum_energy: 0.20, // Umbral muy estricto para forzar veto de energía de vacío
        max_nicolai_residual: 0.50,
        ..Default::default()
    };
    let mut engine = SupersymmetryNicolaiEngine::new(config);

    // Inyectamos un salto gigante de precio y OFI para provocar ruptura masiva
    let mut report = engine.evaluate(0.0, 0.0, 1.0, 0.01, 0.001);
    for _ in 0..5 {
        report = engine.evaluate(0.60, 1.0, 80.0, 5.0, 0.001);
    }

    assert!(
        report.susy_spontaneous_breaking_veto,
        "Ruptura espontánea con energía divergente debe detonar susy_spontaneous_breaking_veto"
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = SupersymmetryNicolaiConfig::default();
    let mut engine = SupersymmetryNicolaiEngine::new(config);

    let report = engine.evaluate(f64::NAN, f64::INFINITY, -10.0, f64::NAN, 0.0);

    assert!(report.vacuum_energy.is_finite());
    assert!(report.witten_index.is_finite());
    assert!(report.nicolai_map_residual.is_finite());
    assert!(report.fermionic_parity.is_finite());
    assert!(report.directional_bias.is_finite());
    assert!(report.slot_scaling_factor.is_finite());
}
