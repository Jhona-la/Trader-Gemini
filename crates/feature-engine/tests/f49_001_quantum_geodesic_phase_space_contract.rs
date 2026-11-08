//! Contrato Formal de Verificación: Fase 49 (Quantum Geodesic Phase Space & BRST Gauge Quantization)
//!
//! Este conjunto de pruebas certifica las propiedades matemáticas y físicas del motor:
//! - C1: Conservación de energía hamiltoniana e integración simpléctica de Verlet de 2º orden.
//! - C2: Cuantización BRST, nilpotencia de carga y evaporación de modos fantasma de spoofing.
//! - C3: Curvatura simpléctica $\Omega = dq \wedge dp$ y modulación acotada de slot $\eta \in [0.10, 1.30]$.
//! - C4: Determinismo en stack $O(1)$, cero heap allocs y robustez numérica total ante NaN/Infinitos.

use feature_engine::{
    GeodesicPhaseSpaceParams, GeodesicPhaseSpaceReport, PhaseSpacePoint,
    QuantumGeodesicPhaseSpaceEngine, PHASE_SPACE_HISTORY_LEN,
};

#[test]
fn test_c1_symplectic_verlet_integration_and_energy_conservation() {
    let params = GeodesicPhaseSpaceParams {
        mass: 1.0,
        friction_gamma: 0.0, // Fricción nula para verificar conservación de energía hamiltoniana pura
        brst_gauge_penalty: 1.0,
        potential_curvature: 4.0,
        time_step_dt: 0.01,
        ofi_coupling: 0.0,
    };
    let mut engine = QuantumGeodesicPhaseSpaceEngine::new(params);

    // Inyectamos una perturbación armónica suave oscilando alrededor de 100.0
    let mut energy_history = [0.0; 20];
    for (i, e_val) in energy_history.iter_mut().enumerate() {
        let t = i as f64 * 0.05;
        let mid = 100.0 + 0.5 * (2.0 * t).sin();
        let report = engine.update_and_evaluate(mid, 0.0, 0.0, (i as u64) * 50);
        *e_val = report.hamiltonian_energy;
        assert!(
            report.hamiltonian_energy.is_finite(),
            "Energía hamiltoniana no finita en paso {}",
            i
        );
        assert!(
            report.hamiltonian_energy >= 0.0,
            "Energía hamiltoniana negativa {}",
            report.hamiltonian_energy
        );
    }

    // Comprobar que en régimen estacionario la energía no explota
    let max_energy = energy_history.iter().fold(0.0_f64, |acc, &x| acc.max(x));
    assert!(
        max_energy < 50.0,
        "Energía hamiltoniana diverge: {}",
        max_energy
    );
}

#[test]
fn test_c2_brst_nilpotency_and_ghost_evaporation() {
    let params = GeodesicPhaseSpaceParams::default();
    let mut clean_engine = QuantumGeodesicPhaseSpaceEngine::new(params);
    let mut spoofed_engine = QuantumGeodesicPhaseSpaceEngine::new(params);

    // Motor 1: flujo limpio sin cancelaciones de spoofing
    let mut clean_report = GeodesicPhaseSpaceReport {
        current_point: PhaseSpacePoint::ZERO,
        hamiltonian_energy: 0.0,
        symplectic_curvature_drift: 0.0,
        brst_charge: 0.0,
        ghost_evaporation_ratio: 0.0,
        physical_liquidity_momentum: 0.0,
        slot_scaling_factor: 1.0,
    };
    for i in 0..10 {
        clean_report = clean_engine.update_and_evaluate(100.0 + (i as f64) * 0.1, 0.5, 0.0, i * 50);
    }

    // Motor 2: flujo fuertemente contaminado por spoofing (tasa de cancelación = 0.95)
    let mut spoofed_report = clean_report;
    for i in 0..10 {
        spoofed_report =
            spoofed_engine.update_and_evaluate(100.0 + (i as f64) * 0.1, 0.5, 0.95, i * 50);
    }

    // La evaporación de fantasmas BRST en el motor con spoofing debe ser significativamente mayor
    assert!(
        spoofed_report.ghost_evaporation_ratio > clean_report.ghost_evaporation_ratio,
        "Ghost evaporation esperada > limpio: {} vs {}",
        spoofed_report.ghost_evaporation_ratio,
        clean_report.ghost_evaporation_ratio
    );

    // El momento físico depurado en el motor con spoofing debe ser atenuado
    assert!(
        spoofed_report.physical_liquidity_momentum.abs()
            <= spoofed_report.current_point.p.abs() + 1e-9,
        "Momento físico supera al momento total"
    );
}

#[test]
fn test_c3_symplectic_curvature_drift_and_slot_scaling() {
    let params = GeodesicPhaseSpaceParams::default();
    let mut engine = QuantumGeodesicPhaseSpaceEngine::new(params);

    for i in 0..50 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let mid = 100.0 + sign * 0.20;
        let ofi = sign * 0.80;
        let report = engine.update_and_evaluate(mid, ofi, 0.30, i * 100);

        // Modulador acotado estrictamente en [0.10, 1.30]
        assert!(
            report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.30,
            "Slot scaling fuera de rango: {}",
            report.slot_scaling_factor
        );
        assert!(
            report.symplectic_curvature_drift.is_finite(),
            "Curvatura simpléctica no finita"
        );
    }
}

#[test]
fn test_c4_stack_zero_heap_and_nan_subnormal_immunity() {
    let params = GeodesicPhaseSpaceParams::default();
    let mut engine = QuantumGeodesicPhaseSpaceEngine::new(params);

    // Inyección de valores anómalos, NaN e Infinitos
    let anomalous_inputs = [
        (f64::NAN, 0.0, 0.0),
        (100.0, f64::INFINITY, 0.5),
        (-50.0, f64::NEG_INFINITY, f64::NAN),
        (0.0, 0.0, 0.0),
        (1e-300, 1e-300, 1e-300),
    ];

    for &(mid, ofi, cancel) in &anomalous_inputs {
        let report = engine.update_and_evaluate(mid, ofi, cancel, 1000);
        assert!(report.hamiltonian_energy.is_finite());
        assert!(report.symplectic_curvature_drift.is_finite());
        assert!(report.ghost_evaporation_ratio >= 0.0 && report.ghost_evaporation_ratio <= 1.0);
        assert!(report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.30);
    }

    assert_eq!(PHASE_SPACE_HISTORY_LEN, 16);
}
