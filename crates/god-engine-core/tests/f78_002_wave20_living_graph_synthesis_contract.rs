//! Contrato Formal de Síntesis Holística: Ola 20 (Fases 76, 77, 78) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica de Gravedad JT 2D (Schwarzian), Holonomía Cuántica de Berry
//! y el Estabilizador Topológico de Toric Code en `OmniscientLivingGraphOrchestrator`.

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave20_triad_holistic_orchestration() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.10,
            relative_weight: 1.05,
        },
        AssetColorState {
            normalized_return: -0.0002,
            order_flow_imbalance: -0.05,
            relative_weight: 0.98,
        },
        AssetColorState {
            normalized_return: 0.0003,
            order_flow_imbalance: 0.02,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Pulso nominal de mercado estable con spread ajustado
    let report = orchestrator.evaluate_and_synthesize(
        0,
        99.99,
        100.01,
        10.0,
        10.0,
        1_700_000_000_000,
        &triad,
        13.0,
        20.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    // 1. Verificación Fase 76: Gravedad JT 2D & Schwarzian
    assert!(report.jt_schwarzian_report.schwarzian_derivative.is_finite());
    assert!(report.jt_schwarzian_report.schwarzian_energy.is_finite());
    assert!(report.jt_schwarzian_report.density_of_states_rho > 0.0);
    assert!(!report.jt_schwarzian_report.schwarzian_singularity_detected);
    assert!(report.jt_schwarzian_report.slot_scale_factor >= 0.15);

    // 2. Verificación Fase 77: Fase de Berry & Holonomía de Fubini-Study
    assert!(report.berry_phase_report.fubini_study_distance.is_finite());
    assert!(report.berry_phase_report.berry_phase.is_finite());
    assert!(report.berry_phase_report.berry_curvature_magnitude > 0.0);
    assert!(!report.berry_phase_report.adiabatic_breakdown);
    assert!(report.berry_phase_report.holographic_conviction_modulator >= 0.45);

    // 3. Verificación Fase 78: Estabilizador Topológico de Toric Code
    assert!(report.toric_code_report.anyon_defect_density.is_finite());
    assert!(!report.toric_code_report.hard_veto_induced);
    assert!(report.toric_code_report.slot_scale_factor >= 0.10);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave20_toric_code_braiding_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.10,
            relative_weight: -1.0, // Induce cross_corr < 0
        },
        AssetColorState {
            normalized_return: 0.002,
            order_flow_imbalance: -0.05,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: -0.001,
            order_flow_imbalance: 0.02,
            relative_weight: 0.20,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Forzar spread_bps > 15.0 para invertir los 16 enlaces verticales (genera devanado de Wilson Wy = -1)
    // Usamos bid: 99.90, ask: 100.10 -> spread = 20.0 bps > 15.0 bps
    let report = orchestrator.evaluate_and_synthesize(
        0,
        99.90,
        100.10,
        10.0,
        10.0,
        1_700_000_000_000,
        &triad,
        13.0,
        20.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert!(report.toric_code_report.hard_veto_induced);
    assert_eq!(report.toric_code_report.slot_scale_factor, 0.0);
    assert!(!report.ville_ruin_bound_satisfied);

    for dec in report.multi_slot_evaluation.decisions.iter() {
        assert_eq!(dec.signal, SignalType::Flat);
        assert!(dec.vetoed);
        assert_eq!(dec.veto_reason, Some("TORIC_CODE_BRAIDING_VETO"));
    }
}

#[test]
fn test_c3_wave20_schwarzian_singularity_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.10,
            relative_weight: 1.05,
        },
        AssetColorState {
            normalized_return: -0.0002,
            order_flow_imbalance: -0.05,
            relative_weight: 0.98,
        },
        AssetColorState {
            normalized_return: 0.0003,
            order_flow_imbalance: 0.02,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Inicializar estado estable
    orchestrator.evaluate_and_synthesize(
        0, 99.99, 100.01, 10.0, 10.0, 1_700_000_000_000,
        &triad, 13.0, 20.0, 0.02, &mut slot_orchestrator,
        &open_slots, &open_taus, &open_is_long,
    );
    orchestrator.evaluate_and_synthesize(
        0, 99.99, 100.01, 10.0, 10.0, 1_700_000_000_010,
        &triad, 13.0, 20.0, 0.02, &mut slot_orchestrator,
        &open_slots, &open_taus, &open_is_long,
    );

    // Shock abrupto en tiempo hiper-reducido (dt = 1 ms) y desbalance OFI violento que dispara aceleración
    let report_shock = orchestrator.evaluate_and_synthesize(
        0, 99.99, 100.01, 1000.0, 0.1, 1_700_000_000_011,
        &triad, 13.0, 20.0, 0.02, &mut slot_orchestrator,
        &open_slots, &open_taus, &open_is_long,
    );

    assert!(report_shock.jt_schwarzian_report.schwarzian_derivative.is_finite());
    assert!(report_shock.jt_schwarzian_report.schwarzian_energy > 0.0);
}

#[test]
fn test_c4_wave20_berry_phase_topological_conviction() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.10,
            relative_weight: 1.05,
        },
        AssetColorState {
            normalized_return: -0.0002,
            order_flow_imbalance: -0.05,
            relative_weight: 0.98,
        },
        AssetColorState {
            normalized_return: 0.0003,
            order_flow_imbalance: 0.02,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Secuencia que describe un ciclo en el espacio de parámetros (spread, vol, ofi)
    let p1 = orchestrator.evaluate_and_synthesize(
        0, 99.99, 100.01, 10.0, 10.0, 1_700_000_000_000,
        &triad, 13.0, 20.0, 0.02, &mut slot_orchestrator,
        &open_slots, &open_taus, &open_is_long,
    );
    let p2 = orchestrator.evaluate_and_synthesize(
        0, 99.98, 100.02, 20.0, 5.0, 1_700_000_000_100,
        &triad, 13.0, 20.0, 0.02, &mut slot_orchestrator,
        &open_slots, &open_taus, &open_is_long,
    );
    let p3 = orchestrator.evaluate_and_synthesize(
        0, 99.99, 100.01, 5.0, 20.0, 1_700_000_000_200,
        &triad, 13.0, 20.0, 0.02, &mut slot_orchestrator,
        &open_slots, &open_taus, &open_is_long,
    );

    assert!(p1.berry_phase_report.holographic_conviction_modulator >= 0.45);
    assert!(p2.berry_phase_report.holographic_conviction_modulator <= 1.40);
    assert!(p3.berry_phase_report.berry_curvature_magnitude > 0.0);
}
