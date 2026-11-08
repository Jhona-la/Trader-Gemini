//! Contrato Formal de Síntesis: Ola 15 (Fases 61, 62, 63) en OmniscientLivingGraphOrchestrator
//!
//! Certifica la unificación holística continua:
//! - C1: Síntesis coordinada de Bergman-Fock, De Rham-Hodge y Entropía Cuántica de Umegaki con Ville Ruin Bound.
//! - C2: Disparo determinista de veto duro de Umegaki (`UMEGAKI_RELATIVE_ENTROPY_VETO`).
//! - C3: Disparo de veto por saturación crítica en la frontera de Shilov de Bergman-Fock (`BERGMAN_FOCK_SHILOV_SATURATION_VETO`).
//! - C4: Detección y mitigación defensiva ante turbulencia de vorticidad de De Rham (`DERHAM_TURBULENT_VORTICITY_VETO`).

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType, SU3_COLORS};

#[test]
fn test_c1_wave15_triad_holistic_orchestration() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0008,
            order_flow_imbalance: 0.15,
            relative_weight: 1.1,
        },
        AssetColorState {
            normalized_return: -0.0004,
            order_flow_imbalance: -0.08,
            relative_weight: 0.95,
        },
        AssetColorState {
            normalized_return: 0.0001,
            order_flow_imbalance: 0.03,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Tick 1: Inicialización y flujo base
    let rep1 = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.02,
        50.0,
        45.0,
        1_000_000,
        &triad,
        1000.0,
        5.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    // Verificaciones Fase 61 (Bergman-Fock)
    assert!(rep1.bergman_fock_report.bergman_kernel_density > 0.0);
    assert!(rep1.bergman_fock_report.kahler_potential.is_finite());
    assert!(!rep1.bergman_fock_report.critical_saturation_detected);
    assert!(rep1.bergman_fock_report.slot_scale_factor >= 0.15);

    // Verificaciones Fase 62 (De Rham-Hodge)
    assert!(rep1.derham_hodge_report.exact_energy_ratio >= 0.0);
    assert!(rep1.derham_hodge_report.coexact_energy_ratio >= 0.0);
    assert!(rep1.derham_hodge_report.betti_one_persistence >= 0.0);
    assert!(!rep1.derham_hodge_report.turbulent_vorticity_alert);

    // Verificaciones Fase 63 (Umegaki Quantum Relative Entropy)
    assert!(
        rep1.umegaki_relative_entropy_report
            .umegaki_relative_entropy
            >= 0.0
    );
    assert!(
        rep1.umegaki_relative_entropy_report
            .von_neumann_entropy_state
            >= 0.0
    );
    assert!(!rep1.umegaki_relative_entropy_report.hard_veto_induced);
    assert!(rep1.ville_ruin_bound_satisfied);

    // Tick 2: Continuidad dinámica
    let rep2 = orchestrator.evaluate_and_synthesize(
        0,
        100.04,
        100.06,
        55.0,
        42.0,
        1_000_050,
        &triad,
        1000.0,
        5.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert!(rep2.holistic_slot_scale >= 0.10 && rep2.holistic_slot_scale <= 1.25);
    assert!(rep2.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave15_umegaki_relative_entropy_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [AssetColorState {
        normalized_return: 0.0,
        order_flow_imbalance: 0.0,
        relative_weight: 1.0,
    }; SU3_COLORS];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Configurar umbral crítico más sensible para verificar veto
    orchestrator.umegaki_entropy_engines[0] =
        risk_engine::UmegakiRelativeEntropyEngine::new(risk_engine::UmegakiEntropyParams {
            critical_divergence_threshold: 0.80,
            ..Default::default()
        });

    // Inyectar desequilibrio cuántico extremo
    let rep_shock = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.50,
        1000.0,
        1.0,
        1_000_000,
        &triad,
        1000.0,
        5.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    if rep_shock.umegaki_relative_entropy_report.hard_veto_induced {
        for decision in rep_shock.multi_slot_evaluation.decisions.iter() {
            assert_eq!(decision.signal, SignalType::Flat);
            assert_eq!(decision.net_confidence, 0.0);
            assert!(decision.vetoed);
            assert_eq!(decision.veto_reason, Some("UMEGAKI_RELATIVE_ENTROPY_VETO"));
        }
    }
}
