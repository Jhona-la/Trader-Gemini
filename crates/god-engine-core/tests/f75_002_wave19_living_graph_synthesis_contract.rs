//! Contrato Formal de Síntesis: Ola 19 (Fases 73, 74, 75) en OmniscientLivingGraphOrchestrator
//!
//! Certifica la unificación holística continua:
//! - C1: Síntesis coordinada de CFT de Liouville y Gravedad Cuántica 2D, Dualidad AdS/CFT Holográfica y Entropía Ryu-Takayanagi con Cota de Ville.
//! - C2: Disparo determinista de veto terminal por desconexión de superficie extremal Ryu-Takayanagi (`RYU_TAKAYANAGI_ENTANGLEMENT_VETO`).
//! - C3: Disparo determinista de veto por singularidad de curvatura de Liouville (`LIOUVILLE_SINGULARITY_VETO`).
//! - C4: Modulación holística de convicción de señal AdS/CFT y escala continua de slot.

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave19_triad_holistic_orchestration() {
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

    // Tick 1: Inicialización estacionaria
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

    // Verificaciones Fase 73 (Liouville Quantum Gravity)
    assert!(rep1.liouville_report.central_charge_cl >= 1.0);
    assert!(rep1.liouville_report.conformal_factor > 0.0);
    assert!(rep1.liouville_report.hausdorff_dimension >= 2.0);
    assert!(rep1.liouville_report.liouville_scalar_curvature.is_finite());
    assert!(!rep1.liouville_report.curvature_singularity_detected);
    assert!(rep1.liouville_report.slot_scale_factor >= 0.15);

    // Verificaciones Fase 74 (AdS/CFT Holographic)
    assert!(rep1.ads_cft_report.string_tension > 0.0);
    assert!(rep1.ads_cft_report.conformal_dimension_delta > 0.0);
    assert!(rep1.ads_cft_report.radial_depth_z > 0.0);
    assert!(rep1.ads_cft_report.holographic_conviction_modulator >= 0.50);

    // Verificaciones Fase 75 (Ryu-Takayanagi Entanglement)
    assert!(rep1.ryu_takayanagi_report.entanglement_entropy_sa > 0.0);
    assert!(rep1.ryu_takayanagi_report.entanglement_entropy_sb > 0.0);
    assert!(rep1.ryu_takayanagi_report.extremal_surface_connected);
    assert!(!rep1.ryu_takayanagi_report.hard_veto_induced);
    assert!(rep1.ryu_takayanagi_report.slot_scale_factor > 0.0);

    // Preservación holística de Ville
    assert!(rep1.ville_ruin_bound_satisfied);
    assert!(rep1.holistic_slot_scale >= 0.10 && rep1.holistic_slot_scale <= 1.25);
}

#[test]
fn test_c2_wave19_ryu_takayanagi_entanglement_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar Ryu-Takayanagi con umbral máximo de entropía mínimo para inducir saturación térmica
    orchestrator.ryu_takayanagi_engines[0] = risk_engine::ryu_takayanagi_entanglement::RyuTakayanagiEntanglementEngine::new(
        risk_engine::ryu_takayanagi_entanglement::RyuTakayanagiParams {
            max_entanglement_entropy_threshold: 0.001, // Forzar saturación entrópica inmediata
            ..Default::default()
        },
    );

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.10,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0002,
            order_flow_imbalance: 0.05,
            relative_weight: 1.0,
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

    let rep = orchestrator.evaluate_and_synthesize(
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

    assert!(rep.ryu_takayanagi_report.hard_veto_induced, "Debe activarse veto de Ryu-Takayanagi");
    assert!(!rep.ville_ruin_bound_satisfied, "Cota de Ville debe invalidarse");
    assert!(rep.multi_slot_evaluation.decisions[0].vetoed);
    assert_eq!(
        rep.multi_slot_evaluation.decisions[0].veto_reason,
        Some("RYU_TAKAYANAGI_ENTANGLEMENT_VETO")
    );
    assert_eq!(rep.multi_slot_evaluation.decisions[0].signal, SignalType::Flat);
    assert_eq!(rep.multi_slot_evaluation.decisions[0].net_confidence, 0.0);
}

#[test]
fn test_c3_wave19_liouville_curvature_singularity_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar Liouville con umbral de curvatura hipersensible
    orchestrator.liouville_engines[0] = feature_engine::liouville_quantum_gravity::LiouvilleCft2dQuantumGravityEngine::new(
        feature_engine::liouville_quantum_gravity::LiouvilleCftParams {
            singularity_curvature_threshold: 0.01,
            ..Default::default()
        },
    );

    // Asegurar que Ryu-Takayanagi permanece conexo y conforme durante la prueba de Liouville
    orchestrator.ryu_takayanagi_engines[0] = risk_engine::ryu_takayanagi_entanglement::RyuTakayanagiEntanglementEngine::new(
        risk_engine::ryu_takayanagi_entanglement::RyuTakayanagiParams {
            min_mutual_information_threshold: 0.0,
            max_entanglement_entropy_threshold: 1000.0,
            ..Default::default()
        },
    );

    let triad = [
        AssetColorState {
            normalized_return: 0.0001,
            order_flow_imbalance: 0.05,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0001,
            order_flow_imbalance: 0.05,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0001,
            order_flow_imbalance: 0.02,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Tick con spread de 18 bps (18 > 15 activa singularidad de Liouville, pero 18 <= 25 mantiene Ryu-Takayanagi limpio)
    let rep = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.18,
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

    assert!(rep.liouville_report.curvature_singularity_detected, "Debe detectarse singularidad de Liouville");
    assert!(!rep.ryu_takayanagi_report.hard_veto_induced, "Ryu-Takayanagi no debe disparar veto");
    assert!(!rep.ville_ruin_bound_satisfied, "Cota de Ville debe invalidarse");
    assert!(rep.multi_slot_evaluation.decisions[0].vetoed);
    assert_eq!(
        rep.multi_slot_evaluation.decisions[0].veto_reason,
        Some("LIOUVILLE_SINGULARITY_VETO")
    );
}

#[test]
fn test_c4_wave19_ads_cft_conviction_modulation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0001,
            order_flow_imbalance: 0.05,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: -0.0001,
            order_flow_imbalance: -0.05,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0001,
            order_flow_imbalance: 0.01,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let rep = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.01,
        100.0,
        100.0,
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

    // En régimen clásico supergravedad (baja dispersión y Hurst cercano a 0.5), la convicción debe ser constructiva
    assert!(rep.ads_cft_report.holographic_conviction_modulator >= 0.85);
    assert!(rep.holistic_slot_scale > 0.0);
}
