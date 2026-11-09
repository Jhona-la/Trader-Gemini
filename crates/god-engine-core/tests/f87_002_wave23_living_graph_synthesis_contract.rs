//! Contrato Formal de Síntesis Holística: Ola 23 (Fases 85, 86, 87) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica en `OmniscientLivingGraphOrchestrator`:
//! - Invariante de Chern-Simons-Witten & Homología Khovanov-Floer (Fase 85)
//! - Operadores de Densidad de Producto Matricial (MPDO) & Lindblad DMRG (Fase 86)
//! - Complejidad Holográfica WDW & Cota de Lloyd de Capacidad de Procesamiento (Fase 87)

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave23_triad_holistic_orchestration() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.10,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: -0.0002,
            order_flow_imbalance: -0.05,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: 0.0003,
            order_flow_imbalance: 0.02,
            relative_weight: 0.50,
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

    // 1. Verificación Fase 85: Invariante Chern-Simons-Witten
    assert!(
        report
            .chern_simons_knot_report
            .jones_polynomial_val
            .is_finite()
    );
    assert!(
        report
            .chern_simons_knot_report
            .khovanov_quantum_degree
            .is_finite()
    );
    assert!(!report.chern_simons_knot_report.knot_collapse_veto);
    assert!(report.chern_simons_knot_report.slot_scale_factor >= 0.10);

    // 2. Verificación Fase 86: MPDO & Ecuación Maestra de Lindblad
    assert!(report.mpdo_report.quantum_purity >= 0.50);
    assert!(report.mpdo_report.von_neumann_entropy.is_finite());
    assert!(report.mpdo_report.coherent_fidelity.is_finite());
    assert!(report.mpdo_report.directional_signal.is_finite());
    assert!(!report.mpdo_report.dissipative_decoherence_veto);
    assert!(report.mpdo_report.slot_scale_factor >= 0.15);

    // 3. Verificación Fase 87: Complejidad Holográfica WDW & Cota de Lloyd
    assert!(report.holographic_lloyd_report.holographic_complexity > 0.0);
    assert!(
        report
            .holographic_lloyd_report
            .lloyd_computation_bound
            .is_finite()
    );
    assert!(!report.holographic_lloyd_report.lloyd_bound_exhaustion_veto);
    assert!(report.holographic_lloyd_report.slot_scale_factor >= 0.10);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave23_chern_simons_knot_collapse_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar el motor Chern-Simons con umbral de complejidad estricto para forzar colapso
    orchestrator.chern_simons_knot_engines[0] =
        feature_engine::ChernSimonsWittenKnotEngine::new(feature_engine::ChernSimonsWittenConfig {
            min_knot_complexity_floor: 100.0,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let report = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.02,
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

    assert!(report.chern_simons_knot_report.knot_collapse_veto);
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in report.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("CHERN_SIMONS_WITTEN_KNOT_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c3_wave23_mpdo_dissipative_decoherence_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Forzar condición de decoherencia térmica con piso de pureza elevado
    orchestrator.mpdo_engines[0] =
        signal_engine::MpdoDissipativeDmrgEngine::new(signal_engine::MpdoDissipativeConfig {
            critical_purity_floor: 0.999,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let report = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.02,
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

    assert!(report.mpdo_report.dissipative_decoherence_veto);
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in report.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("MPDO_DISSIPATIVE_DECOHERENCE_VETO")
        );
    }
}

#[test]
fn test_c4_wave23_holographic_lloyd_bound_exhaustion_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar cota de Lloyd con ratio crítico nulo para forzar agotamiento de velocidad computacional
    orchestrator.holographic_lloyd_engines[0] =
        risk_engine::HolographicLloydComplexityEngine::new(risk_engine::HolographicLloydConfig {
            critical_saturation_ratio: 0.0,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.1,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let report = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.02,
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

    assert!(report.holographic_lloyd_report.lloyd_bound_exhaustion_veto);
    assert_eq!(report.holographic_lloyd_report.slot_scale_factor, 0.0);
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in report.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("HOLOGRAPHIC_LLOYD_BOUND_EXHAUSTION_VETO")
        );
    }
}

#[test]
fn test_c5_wave23_deterministic_replay_and_ruin_bound() {
    let mut orchestrator_a = OmniscientLivingGraphOrchestrator::new(3);
    let mut orchestrator_b = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0004,
            order_flow_imbalance: 0.08,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: -0.0001,
            order_flow_imbalance: -0.03,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: 0.0002,
            order_flow_imbalance: 0.01,
            relative_weight: 0.50,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let rep_a = orchestrator_a.evaluate_and_synthesize(
        0,
        100.0,
        100.02,
        15.0,
        15.0,
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

    let rep_b = orchestrator_b.evaluate_and_synthesize(
        0,
        100.0,
        100.02,
        15.0,
        15.0,
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

    assert_eq!(
        rep_a.chern_simons_knot_report.jones_polynomial_val,
        rep_b.chern_simons_knot_report.jones_polynomial_val
    );
    assert_eq!(
        rep_a.mpdo_report.quantum_purity,
        rep_b.mpdo_report.quantum_purity
    );
    assert_eq!(
        rep_a.holographic_lloyd_report.holographic_complexity,
        rep_b.holographic_lloyd_report.holographic_complexity
    );
    assert_eq!(rep_a.holistic_slot_scale, rep_b.holistic_slot_scale);
    assert_eq!(
        rep_a.ville_ruin_bound_satisfied,
        rep_b.ville_ruin_bound_satisfied
    );
}
