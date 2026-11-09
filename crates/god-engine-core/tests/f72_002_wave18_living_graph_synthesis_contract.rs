//! Contrato Formal de Síntesis: Ola 18 (Fases 70, 71, 72) en OmniscientLivingGraphOrchestrator
//!
//! Certifica la unificación holística continua:
//! - C1: Síntesis coordinada de Gross-Pitaevskii BEC, MF-DFA Multifractal Hurst y Lindblad Master Decoherence con Ville Ruin Bound.
//! - C2: Disparo determinista de veto terminal de Decoherencia Cuántica de Lindblad (`LINDBLAD_DECOHERENCE_VETO`).
//! - C3: Disparo determinista de veto por Cavitación de Vórtice Cuántico de Gross-Pitaevskii (`GROSS_PITAEVSKII_VORTEX_VETO`).
//! - C4: Modulación de convicción de señal y consistencia espectral multifractal MF-DFA.

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave18_triad_holistic_orchestration() {
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

    // Verificaciones Fase 70 (Gross-Pitaevskii BEC)
    assert!(
        rep1.bec_report.condensate_fraction >= 0.0 && rep1.bec_report.condensate_fraction <= 1.0
    );
    assert!(rep1.bec_report.central_density > 0.0);
    assert!(rep1.bec_report.bogoliubov_sound_speed > 0.0);
    assert!(rep1.bec_report.healing_length > 0.0);
    assert!(!rep1.bec_report.quantum_vortex_shedding_detected);
    assert!(rep1.bec_report.slot_scale_factor >= 0.15);

    // Verificaciones Fase 71 (MF-DFA Multifractal Hurst)
    assert!(rep1.mfdfa_report.generalized_hurst_h2 >= 0.01);
    assert!(rep1.mfdfa_report.multifractal_spectrum_width >= 0.0);
    assert!(rep1.mfdfa_report.signal_conviction_modulator >= 0.50);

    // Verificaciones Fase 72 (Lindblad Master Decoherence)
    assert!(rep1.lindblad_report.purity >= 0.333 && rep1.lindblad_report.purity <= 1.0001);
    assert!(rep1.lindblad_report.von_neumann_entropy >= 0.0);
    assert!(!rep1.lindblad_report.hard_veto_induced);
    assert!(rep1.lindblad_report.slot_scale_factor > 0.0);

    // Preservación holística de Ville
    assert!(rep1.ville_ruin_bound_satisfied);
    assert!(rep1.holistic_slot_scale >= 0.10 && rep1.holistic_slot_scale <= 1.25);
}

#[test]
fn test_c2_wave18_lindblad_decoherence_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar parámetros de Lindblad para inducir veto estricto ante perturbación térmica
    orchestrator.lindblad_engines[0] =
        risk_engine::lindblad_master_decoherence::LindbladMasterDecoherenceEngine::new(
            risk_engine::lindblad_master_decoherence::LindbladMasterParams {
                min_purity_threshold: 0.95, // Exigir pureza casi perfecta
                dt: 0.1,
                ..Default::default()
            },
        );
    // Aislar la prueba de Lindblad asegurando que Ryu-Takayanagi no anticipe el veto por dispersión
    orchestrator.ryu_takayanagi_engines[0] =
        risk_engine::RyuTakayanagiEntanglementEngine::new(risk_engine::RyuTakayanagiParams {
            max_entanglement_entropy_threshold: 100.0,
            min_mutual_information_threshold: -1.0,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.05,
            order_flow_imbalance: 0.8,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: -0.05,
            order_flow_imbalance: -0.8,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.02,
            order_flow_imbalance: 0.3,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Forzar desfasamiento severo con spread controlado (10 bps) y ticks disipativos
    let mut rep = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.10,
        10.0,
        10.0,
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

    for i in 1..10 {
        rep = orchestrator.evaluate_and_synthesize(
            0,
            100.0 + (i as f64) * 0.1,
            100.10 + (i as f64) * 0.1,
            10.0,
            10.0,
            1_000_000 + i * 100,
            &triad,
            1000.0,
            5.0,
            0.02,
            &mut slot_orchestrator,
            &open_slots,
            &open_taus,
            &open_is_long,
        );
    }

    assert!(
        rep.lindblad_report.hard_veto_induced,
        "Debe activarse veto de decoherencia de Lindblad"
    );
    assert!(
        !rep.ville_ruin_bound_satisfied,
        "Cota de Ville debe invalidarse bajo colapso de coherencia"
    );
    assert_eq!(
        rep.multi_slot_evaluation.decisions[0].signal,
        SignalType::Flat
    );
    assert_eq!(
        rep.multi_slot_evaluation.decisions[0].veto_reason,
        Some("LINDBLAD_DECOHERENCE_VETO")
    );
}

#[test]
fn test_c3_wave18_gross_pitaevskii_vortex_cavitation_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar umbrales de BEC para disparar cavitación ante spread ensanchado
    orchestrator.bec_engines[0] =
        feature_engine::gross_pitaevskii_bec::GrossPitaevskiiBecEngine::new(
            feature_engine::gross_pitaevskii_bec::GrossPitaevskiiParams {
                min_condensate_fraction_threshold: 0.90,
                ..Default::default()
            },
        );
    // Aislar la prueba de BEC asegurando que Liouville y Ryu-Takayanagi no anticipen el veto ante el shock
    orchestrator.liouville_engines[0] = feature_engine::LiouvilleCft2dQuantumGravityEngine::new(
        feature_engine::LiouvilleCftParams {
            singularity_curvature_threshold: 100.0,
            ..Default::default()
        },
    );
    orchestrator.ryu_takayanagi_engines[0] =
        risk_engine::RyuTakayanagiEntanglementEngine::new(risk_engine::RyuTakayanagiParams {
            max_entanglement_entropy_threshold: 100.0,
            min_mutual_information_threshold: -1.0,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Tick base inicial para calibrar referencia de precio
    let _ = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.01,
        50.0,
        45.0,
        1_000_000,
        &triad,
        1000.0,
        5.0,
        0.0,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    // Shock violento de precio (+8.0%) y spread controlado (13.8 bps) que colapsa la fracción de condensado y nuclea vórtice
    let rep = orchestrator.evaluate_and_synthesize(
        0,
        108.0,
        108.15,
        5.0,
        5.0,
        1_000_100,
        &triad,
        1000.0,
        5.0,
        0.08,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert!(
        rep.bec_report.quantum_vortex_shedding_detected,
        "Debe detectarse vórtice/cavitación en BEC"
    );
    assert!(!rep.ville_ruin_bound_satisfied);
    assert_eq!(
        rep.multi_slot_evaluation.decisions[0].signal,
        SignalType::Flat
    );
    assert_eq!(
        rep.multi_slot_evaluation.decisions[0].veto_reason,
        Some("GROSS_PITAEVSKII_VORTEX_VETO")
    );
}

#[test]
fn test_c4_wave18_mfdfa_multifractal_conviction_modulation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.10,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: -0.0002,
            order_flow_imbalance: -0.05,
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

    // Serie suave de 20 ticks
    let mut rep = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.01,
        50.0,
        50.0,
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

    for i in 1..20 {
        rep = orchestrator.evaluate_and_synthesize(
            0,
            100.0 + (i as f64) * 0.005,
            100.01 + (i as f64) * 0.005,
            50.0,
            50.0,
            1_000_000 + i * 50,
            &triad,
            1000.0,
            5.0,
            0.02,
            &mut slot_orchestrator,
            &open_slots,
            &open_taus,
            &open_is_long,
        );
    }

    assert!(rep.mfdfa_report.signal_conviction_modulator >= 0.50);
    assert!(rep.mfdfa_report.signal_conviction_modulator <= 1.35);
    assert!(rep.holistic_slot_scale > 0.0);
}
