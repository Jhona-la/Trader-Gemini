//! Contrato Formal de Síntesis: Ola 17 (Fases 67, 68, 69) en OmniscientLivingGraphOrchestrator
//!
//! Certifica la unificación holística continua:
//! - C1: Síntesis coordinada de Ashtekar LQG, NLS Peregrine Solitons y Polyakov Schwarzschild Horizon con Ville Ruin Bound.
//! - C2: Disparo determinista de veto terminal de Horizonte de Sucesos de Schwarzschild (`POLYAKOV_SCHWARZSCHILD_EVENT_HORIZON_VETO`).
//! - C3: Disparo determinista de veto por colapso de área cuántica de Ashtekar (`ASHTEKAR_AREA_SINGULARITY_VETO`).
//! - C4: Detección y amplificación de convicción por ondas rebeldes de Peregrine (NLS Rogue Waves).

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType, SU3_COLORS};

#[test]
fn test_c1_wave17_triad_holistic_orchestration() {
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

    // Tick 1: Inicialización y flujo estacionario
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

    // Verificaciones Fase 67 (Ashtekar-Barbero LQG Area Spectrum)
    assert!(rep1.ashtekar_lqg_report.total_quantized_area >= 0.0);
    assert!(rep1.ashtekar_lqg_report.minimum_area_gap > 0.0);
    assert!(rep1.ashtekar_lqg_report.quantum_volume_density >= 0.0);
    assert!(rep1
        .ashtekar_lqg_report
        .wilson_loop_holonomy_trace
        .is_finite());
    assert!(!rep1.ashtekar_lqg_report.ashtekar_area_singularity_detected);
    assert!(rep1.ashtekar_lqg_report.slot_scale_factor >= 0.15);

    // Verificaciones Fase 68 (NLS Peregrine Soliton Rogue Waves)
    assert!(rep1.nls_peregrine_report.envelope_amplitude >= 0.0);
    assert!(rep1.nls_peregrine_report.quantum_phase.is_finite());
    assert!(rep1.nls_peregrine_report.benjamin_feir_index >= 0.0);
    assert!(rep1.nls_peregrine_report.signal_conviction_modulator >= 0.20);

    // Verificaciones Fase 69 (Polyakov Quantum Action & Schwarzschild Horizon)
    assert!(rep1.polyakov_schwarzschild_report.schwarzschild_radius >= 0.0);
    assert!(rep1.polyakov_schwarzschild_report.phase_space_radius > 0.0);
    assert!(rep1.polyakov_schwarzschild_report.horizon_distance > 0.0);
    assert!(
        !rep1
            .polyakov_schwarzschild_report
            .event_horizon_veto_induced
    );
    assert!(rep1.polyakov_schwarzschild_report.slot_scale_factor > 0.0);

    // Tick 2: Continuidad dinámica y preservación de Ville
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
fn test_c2_wave17_polyakov_schwarzschild_event_horizon_terminal_veto() {
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

    // Configurar motor de Schwarzschild con constante gravitacional crítica para inducir colapso
    orchestrator.polyakov_schwarzschild_engines[0] =
        risk_engine::PolyakovSchwarzschildHorizonEngine::new(
            risk_engine::PolyakovSchwarzschildParams {
                g_eff: 5.0, // Alta gravedad efectiva: radio de Schwarzschild masivo
                ..Default::default()
            },
        );

    // Shock de mercado con fricción alta (spread 50 bps, vol extrema)
    let rep_shock = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.50,
        1000.0,
        1.0,
        1_000_000,
        &triad,
        1000.0,
        50.0, // Máximo apalancamiento
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert!(
        rep_shock
            .polyakov_schwarzschild_report
            .event_horizon_veto_induced
    );
    assert!(!rep_shock.ville_ruin_bound_satisfied);
    for decision in rep_shock.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert_eq!(decision.net_confidence, 0.0);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("POLYAKOV_SCHWARZSCHILD_EVENT_HORIZON_VETO")
        );
    }
}

#[test]
fn test_c3_wave17_ashtekar_area_singularity_veto() {
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

    // Configurar LQG de Ashtekar con umbral crítico elevado para detectar colapso de área
    orchestrator.ashtekar_lqg_engines[0] =
        feature_engine::AshtekarLqgAreaEngine::new(feature_engine::AshtekarLqgParams {
            critical_area_threshold: 50.0, // Forzar clasificación de singularidad por debajo de 50.0
            ..Default::default()
        });

    let rep_shock = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.10,
        100.0,
        20.0,
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

    assert!(
        rep_shock
            .ashtekar_lqg_report
            .ashtekar_area_singularity_detected
    );
    for decision in rep_shock.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert_eq!(decision.net_confidence, 0.0);
        assert!(decision.vetoed);
        assert_eq!(decision.veto_reason, Some("ASHTEKAR_AREA_SINGULARITY_VETO"));
    }
}

#[test]
fn test_c4_wave17_nls_peregrine_rogue_wave_amplification() {
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

    // Configurar NLS Peregrine para detección en resonancia analítica (x=0, t=0)
    orchestrator.nls_peregrine_engines[0] =
        signal_engine::NlsPeregrineSolitonEngine::new(signal_engine::NlsPeregrineParams {
            min_bfi_threshold: 0.30,
            ..Default::default()
        });

    // spread = 4.0 bps -> t = 0; ofi = 0.0 -> x = 0; amplitude = 3.0 >= 2.20
    let rep = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.04, // spread 4 bps en precio 100
        50.0,
        50.0, // OFI = 0.0
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

    assert!(rep.nls_peregrine_report.envelope_amplitude >= 2.20);
    assert!(rep.nls_peregrine_report.benjamin_feir_index >= 0.30);
    assert!(rep.nls_peregrine_report.peregrine_rogue_wave_detected);
    assert!(rep.nls_peregrine_report.signal_conviction_modulator > 1.15);
}
