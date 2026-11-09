//! Contrato Formal de Síntesis Holística: Ola 22 (Fases 82, 83, 84) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica en `OmniscientLivingGraphOrchestrator`:
//! - Cromodinámica Cuántica en Red & Holonomía Wilson-Polyakov (Fase 82)
//! - Cosmología Cuántica de Wheeler-DeWitt & Restricción Mini-Superespacio (Fase 83)
//! - Cuantización de Área de Bekenstein-Mukhanov & Espectro Discreto (Fase 84)

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave22_triad_holistic_orchestration() {
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

    // 1. Verificación Fase 82: Cromodinámica Cuántica en Red (Lattice QCD)
    assert!(report.lattice_qcd_report.string_tension_sigma >= 0.0);
    assert!(
        report
            .lattice_qcd_report
            .polyakov_loop_expectation
            .is_finite()
    );
    assert!(report.lattice_qcd_report.average_plaquette.is_finite());
    assert!(!report.lattice_qcd_report.lattice_confinement_collapse_veto);
    assert!(report.lattice_qcd_report.slot_scale_factor >= 0.10);

    // 2. Verificación Fase 83: Cosmología Cuántica de Wheeler-DeWitt
    assert!(report.wheeler_dewitt_report.wave_amplitude > 0.0);
    assert!(report.wheeler_dewitt_report.potential_v.is_finite());
    assert!(
        report
            .wheeler_dewitt_report
            .probability_current_j_phi
            .is_finite()
    );
    assert!(report.wheeler_dewitt_report.directional_signal.is_finite());
    assert!(!report.wheeler_dewitt_report.big_crunch_veto);
    assert!(report.wheeler_dewitt_report.slot_scale_factor >= 0.15);

    // 3. Verificación Fase 84: Cuantización de Área Bekenstein-Mukhanov
    assert!(report.bekenstein_mukhanov_report.quantized_area_an > 0.0);
    assert!(report.bekenstein_mukhanov_report.bekenstein_entropy_sn > 0.0);
    assert!(report.bekenstein_mukhanov_report.bohr_transition_frequency > 0.0);
    assert!(
        !report
            .bekenstein_mukhanov_report
            .bekenstein_evaporation_veto
    );
    assert!(report.bekenstein_mukhanov_report.slot_scale_factor >= 0.10);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave22_lattice_qcd_confinement_collapse_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar el motor Lattice QCD con umbral estricto para forzar colapso de confinamiento
    orchestrator.lattice_qcd_engines[0] =
        feature_engine::LatticeQcdWilsonPolyakovEngine::new(feature_engine::LatticeQcdConfig {
            critical_action_threshold: 0.01,
            polyakov_deconfinement_floor: 0.99,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.05,
            order_flow_imbalance: 0.95,
            relative_weight: 1.5,
        },
        AssetColorState {
            normalized_return: -0.04,
            order_flow_imbalance: -0.90,
            relative_weight: 0.5,
        },
        AssetColorState {
            normalized_return: 0.02,
            order_flow_imbalance: 0.40,
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

    assert!(report.lattice_qcd_report.lattice_confinement_collapse_veto);
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in report.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("LATTICE_GAUGE_CONFINEMENT_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c3_wave22_wheeler_dewitt_big_crunch_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Forzar condición Big Crunch con suelo de escala crítico inalcanzable
    orchestrator.wheeler_dewitt_engines[0] =
        signal_engine::WheelerDeWittMinisuperspaceEngine::new(signal_engine::WheelerDeWittConfig {
            scale_factor_crunch_floor: 100.0,
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

    assert!(report.wheeler_dewitt_report.big_crunch_veto);
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in report.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(decision.veto_reason, Some("WHEELER_DEWITT_BIG_CRUNCH_VETO"));
    }
}

#[test]
fn test_c4_wave22_bekenstein_mukhanov_area_evaporation_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar nivel cuántico mínimo alto para inducir evaporación crítica
    orchestrator.bekenstein_mukhanov_engines[0] =
        risk_engine::BekensteinMukhanovQuantizationEngine::new(
            risk_engine::BekensteinMukhanovConfig {
                critical_floor_quantum_n: 10_000,
                ..Default::default()
            },
        );

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

    assert!(
        report
            .bekenstein_mukhanov_report
            .bekenstein_evaporation_veto
    );
    assert_eq!(report.bekenstein_mukhanov_report.slot_scale_factor, 0.0);
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in report.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("BEKENSTEIN_MUKHANOV_AREA_EVAPORATION_VETO")
        );
    }
}

#[test]
fn test_c5_wave22_deterministic_replay_and_ruin_bound() {
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
        rep_a.lattice_qcd_report.string_tension_sigma,
        rep_b.lattice_qcd_report.string_tension_sigma
    );
    assert_eq!(
        rep_a.wheeler_dewitt_report.wave_amplitude,
        rep_b.wheeler_dewitt_report.wave_amplitude
    );
    assert_eq!(
        rep_a.bekenstein_mukhanov_report.quantized_area_an,
        rep_b.bekenstein_mukhanov_report.quantized_area_an
    );
    assert_eq!(rep_a.holistic_slot_scale, rep_b.holistic_slot_scale);
    assert_eq!(
        rep_a.ville_ruin_bound_satisfied,
        rep_b.ville_ruin_bound_satisfied
    );
}
