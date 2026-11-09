//! Contrato Formal de Síntesis Holística: Ola 26 (Fases 94, 95, 96) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica en `OmniscientLivingGraphOrchestrator`:
//! - Geometría Simpléctica Infinita & Cohomología de Floer Cuántica (Fase 94)
//! - Supersimetría $\mathcal{N}=1$ & Ecuación de Nicolai (Fase 95)
//! - Geometría de Calabi-Yau Cuaterniónica & Hiperkähler (Fase 96)

use feature_engine::SymplecticFloerConfig;
use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use risk_engine::QuaternionicHyperkahlerConfig;
use signal_engine::SupersymmetryNicolaiConfig;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave26_triad_holistic_orchestration() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0006,
            order_flow_imbalance: 0.12,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: -0.0001,
            order_flow_imbalance: -0.04,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: 0.0002,
            order_flow_imbalance: 0.03,
            relative_weight: 0.50,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Pulso nominal de mercado estable con spread ajustado
    let report = orchestrator.evaluate_and_synthesize(
        0,
        99.995,
        100.005,
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

    // 1. Verificación Fase 94: Cohomología de Floer Simpléctica
    assert!(report.symplectic_floer_report.symplectic_action.is_finite());
    assert!(report.symplectic_floer_report.floer_energy >= 0.0);
    assert!(
        !report
            .symplectic_floer_report
            .floer_homology_obstruction_veto
    );
    assert!(report.symplectic_floer_report.slot_scaling_factor >= 0.10);

    // 2. Verificación Fase 95: Supersimetría & Ecuación de Nicolai
    assert!(report.supersymmetry_nicolai_report.vacuum_energy >= 0.0);
    assert!(report.supersymmetry_nicolai_report.witten_index.is_finite());
    assert!(
        !report
            .supersymmetry_nicolai_report
            .susy_spontaneous_breaking_veto
    );
    assert!(report.supersymmetry_nicolai_report.slot_scaling_factor >= 0.15);

    // 3. Verificación Fase 96: Geometría Hiperkähler Cuaterniónica
    assert!(report.quaternionic_hyperkahler_report.quaternion_norm > 0.0);
    assert!(report.quaternionic_hyperkahler_report.moment_map_norm >= 0.0);
    assert!(
        !report
            .quaternionic_hyperkahler_report
            .moment_map_singularity_veto
    );
    assert!(report.quaternionic_hyperkahler_report.slot_scaling_factor >= 0.10);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave26_floer_homology_obstruction_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor de Floer con umbral estricto para forzar el veto
    orchestrator.prev_mid_prices[0] = 90.0;
    orchestrator.floer_engines[0] =
        feature_engine::SymplecticFloerCohomologyEngine::new(SymplecticFloerConfig {
            max_floer_energy: 0.0001,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.10,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.50,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let report = orchestrator.evaluate_and_synthesize(
        0,
        99.995,
        100.005,
        50.0,
        5.0,
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
            .symplectic_floer_report
            .floer_homology_obstruction_veto
    );
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Obstrucción de Floer debe invalidar la condición de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("FLOER_HOMOLOGY_OBSTRUCTION_VETO")
        );
    }
}

#[test]
fn test_c3_wave26_susy_spontaneous_breaking_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor de supersimetría con umbral mínimo para probar la propagación del veto
    orchestrator.prev_mid_prices[0] = 90.0;
    orchestrator.susy_engines[0] =
        signal_engine::SupersymmetryNicolaiEngine::new(SupersymmetryNicolaiConfig {
            max_vacuum_energy: 0.0001,
            max_nicolai_residual: 0.0001,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.05,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.50,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let report = orchestrator.evaluate_and_synthesize(
        0,
        99.995,
        100.005,
        50.0,
        5.0,
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
            .supersymmetry_nicolai_report
            .susy_spontaneous_breaking_veto
    );
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Ruptura espontánea de supersimetría debe invalidar la condición de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(decision.veto_reason, Some("SUSY_SPONTANEOUS_BREAKING_VETO"));
    }
}

#[test]
fn test_c4_wave26_hyperkahler_moment_map_singularity_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor hiperkähler con umbral mínimo para probar la propagación del veto
    orchestrator.hyperkahler_engines[0] =
        risk_engine::QuaternionicHyperkahlerEngine::new(QuaternionicHyperkahlerConfig {
            max_moment_map_norm: 0.0001,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.05,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.50,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let report = orchestrator.evaluate_and_synthesize(
        0,
        99.995,
        100.005,
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
            .quaternionic_hyperkahler_report
            .moment_map_singularity_veto
    );
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Singularidad de mapa de momentos hiperkähler debe invalidar la condición de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("HYPERKAHLER_MOMENT_MAP_SINGULARITY_VETO")
        );
    }
}

#[test]
fn test_c5_wave26_nan_infinity_immunity_and_zero_panic_guarantee() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: f64::NAN,
            order_flow_imbalance: f64::INFINITY,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.50,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let report = orchestrator.evaluate_and_synthesize(
        0,
        f64::NAN,
        f64::INFINITY,
        -1.0,
        -1.0,
        1_700_000_000_000,
        &triad,
        -10.0,
        20.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert!(report.symplectic_floer_report.symplectic_action.is_finite());
    assert!(
        report
            .supersymmetry_nicolai_report
            .vacuum_energy
            .is_finite()
    );
    assert!(
        report
            .quaternionic_hyperkahler_report
            .quaternion_norm
            .is_finite()
    );
    assert!(report.holistic_slot_scale.is_finite());
}
