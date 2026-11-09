//! Contrato Formal de Síntesis Holística: Ola 25 (Fases 91, 92, 93) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica en `OmniscientLivingGraphOrchestrator`:
//! - Geometría del Amplituhedro & Amplitudes Parke-Taylor MHV (Fase 91)
//! - Cosmología Inflacionaria de Starobinsky (R + R^2) & Dinámica de Inflatón (Fase 92)
//! - Geometría de Información de Fisher-Rao & Variedad \alpha-Dual de Amari (Fase 93)

use feature_engine::AmplituhedronOrderFlowConfig;
use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use risk_engine::FisherRaoAmariConfig;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use signal_engine::StarobinskyConfig;
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave25_triad_holistic_orchestration() {
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

    // Pulso nominal de mercado estable con spread ajustado (100.00 a 100.01 = 1 bps)
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

    // 1. Verificación Fase 91: Amplituhedro & Amplitudes Parke-Taylor MHV
    assert!(report.amplituhedron_order_flow_report.polytope_volume > 0.0);
    assert!(report
        .amplituhedron_order_flow_report
        .parke_taylor_amplitude
        .is_finite());
    assert!(
        !report
            .amplituhedron_order_flow_report
            .collinear_singularity_veto
    );
    assert!(report.amplituhedron_order_flow_report.slot_scaling_factor >= 0.10);

    // 2. Verificación Fase 92: Cosmología Inflacionaria de Starobinsky
    assert!(report.starobinsky_inflationary_report.effective_potential >= 0.0);
    assert!(report
        .starobinsky_inflationary_report
        .epsilon_slow_roll
        .is_finite());
    assert!(
        !report
            .starobinsky_inflationary_report
            .reheating_collapse_veto
    );
    assert!(report.starobinsky_inflationary_report.slot_scaling_factor >= 0.15);

    // 3. Verificación Fase 93: Geometría de Información de Fisher-Rao & Amari
    assert!(report.fisher_rao_amari_report.fisher_rao_geodesic_distance >= 0.0);
    assert!(report.fisher_rao_amari_report.amari_alpha_divergence >= 0.0);
    assert!(!report.fisher_rao_amari_report.information_collapse_veto);
    assert!(report.fisher_rao_amari_report.slot_scaling_factor >= 0.10);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave25_amplituhedron_collinear_singularity_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor del Amplituhedro con umbral estricto para probar la propagación del veto
    orchestrator.amplituhedron_engines[0] =
        feature_engine::AmplituhedronOrderFlowEngine::new(AmplituhedronOrderFlowConfig {
            min_polytope_volume: 0.90,
            min_spinor_bracket_threshold: 0.50,
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
            .amplituhedron_order_flow_report
            .collinear_singularity_veto
    );
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Singularidad colineal debe invalidar la cota de ruina de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("AMPLITUHEDRON_COLLINEAR_SINGULARITY_VETO")
        );
    }
}

#[test]
fn test_c3_wave25_starobinsky_reheating_collapse_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor de Starobinsky con umbral estricto para probar la propagación del veto
    orchestrator.starobinsky_engines[0] =
        signal_engine::StarobinskyInflationaryEngine::new(StarobinskyConfig {
            max_epsilon_slow_roll: 0.0001,
            max_tensor_to_scalar_ratio: 0.0001,
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
            .starobinsky_inflationary_report
            .reheating_collapse_veto
    );
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Colapso de recalentamiento violento debe invalidar la condición de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("STAROBINSKY_REHEATING_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c4_wave25_fisher_rao_information_collapse_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor de Fisher-Rao con umbral mínimo para probar la propagación del veto
    orchestrator.fisher_rao_amari_engines[0] =
        risk_engine::FisherRaoAmariManifoldEngine::new(FisherRaoAmariConfig {
            max_fisher_rao_geodesic_distance: 0.001,
            max_amari_divergence: 0.001,
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

    assert!(report.fisher_rao_amari_report.information_collapse_veto);
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Colapso de información de Fisher-Rao debe invalidar la condición de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("FISHER_RAO_INFORMATION_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c5_wave25_nan_infinity_immunity_and_zero_panic_guarantee() {
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

    assert!(report
        .amplituhedron_order_flow_report
        .polytope_volume
        .is_finite());
    assert!(report
        .starobinsky_inflationary_report
        .effective_potential
        .is_finite());
    assert!(report
        .fisher_rao_amari_report
        .fisher_rao_geodesic_distance
        .is_finite());
    assert!(report.holistic_slot_scale.is_finite());
}
