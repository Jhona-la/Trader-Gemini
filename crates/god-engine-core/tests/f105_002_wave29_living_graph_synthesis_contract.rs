//! Contrato Formal de Síntesis Holística: Ola 29 (Fases 103, 104, 105) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica en `OmniscientLivingGraphOrchestrator`:
//! - Teoría de Cuerdas Heterótica $E_8 \times E_8$ & Instantones Calabi-Yau (Fase 103)
//! - Entrelazamiento Cuántico Topological Entanglement Entropy & Fractones X-Cube (Fase 104)
//! - Métrica de Agujero Negro Kerr-Newman & Termodinámica de Bardeen-Carter-Hawking (Fase 105)

use feature_engine::HeteroticE8CalabiYauConfig;
use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use risk_engine::{KerrNewmanConfig, WheelerFeynmanAbsorberConfig, WheelerFeynmanAbsorberEngine};
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use signal_engine::FractonTopologicalConfig;
use strategy_core::{AssetColorState, SignalType};

/// Integracion origin/main (W31): el salto de precio de estos fixtures tambien dispara el veto
/// Dirac de Wheeler-Feynman (ola 31), que encabeza la cadena de etiquetas. Se apaga ese motor
/// para que cada caso mida solo el veto de la ola 29; la decision de vetar no cambia.
fn aislar_de_la_ola_31(orchestrator: &mut OmniscientLivingGraphOrchestrator) {
    orchestrator.wheeler_feynman_engines[0] =
        WheelerFeynmanAbsorberEngine::new(WheelerFeynmanAbsorberConfig {
            radiation_damping_coupling: 0.0,
            max_absorber_deficit: 1.0,
            max_runaway_jerk_threshold: f64::MAX,
            ..Default::default()
        });
}

#[test]
fn test_c1_wave29_triad_holistic_orchestration() {
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
            normalized_return: -0.0001,
            order_flow_imbalance: -0.03,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: 0.0002,
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

    // 1. Verificación Fase 103: Heterotic E8 x E8 String Theory
    assert!(report.heterotic_e8_report.green_schwarz_anomaly >= 0.0);
    assert!(!report.heterotic_e8_report.anomaly_cancellation_veto);
    assert!(report.heterotic_e8_report.slot_scaling_factor >= 0.10);

    // 2. Verificación Fase 104: Fracton Topological Order
    assert!(
        report
            .fracton_topological_report
            .topological_entanglement_entropy
            >= 0.0
    );
    assert!(!report.fracton_topological_report.fracton_collapse_veto);
    assert!(report.fracton_topological_report.slot_scaling_factor >= 0.10);

    // 3. Verificación Fase 105: Kerr-Newman Black Hole Metric
    assert!(report.kerr_newman_report.event_horizon_r_plus > 0.0);
    assert!(report.kerr_newman_report.bekenstein_hawking_entropy > 0.0);
    assert!(!report.kerr_newman_report.naked_singularity_veto);
    assert!(!report.kerr_newman_report.hawking_thermal_evaporation_veto);
    assert!(report.kerr_newman_report.slot_scaling_factor >= 0.10);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave29_heterotic_e8_anomaly_cancellation_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    aislar_de_la_ola_31(&mut orchestrator);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor Heterotic E8 con umbral estricto para forzar veto de anomalía
    orchestrator.heterotic_e8_engines[0] =
        feature_engine::HeteroticE8CalabiYauEngine::new(HeteroticE8CalabiYauConfig {
            anomaly_threshold: 0.000001,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.005,
            order_flow_imbalance: 0.20,
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

    // Calentamiento para llenar el buffer histórico de 8 estados y registrar curvatura
    for i in 1..=10 {
        orchestrator.evaluate_and_synthesize(
            0,
            100.0 + (i as f64) * 0.10,
            100.01 + (i as f64) * 0.10,
            60.0 + (i as f64) * 2.0,
            40.0 - (i as f64) * 1.0,
            1_700_000_000_000 + i * 1000,
            &triad,
            13.0,
            20.0,
            0.02,
            &mut slot_orchestrator,
            &open_slots,
            &open_taus,
            &open_is_long,
        );
    }

    let report = orchestrator.evaluate_and_synthesize(
        0,
        102.0,
        102.01,
        80.0,
        20.0,
        1_700_000_020_000,
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
        report.heterotic_e8_report.anomaly_cancellation_veto,
        "Debe detonar HETEROTIC_ANOMALY_CANCELLATION_VETO"
    );
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Cota de Ville debe ser false bajo veto"
    );

    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("HETEROTIC_ANOMALY_CANCELLATION_VETO")
        );
    }
}

#[test]
fn test_c3_wave29_fracton_topological_collapse_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    aislar_de_la_ola_31(&mut orchestrator);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor Fracton Topological con umbral inalcanzable para forzar veto de colapso
    orchestrator.fracton_topological_engines[0] =
        signal_engine::FractonTopologicalOrderEngine::new(FractonTopologicalConfig {
            topological_entropy_threshold: 5.0,
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

    // Calentamiento para llenar el buffer histórico de 8 estados y registrar curvatura
    for i in 1..=10 {
        orchestrator.evaluate_and_synthesize(
            0,
            100.0 + (i as f64) * 0.10,
            100.01 + (i as f64) * 0.10,
            60.0 + (i as f64) * 2.0,
            40.0 - (i as f64) * 1.0,
            1_700_000_000_000 + i * 1000,
            &triad,
            13.0,
            20.0,
            0.02,
            &mut slot_orchestrator,
            &open_slots,
            &open_taus,
            &open_is_long,
        );
    }

    let report = orchestrator.evaluate_and_synthesize(
        0,
        102.0,
        102.01,
        80.0,
        20.0,
        1_700_000_020_000,
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
        report.fracton_topological_report.fracton_collapse_veto,
        "Debe detonar FRACTON_TOPOLOGICAL_COLLAPSE_VETO"
    );
    assert!(!report.ville_ruin_bound_satisfied);

    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("FRACTON_TOPOLOGICAL_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c4_wave29_kerr_newman_naked_singularity_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    aislar_de_la_ola_31(&mut orchestrator);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor Kerr-Newman con margen cósmico imposible para forzar veto de censura cósmica
    orchestrator.kerr_newman_engines[0] =
        risk_engine::KerrNewmanBlackHoleEngine::new(KerrNewmanConfig {
            cosmic_censorship_margin: 1000.0,
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

    // Calentamiento para llenar el buffer histórico de 8 estados y registrar curvatura
    for i in 1..=10 {
        orchestrator.evaluate_and_synthesize(
            0,
            100.0 + (i as f64) * 0.10,
            100.01 + (i as f64) * 0.10,
            60.0 + (i as f64) * 2.0,
            40.0 - (i as f64) * 1.0,
            1_700_000_000_000 + i * 1000,
            &triad,
            13.0,
            20.0,
            0.02,
            &mut slot_orchestrator,
            &open_slots,
            &open_taus,
            &open_is_long,
        );
    }

    let report = orchestrator.evaluate_and_synthesize(
        0,
        102.0,
        102.01,
        80.0,
        20.0,
        1_700_000_020_000,
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
        report.kerr_newman_report.naked_singularity_veto,
        "Debe detonar KERR_NEWMAN_NAKED_SINGULARITY_VETO"
    );
    assert!(!report.ville_ruin_bound_satisfied);

    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("KERR_NEWMAN_NAKED_SINGULARITY_VETO")
        );
    }
}

#[test]
fn test_c5_wave29_slot_scaling_multiplicative_impact() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0001,
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
        100.0,
        100.005,
        20.0,
        20.0,
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

    assert!(report.holistic_slot_scale >= 0.10);
    assert!(report.holistic_slot_scale <= 1.25);
    assert!(report.heterotic_e8_report.slot_scaling_factor >= 0.10);
    assert!(report.fracton_topological_report.slot_scaling_factor >= 0.10);
    assert!(report.kerr_newman_report.slot_scaling_factor >= 0.10);
}
