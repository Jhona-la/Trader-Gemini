//! Contrato Formal de Síntesis Holística: Ola 24 (Fases 88, 89, 90) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica en `OmniscientLivingGraphOrchestrator`:
//! - Cuantización Geométrica de Kostant-Souriau & Variedad Precuántica (Fase 88)
//! - Cicatrices Cuánticas Many-Body (QMBS) & Evasión de Termalización (Fase 89)
//! - Superficies Cuánticas Extremales (QES) & Fórmula de Engelhardt-Wall (Fase 90)

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType};

/// Integracion origin/main: los vetos de olas posteriores (Donaldson-Thomas, ola 27; DRGT, ola 28;
/// heterotico, ola 29; Dirac de Wheeler-Feynman, ola 31) van antes en la cadena de etiquetas y estos
/// fixtures tambien los disparan. Se apagan esos motores para que cada caso mida solo el veto de la
/// ola 24; la decision de vetar no cambia.
fn aislar_de_olas_posteriores(orchestrator: &mut OmniscientLivingGraphOrchestrator) {
    orchestrator.donaldson_thomas_engines[0] =
        signal_engine::DonaldsonThomasQuantumEngine::new(signal_engine::DonaldsonThomasConfig {
            wall_crossing_threshold: f64::MAX,
            ..Default::default()
        });
    orchestrator.heterotic_e8_engines[0] =
        feature_engine::HeteroticE8CalabiYauEngine::new(feature_engine::HeteroticE8CalabiYauConfig {
            anomaly_threshold: f64::MAX,
            ..Default::default()
        });
    orchestrator.wheeler_feynman_engines[0] =
        risk_engine::WheelerFeynmanAbsorberEngine::new(risk_engine::WheelerFeynmanAbsorberConfig {
            radiation_damping_coupling: 0.0,
            max_absorber_deficit: 1.0,
            max_runaway_jerk_threshold: f64::MAX,
            ..Default::default()
        });
    orchestrator.drgt_gravity_engines[0] =
        risk_engine::DrgtMassiveGravityEngine::new(risk_engine::DrgtMassiveGravityConfig {
            graviton_mass_mg: 1e-3,
            higuchi_margin: 0.0,
            ..Default::default()
        });
}

#[test]
fn test_c1_wave24_triad_holistic_orchestration() {
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

    // 1. Verificación Fase 88: Kostant-Souriau
    assert!(
        report
            .kostant_souriau_report
            .prequantum_observable_val
            .is_finite()
    );
    assert!(
        report
            .kostant_souriau_report
            .symplectic_curvature_norm
            .is_finite()
    );
    assert!(
        report
            .kostant_souriau_report
            .reeb_contact_holonomy
            .is_finite()
    );
    assert!(!report.kostant_souriau_report.prequantum_collapse_veto);
    assert!(report.kostant_souriau_report.slot_scale_factor >= 0.10);

    // 2. Verificación Fase 89: QMBS
    assert!(report.qmbs_report.revival_fidelity >= 0.0);
    assert!(report.qmbs_report.spectral_form_factor.is_finite());
    assert!(report.qmbs_report.scarred_directional_signal.is_finite());
    assert!(!report.qmbs_report.ergodic_collapse_veto);
    assert!(report.qmbs_report.slot_scale_factor >= 0.15);

    // 3. Verificación Fase 90: QES
    assert!(report.qes_report.generalized_entropy_min > 0.0);
    assert!(report.qes_report.qes_radius.is_finite());
    assert!(!report.qes_report.extremal_surface_collapse_veto);
    assert!(report.qes_report.slot_scale_factor >= 0.10);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave24_kostant_souriau_prequantum_collapse_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    aislar_de_olas_posteriores(&mut orchestrator);
    orchestrator.kostant_souriau_engines[0] =
        feature_engine::KostantSouriauPrequantumEngine::new(feature_engine::KostantSouriauConfig {
            min_polarization_coherence: 1.0,
            max_symplectic_curvature_drift: 0.0,
            ..Default::default()
        });
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.001,
            order_flow_imbalance: 0.10,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Forzamos colapso precuántico con choque de polarización anarmónica extrema
    let mut report = orchestrator.evaluate_and_synthesize(
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

    // Aplicamos perturbaciones no simplécticas violentas
    for i in 1..8 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        report = orchestrator.evaluate_and_synthesize(
            0,
            100.0 + sign * 5.0,
            100.02 + sign * 5.0,
            0.1,
            0.1,
            1_700_000_000_000 + i * 100,
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

    assert!(report.kostant_souriau_report.prequantum_collapse_veto);
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Colapso precuántico debe invalidar la condición de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("KOSTANT_SOURIAU_PREQUANTUM_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c3_wave24_qmbs_ergodic_collapse_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    aislar_de_olas_posteriores(&mut orchestrator);
    orchestrator.qmbs_engines[0] =
        signal_engine::QuantumManyBodyScarsEngine::new(signal_engine::QuantumManyBodyScarsConfig {
            min_revival_fidelity: 1.0,
            ..Default::default()
        });
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Evaluamos bajo ruido térmico caótico extremo
    let mut report = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.05,
        1.0,
        1.0,
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

    for i in 1..10 {
        report = orchestrator.evaluate_and_synthesize(
            0,
            100.0 + (i as f64) * 0.5,
            100.05 + (i as f64) * 0.5,
            0.5,
            0.5,
            1_700_000_000_000 + i * 200,
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

    assert!(report.qmbs_report.ergodic_collapse_veto);
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Colapso ergódico total debe invalidar la condición de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("QUANTUM_SCARS_ERGODIC_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c4_wave24_qes_extremal_surface_collapse_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Colapso de capital pegado al floor crítico de liquidación
    let report = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.01,
        10.0,
        10.0,
        1_700_000_000_000,
        &triad,
        10.02, // Capital prácticamente en el suelo de $10 USD
        20.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert!(report.qes_report.extremal_surface_collapse_veto);
    assert!(
        !report.ville_ruin_bound_satisfied,
        "Colapso de superficie extremal de capital debe invalidar la cota de Ville"
    );
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("QUANTUM_EXTREMAL_SURFACE_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c5_wave24_ville_ruin_bound_invariant() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0001,
            order_flow_imbalance: 0.02,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
        AssetColorState {
            normalized_return: 0.0,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let report = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.01,
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

    assert!(report.ville_ruin_bound_satisfied);
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
}
