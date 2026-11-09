//! Contrato Formal de Síntesis Holística: Ola 21 (Fases 79, 80, 81) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica de Modelos Matriciales & Jerarquía KP (Fase 79),
//! Trenzado de Anyones No Abelianos & Polinomio de Jones (Fase 80),
//! y la Curva de Page & Fórmula de Isla Holográfica (Fase 81) en `OmniscientLivingGraphOrchestrator`.

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave21_triad_holistic_orchestration() {
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

    // 1. Verificación Fase 79: Modelos Matriciales & Jerarquía KP
    assert!(report.matrix_model_kp_report.tau_function_val > 0.0);
    assert!(
        report
            .matrix_model_kp_report
            .soliton_potential_u
            .is_finite()
    );
    assert!(report.matrix_model_kp_report.kp_curvature_uxx.is_finite());
    assert!(!report.matrix_model_kp_report.kp_soliton_shock_detected);
    assert!(report.matrix_model_kp_report.slot_scale_factor >= 0.20);

    // 2. Verificación Fase 80: Trenzado de Anyones No Abelianos & Polinomio de Jones
    assert!(report.non_abelian_braiding_report.jones_polynomial_norm >= 0.5);
    assert!(
        report
            .non_abelian_braiding_report
            .braid_holonomy_phase
            .is_finite()
    );
    assert!(
        !report
            .non_abelian_braiding_report
            .braid_decoherence_detected
    );
    assert!(
        report
            .non_abelian_braiding_report
            .holographic_conviction_modulator
            >= 0.40
    );

    // 3. Verificación Fase 81: Curva de Page & Fórmula de Isla Holográfica
    assert!(report.page_curve_island_report.bekenstein_hawking_entropy > 0.0);
    assert!(report.page_curve_island_report.page_entropy.is_finite());
    assert!(!report.page_curve_island_report.island_veto_induced);
    assert!(report.page_curve_island_report.quantum_island_present);
    assert!(report.page_curve_island_report.slot_scale_factor >= 0.15);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave21_kp_soliton_shock_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar el motor KP con umbral estricto para forzar shock
    orchestrator.matrix_model_kp_engines[0] =
        feature_engine::MatrixModelKpHierarchyEngine::new(feature_engine::MatrixModelKpConfig {
            critical_residual_threshold: 0.01,
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

    // Evaluación con fuerte gradiente de precio
    let _ = orchestrator.evaluate_and_synthesize(
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

    let report2 = orchestrator.evaluate_and_synthesize(
        0,
        110.0,
        110.02,
        10.0,
        10.0,
        1_700_000_001_000,
        &triad,
        13.0,
        20.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert!(report2.matrix_model_kp_report.kp_soliton_shock_detected);
    assert!(!report2.ville_ruin_bound_satisfied);
    for decision in report2.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(decision.veto_reason, Some("KP_SOLITON_SHOCK_VETO"));
    }
}

#[test]
fn test_c3_wave21_page_curve_island_collapse_hard_veto() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configurar motor de isla con piso alto y min_area alta para inducir colapso
    orchestrator.page_curve_island_engines[0] =
        risk_engine::PageCurveIslandRiskEngine::new(risk_engine::PageCurveIslandConfig {
            emergency_floor_usd: 95.0, // Cerca del equity de $100
            min_island_area: 0.99,     // Exige descorrelación casi absoluta
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
        0.50, // Volatilidad alta (radiación severa)
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    println!("page report: {:?}", report.page_curve_island_report);

    assert!(report.page_curve_island_report.island_veto_induced);
    assert_eq!(report.page_curve_island_report.slot_scale_factor, 0.0);
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in report.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("PAGE_CURVE_INFORMATION_COLLAPSE_VETO")
        );
    }
}

#[test]
fn test_c4_wave21_deterministic_replay_and_ruin_bound() {
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
        rep_a.matrix_model_kp_report.tau_function_val,
        rep_b.matrix_model_kp_report.tau_function_val
    );
    assert_eq!(
        rep_a.non_abelian_braiding_report.jones_polynomial_norm,
        rep_b.non_abelian_braiding_report.jones_polynomial_norm
    );
    assert_eq!(
        rep_a.page_curve_island_report.page_entropy,
        rep_b.page_curve_island_report.page_entropy
    );
    assert_eq!(rep_a.holistic_slot_scale, rep_b.holistic_slot_scale);
    assert_eq!(
        rep_a.ville_ruin_bound_satisfied,
        rep_b.ville_ruin_bound_satisfied
    );
}
