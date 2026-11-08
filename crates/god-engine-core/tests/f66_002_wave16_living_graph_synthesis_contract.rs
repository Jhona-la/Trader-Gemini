//! Contrato Formal de Síntesis: Ola 16 (Fases 64, 65, 66) en OmniscientLivingGraphOrchestrator
//!
//! Certifica la unificación holística continua:
//! - C1: Síntesis coordinada de PEPS Hypergraph Ricci, Solitones KdV y Monopolos de Dirac-Chern con Ville Ruin Bound.
//! - C2: Disparo determinista de veto duro de Monopolo de Dirac (`YANG_MILLS_DIRAC_MONOPOLE_VETO`).
//! - C3: Disparo de veto por dispersión hiperbólica crítica en red tensorial PEPS (`PEPS_HYPERBOLIC_DISPERSION_VETO`).
//! - C4: Dinámica no lineal y detección de solitones de KdV.

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::{AssetColorState, SignalType, SU3_COLORS};

#[test]
fn test_c1_wave16_triad_holistic_orchestration() {
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

    // Tick 1: Inicialización y flujo base
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

    // Verificaciones Fase 64 (PEPS Hypergraph Ricci)
    assert!(rep1.peps_hypergraph_report.peps_entanglement_entropy >= 0.0);
    assert!(rep1
        .peps_hypergraph_report
        .ollivier_ricci_curvature
        .is_finite());
    assert!(rep1.peps_hypergraph_report.hypergraph_spectral_dimension > 0.0);
    assert!(!rep1.peps_hypergraph_report.hyperbolic_dispersion_detected);
    assert!(rep1.peps_hypergraph_report.slot_scale_factor >= 0.10);

    // Verificaciones Fase 65 (KdV Soliton Dispersion)
    assert!(rep1.kdv_soliton_report.soliton_amplitude >= 0.0);
    assert!(rep1
        .kdv_soliton_report
        .nonlinearity_dispersion_balance_ratio
        .is_finite());
    assert!(rep1.kdv_soliton_report.kdv_hamiltonian_energy.is_finite());

    // Verificaciones Fase 66 (Dirac Monopole Chern Bundle)
    assert!(rep1.dirac_monopole_report.gauge_field_curvature_flux >= 0.0);
    assert!(rep1.dirac_monopole_report.monopole_charge_density >= 0.0);
    assert!(!rep1.dirac_monopole_report.monopole_veto_induced);

    // Tick 2: Continuidad dinámica y convergencia de operadores espectrales
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
fn test_c2_wave16_dirac_monopole_hard_veto() {
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

    // Configurar umbrales hipersensibles para forzar detección de monopolo de Dirac
    orchestrator.dirac_monopole_engines[0] =
        risk_engine::DiracMonopoleChernBundleEngine::new(risk_engine::DiracMonopoleParams {
            critical_flux_threshold: 0.00001,
            ..Default::default()
        });

    let rep_shock = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        101.50,
        1000.0,
        1.0,
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

    assert!(rep_shock.dirac_monopole_report.monopole_veto_induced);
    for decision in rep_shock.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert_eq!(decision.net_confidence, 0.0);
        assert!(decision.vetoed);
        assert_eq!(decision.veto_reason, Some("YANG_MILLS_DIRAC_MONOPOLE_VETO"));
    }
}

#[test]
fn test_c3_wave16_peps_hyperbolic_dispersion_veto() {
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

    // Configurar PEPS con umbral hiperbólico restrictivo (Ricci menor que critical activa veto)
    orchestrator.peps_ricci_engines[0] =
        feature_engine::PepsHypergraphRicciEngine::new(feature_engine::PepsRicciParams {
            critical_negative_ricci: 0.80,
            ..Default::default()
        });

    let rep_shock = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.80,
        500.0,
        5.0,
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
            .peps_hypergraph_report
            .hyperbolic_dispersion_detected
    );
    for decision in rep_shock.multi_slot_evaluation.decisions.iter() {
        assert_eq!(decision.signal, SignalType::Flat);
        assert_eq!(decision.net_confidence, 0.0);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("PEPS_HYPERBOLIC_DISPERSION_VETO")
        );
    }
}

#[test]
fn test_c4_wave16_kdv_soliton_dispersion_conviction() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0012,
            order_flow_imbalance: 0.35,
            relative_weight: 1.2,
        },
        AssetColorState {
            normalized_return: 0.0009,
            order_flow_imbalance: 0.28,
            relative_weight: 1.1,
        },
        AssetColorState {
            normalized_return: 0.0006,
            order_flow_imbalance: 0.19,
            relative_weight: 1.05,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    let rep = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.05,
        60.0,
        30.0,
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

    assert!(rep.kdv_soliton_report.soliton_amplitude >= 0.0);
    assert!(rep
        .kdv_soliton_report
        .nonlinearity_dispersion_balance_ratio
        .is_finite());
    assert!(rep.kdv_soliton_report.soliton_impulse_signal.is_finite());
}
