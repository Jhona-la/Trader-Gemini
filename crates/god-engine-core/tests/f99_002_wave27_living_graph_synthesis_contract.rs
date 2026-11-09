//! Contrato Formal de Síntesis Holística: Ola 27 (Fases 97, 98, 99) — Grafo Vivo Omnisciente
//!
//! Verifica la integración sinérgica en `OmniscientLivingGraphOrchestrator`:
//! - Geometría No Conmutativa de Connes-Kreimer & Renormalización Hopf (Fase 97)
//! - Correspondencia de Donaldson-Thomas Cuántica & Vértice MacMahon (Fase 98)
//! - Gravitación 4D Cuántica Asintóticamente Segura de Weinberg & Punto Fijo Reuter (Fase 99)

use feature_engine::ConnesKreimerConfig;
use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use risk_engine::AsymptoticSafetyConfig;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use signal_engine::DonaldsonThomasConfig;
use strategy_core::{AssetColorState, SignalType};

#[test]
fn test_c1_wave27_triad_holistic_orchestration() {
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

    // 1. Verificación Fase 97: Connes-Kreimer Hopf Algebra
    assert!(report.connes_kreimer_report.uv_divergence_residue >= 0.0);
    assert!(report
        .connes_kreimer_report
        .renormalized_character
        .is_finite());
    assert!(!report.connes_kreimer_report.connes_kreimer_divergence_veto);
    assert!(report.connes_kreimer_report.slot_scaling_factor >= 0.10);

    // 2. Verificación Fase 98: Donaldson-Thomas Quantum Correspondence
    assert!(report.donaldson_thomas_report.dt_invariant_density >= 0.0);
    assert!(report.donaldson_thomas_report.macmahon_partition_energy >= 0.0);
    assert!(
        !report
            .donaldson_thomas_report
            .donaldson_thomas_wall_crossing_veto
    );
    assert!(report.donaldson_thomas_report.slot_scaling_factor >= 0.10);

    // 3. Verificación Fase 99: Asymptotic Safety Reuter Flow
    assert!(
        report
            .asymptotic_safety_report
            .dimensionless_newton_coupling
            > 0.0
    );
    assert!(report.asymptotic_safety_report.ngfp_distance >= 0.0);
    assert!(
        !report
            .asymptotic_safety_report
            .asymptotic_safety_singularity_veto
    );
    assert!(report.asymptotic_safety_report.slot_scaling_factor >= 0.10);

    // 4. Modulación global de slots y cota de Ville
    assert!(report.holistic_slot_scale >= 0.10 && report.holistic_slot_scale <= 1.25);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave27_connes_kreimer_divergence_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor de Connes-Kreimer con umbral ultra sensible para forzar el veto
    orchestrator.prev_mid_prices[0] = 80.0;
    orchestrator.connes_kreimer_engines[0] =
        feature_engine::ConnesKreimerHopfEngine::new(ConnesKreimerConfig {
            max_divergence_residue: 0.0001,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.05,
            order_flow_imbalance: 2.0,
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
        99.90,
        100.10,
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
        report.connes_kreimer_report.connes_kreimer_divergence_veto,
        "Divergencia ultravioleta debe detonar CONNES_KREIMER_DIVERGENCE_VETO"
    );
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(decision.veto_reason, Some("CONNES_KREIMER_DIVERGENCE_VETO"));
    }
}

#[test]
fn test_c3_wave27_donaldson_thomas_wall_crossing_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor de Donaldson-Thomas con umbral ultra sensible para forzar el veto
    orchestrator.prev_mid_prices[0] = 80.0;
    orchestrator.donaldson_thomas_engines[0] =
        signal_engine::DonaldsonThomasQuantumEngine::new(DonaldsonThomasConfig {
            wall_crossing_threshold: 0.0001,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.05,
            order_flow_imbalance: 2.0,
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
        99.90,
        100.10,
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
            .donaldson_thomas_report
            .donaldson_thomas_wall_crossing_veto,
        "Salto de estabilidad debe detonar DONALDSON_THOMAS_WALL_CROSSING_VETO"
    );
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("DONALDSON_THOMAS_WALL_CROSSING_VETO")
        );
    }
}

#[test]
fn test_c4_wave27_asymptotic_safety_singularity_veto_propagation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    // Configuramos el motor de Asymptotic Safety con umbral ultra sensible para forzar el veto
    orchestrator.prev_mid_prices[0] = 80.0;
    orchestrator.asymptotic_safety_engines[0] =
        risk_engine::AsymptoticSafetyReuterEngine::new(AsymptoticSafetyConfig {
            max_ngfp_distance: 0.0001,
            ..Default::default()
        });

    let triad = [
        AssetColorState {
            normalized_return: 0.05,
            order_flow_imbalance: 2.0,
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
        99.90,
        100.10,
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
            .asymptotic_safety_report
            .asymptotic_safety_singularity_veto,
        "Colapso métrico debe detonar ASYMPTOTIC_SAFETY_SINGULARITY_VETO"
    );
    assert!(!report.ville_ruin_bound_satisfied);
    for decision in &report.multi_slot_evaluation.decisions {
        assert_eq!(decision.signal, SignalType::Flat);
        assert!(decision.vetoed);
        assert_eq!(
            decision.veto_reason,
            Some("ASYMPTOTIC_SAFETY_SINGULARITY_VETO")
        );
    }
}

#[test]
fn test_c5_wave27_nan_infinity_immunity_and_zero_panic_guarantee() {
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
        .connes_kreimer_report
        .uv_divergence_residue
        .is_finite());
    assert!(report
        .donaldson_thomas_report
        .dt_invariant_density
        .is_finite());
    assert!(report
        .asymptotic_safety_report
        .dimensionless_newton_coupling
        .is_finite());
    assert!(report.holistic_slot_scale.is_finite());
}
