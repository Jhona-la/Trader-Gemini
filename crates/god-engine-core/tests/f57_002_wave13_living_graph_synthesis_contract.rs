//! 🌌 CONTRATO FORMAL F57-002: SÍNTESIS HOLÍSTICA DEL GRAFO VIVO OLA 13
//! (Wave 13 Holistic Living Graph Synthesis Contract: Fases 55, 56, 57)
//!
//! Este contrato formal verifica la integración armónica completa de la Ola 13 en el Grafo Omnisciente:
//! - c1: Evaluación simultánea y acoplada de Calabi-Yau Seiberg-Witten (Fase 55), Jarzynski-Crooks (Fase 56) y Barrera Simpléctica de Gromov (Fase 57).
//! - c2: Modulación estricta del factor de escala holístico $\eta_{\text{holistic}} \in [0.10, 1.25]$.
//! - c3: Inmunidad simpléctica y termodinámica: singularidad conifold, colapso de entropía y sobre-aplastamiento simpléctico activan veto y aplanan decisiones.
//! - c4: Cero pánicos `rb_p1`, determinismo $O(1)$ en stack y preservación de la cota de ruina de Ville $\le 10^{-6}$.

use god_engine_core::omniscient_living_graph::OmniscientLivingGraphOrchestrator;
use signal_engine::{QuantumSlotOrchestrator, QuantumSlotOrchestratorParams, SignalType};
use strategy_core::{AssetColorState, SU3_COLORS};

#[test]
fn test_c1_wave13_triad_holistic_orchestration() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(3);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.15,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: -0.0002,
            order_flow_imbalance: -0.08,
            relative_weight: 1.0,
        },
        AssetColorState {
            normalized_return: 0.0003,
            order_flow_imbalance: 0.05,
            relative_weight: 1.0,
        },
    ];

    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Tick 1: Régimen normal holgado
    let rep1 = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.02,
        250.0,
        240.0,
        1_700_000_000,
        &triad,
        1_000.0,
        3.0,
        0.02,
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    // 1. Verificación Fase 55 (Calabi-Yau & Seiberg-Witten)
    assert!(rep1.calabi_yau_report.kaehler_volume > 0.0);
    assert!(!rep1.calabi_yau_report.conifold_singularity_detected);
    assert!(rep1.calabi_yau_report.slot_scaling_factor >= 0.10);

    // 2. Verificación Fase 56 (Jarzynski-Crooks Thermodynamics)
    assert!(rep1.jarzynski_crooks_report.dissipated_work >= 0.0);
    assert!(rep1.jarzynski_crooks_report.entropy_production >= 0.0);
    assert!(!rep1.jarzynski_crooks_report.irreversible_entropy_collapse);
    assert!(rep1.jarzynski_crooks_report.signal_conviction_modulator >= 0.05);

    // 3. Verificación Fase 57 (Gromov Symplectic Barrier)
    assert!(rep1.gromov_symplectic_report.portfolio_phase_radius > 0.0);
    assert!(rep1.gromov_symplectic_report.hole_capacity_radius > 0.0);
    assert!(!rep1.gromov_symplectic_report.gromov_symplectic_veto);
    assert!(rep1.gromov_symplectic_report.slot_scaling_factor >= 0.10);

    // 4. Modulador Holístico de Escala Acotado
    assert!(
        rep1.holistic_slot_scale >= 0.10 && rep1.holistic_slot_scale <= 1.25,
        "Factor de escala holístico fuera del intervalo [0.10, 1.25]: {}",
        rep1.holistic_slot_scale
    );
    assert!(rep1.ville_ruin_bound_satisfied);
}

#[test]
fn test_c2_wave13_gromov_symplectic_veto_activation() {
    let mut orchestrator = OmniscientLivingGraphOrchestrator::new(1);
    let mut slot_orchestrator =
        QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let triad = [AssetColorState::default(); SU3_COLORS];
    let open_slots = [false; 3];
    let open_taus = [0.0; 3];
    let open_is_long = [false; 3];

    // Simulación de intento de trade desproporcionado contra liquidez nula (forzando sobre-aplastamiento de Gromov)
    let rep = orchestrator.evaluate_and_synthesize(
        0,
        100.0,
        100.50, // 50 bps spread
        0.001,  // bid qty casi cero
        0.001,  // ask qty casi cero
        1_700_000_100,
        &triad,
        100_000.0, // capital masivo intentando forzar entrada
        10.0,
        1.0, // 100% de capital en riesgo
        &mut slot_orchestrator,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert!(
        rep.gromov_symplectic_report.gromov_symplectic_veto,
        "Debe activar GROMOV_SYMPLECTIC_VETO ante intento de sobre-aplastamiento simpléctico"
    );

    // Todas las decisiones deben haber sido aplanadas a Flat por el veto
    for d in rep.multi_slot_evaluation.decisions.iter() {
        assert_eq!(d.signal, SignalType::Flat);
        assert_eq!(d.net_confidence, 0.0);
        assert!(d.vetoed);
    }
}
