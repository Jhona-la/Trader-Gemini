//! Contrato Formal de Propiedades Matemáticas e Invariantes: Fase 36 (§36)
//! Gobernador de Veto Adaptativo Cuántico y Controlador Conformal de Riesgo
//!
//! - C1: Inviolabilidad de veto duro por colapso de margen de ruina de Ville ($\le 0.001$).
//! - C2: Inviolabilidad de veto duro por colapso de piso dinámico de Grossman-Zhou ($\le 0.0005$).
//! - C3: Expansión suave de elasticidad dinámica de spread ceiling ante volatilidad continua.
//! - C4: Concesión de entrada elástica ante convicción tensorial cuántica fuerte ($|\Psi| \ge 0.65$).
//! - C5: Calibración conformal de riesgo online mediante cuantiles de pérdidas.
//! - C6: Invarianza ante valores patológicos (cero pánicos `rb_p1`).

use risk_engine::{
    AdaptiveQuantumVetoGovernor, AdaptiveQuantumVetoParams, MarketQuantumStateInput,
};

#[test]
fn c1_hard_veto_inviolability_ville_ruin() {
    let governor = AdaptiveQuantumVetoGovernor::default();

    let input = MarketQuantumStateInput {
        current_spread_bps: 5.0,
        continuous_effective_vol: 0.015,
        reference_vol: 0.015,
        quantum_tensor_conviction: 0.90,
        shockwave_amplitude: 0.50,
        air_pocket_active: false,
        behavioral_resonance: 0.50,
        grossman_zhou_floor_distance: 0.50,
        ville_ruin_headroom: 0.0005, // Violación de ruina de Ville
    };

    let decision = governor.evaluate_governance(&input);
    assert!(decision.is_hard_vetoed);
    assert_eq!(decision.sizing_multiplier, 0.0);
    assert_eq!(
        decision.hard_veto_reason,
        "VILLE_SUPERMARTINGALE_RUIN_BOUND_EXCEEDED"
    );
}

#[test]
fn c2_hard_veto_inviolability_grossman_zhou_floor() {
    let governor = AdaptiveQuantumVetoGovernor::default();

    let input = MarketQuantumStateInput {
        current_spread_bps: 5.0,
        continuous_effective_vol: 0.015,
        reference_vol: 0.015,
        quantum_tensor_conviction: 0.90,
        shockwave_amplitude: 0.50,
        air_pocket_active: false,
        behavioral_resonance: 0.50,
        grossman_zhou_floor_distance: 0.0001, // Colapso al piso Grossman-Zhou
        ville_ruin_headroom: 1.0,
    };

    let decision = governor.evaluate_governance(&input);
    assert!(decision.is_hard_vetoed);
    assert_eq!(decision.sizing_multiplier, 0.0);
    assert_eq!(decision.hard_veto_reason, "GROSSMAN_ZHOU_FLOOR_COLLAPSE");
}

#[test]
fn c3_dynamic_spread_elasticity_expansion() {
    let governor = AdaptiveQuantumVetoGovernor::default();

    let input_low_vol = MarketQuantumStateInput {
        current_spread_bps: 6.0,
        continuous_effective_vol: 0.015,
        reference_vol: 0.015,
        quantum_tensor_conviction: 0.40,
        shockwave_amplitude: 0.0,
        air_pocket_active: false,
        behavioral_resonance: 0.0,
        grossman_zhou_floor_distance: 0.50,
        ville_ruin_headroom: 1.0,
    };

    let input_high_vol = MarketQuantumStateInput {
        current_spread_bps: 6.0,
        continuous_effective_vol: 0.045, // 3x volatilidad
        reference_vol: 0.015,
        quantum_tensor_conviction: 0.40,
        shockwave_amplitude: 0.0,
        air_pocket_active: false,
        behavioral_resonance: 0.0,
        grossman_zhou_floor_distance: 0.50,
        ville_ruin_headroom: 1.0,
    };

    let decision_low = governor.evaluate_governance(&input_low_vol);
    let decision_high = governor.evaluate_governance(&input_high_vol);

    assert_eq!(decision_low.dynamic_spread_ceiling_bps, 8.0);
    assert!(decision_high.dynamic_spread_ceiling_bps > decision_low.dynamic_spread_ceiling_bps);
    assert!(decision_high.dynamic_spread_ceiling_bps <= 35.0);
}

#[test]
fn c4_elastic_entry_granted_on_quantum_conviction() {
    let governor = AdaptiveQuantumVetoGovernor::default();

    // Spread = 14 bps (por encima del base 8 bps, pero por debajo de 35 bps)
    let input_no_quantum = MarketQuantumStateInput {
        current_spread_bps: 14.0,
        continuous_effective_vol: 0.015,
        reference_vol: 0.015,
        quantum_tensor_conviction: 0.20, // Convicción débil
        shockwave_amplitude: 0.0,
        air_pocket_active: false,
        behavioral_resonance: 0.0,
        grossman_zhou_floor_distance: 0.50,
        ville_ruin_headroom: 1.0,
    };

    let input_with_quantum = MarketQuantumStateInput {
        current_spread_bps: 14.0,
        continuous_effective_vol: 0.015,
        reference_vol: 0.015,
        quantum_tensor_conviction: 0.85, // Convicción hiper-fuerte
        shockwave_amplitude: 0.0,
        air_pocket_active: false,
        behavioral_resonance: 0.0,
        grossman_zhou_floor_distance: 0.50,
        ville_ruin_headroom: 1.0,
    };

    let decision_no_q = governor.evaluate_governance(&input_no_quantum);
    let decision_with_q = governor.evaluate_governance(&input_with_quantum);

    // Sin convicción cuántica, la orden se veta suavemente (sizing = 0)
    assert_eq!(decision_no_q.sizing_multiplier, 0.0);
    assert!(!decision_no_q.elastic_entry_granted);

    // Con convicción cuántica, se autoriza entrada elástica con sizing modulado
    assert!(decision_with_q.elastic_entry_granted);
    assert!(decision_with_q.sizing_multiplier > 0.0);
    assert!(decision_with_q.sizing_multiplier <= 0.85);
}

#[test]
fn c5_conformal_risk_calibration_online() {
    let mut governor = AdaptiveQuantumVetoGovernor::new(AdaptiveQuantumVetoParams {
        conformal_alpha: 0.10,
        ..Default::default()
    });

    // Registrar 20 eventos de pérdidas calibradas
    for i in 1..=20 {
        governor.record_closed_loss_normalized(i as f64 * 0.02);
    }

    let input = MarketQuantumStateInput {
        current_spread_bps: 5.0,
        continuous_effective_vol: 0.015,
        reference_vol: 0.015,
        quantum_tensor_conviction: 0.50,
        shockwave_amplitude: 0.0,
        air_pocket_active: false,
        behavioral_resonance: 0.0,
        grossman_zhou_floor_distance: 0.50,
        ville_ruin_headroom: 1.0,
    };

    let decision = governor.evaluate_governance(&input);
    assert!(decision.conformal_risk_score > 0.0);
    assert!(decision.conformal_risk_score <= 1.0);
    assert!(decision.sizing_multiplier > 0.0);
}

#[test]
fn c6_zero_panic_boundary_invariance() {
    let governor = AdaptiveQuantumVetoGovernor::default();

    let pathological = MarketQuantumStateInput {
        current_spread_bps: f64::NAN,
        continuous_effective_vol: f64::INFINITY,
        reference_vol: 0.0,
        quantum_tensor_conviction: f64::NEG_INFINITY,
        shockwave_amplitude: f64::NAN,
        air_pocket_active: true,
        behavioral_resonance: f64::NAN,
        grossman_zhou_floor_distance: -1.0,
        ville_ruin_headroom: -10.0,
    };

    let decision = governor.evaluate_governance(&pathological);
    assert!(decision.dynamic_spread_ceiling_bps.is_finite());
    assert!(decision.sizing_multiplier.is_finite());
    assert!(decision.conformal_risk_score.is_finite());
}
