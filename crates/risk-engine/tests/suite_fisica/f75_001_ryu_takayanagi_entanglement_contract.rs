//! # Contrato Formal F75-001: Entropía de Ryu-Takayanagi y Superficies Extremales de Entrelazamiento Holográfico
//!
//! Valida la fórmula geométrica de Ryu-Takayanagi en $AdS_3$, subaditividad de la entropía $S_{A \cup B} \le S_A + S_B$,
//! no negatividad de la información mutua $I(A:B) \ge 0$, detección de desconexión de superficies extremales y resiliencia `rb_p1`.

use risk_engine::ryu_takayanagi_entanglement::{
    RyuTakayanagiEntanglementEngine, RyuTakayanagiParams,
};

#[test]
fn test_c1_ryu_takayanagi_entropy_and_mutual_information() {
    let params = RyuTakayanagiParams::default();
    let mut engine = RyuTakayanagiEntanglementEngine::new(params);

    let rep = engine.update(0.02, 0.02, 0.85, 1.2);

    // Entropías individuales de Ryu-Takayanagi no negativas
    assert!(
        rep.entanglement_entropy_sa >= 0.0,
        "La entropía de Ryu-Takayanagi S_A debe ser no negativa"
    );
    assert!(
        rep.entanglement_entropy_sb >= 0.0,
        "La entropía de Ryu-Takayanagi S_B debe ser no negativa"
    );

    // Teorema de Subaditividad Fuerte: S_{A \cup B} <= S_A + S_B
    assert!(
        rep.entanglement_entropy_union <= rep.entanglement_entropy_sa + rep.entanglement_entropy_sb + 1e-10,
        "Debe cumplirse la subaditividad de la entropía cuántica de Ryu-Takayanagi"
    );

    // Información Mutua Holográfica no negativa: I(A:B) >= 0
    assert!(
        rep.holographic_mutual_information >= 0.0,
        "La información mutua holográfica debe ser no negativa (obtenido: {})",
        rep.holographic_mutual_information
    );
}

#[test]
fn test_c2_calm_entangled_phase_and_constructive_slot() {
    let params = RyuTakayanagiParams::default();
    let mut engine = RyuTakayanagiEntanglementEngine::new(params);

    // Fase conexa fuertemente entrelazada: alta correlación cruzada (0.95), spread mínimo (0.5 bps), baja dispersión (0.01)
    let rep = engine.update(0.01, 0.01, 0.95, 0.5);

    assert!(
        rep.extremal_surface_connected,
        "La superficie extremal de Ryu-Takayanagi debe permanecer conexa en régimen de alta coherencia"
    );
    assert!(
        !rep.hard_veto_induced,
        "No debe haber veto duro de Ryu-Takayanagi en fase conexa entrelazada"
    );
    assert!(
        rep.holographic_mutual_information >= params.min_mutual_information_threshold,
        "La información mutua holográfica debe superar el umbral crítico"
    );
    assert!(
        rep.slot_scale_factor >= 0.85,
        "El modulador de slot debe ser constructivo (obtenido: {})",
        rep.slot_scale_factor
    );
}

#[test]
fn test_c3_entanglement_disconnection_triggers_hard_veto() {
    let params = RyuTakayanagiParams::default();
    let mut engine = RyuTakayanagiEntanglementEngine::new(params);

    // Transición de desconexión topológica (Hawking-Page): descorrelación total (0.0) y spread severo (30.0 bps)
    let rep = engine.update(0.05, 0.05, 0.0, 30.0);

    assert!(
        rep.hard_veto_induced,
        "Debe inducirse RYU_TAKAYANAGI_ENTANGLEMENT_VETO ante fragmentación de la superficie extremal"
    );
    assert_eq!(
        rep.slot_scale_factor, 0.0,
        "El factor de escala de slot debe colapsar rígidamente a 0.0 bajo veto de Ryu-Takayanagi"
    );
}

#[test]
fn test_c4_zero_panic_rb_p1_resilience() {
    let params = RyuTakayanagiParams::default();
    let mut engine = RyuTakayanagiEntanglementEngine::new(params);

    let pathological_inputs = [
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::INFINITY),
        (1e-308, -1e-308, 0.0, -20.0),
        (-0.5, 1.0, 500.0, 100.0),
    ];

    for (disp_a, disp_b, corr, spread) in pathological_inputs {
        let rep = engine.update(disp_a, disp_b, corr, spread);
        assert!(rep.entanglement_entropy_sa.is_finite());
        assert!(rep.entanglement_entropy_sb.is_finite());
        assert!(rep.entanglement_entropy_union.is_finite());
        assert!(rep.holographic_mutual_information.is_finite());
        assert!(rep.slot_scale_factor.is_finite());
    }
}
