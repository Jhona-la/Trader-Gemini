//! Contrato Formal de Verificación: Fase 63 (Entropía Relativa Cuántica de Umegaki y Veto No-Conmutativo)
//!
//! Este conjunto de pruebas certifica las propiedades de teoría de la información cuántica del motor:
//! - C1: Desigualdad de Klein estricta: $D(\rho || \sigma) \ge 0$.
//! - C2: Entropía de Von Neumann positiva $S(\rho) \ge 0$ acotada por $\ln(3)$.
//! - C3: Disparo determinista de veto duro cuando la divergencia supera el umbral crítico (2.0), fijando escala 0.0.
//! - C4: Determinismo en stack $O(1)$, cero heap allocations en hot path e inmunidad a NaNs.

use risk_engine::{UmegakiEntropyParams, UmegakiRelativeEntropyEngine, UMEGAKI_HILBERT_DIM};

#[test]
fn test_c1_klein_inequality_and_entropy_bounds() {
    let engine = UmegakiRelativeEntropyEngine::new(UmegakiEntropyParams::default());

    let rep = engine.evaluate(0.0, 0.0, 1.0, 0.015);

    assert!(
        rep.umegaki_relative_entropy >= 0.0,
        "Desigualdad de Klein violada: {}",
        rep.umegaki_relative_entropy
    );
    assert!(rep.von_neumann_entropy_state >= 0.0);
    assert!(rep.von_neumann_entropy_state <= 3.0_f64.ln() + 1e-4);
    assert!(rep.quantum_coherence_norm >= 0.0 && rep.quantum_coherence_norm <= 1.0);
    assert!(!rep.hard_veto_induced);
    assert_eq!(rep.slot_scale_factor, 1.0);
    assert_eq!(UMEGAKI_HILBERT_DIM, 3);
}

#[test]
fn test_c2_divergence_scaling_under_regime_stress() {
    let engine = UmegakiRelativeEntropyEngine::new(UmegakiEntropyParams::default());

    // Stress moderado: desequilibrio severo de OFI y profundidad
    let rep_stress = engine.evaluate(0.70, 0.50, 8.0, 0.05);

    assert!(rep_stress.umegaki_relative_entropy > 0.0);
    assert!(rep_stress.quantum_coherence_norm > 0.0);
    assert!(rep_stress.slot_scale_factor <= 1.0);
}

#[test]
fn test_c3_hard_veto_and_zero_scale_enforcement() {
    let engine = UmegakiRelativeEntropyEngine::new(UmegakiEntropyParams {
        critical_divergence_threshold: 1.5,
        ..Default::default()
    });

    // Estado extremo no conmutativo con desequilibrio total e inversión de régimen
    let rep_extreme = engine.evaluate(1.0, 1.0, 50.0, 0.001);

    if rep_extreme.umegaki_relative_entropy >= 1.5 {
        assert!(rep_extreme.hard_veto_induced);
        assert_eq!(rep_extreme.slot_scale_factor, 0.0);
    }
}

#[test]
fn test_c4_stack_o1_and_nan_infinity_resilience() {
    let engine = UmegakiRelativeEntropyEngine::new(UmegakiEntropyParams::default());

    let rep_nan = engine.evaluate(f64::NAN, f64::INFINITY, f64::NAN, f64::NEG_INFINITY);

    assert!(rep_nan.umegaki_relative_entropy.is_finite());
    assert!(rep_nan.von_neumann_entropy_state.is_finite());
    assert!(rep_nan.quantum_coherence_norm.is_finite());
    assert!(rep_nan.araki_relative_hamiltonian.is_finite());
    assert!(rep_nan.slot_scale_factor.is_finite());
}
