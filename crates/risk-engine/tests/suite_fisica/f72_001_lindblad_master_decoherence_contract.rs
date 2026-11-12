//! Test de contrato F72-001: Ecuación de Master de Lindblad y Decoherencia Cuántica en Tríada de Activos.
//!
//! Verifica las invariantes fundamentales de la Fase 72:
//! 1. Preservación de propiedades cuánticas de densidad: pureza $\mathcal{P} \in [0.333, 1.0]$ y entropía no negativa.
//! 2. Dinámica disipativa continua bajo calma con pureza alta y escala constructiva.
//! 3. Inducción de veto duro de decoherencia (`LINDBLAD_DECOHERENCE_VETO`) ante ruptura de coherencia y colapso de slot a 0.0.
//! 4. Invariante `rb_p1`: cero pánicos ante valores patológicos (NaN, Inf, extremos).

use risk_engine::lindblad_master_decoherence::{
    LindbladDecoherenceReport, LindbladMasterDecoherenceEngine, LindbladMasterParams,
};

#[test]
fn test_c1_lindblad_density_matrix_purity_bounded() {
    let params = LindbladMasterParams::default();
    let mut engine = LindbladMasterDecoherenceEngine::new(params);

    for i in 0..20 {
        let btc = 0.001 * (i as f64);
        let eth = 0.0008 * (i as f64);
        let sol = 0.0012 * (i as f64);
        let rep = engine.update(btc, eth, sol, 0.01, 1.2);

        assert!(rep.purity >= 0.3333333333 && rep.purity <= 1.000000001, "Pureza debe estar en [1/3, 1]");
        assert!(rep.von_neumann_entropy >= 0.0, "Entropía de Von Neumann debe ser no negativa");
        assert!(rep.l1_quantum_coherence >= 0.0, "Coherencia cuántica l1 debe ser no negativa");
    }
}

#[test]
fn test_c2_calm_coherence_constructive_slot_scaling() {
    let params = LindbladMasterParams::default();
    let mut engine = LindbladMasterDecoherenceEngine::new(params);

    // Tríada altamente correlacionada en calma: bajo spread (0.5 bps), baja dispersión (0.002)
    let rep = engine.update(0.0002, 0.0002, 0.0002, 0.002, 0.5);

    assert!(!rep.hard_veto_induced, "No debe haber veto duro en régimen de alta coherencia");
    assert!(rep.purity > 0.80, "Pureza debe mantenerse alta en régimen coherente");
    assert!(rep.slot_scale_factor >= 0.80, "Slot scale factor debe ser constructivo");
}

#[test]
fn test_c3_severe_decoherence_triggers_hard_veto() {
    let params = LindbladMasterParams::default();
    let mut engine = LindbladMasterDecoherenceEngine::new(params);

    // Choque disipativo violento: spread enorme (40.0 bps) y dispersión descorrelacionada extrema (0.80)
    let mut rep = LindbladDecoherenceReport::default();
    for _ in 0..15 {
        rep = engine.update(0.10, -0.15, 0.25, 0.80, 40.0);
    }

    assert!(rep.hard_veto_induced, "Debe inducirse LINDBLAD_DECOHERENCE_VETO ante dispersión extrema");
    assert_eq!(rep.slot_scale_factor, 0.0, "El factor de slot debe colapsar rígidamente a 0.0 bajo veto duro");
}

#[test]
fn test_c4_zero_panic_rb_p1_resilience() {
    let params = LindbladMasterParams::default();
    let mut engine = LindbladMasterDecoherenceEngine::new(params);

    let pathological = [
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0, -5.0),
        (1e20, -1e20, 1e20, 1e20, 1e20),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ];

    for &(btc, eth, sol, disp, spread) in &pathological {
        let rep = engine.update(btc, eth, sol, disp, spread);
        assert!(rep.purity.is_finite());
        assert!(rep.von_neumann_entropy.is_finite());
        assert!(rep.decoherence_rate.is_finite());
        assert!(rep.slot_scale_factor.is_finite());
    }
}
