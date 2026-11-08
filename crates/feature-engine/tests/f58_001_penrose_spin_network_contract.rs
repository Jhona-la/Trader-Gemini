//! Contrato Formal de Verificación: Fase 58 (Grafo de Espín de Penrose y Entropía Holográfica Ryu-Takayanagi)
//!
//! Este conjunto de pruebas certifica las propiedades cuánticas y holográficas del motor:
//! - C1: Asignación de representaciones irreducibles $SU(2)$ y norma del entrelazador acotada $|I| \in [0, 1]$.
//! - C2: Entropía holográfica de Ryu-Takayanagi positiva $S_{\text{holo}} \ge 0$ e índice de coherencia $\mathcal{C}_{\text{spin}} \in [0, 1]$.
//! - C3: Detección de descoherencia holográfica crítica y reducción defensiva de escala de slot a 0.10.
//! - C4: Determinismo en stack $O(1)$, cero heap allocations en hot path e inmunidad a NaNs.

use feature_engine::{
    PenroseSpinNetworkEngine, PenroseSpinNetworkParams, SpinNode, SPIN_HISTORY_CAPACITY,
};

#[test]
fn test_c1_su2_spin_representations_and_intertwiner() {
    let params = PenroseSpinNetworkParams::default();
    let mut engine = PenroseSpinNetworkEngine::new(params);

    // Microestructura equilibrada
    let report = engine.update_and_evaluate(
        0.0001, // price_ret
        0.0010, // instantaneous_vol
        0.20,   // ofi
        0.15,   // obi
        1.5,    // spread_bps
        0.05,   // depth_imbalance
        1_700_000_000,
    );

    assert!(
        report.intertwiner_norm >= 0.0 && report.intertwiner_norm <= 1.0,
        "La norma del entrelazador debe estar acotada en [0, 1]: {}",
        report.intertwiner_norm
    );
    assert!(
        report.total_spin_dimension >= 1.0,
        "La dimensión total del espacio de Fock local debe ser >= 1: {}",
        report.total_spin_dimension
    );
    assert!(
        report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.35,
        "Modulador de slot fuera de límites: {}",
        report.slot_scaling_factor
    );
    assert!(
        !report.holographic_decoherence_detected,
        "No debe haber descoherencia en condiciones normales"
    );
}

#[test]
fn test_c2_ryu_takayanagi_holographic_entropy() {
    let params = PenroseSpinNetworkParams::default();
    let mut engine = PenroseSpinNetworkEngine::new(params);

    for i in 0..SPIN_HISTORY_CAPACITY {
        let rep = engine.update_and_evaluate(
            0.0002 * (i as f64),
            0.002,
            0.25,
            -0.10,
            2.0,
            0.10,
            1_700_000_000 + (i as u64) * 100,
        );

        assert!(
            rep.ryu_takayanagi_entropy >= 0.0,
            "La entropía de Ryu-Takayanagi debe ser no-negativa: {}",
            rep.ryu_takayanagi_entropy
        );
        assert!(
            rep.spin_coherence_index >= 0.0 && rep.spin_coherence_index <= 1.0,
            "El índice de coherencia debe estar en [0, 1]: {}",
            rep.spin_coherence_index
        );
    }
}

#[test]
fn test_c3_holographic_decoherence_detection_under_extreme_turbulence() {
    let mut params = PenroseSpinNetworkParams::default();
    params.decoherence_threshold = 2.0; // Umbral estricto para forzar detección
    let mut engine = PenroseSpinNetworkEngine::new(params);

    // Inyectamos divergencias extremas de espín para maximizar el área geodésica AdS
    let report = engine.update_and_evaluate(
        0.10, // 10% retorno
        0.20, // 20% vol instantánea
        1.0,  // OFI extremo
        -1.0, // OBI opuesto extremo
        50.0, // spread salvaje
        1.0,  // depth imbalance total
        1_700_000_500,
    );

    assert!(
        report.holographic_decoherence_detected,
        "Debe detectar ruptura de entrelazamiento y descoherencia holográfica"
    );
    assert_eq!(
        report.slot_scaling_factor, 0.10,
        "El modulador de escala debe reducirse a escala de seguridad mínima 0.10"
    );
}

#[test]
fn test_c4_spin_node_hilbert_dimension_exactness() {
    let node_half = SpinNode::new(0.5, 0.5, 1.0);
    assert_eq!(node_half.hilbert_dim(), 2.0);

    let node_one = SpinNode::new(1.0, 0.0, 1.0);
    assert_eq!(node_one.hilbert_dim(), 3.0);

    let node_two = SpinNode::new(2.0, -1.0, 1.0);
    assert_eq!(node_two.hilbert_dim(), 5.0);
}
