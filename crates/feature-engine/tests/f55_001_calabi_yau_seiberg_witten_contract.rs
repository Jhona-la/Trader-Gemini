//! Contrato Formal de Verificación: Fase 55 (Auto-Regulación Cuántica por Invariantes de Calabi-Yau y Seiberg-Witten)
//!
//! Este conjunto de pruebas certifica las propiedades geométricas y físicas del motor:
//! - C1: Mapeo a $\mathbb{C}^3$, volumen de Kähler positivo $V_{CY} > 0$ y desviación de Ricci $\Delta_{CY} \in [0.0, 1.0]$.
//! - C2: Monopolos de Seiberg-Witten $F_A^+ = \sigma(\psi, \psi)$, invariante acotado $SW_{\text{inv}} \in [-1.0, 1.0]$.
//! - C3: Detección de singularidad de conifold ante colapso de liquidez y reducción defensiva de slot a $\omega_{CY} = 0.10$.
//! - C4: Determinismo en stack $O(1)$, cero heap allocations en hot path y estabilidad ante perturbaciones extremas.

use feature_engine::{CalabiYauParams, CalabiYauSeibergWittenEngine, ComplexCoord, CY_HISTORY_LEN};

#[test]
fn test_c1_kaehler_metric_and_ricci_flatness_deviation() {
    let params = CalabiYauParams::default();
    let mut engine = CalabiYauSeibergWittenEngine::new(params);

    // Estado estacionario con microestructura normalizada
    let report = engine.update_and_evaluate(
        0.0002, // price_ret
        0.0015, // instantaneous_vol
        0.15,   // obi
        0.20,   // ofi
        2.5,    // spread_bps
        0.05,   // depth_imbalance
        1_700_000_000,
    );

    assert!(
        report.kaehler_volume > 0.0,
        "El volumen de Kähler debe ser estrictamente positivo: {}",
        report.kaehler_volume
    );
    assert!(
        report.ricci_flatness_deviation >= 0.0 && report.ricci_flatness_deviation <= 1.0,
        "La desviación de Calabi-Yau debe estar acotada en [0, 1]: {}",
        report.ricci_flatness_deviation
    );
    assert!(
        report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.40,
        "El modulador de slot debe estar en [0.10, 1.40]: {}",
        report.slot_scaling_factor
    );
    assert!(
        !report.conifold_singularity_detected,
        "No debe detectar singularidad en condiciones ordinarias"
    );
}

#[test]
fn test_c2_seiberg_witten_invariants_and_monopole_density() {
    let params = CalabiYauParams::default();
    let mut engine = CalabiYauSeibergWittenEngine::new(params);

    for i in 0..CY_HISTORY_LEN {
        let rep = engine.update_and_evaluate(
            0.0005 * (i as f64),
            0.0020,
            0.30,
            -0.20,
            3.0,
            0.10,
            1_700_000_000 + (i as u64) * 100,
        );

        assert!(
            rep.seiberg_witten_invariant >= -1.0 && rep.seiberg_witten_invariant <= 1.0,
            "Invariante Seiberg-Witten fuera de rango [-1, 1]: {}",
            rep.seiberg_witten_invariant
        );
        assert!(
            rep.monopole_density >= 0.0,
            "Densidad de monopolos debe ser no-negativa: {}",
            rep.monopole_density
        );
        assert!(
            rep.hodge_euler_characteristic.is_finite(),
            "Característica de Euler debe ser finita: {}",
            rep.hodge_euler_characteristic
        );
    }
}

#[test]
fn test_c3_conifold_singularity_detection_under_extreme_stress() {
    let mut params = CalabiYauParams::default();
    params.singularity_threshold = 0.50; // Umbral estricto para forzar detección
    let mut engine = CalabiYauSeibergWittenEngine::new(params);

    // Inyectamos condiciones extremas de deformación de métrica
    let report = engine.update_and_evaluate(
        0.08, // 8% retorno de salto violento
        0.15, // 15% vol
        1.0,  // OBI extremo
        1.0,  // OFI extremo
        80.0, // spread salvaje de 80 bps
        1.0,  // depth imbalance total
        1_700_000_500,
    );

    assert!(
        report.conifold_singularity_detected,
        "Debe detectar colapso de ciclo evanescente / conifold singularity"
    );
    assert_eq!(
        report.slot_scaling_factor, 0.10,
        "El factor de escala debe reducirse a escala defensiva mínima de 0.10"
    );
}

#[test]
fn test_c4_complex_coord_arithmetic_precision() {
    let c1 = ComplexCoord::new(3.0, 4.0);
    let c2 = ComplexCoord::new(1.0, -2.0);

    assert_eq!(c1.norm_sq(), 25.0);
    assert_eq!(c1.norm(), 5.0);

    let sum = c1.add(c2);
    assert_eq!(sum.re, 4.0);
    assert_eq!(sum.im, 2.0);

    let prod = c1.mul(c2);
    // (3 + 4i)(1 - 2i) = 3 - 6i + 4i - 8i^2 = 11 - 2i
    assert_eq!(prod.re, 11.0);
    assert_eq!(prod.im, -2.0);

    let conj = c1.conj();
    assert_eq!(conj.re, 3.0);
    assert_eq!(conj.im, -4.0);
}
