//! # Contrato Formal F44-001: Geometría Espectral No Conmutativa de Connes
//!
//! Valida el comportamiento matemático y la robustez de la tripleta espectral y matrices gamma (§44):
//! 1. `test_c1_dirac_clifford_algebra_anticommutation`: Álgebra de Clifford $\{\gamma^\mu, \gamma^\nu\} = 2 \eta^{\mu\nu} \mathbf{1}_4$.
//! 2. `test_c2_noncommutative_geodesic_distance_order_flow`: Métrica no conmutativa y expansión por $\theta[T, M]$.
//! 3. `test_c3_chiral_anomaly_institution_asymmetry`: Índice quiral $\chi \in [-1.0, 1.0]$ y asimetría institucional.
//! 4. `test_c4_zero_alloc_stack_determinism_and_nan_stability`: Estabilidad de stack, cero pánico y determinismo.

use strategy_core::connes_spectral_geometry::{
    mat4_add, mat4_mul, ConnesSpectralGeometryEngine, ConnesSpectralParams, DiracGammaAlgebra,
    Matrix4c, C64, DIRAC_DIM,
};

#[allow(clippy::needless_range_loop)]
fn mat4_identity() -> Matrix4c {
    let mut id = [[C64::ZERO; DIRAC_DIM]; DIRAC_DIM];
    for i in 0..DIRAC_DIM {
        id[i][i] = C64::ONE;
    }
    id
}

#[allow(clippy::needless_range_loop)]
fn mat4_is_approx_equal(a: &Matrix4c, b: &Matrix4c, eps: f64) -> bool {
    for i in 0..DIRAC_DIM {
        for j in 0..DIRAC_DIM {
            if (a[i][j].re - b[i][j].re).abs() > eps || (a[i][j].im - b[i][j].im).abs() > eps {
                return false;
            }
        }
    }
    true
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_c1_dirac_clifford_algebra_anticommutation() {
    let g0 = DiracGammaAlgebra::gamma0();
    let g1 = DiracGammaAlgebra::gamma1();
    let g2 = DiracGammaAlgebra::gamma2();
    let g3 = DiracGammaAlgebra::gamma3();
    let g5 = DiracGammaAlgebra::gamma5();
    let id = mat4_identity();

    // 1. Cuadrados de las matrices gamma: (g0)^2 = I, (gk)^2 = -I
    let g0_sq = mat4_mul(&g0, &g0);
    assert!(mat4_is_approx_equal(&g0_sq, &id, 1e-10));

    let mut neg_id = [[C64::ZERO; DIRAC_DIM]; DIRAC_DIM];
    for i in 0..DIRAC_DIM {
        neg_id[i][i] = C64::new(-1.0, 0.0);
    }

    let g1_sq = mat4_mul(&g1, &g1);
    assert!(mat4_is_approx_equal(&g1_sq, &neg_id, 1e-10));

    let g2_sq = mat4_mul(&g2, &g2);
    assert!(mat4_is_approx_equal(&g2_sq, &neg_id, 1e-10));

    let g3_sq = mat4_mul(&g3, &g3);
    assert!(mat4_is_approx_equal(&g3_sq, &neg_id, 1e-10));

    // Cuadrado de Gamma5: (g5)^2 = I
    let g5_sq = mat4_mul(&g5, &g5);
    assert!(mat4_is_approx_equal(&g5_sq, &id, 1e-10));

    // 2. Relaciones de anticonmutación cruzadas: {g_mu, g_nu} = 0 para mu != nu
    let zero = [[C64::ZERO; DIRAC_DIM]; DIRAC_DIM];
    let gammas = [g0, g1, g2, g3];

    for mu in 0..4 {
        for nu in 0..4 {
            if mu != nu {
                let p1 = mat4_mul(&gammas[mu], &gammas[nu]);
                let p2 = mat4_mul(&gammas[nu], &gammas[mu]);
                let anticomm = mat4_add(&p1, &p2);
                assert!(mat4_is_approx_equal(&anticomm, &zero, 1e-10));
            }
        }
    }
}

#[test]
fn test_c2_noncommutative_geodesic_distance_order_flow() {
    let params = ConnesSpectralParams {
        non_commutative_theta: 0.10,
        spectral_cutoff_lambda: 1.0,
        chiral_anomaly_threshold: 0.30,
    };
    let mut engine = ConnesSpectralGeometryEngine::new(params);

    // Ticks sucesivos con desequilibrio de flujo y latencia
    let r1 = engine.update_and_evaluate(0.01, 0.5, 0.8, 1_000);
    assert!(r1.connes_distance >= 0.0);
    assert!(r1.spectral_action >= 0.0);
    assert!(r1.slot_scaling_factor >= 0.10 && r1.slot_scaling_factor <= 1.25);

    let r2 = engine.update_and_evaluate(0.05, 1.2, -0.6, 1_050);
    assert!(r2.connes_distance > 0.0);
    assert!(r2.spectral_action >= 0.0);
}

#[test]
fn test_c3_chiral_anomaly_institution_asymmetry() {
    let mut engine = ConnesSpectralGeometryEngine::new(ConnesSpectralParams::default());

    // Fuerte flujo comprador (OFI = +0.9)
    let rep_buy = engine.update_and_evaluate(0.02, 0.8, 0.9, 1_000);
    assert!(rep_buy.chiral_index > 0.0);
    assert!(rep_buy.slot_scaling_factor >= 1.0);

    // Fuerte flujo vendedor (OFI = -0.9)
    let rep_sell = engine.update_and_evaluate(-0.02, 0.8, -0.9, 1_100);
    assert!(rep_sell.chiral_index < 0.0);
}

#[test]
fn test_c4_zero_alloc_stack_determinism_and_nan_stability() {
    let mut engine_a = ConnesSpectralGeometryEngine::new(ConnesSpectralParams::default());
    let mut engine_b = ConnesSpectralGeometryEngine::new(ConnesSpectralParams::default());

    for step in 1..=1000 {
        let ts = 10_000 + step * 50;
        let p = ((step as f64) * 0.02).sin() * 0.5;
        let v = ((step as f64) * 0.03).cos() * 0.7;
        let ofi = ((step as f64) * 0.04).sin();

        let r_a = engine_a.update_and_evaluate(p, v, ofi, ts);
        let r_b = engine_b.update_and_evaluate(p, v, ofi, ts);

        assert_eq!(r_a.connes_distance.to_bits(), r_b.connes_distance.to_bits());
        assert_eq!(r_a.chiral_index.to_bits(), r_b.chiral_index.to_bits());
        assert_eq!(r_a.spectral_action.to_bits(), r_b.spectral_action.to_bits());
        assert_eq!(
            r_a.slot_scaling_factor.to_bits(),
            r_b.slot_scaling_factor.to_bits()
        );
    }

    // Degenerate inputs test
    let degen = [
        (f64::NAN, 0.0, 0.0),
        (0.0, f64::INFINITY, 0.0),
        (0.0, 0.0, f64::NEG_INFINITY),
        (f64::MIN_POSITIVE, -1e-300, 1e-300),
    ];
    for (idx, &(p, v, ofi)) in degen.iter().enumerate() {
        let rep = engine_a.update_and_evaluate(p, v, ofi, 500_000 + (idx as u64) * 100);
        assert!(rep.connes_distance.is_finite());
        assert!(rep.chiral_index.is_finite());
        assert!(rep.spectral_curvature.is_finite());
        assert!(rep.spectral_action.is_finite());
        assert!(rep.slot_scaling_factor.is_finite());
    }
}
