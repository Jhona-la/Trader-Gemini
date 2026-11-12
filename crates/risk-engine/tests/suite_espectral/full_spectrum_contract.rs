//! A spectrum diagnostic must not weaken the correlation input contract.
use risk_engine::random_matrix::{effective_bets, full_spectrum, largest_eigenvalue};

fn equicorrelation(n: usize, rho: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect()
}

#[test]
fn covariance_scale_is_rejected_by_every_spectral_entrypoint() {
    let c = vec![
        vec![4.0, 0.0, 0.0],
        vec![0.0, 4.0, 0.0],
        vec![0.0, 0.0, 4.0],
    ];
    assert!(largest_eigenvalue(&c).is_none());
    assert!(full_spectrum(&c).is_none(), "4I is not a correlation");
    assert!(effective_bets(&c, 512).is_none());
}

#[test]
fn material_asymmetry_is_not_averaged_into_a_different_matrix() {
    for c in [
        vec![vec![1.0, 2.0], vec![-2.0, 1.0]],
        vec![vec![1.0, 0.8], vec![0.2, 1.0]],
    ] {
        assert!(largest_eigenvalue(&c).is_none());
        assert!(
            full_spectrum(&c).is_none(),
            "invalid input must not be symmetrized"
        );
        assert!(effective_bets(&c, 512).is_none());
    }
}

#[test]
fn malformed_nonfinite_and_out_of_range_inputs_are_rejected() {
    let mut cases = vec![
        vec![],
        vec![vec![1.0]],
        vec![vec![1.0], vec![0.0, 1.0]],
        equicorrelation(2, 1.1),
    ];
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cases.push(equicorrelation(2, x));
        cases.push(vec![vec![x, 0.0], vec![0.0, 1.0]]);
    }
    for c in cases {
        assert!(full_spectrum(&c).is_none(), "{c:?}");
        assert!(effective_bets(&c, 512).is_none());
    }
}

#[test]
fn indefinite_matrices_do_not_acquire_diversification_evidence() {
    let star = vec![
        vec![1.0, 0.9, 0.9],
        vec![0.9, 1.0, 0.0],
        vec![0.9, 0.0, 1.0],
    ];
    for c in [star, equicorrelation(5, -0.4)] {
        assert!(full_spectrum(&c).is_none());
        assert!(effective_bets(&c, 512).is_none());
    }
}

#[test]
fn complete_spectrum_preserves_analytic_eigenvalues_and_trace() {
    for n in [2, 5, 30] {
        for rho in [0.0, -0.5 / (n - 1) as f64, 0.4, 1.0] {
            let c = equicorrelation(n, rho);
            let eig = full_spectrum(&c).unwrap();
            let mut expected = vec![1.0 - rho; n];
            expected[0] = 1.0 + (n - 1) as f64 * rho;
            expected.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(eig.len(), n);
            assert!((eig.iter().sum::<f64>() - n as f64).abs() < 1e-10);
            for (actual, target) in eig.iter().zip(expected) {
                assert!((actual - target).abs() < 1e-10, "{eig:?}");
            }
            assert!((eig[0] - largest_eigenvalue(&c).unwrap()).abs() < 1e-12);
        }
    }
}

#[test]
fn permutation_and_position_signs_do_not_change_spectrum() {
    let c = vec![
        vec![1.0, 0.8, 0.3],
        vec![0.8, 1.0, 0.2],
        vec![0.3, 0.2, 1.0],
    ];
    let reference = full_spectrum(&c).unwrap();
    for permutation in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
        for mask in 0..8 {
            let signed: Vec<Vec<f64>> = (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| {
                            let sign_i = if mask & (1 << i) == 0 { 1.0 } else { -1.0 };
                            let sign_j = if mask & (1 << j) == 0 { 1.0 } else { -1.0 };
                            sign_i * sign_j * c[permutation[i]][permutation[j]]
                        })
                        .collect()
                })
                .collect();
            let actual = full_spectrum(&signed).unwrap();
            for (a, b) in actual.iter().zip(&reference) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn retained_mode_participation_is_not_total_portfolio_dimension() {
    assert!(effective_bets(&equicorrelation(6, 0.0), 512).is_none());
    assert!((effective_bets(&equicorrelation(6, 1.0), 512).unwrap() - 1.0).abs() < 1e-10);
    let blocks: Vec<Vec<f64>> = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| if i / 3 == j / 3 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    assert!((effective_bets(&blocks, 512).unwrap() - 2.0).abs() < 1e-10);
    assert!(effective_bets(&blocks, 6).is_none());
}
