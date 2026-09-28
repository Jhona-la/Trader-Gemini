//! Adversarial contracts: numerical validity is not statistical independence.
use risk_engine::random_matrix::{largest_eigenvalue, systematic_mode, MppVerdict};

fn equicorrelation(n: usize, rho: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect()
}

#[test]
fn anticorrelation_dominant_mode_is_not_orthogonalized_away() {
    let c = vec![vec![1.0, -0.9], vec![-0.9, 1.0]];
    let lambda = largest_eigenvalue(&c).expect("valid positive definite correlation");
    assert!((lambda - 1.9).abs() < 1e-12, "expected 1.9, got {lambda}");
    assert!(matches!(
        systematic_mode(&c, 512),
        Some(MppVerdict::SystematicMode { .. })
    ));
}

#[test]
fn singular_anticorrelation_has_a_nonzero_dominant_mode() {
    let c = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
    assert!((largest_eigenvalue(&c).unwrap() - 2.0).abs() < 1e-12);
}

#[test]
fn changing_exposure_signs_preserves_the_spectrum() {
    let c = vec![
        vec![1.0, 0.8, 0.3],
        vec![0.8, 1.0, 0.2],
        vec![0.3, 0.2, 1.0],
    ];
    let lambda = largest_eigenvalue(&c).unwrap();
    for mask in 0..8 {
        let signed: Vec<Vec<f64>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        let si = if mask & (1 << i) == 0 { 1.0 } else { -1.0 };
                        let sj = if mask & (1 << j) == 0 { 1.0 } else { -1.0 };
                        si * sj * c[i][j]
                    })
                    .collect()
            })
            .collect();
        assert!((largest_eigenvalue(&signed).unwrap() - lambda).abs() < 1e-10);
    }
}

#[test]
fn indefinite_star_is_not_a_correlation_matrix() {
    let c = vec![
        vec![1.0, 0.9, 0.9],
        vec![0.9, 1.0, 0.0],
        vec![0.9, 0.0, 1.0],
    ];
    assert!(largest_eigenvalue(&c).is_none());
    assert!(systematic_mode(&c, 512).is_none());
}

#[test]
fn asymmetric_input_is_not_silently_accepted() {
    assert!(largest_eigenvalue(&vec![vec![1.0, 0.8], vec![0.2, 1.0]]).is_none());
}

#[test]
fn covariance_without_unit_diagonal_is_not_a_correlation() {
    assert!(largest_eigenvalue(&vec![vec![2.0, 0.0], vec![0.0, 2.0]]).is_none());
}

#[test]
fn out_of_range_correlation_is_rejected() {
    assert!(largest_eigenvalue(&equicorrelation(2, 1.1)).is_none());
}

#[test]
fn non_finite_entries_are_rejected() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(largest_eigenvalue(&equicorrelation(2, invalid)).is_none());
    }
}

#[test]
fn identity_and_rank_one_positive_matrix_are_valid() {
    for n in [2, 3, 8, 30] {
        assert!((largest_eigenvalue(&equicorrelation(n, 0.0)).unwrap() - 1.0).abs() < 1e-12);
        assert!((largest_eigenvalue(&equicorrelation(n, 1.0)).unwrap() - n as f64).abs() < 1e-10);
    }
}

#[test]
fn negative_equicorrelation_has_repeated_dominant_eigenvalues() {
    // Spectrum: 1 + 4*rho = 0.2, and 1-rho = 1.2 with multiplicity four.
    assert!((largest_eigenvalue(&equicorrelation(5, -0.2)).unwrap() - 1.2).abs() < 1e-12);
}

#[test]
fn balanced_long_short_factor_does_not_disappear() {
    let n = 30;
    let mut c = equicorrelation(n, 0.4);
    for i in 0..n {
        for j in 0..n {
            if (i < n / 2) != (j < n / 2) {
                c[i][j] = -c[i][j];
            }
        }
    }
    assert!((largest_eigenvalue(&c).unwrap() - 12.6).abs() < 1e-10);
}

#[test]
fn impossible_negative_equicorrelation_is_rejected() {
    // k + k(k-1)*rho = -3, not a realizable equal-risk portfolio variance.
    assert!(largest_eigenvalue(&equicorrelation(5, -0.4)).is_none());
}

#[test]
fn insufficient_sample_and_ragged_inputs_produce_no_verdict() {
    assert!(systematic_mode(&equicorrelation(3, 0.2), 3).is_none());
    assert!(largest_eigenvalue(&[]).is_none());
    assert!(largest_eigenvalue(&vec![vec![1.0], vec![0.0, 1.0]]).is_none());
}
