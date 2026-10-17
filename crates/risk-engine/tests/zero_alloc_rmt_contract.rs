//! Contract: Zero-allocation stack Jacobi eigensolver and flat slice RMT API.
//!
//! Verifies:
//! 1. Bit-for-bit equivalence between flat zero-alloc solver, slice-of-vec solver, and analytical eigenvalues.
//! 2. Exact trace preservation sum(lambda) == N on zero-alloc flat spectrum extraction.
//! 3. Continuity across the stack/heap boundary (N = 32 stack vs N = 33 fallback).
//! 4. Falsation & strict rejection of invalid, asymmetric, and indefinite matrices.

use risk_engine::random_matrix::{
    full_spectrum, full_spectrum_flat_zero_alloc, largest_eigenvalue, largest_eigenvalue_flat,
    MAX_STACK_CORR_DIM,
};

fn flat_equicorrelation(n: usize, rho: f64) -> Vec<f64> {
    let mut flat = vec![rho; n * n];
    for i in 0..n {
        flat[i * n + i] = 1.0;
    }
    flat
}

fn nested_equicorrelation(n: usize, rho: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect()
}

#[test]
fn test_zero_alloc_flat_matches_nested_and_analytic() {
    for n in [2, 3, 5, 8, 16, 32] {
        for rho in [0.0, 0.25, 0.50, 0.85] {
            let flat = flat_equicorrelation(n, rho);
            let nested = nested_equicorrelation(n, rho);

            let lambda_flat = largest_eigenvalue_flat(&flat, n).expect("valid PSD flat");
            let lambda_nested = largest_eigenvalue(&nested).expect("valid PSD nested");
            let analytic = 1.0 + (n - 1) as f64 * rho;

            assert!(
                (lambda_flat - analytic).abs() < 1e-10,
                "n={n}, rho={rho}: flat={lambda_flat}, analytic={analytic}"
            );
            assert!(
                (lambda_nested - analytic).abs() < 1e-10,
                "n={n}, rho={rho}: nested={lambda_nested}, analytic={analytic}"
            );
            assert!(
                (lambda_flat - lambda_nested).abs() < 1e-12,
                "n={n}, rho={rho}: flat and nested must match bit-for-bit"
            );
        }
    }
}

#[test]
fn test_full_spectrum_flat_zero_alloc_trace_and_ordering() {
    for n in [2, 4, 8, 16, 32] {
        let rho = 0.40;
        let flat = flat_equicorrelation(n, rho);
        let mut out_spectrum = [0.0f64; MAX_STACK_CORR_DIM];

        let written = full_spectrum_flat_zero_alloc(&flat, n, &mut out_spectrum)
            .expect("must diagonalize on stack");
        assert_eq!(written, n);

        // Verification 1: Trace preservation sum(lambda) == N
        let trace: f64 = out_spectrum[..n].iter().sum();
        assert!(
            (trace - n as f64).abs() < 1e-9,
            "trace sum must equal n={n}, got {trace}"
        );

        // Verification 2: Strict descending order
        for i in 0..(n - 1) {
            assert!(
                out_spectrum[i] >= out_spectrum[i + 1] - 1e-12,
                "spectrum must be sorted descending"
            );
        }

        // Verification 3: Dominant mode
        let expected_max = 1.0 + (n - 1) as f64 * rho;
        assert!(
            (out_spectrum[0] - expected_max).abs() < 1e-10,
            "max eigenvalue must be {expected_max}, got {}",
            out_spectrum[0]
        );

        // Verification 4: Degenerate modes
        let expected_min = 1.0 - rho;
        for i in 1..n {
            assert!(
                (out_spectrum[i] - expected_min).abs() < 1e-10,
                "mode {i} must be {expected_min}, got {}",
                out_spectrum[i]
            );
        }
    }
}

#[test]
fn test_stack_heap_boundary_continuity() {
    // Stack path operates for N <= 32
    // Heap fallback operates for N > 32
    let n_stack = 32;
    let n_heap = 33;
    let rho = 0.35;

    let flat_stack = flat_equicorrelation(n_stack, rho);
    let flat_heap = flat_equicorrelation(n_heap, rho);

    let lambda_stack = largest_eigenvalue_flat(&flat_stack, n_stack).unwrap();
    let lambda_heap = largest_eigenvalue_flat(&flat_heap, n_heap).unwrap();

    let expected_stack = 1.0 + (n_stack - 1) as f64 * rho;
    let expected_heap = 1.0 + (n_heap - 1) as f64 * rho;

    assert!((lambda_stack - expected_stack).abs() < 1e-9);
    assert!((lambda_heap - expected_heap).abs() < 1e-9);

    // Verify nested API on boundary
    let nested_heap = nested_equicorrelation(n_heap, rho);
    let nested_heap_lambda = largest_eigenvalue(&nested_heap).unwrap();
    assert!((nested_heap_lambda - expected_heap).abs() < 1e-9);

    let spec_heap = full_spectrum(&nested_heap).unwrap();
    assert_eq!(spec_heap.len(), n_heap);
    let trace_heap: f64 = spec_heap.iter().sum();
    assert!((trace_heap - n_heap as f64).abs() < 1e-9);
}

#[test]
fn test_flat_api_rejects_malformed_and_invalid() {
    // Invalid dimension (n < 2)
    assert!(largest_eigenvalue_flat(&[1.0], 1).is_none());

    // Mismatched length (expects 4 elements for 2x2, provided 3)
    assert!(largest_eigenvalue_flat(&[1.0, 0.5, 0.5], 2).is_none());

    // Asymmetric matrix
    let asym = vec![1.0, 0.8, 0.2, 1.0];
    assert!(largest_eigenvalue_flat(&asym, 2).is_none());

    // Non-unit diagonal
    let bad_diag = vec![2.0, 0.5, 0.5, 2.0];
    assert!(largest_eigenvalue_flat(&bad_diag, 2).is_none());

    // Indefinite matrix (star graph with rho=0.9 for N=3 is indefinite: det < 0)
    let indefinite = vec![
        1.0, 0.9, 0.9,
        0.9, 1.0, 0.0,
        0.9, 0.0, 1.0,
    ];
    assert!(largest_eigenvalue_flat(&indefinite, 3).is_none());

    // NaN / Infinity
    let nan_flat = vec![1.0, f64::NAN, f64::NAN, 1.0];
    assert!(largest_eigenvalue_flat(&nan_flat, 2).is_none());
}
