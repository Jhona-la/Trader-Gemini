//! Contract test for F1-085 (Ola Ω189):
//!
//! F1-085: Freeze `global_max_drawdown` (slot vec[0]) in `SuperGenotype` to eliminate
//! evolutionary noise and align Simulated Annealing with production account circuit breakers.
//!
//! In live production, account drawdown is strictly governed by `freno_duro` (CL-106 / ADR-0017)
//! under hypothesis testing H0 = +0.05 R with 95% confidence and bounded by theoretical
//! maximum d*(c, α) = 0.632 (ADR-0016 / D3). The gene `global_max_drawdown` had zero live
//! callers altering trades, yet mutated randomly across [0.50, 0.99] and was distorted in
//! `evolution.rs` to [0.05, 0.30].
//!
//! This contract verifies:
//! 1. `global_max_drawdown` is strictly frozen across mutations (preserves parent value bit-for-bit).
//! 2. `SuperGenotype::new_random()` initializes `global_max_drawdown` to 0.96 (canonical neutral default).
//! 3. Vector slot 0, bounds [0.50, 0.99], and `SuperGenotype::DIMENSION == 144` are preserved with 100% fidelity.

use quantum_arena::genome::SuperGenotype;

#[test]
fn test_f1_085_global_max_drawdown_remains_frozen_across_mutations() {
    let test_drawdowns = [0.52, 0.632, 0.75, 0.88, 0.96];

    for &dd in &test_drawdowns {
        let mut parent = SuperGenotype::default();
        parent.global_max_drawdown = dd;

        // Rate = 0.0 must preserve exact bit representation
        let child_zero = parent.mutate_cmaes(0.0);
        assert_eq!(
            child_zero.global_max_drawdown.to_bits(),
            dd.to_bits(),
            "global_max_drawdown must not be altered with rate 0.0"
        );

        // High mutation rate across multiple generations and seeds must remain strictly frozen
        for seed in 0..50u64 {
            let child = parent.mutate_cmaes_seeded(0.85, seed);
            assert_eq!(
                child.global_max_drawdown.to_bits(),
                dd.to_bits(),
                "global_max_drawdown must remain frozen across mutations (seed {seed})"
            );
        }
    }
}

#[test]
fn test_f1_085_new_random_initializes_canonical_neutral_default() {
    let expected_dd = 0.96_f64.to_bits();
    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();

    assert_eq!(lo[0], 0.50);
    assert_eq!(hi[0], 0.99);

    for _ in 0..30 {
        let g = SuperGenotype::new_random();
        assert_eq!(
            g.global_max_drawdown.to_bits(),
            expected_dd,
            "new_random must set canonical neutral global_max_drawdown (0.96)"
        );
        assert!(
            g.global_max_drawdown >= lo[0] && g.global_max_drawdown <= hi[0],
            "global_max_drawdown must fall strictly within [0.50, 0.99]"
        );
    }
}

#[test]
fn test_f1_085_layout_dimension_and_vector_slot_0_preserved() {
    let mut g = SuperGenotype::default();
    g.global_max_drawdown = 0.725;

    let vec = g.to_vector();
    assert_eq!(vec.len(), SuperGenotype::DIMENSION);
    assert_eq!(SuperGenotype::DIMENSION, 144);
    assert_eq!(vec[0], 0.725, "Slot 0 must hold global_max_drawdown");

    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    assert_eq!(lo.len(), 144);
    assert_eq!(hi.len(), 144);
    assert_eq!(lo[0], 0.50);
    assert_eq!(hi[0], 0.99);

    let reconstructed = SuperGenotype::try_from_vector(&vec).expect("valid vector deserialization");
    assert_eq!(reconstructed.global_max_drawdown, 0.725);
}
