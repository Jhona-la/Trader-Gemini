//! Contract test for F1-084 (Ola Ω188):
//!
//! F1-084: Freeze dormant Fase 8 portfolio performance redistribution genes in `SuperGenotype`
//! (`portfolio_perf_mult_steepness`, `portfolio_dd_penalty_decay`, `portfolio_perf_mult_min`,
//! `portfolio_perf_mult_max`, `portfolio_perf_mult_center`, slots vec[117..121]).
//!
//! Because `PortfolioOrchestrator::calculate_dynamic_allocation` has zero callers in production
//! and backtesting pipelines, fitness is 100% flat on these 5 dimensions. Freezing them in
//! mutation and initializing them to canonical neutral defaults eliminates ~4% of evolutionary
//! noise while strictly preserving `SuperGenotype::DIMENSION == 144` and binary compatibility.

use quantum_arena::genome::SuperGenotype;
use std::f64::consts::{E, PI};

#[test]
fn test_f1_084_portfolio_perf_genes_remain_frozen_across_mutations() {
    let test_steepness = [3.5, 8.123, 14.5];
    let test_decay = [6.0, 15.707, 24.0];
    let test_min = [0.15, 0.275, 0.45];
    let test_max = [1.2, 1.65, 2.9];
    let test_center = [0.42, 0.55, 0.58];

    for i in 0..test_steepness.len() {
        let mut parent = SuperGenotype::default();
        parent.portfolio_perf_mult_steepness = test_steepness[i];
        parent.portfolio_dd_penalty_decay = test_decay[i];
        parent.portfolio_perf_mult_min = test_min[i];
        parent.portfolio_perf_mult_max = test_max[i];
        parent.portfolio_perf_mult_center = test_center[i];

        // Rate = 0.0 must preserve exact bit representation
        let child_zero = parent.mutate_cmaes(0.0);
        assert_eq!(
            child_zero.portfolio_perf_mult_steepness.to_bits(),
            test_steepness[i].to_bits(),
            "steepness must not be altered with rate 0.0"
        );
        assert_eq!(
            child_zero.portfolio_dd_penalty_decay.to_bits(),
            test_decay[i].to_bits(),
            "decay must not be altered with rate 0.0"
        );
        assert_eq!(
            child_zero.portfolio_perf_mult_min.to_bits(),
            test_min[i].to_bits(),
            "min must not be altered with rate 0.0"
        );
        assert_eq!(
            child_zero.portfolio_perf_mult_max.to_bits(),
            test_max[i].to_bits(),
            "max must not be altered with rate 0.0"
        );
        assert_eq!(
            child_zero.portfolio_perf_mult_center.to_bits(),
            test_center[i].to_bits(),
            "center must not be altered with rate 0.0"
        );

        // High mutation rate across multiple generations and seeds must remain strictly frozen
        for seed in 0..50u64 {
            let child = parent.mutate_cmaes_seeded(0.85, seed);
            assert_eq!(
                child.portfolio_perf_mult_steepness.to_bits(),
                test_steepness[i].to_bits(),
                "portfolio_perf_mult_steepness must remain frozen across mutations (seed {seed})"
            );
            assert_eq!(
                child.portfolio_dd_penalty_decay.to_bits(),
                test_decay[i].to_bits(),
                "portfolio_dd_penalty_decay must remain frozen across mutations (seed {seed})"
            );
            assert_eq!(
                child.portfolio_perf_mult_min.to_bits(),
                test_min[i].to_bits(),
                "portfolio_perf_mult_min must remain frozen across mutations (seed {seed})"
            );
            assert_eq!(
                child.portfolio_perf_mult_max.to_bits(),
                test_max[i].to_bits(),
                "portfolio_perf_mult_max must remain frozen across mutations (seed {seed})"
            );
            assert_eq!(
                child.portfolio_perf_mult_center.to_bits(),
                test_center[i].to_bits(),
                "portfolio_perf_mult_center must remain frozen across mutations (seed {seed})"
            );
        }
    }
}

#[test]
fn test_f1_084_new_random_initializes_canonical_neutral_defaults() {
    let expected_steepness = (E * 3.0_f64).to_bits();
    let expected_decay = (PI * 5.0_f64).to_bits();
    let expected_min = (0.55_f64 * 0.5_f64).to_bits();
    let expected_max = (0.55_f64 * 3.0_f64).to_bits();
    let expected_center = 0.55_f64.to_bits();

    for _ in 0..30 {
        let g = SuperGenotype::new_random();
        assert_eq!(
            g.portfolio_perf_mult_steepness.to_bits(),
            expected_steepness,
            "new_random must set canonical neutral steepness"
        );
        assert_eq!(
            g.portfolio_dd_penalty_decay.to_bits(),
            expected_decay,
            "new_random must set canonical neutral decay"
        );
        assert_eq!(
            g.portfolio_perf_mult_min.to_bits(),
            expected_min,
            "new_random must set canonical neutral min"
        );
        assert_eq!(
            g.portfolio_perf_mult_max.to_bits(),
            expected_max,
            "new_random must set canonical neutral max"
        );
        assert_eq!(
            g.portfolio_perf_mult_center.to_bits(),
            expected_center,
            "new_random must set canonical neutral center"
        );
    }
}

#[test]
fn test_f1_084_layout_dimension_and_vector_slots_117_to_121_preserved() {
    let mut g = SuperGenotype::default();
    g.portfolio_perf_mult_steepness = 9.123;
    g.portfolio_dd_penalty_decay = 17.456;
    g.portfolio_perf_mult_min = 0.345;
    g.portfolio_perf_mult_max = 2.456;
    g.portfolio_perf_mult_center = 0.523;

    let vec = g.to_vector();
    assert_eq!(vec.len(), SuperGenotype::DIMENSION);
    assert_eq!(SuperGenotype::DIMENSION, 144);

    assert_eq!(vec[117], 9.123, "Slot 117 must hold portfolio_perf_mult_steepness");
    assert_eq!(vec[118], 17.456, "Slot 118 must hold portfolio_dd_penalty_decay");
    assert_eq!(vec[119], 0.345, "Slot 119 must hold portfolio_perf_mult_min");
    assert_eq!(vec[120], 2.456, "Slot 120 must hold portfolio_perf_mult_max");
    assert_eq!(vec[121], 0.523, "Slot 121 must hold portfolio_perf_mult_center");

    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    assert_eq!(lo.len(), 144);
    assert_eq!(hi.len(), 144);

    assert_eq!(lo[117], 3.0);
    assert_eq!(hi[117], 15.0);
    assert_eq!(lo[118], 5.0);
    assert_eq!(hi[118], 25.0);
    assert_eq!(lo[119], 0.1);
    assert_eq!(hi[119], 0.5);
    assert_eq!(lo[120], 1.1);
    assert_eq!(hi[120], 3.0);
    assert_eq!(lo[121], 0.4);
    assert_eq!(hi[121], 0.6);

    let reconstructed = SuperGenotype::try_from_vector(&vec).expect("valid vector deserialization");
    assert_eq!(reconstructed.portfolio_perf_mult_steepness, 9.123);
    assert_eq!(reconstructed.portfolio_dd_penalty_decay, 17.456);
    assert_eq!(reconstructed.portfolio_perf_mult_min, 0.345);
    assert_eq!(reconstructed.portfolio_perf_mult_max, 2.456);
    assert_eq!(reconstructed.portfolio_perf_mult_center, 0.523);
}
