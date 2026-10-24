//! Contract test for F1-069 & F1-070 (Ola Ω174):
//!
//! F1-069: Verification and contract enforcement that `maker_only_capital_threshold`
//! is strictly frozen across mutations (avoiding hardcoded 50.0 override) and preserves
//! the 144-dimensional genome layout.
//!
//! F1-070: Unification and total parity of genome vector bounds (`get_lower_bounds` / `get_upper_bounds`),
//! mutation bands (`mutate_with_rng` and `new_random`), and reader domains (`output_bounds` in `horizon_policy.rs`),
//! guaranteeing zero flat fitness exploration and zero reader clipping distortion across all horizons.

use quantum_arena::genome::SuperGenotype;
use quantum_arena::horizon_policy::HorizonParameter;
use quantum_arena::temporal_spectrum::{TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};

#[test]
fn test_f1_069_maker_threshold_freeze_arbitrary_values() {
    let test_thresholds = [12.0, 25.0, 42.5, 100.0, 500.0, 2500.0];

    for &threshold in &test_thresholds {
        let mut parent = SuperGenotype::default();
        parent.maker_only_capital_threshold = threshold;

        // Rate = 0.0 must preserve exact bit representation
        let child_zero = parent.mutate_cmaes(0.0);
        assert_eq!(
            child_zero.maker_only_capital_threshold.to_bits(),
            threshold.to_bits(),
            "maker_only_capital_threshold must not be altered with rate 0.0"
        );

        // High mutation rate across multiple generations must remain frozen
        for seed in 0..20u64 {
            let child = parent.mutate_cmaes_seeded(0.8, seed);
            assert_eq!(
                child.maker_only_capital_threshold, threshold,
                "maker_only_capital_threshold must remain frozen to parent value across mutations"
            );
        }
    }
}

#[test]
fn test_f1_069_layout_slot_108_preserved() {
    let mut g = SuperGenotype::default();
    g.maker_only_capital_threshold = 123.45;

    let vec = g.to_vector();
    assert_eq!(vec.len(), SuperGenotype::DIMENSION);
    assert_eq!(SuperGenotype::DIMENSION, 144);
    assert_eq!(vec[108], 123.45, "Slot 108 must hold maker_only_capital_threshold");

    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    assert_eq!(lo.len(), 144);
    assert_eq!(hi.len(), 144);
    assert_eq!(lo[108], 10.0);
    assert_eq!(hi[108], 5000.0);

    let reconstructed = SuperGenotype::try_from_vector(&vec).expect("valid vector deserialization");
    assert_eq!(reconstructed.maker_only_capital_threshold, 123.45);
}

#[test]
fn test_f1_070_vector_bounds_parity_with_reader_domains() {
    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();

    // ObiThreshold: [0.10, 0.95]
    assert_eq!(lo[12], 0.10, "Slot 12 (scalp_obi_threshold) lower bound");
    assert_eq!(hi[12], 0.95, "Slot 12 (scalp_obi_threshold) upper bound");
    assert_eq!(lo[137], 0.10, "Slot 137 (swing_obi_threshold) lower bound");
    assert_eq!(hi[137], 0.95, "Slot 137 (swing_obi_threshold) upper bound");

    let (obi_lo, obi_hi) = HorizonParameter::ObiThreshold.output_bounds().expect("ObiThreshold bounds");
    assert_eq!(lo[12], obi_lo);
    assert_eq!(hi[12], obi_hi);
    assert_eq!(lo[137], obi_lo);
    assert_eq!(hi[137], obi_hi);

    // TrailActivation: [1.5, 6.0]
    let (act_lo, act_hi) = HorizonParameter::TrailActivation.output_bounds().expect("TrailActivation bounds");
    assert_eq!(lo[44], 1.5, "Slot 44 (scalp_trail_act_atr) lower bound");
    assert_eq!(hi[44], 6.0, "Slot 44 (scalp_trail_act_atr) upper bound");
    assert_eq!(lo[49], 1.5, "Slot 49 (swing_trail_act_atr) lower bound");
    assert_eq!(hi[49], 6.0, "Slot 49 (swing_trail_act_atr) upper bound");
    assert_eq!(lo[44], act_lo);
    assert_eq!(hi[44], act_hi);
    assert_eq!(lo[49], act_lo);
    assert_eq!(hi[49], act_hi);

    // TrailMultiplier: [1.5, 6.0]
    let (mult_lo, mult_hi) = HorizonParameter::TrailMultiplier.output_bounds().expect("TrailMultiplier bounds");
    assert_eq!(lo[48], 1.5, "Slot 48 (scalp_trail_atr_mult_base) lower bound");
    assert_eq!(hi[48], 6.0, "Slot 48 (scalp_trail_atr_mult_base) upper bound");
    assert_eq!(lo[53], 1.5, "Slot 53 (swing_trail_atr_mult_base) lower bound");
    assert_eq!(hi[53], 6.0, "Slot 53 (swing_trail_atr_mult_base) upper bound");
    assert_eq!(lo[48], mult_lo);
    assert_eq!(hi[48], mult_hi);
    assert_eq!(lo[53], mult_lo);
    assert_eq!(hi[53], mult_hi);

    // TrailStep: [0.5, 4.0]
    let (step_lo, step_hi) = HorizonParameter::TrailStep.output_bounds().expect("TrailStep bounds");
    assert_eq!(lo[45], 0.5, "Slot 45 (scalp_trail_step_atr) lower bound");
    assert_eq!(hi[45], 4.0, "Slot 45 (scalp_trail_step_atr) upper bound");
    assert_eq!(lo[50], 0.5, "Slot 50 (swing_trail_step_atr) lower bound");
    assert_eq!(hi[50], 4.0, "Slot 50 (swing_trail_step_atr) upper bound");
    assert_eq!(lo[45], step_lo);
    assert_eq!(hi[45], step_hi);
    assert_eq!(lo[50], step_lo);
    assert_eq!(hi[50], step_hi);
}

#[test]
fn test_f1_070_new_baseline_operates_without_reader_clipping() {
    let g = SuperGenotype::new_baseline(0.0002, 0.0005);

    // Check raw values are within unified bounds
    assert!(g.scalp_obi_threshold >= 0.10 && g.scalp_obi_threshold <= 0.95);
    assert!(g.swing_obi_threshold >= 0.10 && g.swing_obi_threshold <= 0.95);
    assert!(g.scalp_trail_act_atr >= 1.5 && g.scalp_trail_act_atr <= 6.0);
    assert!(g.swing_trail_act_atr >= 1.5 && g.swing_trail_act_atr <= 6.0);
    assert!(g.scalp_trail_step_atr >= 0.5 && g.scalp_trail_step_atr <= 4.0);
    assert!(g.swing_trail_step_atr >= 0.5 && g.swing_trail_step_atr <= 4.0);
    assert!(g.scalp_trail_atr_mult_base >= 1.5 && g.scalp_trail_atr_mult_base <= 6.0);
    assert!(g.swing_trail_atr_mult_base >= 1.5 && g.swing_trail_atr_mult_base <= 6.0);

    // Evaluate readers at anchor scales: must match exactly without distortion
    let obi_fast = g.obi_threshold_at_tau(TAU_ANCHOR_FAST_MS);
    assert!((obi_fast - g.scalp_obi_threshold).abs() < 1e-9, "fast obi reader must yield gene without clipping");

    let (mult_fast, act_fast, step_fast) = g.trail_params_at_tau(TAU_ANCHOR_FAST_MS);
    assert!((mult_fast - g.scalp_trail_atr_mult_base).abs() < 1e-9, "fast trail mult must yield gene without clipping");
    assert!((act_fast - g.scalp_trail_act_atr).abs() < 1e-9, "fast trail act must yield gene without clipping");
    assert!((step_fast - g.scalp_trail_step_atr).abs() < 1e-9, "fast trail step must yield gene without clipping");

    let (mult_slow, act_slow, step_slow) = g.trail_params_at_tau(TAU_ANCHOR_SLOW_MS);
    assert!((mult_slow - g.swing_trail_atr_mult_base).abs() < 1e-9, "slow trail mult must yield gene without clipping");
    assert!((act_slow - g.swing_trail_act_atr).abs() < 1e-9, "slow trail act must yield gene without clipping");
    assert!((step_slow - g.swing_trail_step_atr).abs() < 1e-9, "slow trail step must yield gene without clipping");
}

#[test]
fn test_f1_070_random_and_mutated_genomes_stay_within_bounds() {
    let base = SuperGenotype::default();

    for seed in 0..30u64 {
        let mutated = base.mutate_cmaes_seeded(0.75, seed);

        assert!(mutated.scalp_obi_threshold >= 0.10 && mutated.scalp_obi_threshold <= 0.95);
        assert!(mutated.swing_obi_threshold >= 0.10 && mutated.swing_obi_threshold <= 0.95);
        assert!(mutated.scalp_trail_act_atr >= 1.5 && mutated.scalp_trail_act_atr <= 6.0);
        assert!(mutated.swing_trail_act_atr >= 1.5 && mutated.swing_trail_act_atr <= 6.0);
        assert!(mutated.scalp_trail_step_atr >= 0.5 && mutated.scalp_trail_step_atr <= 4.0);
        assert!(mutated.swing_trail_step_atr >= 0.5 && mutated.swing_trail_step_atr <= 4.0);
        assert!(mutated.scalp_trail_atr_mult_base >= 1.5 && mutated.scalp_trail_atr_mult_base <= 6.0);
        assert!(mutated.swing_trail_atr_mult_base >= 1.5 && mutated.swing_trail_atr_mult_base <= 6.0);

        // Vector roundtrip must be idempotent
        let v = mutated.to_vector();
        let from_vec = SuperGenotype::try_from_vector(&v).expect("try_from_vector must succeed");
        assert_eq!(from_vec.scalp_obi_threshold, mutated.scalp_obi_threshold);
        assert_eq!(from_vec.swing_obi_threshold, mutated.swing_obi_threshold);
        assert_eq!(from_vec.scalp_trail_act_atr, mutated.scalp_trail_act_atr);
        assert_eq!(from_vec.swing_trail_act_atr, mutated.swing_trail_act_atr);
        assert_eq!(from_vec.scalp_trail_step_atr, mutated.scalp_trail_step_atr);
        assert_eq!(from_vec.swing_trail_step_atr, mutated.swing_trail_step_atr);
        assert_eq!(from_vec.scalp_trail_atr_mult_base, mutated.scalp_trail_atr_mult_base);
        assert_eq!(from_vec.swing_trail_atr_mult_base, mutated.swing_trail_atr_mult_base);
    }
}
