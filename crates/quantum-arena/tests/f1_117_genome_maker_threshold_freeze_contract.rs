//! Contract test for F1-117 (Ola Ω156):
//!
//! Contract verifying that `SuperGenotype::mutate_cmaes()` preserves `maker_only_capital_threshold`
//! without mutating it away from the parent's value, preventing evolutionary drift on
//! an inactive host path while strictly preserving the 144-dimensional genome layout.

use quantum_arena::genome::SuperGenotype;

#[test]
fn test_f1_117_genome_maker_threshold_freeze_contract() {
    let mut parent = SuperGenotype::default();
    parent.maker_only_capital_threshold = 42.5;

    // Mutate over multiple iterations/generations with high mutation rate
    for _ in 0..50 {
        let child = parent.mutate_cmaes(0.5);
        assert_eq!(
            child.maker_only_capital_threshold, 42.5,
            "maker_only_capital_threshold must remain frozen to parent value across mutations"
        );
    }
}
