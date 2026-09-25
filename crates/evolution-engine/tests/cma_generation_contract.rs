//! Pure optimizer tests: no trading, promotion, genome files or deployment.
use evolution_engine::cma_es::CmaEsOptimizer;
use rand::{SeedableRng, rngs::StdRng};

type Score = (usize, f64, f64, usize, f64, f64);

fn scores() -> Vec<Score> {
    (0..4)
        .map(|i| (i, 4.0 - i as f64, 1.0, 20, 1.0, 1.0))
        .collect()
}

fn population() -> Vec<Vec<f64>> {
    vec![
        vec![0.2, 0.3],
        vec![0.4, 0.5],
        vec![0.6, 0.7],
        vec![0.8, 0.9],
    ]
}

#[test]
fn duplicated_candidate_cannot_be_selected_as_multiple_parents() {
    let mut opt = CmaEsOptimizer::new(2, 0.2, Some(4));
    let mut evaluations = scores();
    evaluations[1].0 = 0;
    opt.update_backtest_only(&population(), &mut evaluations, 0.0);
    assert_eq!(opt.generation, 0, "duplicate ID mutated optimizer");
    assert_eq!(opt.mean, vec![0.0; 2]);
}

#[test]
fn out_of_range_candidate_is_rejected_without_panicking() {
    let mut opt = CmaEsOptimizer::new(2, 0.2, Some(4));
    let mut evaluations = scores();
    evaluations[0].0 = 4;
    opt.update_backtest_only(&population(), &mut evaluations, 0.0);
    assert_eq!(opt.generation, 0);
}

#[test]
fn malformed_genome_cannot_poison_optimizer_memory() {
    for malformed in [vec![f64::NAN, 0.3], vec![0.2], vec![1.1, 0.3]] {
        let mut opt = CmaEsOptimizer::new(2, 0.2, Some(4));
        let mut pop = population();
        pop[0] = malformed;
        opt.update_backtest_only(&pop, &mut scores(), 0.0);
        assert_eq!(opt.generation, 0, "invalid genome entered optimizer");
        assert_eq!(opt.global_best_fitness, f64::MIN);
    }
}

#[test]
fn unobserved_pso_memories_cannot_attract_samples() {
    let mut base = CmaEsOptimizer::new(2, 0.2, Some(4));
    let mut hybrid = CmaEsOptimizer::new(2, 0.2, Some(4));
    base.mean.fill(0.5);
    hybrid.mean.fill(0.5);
    let mut rng_a = StdRng::seed_from_u64(81);
    let mut rng_b = StdRng::seed_from_u64(81);
    assert_eq!(
        base.sample_population_with_rng(0.7, 0.0, 0.0, &mut rng_a),
        hybrid.sample_population_with_rng(0.7, 1.4, 1.4, &mut rng_b),
        "unevaluated zero vectors exert a hidden attraction"
    );
}

#[test]
fn damping_matches_the_published_csa_equation() {
    for (dimension, lambda) in [(2, 4), (2, 100), (10, 40), (100, 100)] {
        let opt = CmaEsOptimizer::new(dimension, 0.2, Some(lambda));
        let c_sigma = (opt.mueff + 2.0) / (dimension as f64 + opt.mueff + 5.0);
        let expected = 1.0
            + 2.0 * (((opt.mueff - 1.0) / (dimension as f64 + 1.0)).sqrt() - 1.0).max(0.0)
            + c_sigma;
        assert!(
            (opt.d_sigma - expected).abs() < 1e-14,
            "n={dimension}, lambda={lambda}, actual={}, expected={expected}",
            opt.d_sigma
        );
    }
}
