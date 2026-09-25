//! No live evolution, tape, exchange, genome files or deployment.
use evolution_engine::cma_es::CmaEsOptimizer;

#[test]
fn a_worse_reality_gap_cannot_reward_negative_fitness() {
    for gross_pnl in [-10.0, 10.0] {
        let mut opt = CmaEsOptimizer::new(2, 0.2, Some(2));
        let population = vec![vec![0.1, 0.2], vec![0.8, 0.9]];
        let mut scores = vec![
            (0, -10.0, gross_pnl, 20, 2.0, 2.0),
            (1, -10.0, gross_pnl, 20, 2.0, 0.0),
        ];
        opt.update(&population, &mut scores, 0.0005);
        assert_eq!(scores[0].0, 0, "deteriorated candidate won: {scores:?}");
        assert!(scores[1].1 <= -10.0, "penalty improved signed fitness");
        assert_eq!(opt.global_best, population[0]);
    }
}

#[test]
fn numerical_failure_cannot_outrank_a_finite_loss() {
    let mut opt = CmaEsOptimizer::new(2, 0.2, Some(2));
    let population = vec![vec![0.1, 0.2], vec![0.8, 0.9]];
    let mut scores = vec![
        (0, -1e12, 10.0, 20, 2.0, 2.0),
        (1, f64::NAN, 10.0, 20, 2.0, 2.0),
    ];
    opt.update(&population, &mut scores, 0.0005);
    assert_eq!(scores[0].0, 0, "numeric fallback rewarded invalid fitness");
}

#[test]
fn backtest_metadata_is_preserved_but_cannot_change_selection() {
    let population = vec![vec![0.1, 0.2], vec![0.8, 0.9]];
    for growth_metadata in [0.0, 1.0, 100.0, f64::NAN] {
        let mut opt = CmaEsOptimizer::new(2, 0.2, Some(2));
        let mut scores = vec![
            (0, 2.0, 10.0, 20, 2.0, growth_metadata),
            (1, 1.0, 10.0, 20, 2.0, 2.0),
        ];
        opt.update_backtest_only(&population, &mut scores, 0.0005);
        assert_eq!(scores[0].0, 0);
        assert_eq!(scores[0].1, 2.0);
        assert_eq!(scores[0].5.to_bits(), growth_metadata.to_bits());
        assert_eq!(opt.global_best, population[0]);
    }
}

#[test]
fn penalty_is_monotone_for_both_signs_and_preserves_unpenalized_scores() {
    let population = vec![vec![0.1, 0.2], vec![0.8, 0.9]];
    for score in [-1e300, -10.0, -1e-100, 0.0, 1e-100, 10.0, 1e300] {
        let mut previous = score;
        for live in [2.0, 1.0, 0.0, -2.0, -100.0] {
            let mut opt = CmaEsOptimizer::new(2, 0.2, Some(2));
            let mut scores = vec![
                (0, score, 10.0, 20, 2.0, live),
                (1, score, 10.0, 20, 2.0, live),
            ];
            opt.update(&population, &mut scores, 0.0005);
            assert!(scores[0].1 <= previous, "score={score}, live={live}");
            if live == 2.0 {
                assert_eq!(scores[0].1, score);
            }
            previous = scores[0].1;
        }
    }
}

#[test]
fn insufficient_valid_parents_do_not_mutate_optimizer_memory() {
    let mut opt = CmaEsOptimizer::new(2, 0.2, Some(4));
    let population = vec![vec![0.1, 0.2]; 4];
    let mut scores = vec![
        (0, 10.0, 1.0, 20, 2.0, 2.0),
        (1, f64::NAN, 1.0, 20, 2.0, 2.0),
        (2, f64::INFINITY, 1.0, 20, 2.0, 2.0),
        (3, f64::NEG_INFINITY, 1.0, 20, 2.0, 2.0),
    ];
    opt.update(&population, &mut scores, 0.0005);
    assert_eq!(opt.generation, 0);
    assert_eq!(opt.mean, vec![0.0; 2]);
    assert_eq!(opt.sigma, 0.2);
    assert_eq!(opt.personal_best_fitnesses, vec![f64::MIN; 4]);
    assert_eq!(opt.global_best_fitness, f64::MIN);
}

#[test]
fn overflowed_penalty_and_invalid_live_reference_are_not_rewarded() {
    let population = vec![vec![0.1, 0.2], vec![0.8, 0.9]];
    for (score, live) in [(-1e308, -100.0), (100.0, f64::NAN), (100.0, f64::INFINITY)] {
        let mut opt = CmaEsOptimizer::new(2, 0.2, Some(2));
        let mut scores = vec![
            (0, -1e12, 10.0, 20, 2.0, 2.0),
            (1, score, 10.0, 20, 2.0, live),
        ];
        opt.update(&population, &mut scores, 0.0005);
        assert_eq!(scores[0].0, 0);
        assert_eq!(scores[1].1, f64::MIN);
    }
}
