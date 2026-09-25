//! Generation lifecycle and malformed-input boundaries; never runs the live loop.
use evolution_engine::cma_es::{CmaEsOptimizer, CmaInputError, UpdateOutcome};
use rand::{SeedableRng, rngs::StdRng};

fn scores() -> Vec<(usize, f64, f64, usize, f64, f64)> {
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

fn assert_unchanged(before: &CmaEsOptimizer, after: &CmaEsOptimizer) {
    // Debug includes the private supervisor reference. Unlike float equality,
    // this also compares unchanged NaN sentinels in corruption fixtures.
    assert_eq!(format!("{before:?}"), format!("{after:?}"));
}

#[test]
fn unchanged_supervisor_preserves_learning_through_two_generations() {
    let mut opt = CmaEsOptimizer::new(2, 0.2, Some(4));
    opt.mean.fill(0.5);
    let mut rng = StdRng::seed_from_u64(93);
    for generation in 1..=2 {
        let before = opt.sigma;
        let pop = opt.sample_population_with_rng(0.0, 0.0, 0.0, &mut rng);
        assert_eq!(
            opt.update_backtest_only(&pop, &mut scores(), 0.0),
            UpdateOutcome::Updated
        );
        let learned = opt.sigma;
        assert_ne!(learned, before, "fixture must exercise CSA");
        let snapshot = opt.clone();
        let trace = opt.apply_exploration_level(0.2).unwrap();
        assert_eq!(trace.previous_sigma, learned);
        assert_eq!(trace.sigma, learned);
        assert_eq!(opt.generation, generation);
        assert_unchanged(&snapshot, &opt);
        println!(
            "generation={generation} before={before:.16e} learned={learned:.16e} next_sample_sigma={:.16e}",
            opt.sigma
        );
    }
}

#[test]
fn supervisor_changes_compose_relatively_without_resetting_geometry() {
    let mut opt = CmaEsOptimizer::new(2, 0.2, Some(4));
    opt.update_backtest_only(&population(), &mut scores(), 0.0);
    let learned = opt.sigma;
    let before = opt.clone();
    opt.apply_exploration_level(0.3).unwrap();
    assert!((opt.sigma / learned - 1.5).abs() < 1e-14);
    let adjusted = opt.clone();
    opt.apply_exploration_level(0.3).unwrap();
    assert_unchanged(&adjusted, &opt);
    opt.apply_exploration_level(0.2).unwrap();
    assert!((opt.sigma / learned - 1.0).abs() < 1e-14);
    assert_eq!(opt.mean, before.mean);
    assert_eq!(opt.p_c, before.p_c);
    assert_eq!(opt.p_sigma, before.p_sigma);
    assert_eq!(opt.cov_matrix, before.cov_matrix);
    assert_eq!(opt.generation, before.generation);
}

#[test]
fn invalid_levels_are_transactionally_rejected() {
    let mut opt = CmaEsOptimizer::new(2, 0.2, Some(4));
    for level in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let before = opt.clone();
        assert_eq!(
            opt.apply_exploration_level(level),
            Err(CmaInputError::InvalidStepSize)
        );
        assert_unchanged(&before, &opt);
    }
    opt.apply_exploration_level(0.4).unwrap();
    assert_eq!(opt.sigma, 0.4);
}

#[test]
fn true_overflow_and_underflow_do_not_commit_supervisor_state() {
    for (sigma, level) in [(f64::MAX, 2.0), (1e-300, 1e-300)] {
        let mut opt = CmaEsOptimizer::new(2, 1.0, Some(4));
        opt.sigma = sigma;
        let before = opt.clone();
        assert_eq!(
            opt.apply_exploration_level(level),
            Err(CmaInputError::InvalidStepSize)
        );
        assert_unchanged(&before, &opt);
    }
}

#[test]
fn representable_rescaling_survives_intermediate_ratio_overflow() {
    let mut opt = CmaEsOptimizer::new(2, 1e-300, Some(4));
    opt.apply_exploration_level(1e300).unwrap();
    assert!((opt.sigma / 1e300 - 1.0).abs() < 1e-12);
    let before = opt.clone();
    opt.apply_exploration_level(1e300).unwrap();
    assert_unchanged(&before, &opt);
}

#[test]
fn invalid_constructor_parameters_have_explicit_errors() {
    assert!(matches!(
        CmaEsOptimizer::try_new(0, 0.2, Some(4)),
        Err(CmaInputError::InvalidDimension)
    ));
    for lambda in [0, 1] {
        assert!(matches!(
            CmaEsOptimizer::try_new(2, 0.2, Some(lambda)),
            Err(CmaInputError::InvalidPopulationSize)
        ));
    }
    for sigma in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            CmaEsOptimizer::try_new(2, sigma, Some(4)),
            Err(CmaInputError::InvalidStepSize)
        ));
    }
}

#[test]
fn malformed_batch_preserves_every_field_and_score() {
    let base = CmaEsOptimizer::new(2, 0.2, Some(4));
    for (bad_index, expected) in [
        (0, CmaInputError::DuplicateCandidateIndex),
        (10, CmaInputError::CandidateIndexOutOfRange),
    ] {
        for live in [true, false] {
            let mut opt = base.clone();
            let mut evaluations = scores();
            evaluations[1].0 = bad_index;
            let before_scores = evaluations.clone();
            let outcome = if live {
                opt.update(&population(), &mut evaluations, 0.0)
            } else {
                opt.update_backtest_only(&population(), &mut evaluations, 0.0)
            };
            assert_eq!(outcome, UpdateOutcome::Rejected(expected));
            assert_eq!(evaluations, before_scores);
            assert_unchanged(&base, &opt);
        }
    }
}

#[test]
fn incomplete_batches_and_corrupted_shapes_are_explicit_rejections() {
    let mut opt = CmaEsOptimizer::new(2, 0.2, Some(4));
    let mut evaluations = scores();
    let before = opt.clone();
    assert_eq!(
        opt.update_backtest_only(&population()[..3], &mut evaluations, 0.0),
        UpdateOutcome::Rejected(CmaInputError::PopulationSizeMismatch)
    );
    assert_eq!(
        opt.update_backtest_only(&population(), &mut evaluations[..3], 0.0),
        UpdateOutcome::Rejected(CmaInputError::EvaluationCountMismatch)
    );
    assert_unchanged(&before, &opt);
    opt.cov_matrix.pop();
    let before = opt.clone();
    assert_eq!(
        opt.update_backtest_only(&population(), &mut evaluations, 0.0),
        UpdateOutcome::Rejected(CmaInputError::InvalidOptimizerShape)
    );
    assert_unchanged(&before, &opt);
}

#[test]
fn evaluation_arrival_order_does_not_change_genome_attribution() {
    let mut ordered = CmaEsOptimizer::new(2, 0.2, Some(4));
    let mut permuted = ordered.clone();
    let mut evaluations = scores();
    evaluations.reverse();
    assert_eq!(
        ordered.update_backtest_only(&population(), &mut scores(), 0.0),
        UpdateOutcome::Updated
    );
    assert_eq!(
        permuted.update_backtest_only(&population(), &mut evaluations, 0.0),
        UpdateOutcome::Updated
    );
    assert_unchanged(&ordered, &permuted);
    assert_eq!(ordered.global_best, population()[0]);
}

#[test]
fn observed_pso_memories_still_influence_search() {
    let mut base = CmaEsOptimizer::new(2, 0.2, Some(4));
    base.update_backtest_only(&population(), &mut scores(), 0.0);
    let mut hybrid = base.clone();
    let mut rng_a = StdRng::seed_from_u64(81);
    let mut rng_b = StdRng::seed_from_u64(81);
    assert_ne!(
        base.sample_population_with_rng(0.0, 0.0, 0.0, &mut rng_a),
        hybrid.sample_population_with_rng(0.0, 1.4, 1.4, &mut rng_b)
    );
}

#[test]
fn nonfinite_pso_fitness_is_not_evidence() {
    for fitness in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut base = CmaEsOptimizer::new(2, 0.2, Some(4));
        base.mean.fill(0.5);
        let mut hybrid = base.clone();
        hybrid.personal_best_fitnesses.fill(fitness);
        hybrid.global_best_fitness = fitness;
        let mut rng_a = StdRng::seed_from_u64(81);
        let mut rng_b = StdRng::seed_from_u64(81);
        assert_eq!(
            base.sample_population_with_rng(0.0, 0.0, 0.0, &mut rng_a),
            hybrid.sample_population_with_rng(0.0, 1.4, 1.4, &mut rng_b)
        );
    }
}
