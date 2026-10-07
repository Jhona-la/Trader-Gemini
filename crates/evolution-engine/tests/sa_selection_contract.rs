//! Pure SA score-state tests plus an explicit source-control-flow contract.
#[path = "../src/sa_selection.rs"]
mod sa_selection;
#[path = "../src/score_retention.rs"]
mod score_retention;
use sa_selection::SaScoreState;
use score_retention::penalize_signed_score;

#[test]
fn no_champion_exists_before_a_finite_evaluation() {
    let state = SaScoreState::new();
    assert_eq!(state.current_score(), None);
    assert_eq!(state.best_score(), None);
}

#[test]
fn first_finite_loss_below_the_old_sentinel_is_selected() {
    let mut state = SaScoreState::new();
    let score =
        penalize_signed_score(-200000.0, 0.01).unwrap() - 200000.0 - 0.999999999999 * 200000.0;
    assert!(score < -9999999.0);
    let mut sampled = false;
    let decision = state.consider(score, 100.0, || {
        sampled = true;
        0.99
    });
    assert!(decision.accepted && decision.improved_best);
    assert!(
        !sampled,
        "first finite candidate must not need a random escape"
    );
    assert_eq!(state.current_score(), Some(score));
    assert_eq!(state.best_score(), Some(score));
}

#[test]
fn invalid_scores_neither_select_nor_consume_randomness() {
    let mut state = SaScoreState::new();
    let before = (state.current_score(), state.best_score());
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let decision = state.consider(score, 100.0, || panic!("invalid score used RNG"));
        assert!(!decision.accepted && !decision.improved_best);
        assert_eq!((state.current_score(), state.best_score()), before);
    }
}

#[test]
fn a_better_finite_candidate_keeps_the_deterministic_rule() {
    let mut state = SaScoreState::new();
    assert!(state.consider(-10.0, 100.0, || 0.0).accepted);
    let decision = state.consider(10.0, 100.0, || panic!("better score used RNG"));
    assert!(decision.accepted && decision.improved_best);
    assert_eq!(state.best_score(), Some(10.0));
}

#[test]
fn accepted_deterioration_does_not_replace_the_champion() {
    let mut state = SaScoreState::new();
    state.consider(10.0, 100.0, || 0.0);
    let decision = state.consider(9.0, 100.0, || 0.0);
    assert!(decision.accepted && !decision.improved_best);
    assert_eq!(state.current_score(), Some(9.0));
    assert_eq!(state.best_score(), Some(10.0));
}

#[test]
fn rejected_deterioration_keeps_both_evaluated_scores() {
    let mut state = SaScoreState::new();
    state.consider(10.0, 100.0, || 0.0);
    let decision = state.consider(-10.0, 100.0, || 0.99);
    assert!(!decision.accepted && !decision.improved_best);
    assert_eq!(state.current_score(), Some(10.0));
    assert_eq!(state.best_score(), Some(10.0));
}

#[test]
fn legacy_acceptance_probability_is_bitwise_unchanged() {
    for (current, score, temperature) in
        [(10.0, 9.0, 100.0), (-10.0, -20.0, 0.5), (0.0, 0.0, 100.0)]
    {
        let probability = std::f64::consts::E.powf((score - current) / temperature);
        for sample in [0.0, 0.25, 0.9, 0.99] {
            let mut state = SaScoreState::new();
            state.consider(current, temperature, || 0.0);
            assert_eq!(
                state.consider(score, temperature, || sample).accepted,
                sample < probability
            );
        }
    }
}

#[test]
fn every_representable_finite_first_score_has_an_evaluated_champion() {
    for score in [
        f64::MIN,
        -1e300,
        -20_410_000.0,
        -0.0,
        0.0,
        f64::from_bits(1),
        f64::MAX,
    ] {
        let mut state = SaScoreState::new();
        let decision = state.consider(score, 100.0, || panic!("first score used RNG"));
        assert!(decision.accepted && decision.improved_best);
        assert_eq!(state.best_score().unwrap().to_bits(), score.to_bits());
    }
}

#[test]
fn cli_rejection_does_not_skip_cooling_and_requires_a_champion() {
    // Structural callsite guard, not an execution of the evolution CLI.
    let source = include_str!("../../../src/bin/evolution.rs");
    let loop_start = source.find("// RA-SA-F01:").unwrap();
    let loop_end = source.find("let _best_out_pnl").unwrap();
    let selection_block = &source[loop_start..loop_end];
    assert!(
        !selection_block.contains("continue;"),
        "rejection must reach unconditional cooling"
    );
    assert!(selection_block.contains("selection.consider(score, temp"));
    assert!(selection_block.contains("temp *= cooling_rate;"));
    assert!(selection_block.contains("if temp < 0.01"));
    assert!(selection_block.contains("let Some(best_score) = selection.best_score() else"));
}
