//! Standalone mathematical contract: no tapes, models, exchange or evolution process.
#[path = "../src/score_retention.rs"]
mod score_retention;
use score_retention::{ScoreRetentionError, penalize_signed_score};

fn retention(dd: f64, dd_max: f64) -> f64 {
    let threshold = dd_max / 3.0;
    if dd > threshold {
        (-(dd - threshold) * 20.0).exp().clamp(0.01, 1.0)
    } else {
        1.0
    }
}

#[test]
fn unit_retention_preserves_every_finite_score_bitwise() {
    for score in [
        f64::MIN,
        -1e300,
        -1.0,
        -f64::MIN_POSITIVE,
        -0.0,
        0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1.0,
        1e300,
        f64::MAX,
    ] {
        assert_eq!(
            penalize_signed_score(score, 1.0).unwrap().to_bits(),
            score.to_bits()
        );
    }
}

#[test]
fn positive_scores_preserve_the_old_multiplication_bitwise() {
    for score in [
        0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1e-100,
        1.0,
        1e300,
        f64::MAX,
    ] {
        for r in [0.01, 0.125, 0.5, 0.99, 1.0] {
            assert_eq!(
                penalize_signed_score(score, r).unwrap().to_bits(),
                (score * r).to_bits()
            );
        }
    }
}

#[test]
fn a_penalty_never_rewards_a_negative_score() {
    for score in [-1e300, -1053.605156578264, -1.0, -1e-100] {
        for r in [0.01, 0.125, 0.5, 0.99, 1.0] {
            assert!(
                penalize_signed_score(score, r).unwrap() <= score,
                "score={score}, r={r}"
            );
        }
    }
}

#[test]
fn increasing_penalty_is_monotone_for_both_signs() {
    for score in [-1e300, -10.0, -1e-100, -0.0, 0.0, 1e-100, 10.0, 1e300] {
        let mut previous = score;
        for r in [1.0, 0.99, 0.5, 0.125, 0.01] {
            let penalized = penalize_signed_score(score, r).unwrap();
            assert!(
                penalized <= previous,
                "score={score}, r={r}, previous={previous}"
            );
            previous = penalized;
        }
    }
}

#[test]
fn higher_drawdown_cannot_win_at_identical_losing_terminal_capital() {
    let c0: f64 = 13.0;
    let capital: f64 = 11.7;
    let base = (capital / c0).ln().max(-20.0) * 10_000.0;
    let common_loss_penalty = 10_000.0 * (c0 - capital);
    let low = penalize_signed_score(base, retention(0.11, 0.30)).unwrap() - common_loss_penalty;
    let high = penalize_signed_score(base, retention(0.20, 0.30)).unwrap() - common_loss_penalty;
    println!("same_terminal_capital: low_dd_score={low:.15}, high_dd_score={high:.15}");
    assert!(high < low, "higher drawdown won: high={high}, low={low}");
}

#[test]
fn higher_drawdown_still_penalizes_a_profitable_candidate() {
    let base = (14.3_f64 / 13.0).ln() * 10_000.0;
    let low = penalize_signed_score(base, retention(0.11, 0.30)).unwrap();
    let high = penalize_signed_score(base, retention(0.20, 0.30)).unwrap();
    assert!(high < low);
}

#[test]
fn signed_zero_keeps_its_bits_without_inventing_return() {
    for score in [-0.0_f64, 0.0] {
        for r in [0.01, 0.5, 1.0] {
            assert_eq!(
                penalize_signed_score(score, r).unwrap().to_bits(),
                score.to_bits()
            );
        }
    }
}

#[test]
fn invalid_scores_have_an_explicit_error() {
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            penalize_signed_score(score, 0.5),
            Err(ScoreRetentionError::NonFiniteScore)
        );
    }
}

#[test]
fn retention_outside_the_domain_is_not_silently_clamped() {
    for r in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        1.0001,
    ] {
        assert_eq!(
            penalize_signed_score(-10.0, r),
            Err(ScoreRetentionError::InvalidRetention)
        );
    }
}

#[test]
fn a_negative_division_overflow_is_rejected_not_ranked() {
    assert_eq!(
        penalize_signed_score(f64::MIN, 0.01),
        Err(ScoreRetentionError::NonFiniteResult)
    );
}

#[test]
fn representable_subnormal_results_remain_valid() {
    let tiny = f64::from_bits(1);
    assert_eq!(penalize_signed_score(tiny, 0.01).unwrap(), 0.0);
    assert!(penalize_signed_score(-tiny, 0.01).unwrap() <= -tiny);
}

#[test]
fn ordering_is_preserved_at_a_fixed_retention() {
    let scores = [-1e300, -10.0, -1e-100, -0.0, 0.0, 1e-100, 10.0, 1e300];
    for r in [0.01, 0.125, 0.5, 0.99, 1.0] {
        for pair in scores.windows(2) {
            assert!(
                penalize_signed_score(pair[0], r).unwrap()
                    <= penalize_signed_score(pair[1], r).unwrap()
            );
        }
    }
}

#[test]
fn existing_drawdown_schedule_is_retained_and_bounded() {
    // Use the actual binary threshold, not the independently rounded decimal 0.10.
    assert_eq!(retention(0.30 / 3.0, 0.30), 1.0);
    assert_eq!(retention(1.0, 0.30), 0.01);
    let mut previous = 1.0;
    for step in 0..=1000 {
        let r = retention(step as f64 / 1000.0, 0.30);
        assert!((0.01..=1.0).contains(&r));
        assert!(r <= previous);
        previous = r;
    }
}
