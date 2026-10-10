//! Signed score retention for maximization objectives (standalone, no runtime state).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreRetentionError {
    NonFiniteScore,
    InvalidRetention,
    NonFiniteResult,
}

/// Apply a retention factor `r` in `(0, 1]` to a score which is maximized.
///
/// For `score >= 0`, preserve the existing multiplication `score * r` exactly.
/// For a loss (`score < 0`), use `score / r`: multiplication would reward the
/// loss by moving it towards zero. Thus a smaller `r` never improves a score,
/// and `r = 1` is an identity. This is a ranking contract, not a calibrated
/// probability, posterior, risk bound, or claim about future profitability.
/// Invalid inputs and a non-representable result are explicit errors; callers
/// must exclude them from selection rather than fabricate a finite fitness.
pub fn penalize_signed_score(score: f64, retention: f64) -> Result<f64, ScoreRetentionError> {
    if !score.is_finite() {
        return Err(ScoreRetentionError::NonFiniteScore);
    }
    if !retention.is_finite() || retention <= 0.0 || retention > 1.0 {
        return Err(ScoreRetentionError::InvalidRetention);
    }
    let penalized = if score >= 0.0 {
        score * retention
    } else {
        score / retention
    };
    if !penalized.is_finite() {
        return Err(ScoreRetentionError::NonFiniteResult);
    }
    Ok(penalized)
}
