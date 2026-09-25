//! Shared numerical contract, not a statistical certification or trading policy.
//! Log wealth growth is scale invariant; leverage changes the wealth path.
//! The inherited squared drawdown penalty is a preference, not a ruin bound.

pub const DRAWDOWN_LAMBDA: f64 = 2.772_588_722_239_781;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitnessError {
    InvalidInitialCapital,
    InvalidFinalCapital,
    InvalidDrawdown,
    InsufficientTrades { observed: u32, required: u32 },
}

/// Positive finite endpoints can have an unrepresentable ratio but a finite
/// logarithmic difference. Preserve the ordinary ratio path when representable.
pub fn log_capital_growth(initial: f64, final_capital: f64) -> Result<f64, FitnessError> {
    if !initial.is_finite() || initial <= 0.0 {
        return Err(FitnessError::InvalidInitialCapital);
    }
    if !final_capital.is_finite() || final_capital <= 0.0 {
        return Err(FitnessError::InvalidFinalCapital);
    }
    let ratio = final_capital / initial;
    Ok(if ratio.is_finite() && ratio > 0.0 {
        ratio.ln()
    } else {
        final_capital.ln() - initial.ln()
    })
}

/// Separates numeric invalidity and insufficient evidence from measured utility.
/// The caller's minimum-count policy and inherited finite-DD clamp are retained.
/// Neither a sample count nor this scalar proves independent statistical evidence.
pub fn checked_fitness(
    initial: f64,
    final_capital: f64,
    max_drawdown: f64,
    total_trades: u32,
    min_trades_required: u32,
) -> Result<f64, FitnessError> {
    let growth = log_capital_growth(initial, final_capital)?;
    if !max_drawdown.is_finite() {
        return Err(FitnessError::InvalidDrawdown);
    }
    let required = min_trades_required.max(1);
    if total_trades < required {
        return Err(FitnessError::InsufficientTrades {
            observed: total_trades,
            required,
        });
    }
    let dd = max_drawdown.clamp(0.0, 1.0);
    Ok(growth - DRAWDOWN_LAMBDA * dd * dd)
}
