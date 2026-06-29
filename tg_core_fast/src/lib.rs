use pyo3::prelude::*;

/// Calculates the optimal Kelly Fraction.
/// W - (L / R)
#[pyfunction]
fn calculate_kelly(win_prob: f64, payoff_ratio: f64) -> PyResult<(f64, String)> {
    if win_prob <= 0.0 || payoff_ratio <= 0.0 {
        return Ok((0.0, "INVALID_PARAMS".to_string()));
    }
    let loss_prob = 1.0 - win_prob;
    let kelly_f = win_prob - (loss_prob / payoff_ratio);
    
    if kelly_f <= 0.0 {
        return Ok((0.0, "NEGATIVE_EXPECTANCY".to_string()));
    }
    Ok((kelly_f, "PROCEED".to_string()))
}

/// Applies asymmetric compounding rules and micro-sizing checks.
/// Ensures notional size meets Binance minimums ($5.00) for micro accounts.
#[pyfunction]
fn apply_micro_sizing(
    mut base_risk_amount: f64,
    current_capital: f64,
    target_leverage: f64,
    min_notional: f64, // e.g. 5.50
    kelly_f: f64,
) -> PyResult<(f64, f64, String)> { // Returns (risk_amount, notional, status)
    
    // For accounts under $100, we apply full Kelly to the remaining capital limit
    if current_capital < 100.0 {
        let max_allocated = current_capital * 0.95; // Leave 5% buffer
        
        // Base risk amount already contains the Python adaptive multipliers
        // We just ensure it doesn't exceed 95% of current micro-capital
        base_risk_amount = if base_risk_amount < max_allocated {
            base_risk_amount
        } else {
            max_allocated
        };
    }
    
    let mut notional = base_risk_amount * target_leverage;
    
    // Binance Floor Evasion
    if notional < min_notional && notional > 0.0 {
        // If we bump it to min_notional, calculate required risk_amount
        let required_margin = min_notional / target_leverage;
        if required_margin <= current_capital * 0.95 {
            notional = min_notional;
            base_risk_amount = required_margin;
        } else {
            return Ok((0.0, 0.0, "INSUFFICIENT_FUNDS_FOR_FLOOR".to_string()));
        }
    }

    Ok((base_risk_amount, notional, "OK".to_string()))
}

#[pymodule]
fn tg_core_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_kelly, m)?)?;
    m.add_function(wrap_pyfunction!(apply_micro_sizing, m)?)?;
    Ok(())
}
