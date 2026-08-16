/// Escudos Cortafuegos
/// Reglas inquebrantables que bloquean operaciones perdedoras o que violan límites de Exchange.

#[inline(always)]
pub fn check_drawdown_limit(
    current_capital: f64,
    peak_capital: f64,
    genome_max_drawdown_pct: f64,
    base_capital: f64,
    guard_dd_sigmoid_steepness: f64,
    guard_dd_sigmoid_center: f64,
) -> bool {
    if peak_capital <= 0.0 {
        return true; // No hay histórico, se asume seguro inicial
    }

    let current_drawdown = (peak_capital - current_capital) / peak_capital;

    // Continuous Sigmoid: Drawdown limit scales smoothly with capital size relative to base.
    // Small capital (< 3x base) → aggressive limit approaching 0.99
    // Large capital (> 10x base) → strict limit approaching genome_max_drawdown_pct
    // Formula: dd_limit = genome_dd + (0.99 - genome_dd) * sigmoid(-k * (capital/base - midpoint))
    let capital_ratio = current_capital / base_capital.max(1.0);
    let sigmoid_input = -guard_dd_sigmoid_steepness * (capital_ratio - guard_dd_sigmoid_center); // Centered dynamically
    let sigmoid_val = 1.0 / (1.0 + sigmoid_input.exp());
    let dynamic_max_drawdown =
        genome_max_drawdown_pct + (0.99 - genome_max_drawdown_pct) * sigmoid_val;

    // Si el drawdown actual es mayor o igual al límite duro, BLOQUEAR
    current_drawdown < dynamic_max_drawdown
}

#[inline(always)]
pub fn enforce_minimum_notional(
    intended_volume: f64,
    min_notional: f64,
    available_leverage: f64,
) -> (bool, f64) {
    // Binance exige un nominal mínimo de $5.00 USD por orden
    let nominal_value = intended_volume * available_leverage;

    if nominal_value < min_notional {
        // No alcanza, rechazamos (o podríamos forzarlo al mínimo, pero forzar apalancamiento aumenta riesgo de ruina)
        (false, 0.0)
    } else {
        (true, intended_volume)
    }
}
