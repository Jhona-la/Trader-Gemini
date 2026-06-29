/// Escudos Cortafuegos
/// Reglas inquebrantables que bloquean operaciones perdedoras o que violan límites de Exchange.

#[inline(always)]
pub fn check_drawdown_limit(current_capital: f64, peak_capital: f64, max_drawdown_pct: f64) -> bool {
    if peak_capital <= 0.0 {
        return true; // No hay histórico, se asume seguro inicial
    }
    
    let current_drawdown = (peak_capital - current_capital) / peak_capital;
    
    // Si el drawdown actual es mayor o igual al límite duro, BLOQUEAR
    current_drawdown < max_drawdown_pct
}

#[inline(always)]
pub fn enforce_minimum_notional(intended_volume: f64, min_notional: f64, available_leverage: f64) -> (bool, f64) {
    // Binance exige un nominal mínimo de $5.00 USD por orden
    let nominal_value = intended_volume * available_leverage;

    if nominal_value < min_notional {
        // No alcanza, rechazamos (o podríamos forzarlo al mínimo, pero forzar apalancamiento aumenta riesgo de ruina)
        (false, 0.0)
    } else {
        (true, intended_volume)
    }
}
