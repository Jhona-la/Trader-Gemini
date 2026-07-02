/// Fórmula Dinámica de Kelly para Supervivencia Absoluta
/// Kelly = W - [(1 - W) / R]
/// donde W = Probabilidad de acierto (Win Rate), R = Profit Factor (Beneficio / Riesgo)

/// Fórmula Dinámica de Kelly para Supervivencia y Crecimiento Exponencial
#[inline(always)]
pub fn calculate_kelly_fraction(win_rate: f64, profit_factor: f64, current_capital: f64) -> f64 {
    if profit_factor <= 0.0 || win_rate < 0.01 {
        return 0.0;
    }

    let kelly = win_rate - ((1.0 - win_rate) / profit_factor);
    if kelly <= 0.0 {
        return 0.0;
    }

    let mut dynamic_max_risk = win_rate.powi(2); // Auto-adaptable: 100% WR permite 100% riesgo
    
    if current_capital < 50.0 {
        kelly.clamp(0.0, dynamic_max_risk.max(0.10))
    } else if current_capital < 200.0 {
        (kelly * 0.75).clamp(0.0, (dynamic_max_risk * 0.75).max(0.05))
    } else {
        (kelly * 0.50).clamp(0.0, (dynamic_max_risk * 0.50).max(0.01))
    }
}
