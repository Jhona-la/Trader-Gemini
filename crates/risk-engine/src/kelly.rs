/// Fórmula Dinámica de Kelly para Supervivencia Absoluta
/// Kelly = W - [(1 - W) / R]
/// donde W = Probabilidad de acierto (Win Rate), R = Profit Factor (Beneficio / Riesgo)

/// Fórmula Dinámica de Kelly para Supervivencia y Crecimiento Exponencial
#[inline(always)]
pub fn calculate_kelly_fraction(
    win_rate: f64,
    profit_factor: f64,
    current_capital: f64,
    base_capital: f64,
    kelly_survival_cap_ratio: f64,
    kelly_expansion_mult: f64,
) -> f64 {
    if profit_factor <= 0.0 || win_rate < 0.01 {
        return 0.0;
    }

    let kelly = win_rate - ((1.0 - win_rate) / profit_factor);
    if kelly <= 0.0 {
        return 0.0;
    }

    let _dynamic_max_risk = win_rate.powi(2); // Auto-adaptable: 100% WR permite 100% riesgo

    // Asimetría Matemática: Supervivencia vs Expansión Parabólica (Scale-Invariant)
    let capital_ratio = (current_capital / base_capital.max(1.0)).max(0.01);
    let capital_scale = if capital_ratio < kelly_survival_cap_ratio {
        // Modo Supervivencia Adaptativa: Base fraction scaled by win rate
        (0.25 * win_rate * capital_ratio.sqrt()).clamp(0.10, 0.60)
    } else {
        // Modo Expansión Parabólica (Interés Compuesto):
        let expansion = (capital_ratio.log10() * kelly_expansion_mult + 0.6).clamp(0.6, 2.5);
        (0.6 * expansion).clamp(0.6, 1.0)
    };

    // Retornamos el Kelly ajustado asimétricamente
    (kelly * capital_scale).clamp(0.0, 1.0)
}
