/// Fórmula Dinámica de Kelly para Supervivencia y Crecimiento Exponencial
/// Con Profit Factor PF: Edge > 0 si y solo si PF > 1.0.
/// f* = W * (1 - 1 / PF) = W - (1 - W) / R donde R = PF * (1 - W) / W (Payoff Ratio)
#[inline(always)]
pub fn calculate_kelly_fraction(
    win_rate: f64,
    profit_factor: f64,
    current_capital: f64,
    base_capital: f64,
    kelly_survival_cap_ratio: f64,
    kelly_expansion_mult: f64,
) -> f64 {
    // FIX #653: Guarda de finitud estricta previa
    if !win_rate.is_finite() || !profit_factor.is_finite() || !current_capital.is_finite() || !base_capital.is_finite() || !kelly_survival_cap_ratio.is_finite() || !kelly_expansion_mult.is_finite() {
        return 0.0;
    }

    // Si no hay edge estadístico (PF <= 1.0) o el Win Rate es insuficiente, no se arriesga capital.
    if profit_factor <= 1.0 || win_rate < 0.05 {
        return 0.0;
    }

    // FIX #374: Fórmula exacta de Kelly a partir de Profit Factor: f* = W * (1 - 1/PF)
    let kelly = win_rate * (1.0 - (1.0 / profit_factor));
    if kelly <= 0.0 {
        return 0.0;
    }

    // Asimetría Matemática: Para micro-cuentas ($13 USD), dimensionar con Half-Kelly adaptativo
    // asegurando crecimiento geométrico sin asfixia de escala ni riesgo de ruina.
    // FIX #740: Escala de micro-cuenta dinámica derivada de base_capital
    let capital_ratio = (current_capital / base_capital.max(1.0)).max(0.01);
    let micro_account_threshold = base_capital.max(1.0) * 2.30;
    let capital_scale = if current_capital < micro_account_threshold {
        // FIX #378: Half-to-fractional Kelly dinámico modulado por la calidad del edge
        (0.50 * (win_rate / 0.60).clamp(0.7, 1.3)).clamp(0.25, 0.75)
    } else if capital_ratio < kelly_survival_cap_ratio {
        // Modo Supervivencia Adaptativa
        (0.35 * win_rate * capital_ratio.sqrt()).clamp(0.20, 0.60)
    } else {
        // Modo Expansión Parabólica (Interés Compuesto)
        let expansion = (capital_ratio.log10() * kelly_expansion_mult + 0.6).clamp(0.6, 2.0);
        (0.50 * expansion).clamp(0.50, 0.90)
    };

    // Retornamos el Kelly ajustado asimétricamente acotado al tope seguro
    (kelly * capital_scale).clamp(0.0, 0.75)
}
