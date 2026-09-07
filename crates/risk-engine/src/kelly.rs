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
    clamp_min: f64,
    clamp_max: f64,
    strategy_base_fraction: f64,
) -> f64 {
    // FIX #653: Guarda de finitud estricta previa
    if !win_rate.is_finite() || !profit_factor.is_finite() || !current_capital.is_finite() || !base_capital.is_finite() || !kelly_survival_cap_ratio.is_finite() || !kelly_expansion_mult.is_finite() || !clamp_min.is_finite() || !clamp_max.is_finite() || !strategy_base_fraction.is_finite() {
        return 0.0;
    }

    // Sin edge estadístico (PF <= 1.0): NO se congela el sistema a 0.0 para
    // siempre. El PF es una estadística con memoria; un régimen adverso
    // temprano dejaría el backtest/producción plano de forma permanente.
    // En su lugar, una rampa de exploración continua — derivada del piso
    // fraccional del genoma (clamp_min), no de un literal — permite
    // redescubrir edge a medida que el PF se acerca al break-even:
    // PF 0.5 -> 0% de la fracción base; PF 1.0 -> 100% de la fracción base.
    if profit_factor <= 1.0 || win_rate < 0.05 {
        let pf_ramp = ((profit_factor - 0.5).clamp(0.0, 0.5)) / 0.5;
        let wr_ramp = if win_rate < 0.05 { win_rate / 0.05 } else { 1.0 };
        let exploration = clamp_min.max(0.0) * 0.25 * pf_ramp * wr_ramp;
        return exploration.clamp(0.0, clamp_max.max(clamp_min));
    }

    // FIX #374: Fórmula exacta de Kelly a partir de Profit Factor: f* = W * (1 - 1/PF)
    let kelly = win_rate * (1.0 - (1.0 / profit_factor));
    if kelly <= 0.0 {
        return 0.0;
    }

    // Escala de capital: regímenes continuos parametrizados por el GENOMA
    // (kelly_survival_cap_ratio / kelly_expansion_mult), sin la antigua rama
    // de micro-cuenta `base_capital × 2.30`: ese umbral fijo creaba una
    // meseta donde el capital quedaba atrapado en media-Kelly hasta superar
    // 2.3× la base (~$30-40 con base de $13) — el "techo de $40". El modo
    // supervivencia (clamp 0.20-0.60 modulado por win rate y raíz del ratio
    // de capital) ya protege cuentas pequeñas de forma continua.
    let capital_ratio = (current_capital / base_capital.max(1.0)).max(0.01);
    let capital_scale = if capital_ratio < kelly_survival_cap_ratio {
        // Modo Supervivencia Adaptativa
        (0.35 * win_rate * capital_ratio.sqrt()).clamp(0.20, 0.60)
    } else {
        // Modo Expansión Parabólica (Interés Compuesto)
        let expansion = (capital_ratio.log10() * kelly_expansion_mult + 0.6).clamp(0.6, 2.0);
        (0.50 * expansion).clamp(0.50, 0.90)
    };

    // Retornamos el Kelly ajustado asimétricamente acotado al tope seguro
    (kelly * capital_scale).clamp(clamp_min, clamp_max)
}
