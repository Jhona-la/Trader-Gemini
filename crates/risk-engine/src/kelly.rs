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

    // R1.4 — criterios SEPARADOS, no mezclados con `||`:
    //  (a) PF <= 1.0: rampa de exploración (el PF es una estadística con
    //      memoria; congelar a 0.0 dejaría el sistema plano para siempre
    //      tras un régimen adverso temprano). PF 0.5 -> 0; PF 1.0 -> techo.
    //      La escala 1/4 del piso fraccional es la convención estándar de
    //      Kelly fraccional para exploración ultra-conservadora (quarter de
    //      la fracción mínima del genoma — derivación documentada, no
    //      literal suelto).
    //  (b) WR < 0.05 con PF > 1.0: edge legítimo con pocos trades ganados
    //      — ANTES el `||` pisaba el Kelly exacto con ~0; ahora fluye a la
    //      fórmula exacta modulada por wr_ramp.
    if profit_factor <= 1.0 {
        let pf_ramp = ((profit_factor - 0.5).clamp(0.0, 0.5)) / 0.5;
        let wr_ramp = (win_rate / 0.05).clamp(0.0, 1.0);
        let exploration = clamp_min.max(0.0) * 0.25 * pf_ramp * wr_ramp;
        let mx = clamp_max.max(clamp_min.max(0.0));
        return exploration.clamp(0.0, mx);
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
    // R1.4 — el coeficiente de supervivencia ya no es el literal 0.35: es el
    // gen strategy_base_fraction (scalp/swing_kelly_fraction), revivido de
    // su condición de gen muerto. El clamp [0.20, 0.75] es el riel de
    // seguridad del modo (equivalente al comportamiento histórico).
    let survival_coeff = strategy_base_fraction.clamp(0.20, 0.75);
    let capital_scale = if capital_ratio < kelly_survival_cap_ratio {
        // Modo Supervivencia Adaptativa
        (survival_coeff * win_rate * capital_ratio.sqrt()).clamp(0.20, 0.60)
    } else {
        // Modo Expansión Parabólica (Interés Compuesto)
        let expansion = (capital_ratio.log10() * kelly_expansion_mult + 0.6).clamp(0.6, 2.0);
        (0.50 * expansion).clamp(0.50, 0.90)
    };

    // R1.4 — guard de clamp: `f64::clamp` con min > max es pánico; un genoma
    // legacy cargado del espejo (sin pasar por el gate) puede traer cualquier par.
    let (mn, mx) = if clamp_min <= clamp_max {
        (clamp_min, clamp_max)
    } else {
        (clamp_max, clamp_min)
    };
    // Retornamos el Kelly ajustado asimétricamente acotado al tope seguro
    (kelly * capital_scale).clamp(mn, mx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> (f64, f64, f64, f64, f64, f64) {
        // (base_capital, survival_ratio, expansion_mult, clamp_min, clamp_max, strategy_base)
        (13.0, 1.5, 1.0, 0.01, 0.5, 0.35)
    }
    fn call(
        wr: f64,
        pf: f64,
        cap: f64,
        base: f64,
        surv: f64,
        exp: f64,
        cmin: f64,
        cmax: f64,
        sbase: f64,
    ) -> f64 {
        calculate_kelly_fraction(wr, pf, cap, base, surv, exp, cmin, cmax, sbase)
    }

    #[test]
    fn test_r14_exploration_zero_below_pf_floor() {
        let (base, surv, exp, cmin, cmax, sbase) = base_args();
        let f = call(0.5, 0.3, 13.0, base, surv, exp, cmin, cmax, sbase);
        assert_eq!(f, 0.0, "PF 0.5 -> rampa 0: sin apuesta con PF catastrófico");
    }

    #[test]
    fn test_r14_exploration_ramps_with_pf() {
        let (base, surv, exp, cmin, cmax, sbase) = base_args();
        let low = call(0.4, 0.7, 13.0, base, surv, exp, cmin, cmax, sbase);
        let high = call(0.4, 0.99, 13.0, base, surv, exp, cmin, cmax, sbase);
        assert!(high > low, "más cerca del break-even -> más exploración");
        assert!(high <= cmin * 0.25 + 1e-12, "exploración acotada a quarter del piso");
    }

    #[test]
    fn test_r14_low_wr_with_positive_pf_flows_to_exact_kelly() {
        // ANTES: WR<0.05 con PF>1 pisaba el Kelly legítimo con ~0.
        let (base, surv, exp, cmin, cmax, sbase) = base_args();
        let f = call(0.03, 2.0, 100.0, base, surv, exp, cmin, cmax, sbase);
        // El resultado ya no es la rampa de exploración: con PF>1 fluye a la
        // fórmula exacta y su piso clamp_min.
        assert!(f >= cmin - 1e-12, "Kelly exacto con piso genómico, no rampa");
    }

    #[test]
    fn test_r14_clamp_guard_no_panic_on_inverted_genome() {
        let (base, surv, exp, _cmin, _cmax, sbase) = base_args();
        // Genoma legacy corrompido: min > max. Antes: panic en f64::clamp.
        let f = call(0.6, 2.0, 50.0, base, surv, exp, 0.5, 0.1, sbase);
        assert!(f.is_finite());
    }

    #[test]
    fn test_r14_survival_uses_genome_fraction() {
        let (base, surv, exp, cmin, cmax, _sbase) = base_args();
        let low_base = call(0.6, 2.0, 13.0, base, surv, exp, cmin, cmax, 0.20);
        let high_base = call(0.6, 2.0, 13.0, base, surv, exp, cmin, cmax, 0.75);
        assert!(high_base > low_base, "strategy_base_fraction revive como coeficiente");
    }
}
