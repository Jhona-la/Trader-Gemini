/// Fórmula Dinámica de Kelly para Supervivencia y Crecimiento Exponencial
/// Con Profit Factor PF: Edge > 0 si y sólo si PF > 1.0.
/// f* = W * (1 - 1 / PF) = W - (1 - W) / R donde R = PF * (1 - W) / W (Payoff Ratio)
///
/// S-1 (MOTOR UNIVERSAL / ESPECTRALIZACIÓN): los rieles de capital
/// (supervivencia [0.20,0.60], expansión [0.50,0.90]) dejan de ser literales:
/// la BANDA se modula por `spectral_conf` ∈ [0,1] — confianza espectral del
/// mercado (persistencia de la escala dominante mapeada a [0,1]; 0.5 =
/// neutral browniano). Mercado persistente/tendencial ⇒ banda desplazada
/// hacia el crecimiento (más Kelly justificado: la tendencia sostiene las
/// ganancias); anti-persistente/mean-reverting ⇒ banda comprimida (las
/// ganancias no se sostienen:收割 temprano). Con s=0.5 la banda reproduce
/// el comportamiento histórico (0.20/0.60 y 0.50/0.90 ± redondeo).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
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
    spectral_conf: f64,
) -> f64 {
    // FIX #653: Guarda de finitud estricta previa
    if !win_rate.is_finite()
        || !profit_factor.is_finite()
        || !current_capital.is_finite()
        || !base_capital.is_finite()
        || !kelly_survival_cap_ratio.is_finite()
        || !kelly_expansion_mult.is_finite()
        || !clamp_min.is_finite()
        || !clamp_max.is_finite()
        || !strategy_base_fraction.is_finite()
    {
        return 0.0;
    }
    // S-1: s fuera de banda o degenerado ⇒ neutral (0.5).
    let s = if spectral_conf.is_finite() {
        spectral_conf.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let lerp = |a: f64, b: f64| a + (b - a) * s;

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
    // gen kelly_at_tau(τ de la posición), revivido de su condición de gen
    // muerto. El clamp [0.20, 0.75] es el riel de seguridad del modo.
    let survival_coeff = strategy_base_fraction.clamp(0.20, 0.75);
    let capital_scale = if capital_ratio < kelly_survival_cap_ratio {
        // Modo Supervivencia Adaptativa — S-1: banda espectral
        // (anti-persistente [0.15,0.50] ↔ persistente [0.25,0.65]).
        (survival_coeff * win_rate * capital_ratio.sqrt()).clamp(lerp(0.15, 0.25), lerp(0.50, 0.65))
    } else {
        // Modo Expansión Parabólica (Interés Compuesto) — S-1: banda espectral
        // (anti-persistente [0.40,0.80] ↔ persistente [0.55,0.95]).
        let expansion = (capital_ratio.log10() * kelly_expansion_mult + 0.6).clamp(0.6, 2.0);
        (0.50 * expansion).clamp(lerp(0.40, 0.55), lerp(0.80, 0.95))
    };

    // R1.4 — guard de clamp: `f64::clamp` con min > max es pánico; un genoma
    // legacy cargado del espejo (sin pasar por el gate) puede traer cualquier par.
    let (mn, mx) = if clamp_min <= clamp_max {
        (clamp_min, clamp_max)
    } else {
        (clamp_max, clamp_min)
    };

    // CERT-M5-H03 — CONTROL DE RUINA CENTRALIZADO. El bloque QO-M1.2
    // anterior estaba ROTO como tope: la exponencial P(f)=((1−f)/(1+f))^(1/f)
    // es monótona DECRECIENTE en f (piso asintótico e⁻²≈13.5%), y su
    // "f_cap" −ln(P)/2 ≈ 1.50 jamás vinculaba — código muerto disfrazado de
    // protección. El tope VIGENTE (mismo que envelope/bootstrap/micro desde
    // esta auditoría) es el streak-bound FIX #593 + axioma 25%:
    //   f_cap = 1 − SURVIVAL_FLOOR^(1/streak(q)),  streak = ln(200)/ln(q)
    // Vive DESPUÉS del clamp del genoma: ningún multiplicador (streak,
    // neural, régimen) puede empujar por encima.
    let kelly_prelim = (kelly * capital_scale).clamp(mn, mx);
    crate::ruin::clamp_ruin(kelly_prelim, 1.0 - win_rate.clamp(0.0, 1.0))
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
        // S-1: neutral espectral por defecto en los tests legados.
        calculate_kelly_fraction(wr, pf, cap, base, surv, exp, cmin, cmax, sbase, 0.5)
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
        assert!(
            high <= cmin * 0.25 + 1e-12,
            "exploración acotada a quarter del piso"
        );
    }

    #[test]
    fn test_r14_low_wr_with_positive_pf_flows_to_exact_kelly() {
        // ANTES: WR<0.05 con PF>1 pisaba el Kelly legítimo con ~0.
        let (base, surv, exp, cmin, cmax, sbase) = base_args();
        let f = call(0.03, 2.0, 100.0, base, surv, exp, cmin, cmax, sbase);
        // El resultado ya no es la rampa de exploración: con PF>1 fluye a la
        // fórmula exacta y su piso clamp_min.
        assert!(
            f >= cmin - 1e-12,
            "Kelly exacto con piso genómico, no rampa"
        );
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
        assert!(
            high_base > low_base,
            "strategy_base_fraction revive como coeficiente"
        );
    }

    /// S-1: la banda de capital responde a la confianza espectral — mercado
    /// persistente (s→1) sostiene más Kelly que mercado anti-persistente
    /// (s→0), con el mismo edge medido. Caso: expansión plena (ratio 1000×,
    /// capital_scale en el riel superior de cada banda).
    /// CERT-M5-H03: edge moderado (wr=0.55, pf=1.3) para que el preliminar
    /// quede DEBAJO del tope de ruina (axioma 25%) — con edge grande
    /// (wr=0.7/pf=2.0) el tope vinculaba en ambos regímenes y la modulación
    /// quedaba plana por diseño del AXIOMA, no por bug.
    #[test]
    fn test_s1_spectral_band_modulates_kelly() {
        let (base, surv, exp, cmin, cmax, sbase) = base_args();
        let k_persist = calculate_kelly_fraction(
            0.55, 1.3, 13_000.0, base, surv, exp, cmin, cmax, sbase, 1.0,
        );
        let k_anti = calculate_kelly_fraction(
            0.55, 1.3, 13_000.0, base, surv, exp, cmin, cmax, sbase, 0.0,
        );
        assert!(
            k_persist > k_anti,
            "persistencia espectral debe ampliar el Kelly: {k_persist} vs {k_anti}"
        );
        // Y en modo supervivencia (capital bajo) la misma orden.
        let s_persist = calculate_kelly_fraction(
            0.55, 1.3, 13.0, base, surv, exp, cmin, cmax, sbase, 1.0,
        );
        let s_anti =
            calculate_kelly_fraction(0.55, 1.3, 13.0, base, surv, exp, cmin, cmax, sbase, 0.0);
        assert!(s_persist >= s_anti);
    }
}
