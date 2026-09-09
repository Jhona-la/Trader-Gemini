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
    if !current_capital.is_finite() || current_capital <= 0.0 {
        return false; // Total capital loss or corruption -> block
    }
    if !peak_capital.is_finite() || peak_capital <= 0.0 {
        return true; // No hay histórico, se asume seguro inicial
    }

    let current_drawdown = (peak_capital - current_capital) / peak_capital;

    // Continuous Sigmoid: Drawdown limit scales smoothly with capital size relative to base.
    // Small capital (<= $50) → high tolerance (0.75) allowing exponential bootstrap compounding without premature freeze
    // Large capital (> $50) → strict institutional limit approaching genome_max_drawdown_pct (0.02-0.05)
    let capital_ratio = (current_capital / base_capital.max(1.0)).max(0.0);
    let raw_input = guard_dd_sigmoid_steepness * (capital_ratio - guard_dd_sigmoid_center);
    let sigmoid_input = if raw_input.is_finite() {
        raw_input.clamp(-20.0, 20.0)
    } else {
        0.0
    };
    let sigmoid_val = (1.0 / (1.0 + sigmoid_input.exp())).clamp(0.0, 1.0);
    let dynamic_max_drawdown = if current_capital <= 50.0 {
        0.75_f64.max(genome_max_drawdown_pct)
    } else {
        (genome_max_drawdown_pct + (0.85 - genome_max_drawdown_pct) * sigmoid_val).clamp(0.02, 0.90)
    };

    // Si el drawdown actual es mayor o igual al límite duro, BLOQUEAR
    current_drawdown < dynamic_max_drawdown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_drawdown_limit_immunity_to_overflow() {
        // Test with extreme parameters
        let res = check_drawdown_limit(13.0, 13.0, 0.05, 13.0, 1000.0, 5.0);
        assert!(res, "Zero drawdown should always be allowed");

        // Extreme large capital
        let res2 = check_drawdown_limit(1_000_000.0, 1_000_000.0, 0.02, 13.0, 100.0, 2.0);
        assert!(res2);

        // Heavy drawdown exceeding limit
        let res3 = check_drawdown_limit(1.0, 100.0, 0.05, 13.0, 1.0, 1.0);
        assert!(!res3, "99% drawdown must be blocked");

        // NaN and zero/negative capital must be blocked
        assert!(!check_drawdown_limit(f64::NAN, 100.0, 0.05, 13.0, 1.0, 1.0));
        assert!(!check_drawdown_limit(-5.0, 100.0, 0.05, 13.0, 1.0, 1.0));
        assert!(!check_drawdown_limit(0.0, 100.0, 0.05, 13.0, 1.0, 1.0));
    }

    #[test]
    fn test_enforce_minimum_notional() {
        let (ok, _) = enforce_minimum_notional(1.0, 5.0, 10.0); // 1.0 * 10 = $10 >= $5
        assert!(ok);

        let (fail, _) = enforce_minimum_notional(0.2, 5.0, 5.0); // 0.2 * 5 = $1 < $5
        assert!(!fail);

        let (nan_fail, _) = enforce_minimum_notional(f64::NAN, 5.0, 10.0);
        assert!(!nan_fail);
    }

    #[test]
    fn test_streak_drawdown_limit() {
        assert!(check_streak_drawdown_limit(0, 3));
        assert!(check_streak_drawdown_limit(2, 3));
        assert!(!check_streak_drawdown_limit(3, 3));
        assert!(!check_streak_drawdown_limit(5, 3));
    }
}

#[inline(always)]
pub fn enforce_minimum_notional(
    intended_volume: f64,
    min_notional: f64,
    available_leverage: f64,
) -> (bool, f64) {
    if !intended_volume.is_finite()
        || intended_volume <= 0.0
        || !available_leverage.is_finite()
        || available_leverage <= 0.0
    {
        return (false, 0.0);
    }
    // Binance exige un nominal mínimo de $5.00 USD por orden
    let nominal_value = intended_volume * available_leverage;

    if nominal_value < min_notional {
        // No alcanza, rechazamos (o podríamos forzarlo al mínimo, pero forzar apalancamiento aumenta riesgo de ruina)
        (false, 0.0)
    } else {
        (true, intended_volume)
    }
}

/// Protege la cuenta ante rachas de pérdidas consecutivas (Punto #211)
#[inline(always)]
pub fn check_streak_drawdown_limit(consecutive_losses: u32, max_allowed_streak: u32) -> bool {
    let allowed = max_allowed_streak.max(2);
    consecutive_losses < allowed
}
