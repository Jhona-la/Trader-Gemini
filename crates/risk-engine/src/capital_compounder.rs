pub struct CapitalRegimeMetrics {
    pub max_concurrent_positions: usize,
    /// Fracción continua de energía espectral asignada a escalas reactivas
    pub spectral_energy_split: f64,
    /// Alias legacy de compatibilidad
    pub scalp_capital_split: f64,
    pub wealth_factor: f64,
}

pub struct CapitalCompounderEngine;

impl CapitalCompounderEngine {
    /// Returns dynamic capital allocation rules based on continuous math models
    /// rather than hardcoded step functions.
    pub fn get_capital_regime_metrics(
        current_capital: f64,
        current_drawdown: f64,
        base_capital: f64,
    ) -> CapitalRegimeMetrics {
        // FIX #652: Sanitizar parámetros entrantes
        let safe_curr = if current_capital.is_finite() && current_capital > 0.0 {
            current_capital
        } else {
            13.0
        };
        let safe_base = if base_capital.is_finite() && base_capital > 0.0 {
            base_capital
        } else {
            13.0
        };
        let safe_dd = if current_drawdown.is_finite() && current_drawdown >= 0.0 {
            current_drawdown
        } else {
            0.0
        };

        // Continuous wealth factor (1.0 = base, expands exponentially)
        let wealth_ratio = (safe_curr / safe_base.max(1.0)).max(1.0);

        // SISTEMA SUPREMO: Escalado Concurrente Exponencial (Raíz Cuadrada)
        // Sustituimos la asfixia logarítmica (log10). Si cuadruplicamos el capital ($52), abrimos 3 posiciones.
        // Si llegamos a 81x ($1000), abrimos las 10 del Top Epigenético. Esto fuerza el crecimiento de interés compuesto.
        let concurrent_positions = (1.0 + wealth_ratio.sqrt()).floor() as usize;

        // Clamp it to reasonable bounds based on our memory/risk budget (Maximum Top 10 Epigenetic slots)
        let max_concurrent_positions = concurrent_positions.clamp(1, 10);

        // Drawdown continuously penalizes the capital split: 0% penalty at 0 drawdown (ATH)
        // High drawdown (e.g. 0.10+) smoothly suppresses scalp split towards conservative bounds
        let dd_penalty = (safe_dd / 0.10).clamp(0.0, 1.0).powi(2);
        let spectral_energy_split = (0.95 * (1.0 - 0.85 * dd_penalty)).clamp(0.1, 0.95);
        let scalp_capital_split = spectral_energy_split;

        CapitalRegimeMetrics {
            max_concurrent_positions,
            spectral_energy_split,
            scalp_capital_split,
            wealth_factor: wealth_ratio,
        }
    }

    /// Calculates the Kelly-optimal notional position size without human biases
    pub fn calculate_compounding_position_notional(
        capital_bucket: f64,
        win_rate: f64,
        profit_factor: f64,
        confidence: f64,
        hurst_exponent: f64,
        current_drawdown: f64,
        consecutive_wins: u64,
        consecutive_losses: u64,
        correlation_penalty: f64,
        _wealth_factor: f64,
        kelly_clamp_min: f64,
        kelly_clamp_max: f64,
    ) -> f64 {
        if !capital_bucket.is_finite()
            || capital_bucket <= 0.0
            || !win_rate.is_finite()
            || !profit_factor.is_finite()
        {
            return 0.0;
        }

        // FIX #555: Fórmula analítica exacta de Kelly basada en Profit Factor: f* = W * (1 - 1/PF)
        let pf = profit_factor.max(1.01);
        let kelly = (win_rate * (1.0 - 1.0 / pf)).max(0.0);
        if kelly <= 0.0 || win_rate < 0.40 {
            return 0.0;
        }

        // Half Kelly for volatility safety
        let mut target_fraction = (kelly * 0.5).max(0.005);

        // Scale continuously with ML confidence (0.0 to 1.0)
        // Scale continuously with ML confidence (0.0 to 1.0)
        // FIX #1430: Sanitización de confianza, Hurst y penalización de correlación
        let safe_conf = if confidence.is_finite() {
            confidence.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let confidence_scalar = safe_conf.max(0.5);
        target_fraction *= confidence_scalar;

        // Hurst Exponent continuous penalty
        // > 0.5 is trending (good for swing/scalp continuation), < 0.5 is mean reverting
        // We apply a smooth polynomial scalar based on Hurst (if Hurst ~ 0.5 it's random, we reduce size)
        let safe_hurst = if hurst_exponent.is_finite() {
            hurst_exponent.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let hurst_scalar = (2.0 * (safe_hurst - 0.5).abs()).powf(1.5).clamp(0.5, 1.0);
        target_fraction *= hurst_scalar;

        // Consecutive streak multiplier (Momentum scaling)
        // Exponential decay on losses to protect capital dynamically
        let streak_scalar = if consecutive_losses > 0 {
            1.0 / (1.0 + consecutive_losses as f64 * 0.5) // Rapid decay
        } else if consecutive_wins > 0 {
            1.0 + (consecutive_wins as f64 * 0.1).min(0.5) // Small reward
        } else {
            1.0
        };
        target_fraction *= streak_scalar;

        // Correlation penalty is passed down directly (1.0 = no penalty, < 1.0 = highly correlated active positions)
        let safe_corr = if correlation_penalty.is_finite() {
            correlation_penalty.clamp(0.0, 1.0)
        } else {
            1.0
        };
        target_fraction *= safe_corr;

        // Continuous Drawdown penalty
        // As drawdown approaches 15% (0.15), size decays exponentially
        // FIX #612: Proteger contra drawdowns negativos o fluctuaciones flotantes favorables
        let safe_dd = if current_drawdown.is_finite() {
            current_drawdown.max(0.0)
        } else {
            0.0
        };
        let dd_decay = (-safe_dd * 15.0).exp().clamp(0.01, 1.0);
        target_fraction *= dd_decay;

        // Enforce the clamping requested by the arena limits (solo si hay fracción positiva)
        // FIX #703: Ordenamiento defensivo de (min, max) y validación de finitud para evitar pánicos en clamp
        if target_fraction > 0.0 {
            let min_k = if kelly_clamp_min.is_finite() {
                kelly_clamp_min.max(0.0)
            } else {
                0.01
            };
            let max_k = if kelly_clamp_max.is_finite() {
                kelly_clamp_max.max(0.0)
            } else {
                0.25
            };
            let (safe_min, safe_max) = (min_k.min(max_k), min_k.max(max_k));
            target_fraction = target_fraction.clamp(safe_min, safe_max);
        }

        capital_bucket * target_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capital_regime_metrics_continuous_expansion() {
        // Base $13.0 USD
        let base_m = CapitalCompounderEngine::get_capital_regime_metrics(13.0, 0.0, 13.0);
        assert_eq!(base_m.max_concurrent_positions, 2);
        assert!((base_m.scalp_capital_split - 0.95).abs() < 1e-4);
        assert!((base_m.wealth_factor - 1.0).abs() < 1e-4);

        // Quadruple $52.0 USD (wealth ratio = 4, sqrt = 2 -> 3 concurrent positions)
        let quad_m = CapitalCompounderEngine::get_capital_regime_metrics(52.0, 0.0, 13.0);
        assert_eq!(quad_m.max_concurrent_positions, 3);

        // $1000 USD (wealth ratio ~ 76.9, sqrt ~ 8.7 -> 9-10 concurrent positions)
        let big_m = CapitalCompounderEngine::get_capital_regime_metrics(1000.0, 0.0, 13.0);
        assert!(big_m.max_concurrent_positions >= 9);
    }

    #[test]
    fn test_capital_compounder_calculate_position_notional_and_nan_immunity() {
        let size = CapitalCompounderEngine::calculate_compounding_position_notional(
            13.0, 0.65, // 65% WR
            1.8,  // 1.8 PF
            0.85, // confidence
            0.65, // Hurst
            0.0,  // DD
            2,    // wins streak
            0,    // loss streak
            1.0,  // corr penalty
            1.0,  // wealth
            0.01, 0.25,
        );
        assert!(size > 0.0 && size <= 13.0 * 0.25);

        // NaN immunity
        let nan_size = CapitalCompounderEngine::calculate_compounding_position_notional(
            f64::NAN,
            0.65,
            1.8,
            0.85,
            0.65,
            0.0,
            0,
            0,
            1.0,
            1.0,
            0.01,
            0.25,
        );
        assert_eq!(nan_size, 0.0);
    }
}
