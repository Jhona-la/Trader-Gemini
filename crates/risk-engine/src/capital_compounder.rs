pub struct CapitalRegimeMetrics {
    pub max_concurrent_positions: usize,
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
        base_capital: f64
    ) -> CapitalRegimeMetrics {
        // Continuous wealth factor (1.0 = base, expands exponentially)
        let wealth_ratio = (current_capital / base_capital.max(1.0)).max(1.0);
        
        // SISTEMA SUPREMO: Escalado Concurrente Exponencial (Raíz Cuadrada)
        // Sustituimos la asfixia logarítmica (log10). Si cuadruplicamos el capital ($52), abrimos 3 posiciones.
        // Si llegamos a 81x ($1000), abrimos las 10 del Top Epigenético. Esto fuerza el crecimiento de interés compuesto.
        let concurrent_positions = (1.0 + wealth_ratio.sqrt()).floor() as usize;
        
        // Clamp it to reasonable bounds based on our memory/risk budget (Maximum Top 10 Epigenetic slots)
        let max_concurrent_positions = concurrent_positions.clamp(1, 10);
        
        // Drawdown continuously penalizes the capital split (Sigmoid curve)
        // High drawdown (e.g. 0.15) heavily suppresses scalp split towards 0.
        // A drawdown of 0 maintains it near 0.95.
        // We use a generalized logistic function
        let dd_penalty = 1.0 / (1.0 + (-20.0 * (current_drawdown - 0.05)).exp()); // Shifted sigmoid
        let scalp_capital_split = (0.95 * (1.0 - dd_penalty)).clamp(0.1, 0.95);
        
        CapitalRegimeMetrics {
            max_concurrent_positions,
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
        // Continuous Half-Kelly Formula based on live metrics
        // Kelly % = W - ( (1-W) / R ) where R is reward/risk (profit factor approx)
        let kelly = win_rate - ((1.0 - win_rate) / profit_factor.max(0.1));
        
        // Half Kelly for volatility safety
        let mut target_fraction = (kelly * 0.5).max(0.01);
        
        // Scale continuously with ML confidence (0.0 to 1.0)
        let confidence_scalar = confidence.max(0.5); 
        target_fraction *= confidence_scalar;
        
        // Hurst Exponent continuous penalty 
        // > 0.5 is trending (good for swing/scalp continuation), < 0.5 is mean reverting
        // We apply a smooth polynomial scalar based on Hurst (if Hurst ~ 0.5 it's random, we reduce size)
        let hurst_scalar = (2.0 * (hurst_exponent - 0.5).abs()).powf(1.5).clamp(0.5, 1.0);
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
        target_fraction *= correlation_penalty;
        
        // Continuous Drawdown penalty
        // As drawdown approaches 15% (0.15), size decays exponentially
        let dd_decay = (-current_drawdown * 15.0).exp(); 
        target_fraction *= dd_decay;
        
        // Enforce the clamping requested by the arena limits
        target_fraction = target_fraction.clamp(kelly_clamp_min, kelly_clamp_max);
        
        capital_bucket * target_fraction
    }
}
