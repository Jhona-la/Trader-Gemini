/// Legacy weighted-log basket and event-index mean-reversion proxy (#83-#95).
/// Supplied weights are assumed, not estimated cointegration vectors. Neither
/// stationarity nor a physical-time OU model is certified by this helper.
use crate::types::{SignalIntent, SignalType, TradeHorizon};

const MAX_ASSETS: usize = 4;

#[derive(Debug, Clone)]
pub struct MultivariateCointegrationEngine {
    pub weights: [f64; MAX_ASSETS],
    pub mean_spread: f64,
    pub var_spread: f64,
    pub theta_reversion_speed: f64,
    pub count: usize,
    pub z_score_threshold: f64,
    pub memory_decay: f64,
    last_spread: f64,
}

impl MultivariateCointegrationEngine {
    pub fn new(weights: [f64; MAX_ASSETS], z_score_threshold: f64) -> Self {
        Self::new_with_decay(weights, z_score_threshold, 0.98)
    }

    /// Constructor con factor de decaimiento de memoria temporal configurable (Punto #110)
    pub fn new_with_decay(
        weights: [f64; MAX_ASSETS],
        z_score_threshold: f64,
        memory_decay: f64,
    ) -> Self {
        let mut safe_weights = [0.0; MAX_ASSETS];
        for i in 0..MAX_ASSETS {
            safe_weights[i] = if weights[i].is_finite() {
                weights[i]
            } else {
                0.25
            };
        }
        Self {
            weights: safe_weights,
            mean_spread: 0.0,
            var_spread: 1.0,
            theta_reversion_speed: 0.1,
            count: 0,
            z_score_threshold: if z_score_threshold.is_finite() && z_score_threshold > 0.5 {
                z_score_threshold
            } else {
                2.0
            },
            memory_decay: if memory_decay.is_finite() {
                memory_decay.clamp(0.80, 0.999)
            } else {
                0.98
            },
            last_spread: 0.0,
        }
    }

    /// Updates the legacy event-index estimator. The timestamp is not yet used
    /// for a physical-time OU fit. Rejected observations do not mutate state.
    pub fn update_and_evaluate(
        &mut self,
        prices: &[f64; MAX_ASSETS],
        _timestamp_ms: u64,
    ) -> Option<SignalIntent> {
        // Public fields can be changed by callers; do not extend invalid state.
        if !self.mean_spread.is_finite()
            || !self.last_spread.is_finite()
            || !self.var_spread.is_finite()
            || self.var_spread < 0.0
            || !self.theta_reversion_speed.is_finite()
            || self.theta_reversion_speed <= 0.0
            || !self.z_score_threshold.is_finite()
            || self.z_score_threshold <= 0.0
            || !self.memory_decay.is_finite()
            || !(0.0..=1.0).contains(&self.memory_decay)
        {
            return None;
        }
        // 1. Calcular el spread sintético $S_t = \sum w_i \ln(P_i)$
        let mut spread = 0.0;
        for i in 0..MAX_ASSETS {
            if prices[i] <= 0.0 || !prices[i].is_finite() || !self.weights[i].is_finite() {
                return None;
            }
            spread += self.weights[i] * prices[i].ln();
        }

        if !spread.is_finite() {
            return None;
        }
        let next_count = self.count.checked_add(1)?;

        // 2. Legacy hybrid mean/EW variance update, not unbiased sample Welford.
        if next_count == 1 {
            self.mean_spread = spread;
            self.var_spread = 1e-4;
            self.last_spread = spread;
            self.count = next_count;
            return None;
        }

        let diff_spread = spread - self.last_spread;
        if !diff_spread.is_finite() {
            return None;
        }
        // Legacy jump policy in weighted-log units, not a percentage return.
        // A legitimate permanent level shift can still freeze this estimator.
        let max_jump = 10.0 * self.var_spread.sqrt();
        if next_count > 10 && diff_spread.abs() > max_jump.max(1.5) {
            return None; // Rejection alone cannot establish that a tick is corrupt.
        }

        let delta = spread - self.mean_spread;
        let next_mean = self.mean_spread + delta / (next_count as f64).min(500.0);
        let delta2 = spread - next_mean;
        let innovation_product = delta * delta2;
        if !delta.is_finite() || !next_mean.is_finite() || !innovation_product.is_finite() {
            return None;
        }
        let next_var = (self.memory_decay * self.var_spread
            + (1.0 - self.memory_decay) * innovation_product.max(0.0))
        .max(1e-6);

        // 3. Estimación discreta de velocidad de reversión Ornstein-Uhlenbeck: $\Delta S_t = -\theta (S_{t-1} - \mu) + \epsilon_t$
        let spread_deviation = self.last_spread - next_mean;
        let mut next_theta = self.theta_reversion_speed;
        if spread_deviation.abs() > 1e-6 {
            let ratio = -diff_spread / spread_deviation;
            if !spread_deviation.is_finite() || !ratio.is_finite() {
                return None;
            }
            let instantaneous_theta = ratio.clamp(0.01, 2.0);
            next_theta = 0.95 * self.theta_reversion_speed + 0.05 * instantaneous_theta;
        }

        // 4. Calcular Z-Score del spread
        let std_dev = next_var.sqrt();
        let z_score = delta2 / std_dev;
        if !next_var.is_finite() || !next_theta.is_finite() || !z_score.is_finite() {
            return None;
        }
        // Commit all accepted-observation statistics together. This guards the
        // numerical contract, not stationarity or the validity of the OU model.
        self.count = next_count;
        self.mean_spread = next_mean;
        self.var_spread = next_var;
        self.theta_reversion_speed = next_theta;
        self.last_spread = spread;

        // 5. Vida media de reversión: $t_{1/2} = \frac{\ln(2)}{\theta}$
        let half_life_periods = (2.0_f64.ln() / self.theta_reversion_speed).clamp(1.0, 100.0);

        // Si el spread se desvía más allá del umbral y la vida media es razonable (< 50 periodos)
        if half_life_periods <= 50.0 {
            if z_score < -self.z_score_threshold {
                // Spread subvaluado -> Comprar el cluster (Long)
                let confidence = (-z_score / 3.0).clamp(0.5, 0.99);
                return Some(SignalIntent {
                    signal: SignalType::Long,
                    confidence,
                    horizon: TradeHorizon::Continuous,
                    expected_magnitude: 0.015,
                    ..Default::default()
                });
            } else if z_score > self.z_score_threshold {
                // Spread sobrevaluado -> Vender el cluster (Short)
                let confidence = (z_score / 3.0).clamp(0.5, 0.99);
                return Some(SignalIntent {
                    signal: SignalType::Short,
                    confidence,
                    horizon: TradeHorizon::Continuous,
                    expected_magnitude: 0.015,
                    ..Default::default()
                });
            }
        }

        None
    }

    /// Retorna la vida media de reversión calculada
    #[inline(always)]
    pub fn half_life(&self) -> f64 {
        if self.theta_reversion_speed.is_finite() && self.theta_reversion_speed > 0.0 {
            2.0_f64.ln() / self.theta_reversion_speed
        } else {
            10.0
        }
    }

    #[inline(always)]
    pub fn get_half_life(&self) -> f64 {
        self.half_life()
    }

    /// Retorna los pesos normalizados del cluster para cada pata del arbitraje (Long/Short por pata)
    #[inline(always)]
    pub fn get_basket_allocations(&self, signal_type: SignalType) -> [f64; MAX_ASSETS] {
        let sign = match signal_type {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            SignalType::Flat => 0.0,
        };
        let mut alloc = [0.0; MAX_ASSETS];
        if sign == 0.0 || self.weights.iter().any(|w| !w.is_finite()) {
            return alloc;
        }
        let scale = self.weights.iter().map(|w| w.abs()).fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return alloc;
        }
        // L1 normalization in scaled coordinates avoids overflow and an
        // arbitrary absolute floor for very small but nonzero baskets.
        let norm: f64 = self.weights.iter().map(|w| (w / scale).abs()).sum();
        for i in 0..MAX_ASSETS {
            alloc[i] = sign * ((self.weights[i] / scale) / norm);
        }
        alloc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multivariate_cointegration_ou_reversion() {
        let weights = [1.0, -0.5, -0.3, -0.2];
        let mut engine = MultivariateCointegrationEngine::new(weights, 2.0);

        // Alimentar precios estacionarios sintéticos
        let base_prices = [100.0, 50.0, 30.0, 20.0];
        for i in 0..100 {
            let noise = ((i % 5) as f64 - 2.0) * 0.1;
            let prices = [
                base_prices[0] + noise,
                base_prices[1] + noise * 0.5,
                base_prices[2] + noise * 0.3,
                base_prices[3] + noise * 0.2,
            ];
            let _ = engine.update_and_evaluate(&prices, 1000 + i * 1000);
        }

        assert!(engine.var_spread > 0.0);
        assert!(engine.half_life() > 0.0);

        // Forzar shock negativo extremo en el activo principal (spread subvaluado)
        let shock_prices = [70.0, 50.0, 30.0, 20.0];
        let signal = engine.update_and_evaluate(&shock_prices, 200000);

        assert!(signal.is_some());
        let intent = signal.unwrap();
        assert_eq!(intent.signal, SignalType::Long);
        assert_eq!(intent.horizon, TradeHorizon::Continuous);
    }

    #[test]
    fn test_multivariate_cointegration_custom_decay() {
        let weights = [1.0, -0.5, -0.3, -0.2];
        let mut engine = MultivariateCointegrationEngine::new_with_decay(weights, 2.0, 0.95);
        assert_eq!(engine.memory_decay, 0.95);

        let base_prices = [100.0, 50.0, 30.0, 20.0];
        let _ = engine.update_and_evaluate(&base_prices, 1000);
        let _ = engine.update_and_evaluate(&base_prices, 2000);
        assert!(engine.var_spread >= 1e-6);
    }

    #[test]
    fn test_multivariate_cointegration_nan_immunity() {
        let weights = [1.0, -0.5, -0.3, -0.2];
        let mut engine = MultivariateCointegrationEngine::new(weights, 2.0);

        let corrupt_prices = [100.0, f64::NAN, 30.0, 20.0];
        let sig = engine.update_and_evaluate(&corrupt_prices, 1000);
        assert!(sig.is_none());

        let zero_prices = [100.0, 0.0, 30.0, 20.0];
        let sig2 = engine.update_and_evaluate(&zero_prices, 2000);
        assert!(sig2.is_none());
    }

    #[test]
    fn test_multivariate_cointegration_basket_allocations() {
        let weights = [1.0, -0.5, -0.3, -0.2];
        let engine = MultivariateCointegrationEngine::new(weights, 2.0);

        let long_allocs = engine.get_basket_allocations(SignalType::Long);
        assert!((long_allocs[0] - 0.5).abs() < 1e-4);
        assert!((long_allocs[1] - (-0.25)).abs() < 1e-4);

        let short_allocs = engine.get_basket_allocations(SignalType::Short);
        assert!((short_allocs[0] - (-0.5)).abs() < 1e-4);

        let flat_allocs = engine.get_basket_allocations(SignalType::Flat);
        assert_eq!(flat_allocs, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_multivariate_cointegration_short_signal_on_positive_spread_shock() {
        let weights = [1.0, -0.5, -0.3, -0.2];
        let mut engine = MultivariateCointegrationEngine::new(weights, 2.0);

        let base_prices = [100.0, 50.0, 30.0, 20.0];
        for i in 0..100 {
            let noise = ((i % 5) as f64 - 2.0) * 0.1;
            let prices = [
                base_prices[0] + noise,
                base_prices[1] + noise * 0.5,
                base_prices[2] + noise * 0.3,
                base_prices[3] + noise * 0.2,
            ];
            let _ = engine.update_and_evaluate(&prices, 1000 + i * 1000);
        }

        // Forzar shock positivo extremo en el activo principal (spread sobrevaluado -> Short)
        let shock_prices = [150.0, 50.0, 30.0, 20.0];
        let signal = engine.update_and_evaluate(&shock_prices, 300000);

        assert!(signal.is_some());
        let intent = signal.unwrap();
        assert_eq!(intent.signal, SignalType::Short);
        assert_eq!(intent.horizon, TradeHorizon::Continuous);
    }
}
