/// 🌌 MOTOR DE COINTEGRACIÓN MULTIVARIANTE Y ORNSTEIN-UHLENBECK (MULTIVARIATE COINTEGRATION & OU ENGINE)
/// Modela clusters de altcoins cointegradas con estimación de vida media de reversión $t_{1/2} = \frac{\ln(2)}{\theta}$ (#83-#95).
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

    /// Actualiza el estado con un nuevo vector de precios de los activos del cluster
    pub fn update_and_evaluate(
        &mut self,
        prices: &[f64; MAX_ASSETS],
        _timestamp_ms: u64,
    ) -> Option<SignalIntent> {
        // 1. Calcular el spread sintético $S_t = \sum w_i \ln(P_i)$
        let mut spread = 0.0;
        for i in 0..MAX_ASSETS {
            if prices[i] <= 0.0 || !prices[i].is_finite() {
                return None;
            }
            spread += self.weights[i] * prices[i].ln();
        }

        self.count += 1;

        // 2. Actualización online de Welford para media y varianza del spread
        if self.count == 1 {
            self.mean_spread = spread;
            self.var_spread = 1e-4;
            self.last_spread = spread;
            return None;
        }

        let diff_spread = spread - self.last_spread;
        // FIX #577: Rechazar saltos espurios causados por datos corruptos (>150% de movimiento logarítmico en 1 tick)
        let max_jump = 10.0 * self.var_spread.sqrt();
        if self.count > 10 && diff_spread.abs() > max_jump.max(1.5) {
            return None; // Outlier temporal de red: rechazar para no corromper la distribución OU
        }

        let delta = spread - self.mean_spread;
        self.mean_spread += delta / (self.count as f64).min(500.0);
        let delta2 = spread - self.mean_spread;
        self.var_spread = (self.memory_decay * self.var_spread
            + (1.0 - self.memory_decay) * (delta * delta2).max(0.0))
        .max(1e-6);

        // 3. Estimación discreta de velocidad de reversión Ornstein-Uhlenbeck: $\Delta S_t = -\theta (S_{t-1} - \mu) + \epsilon_t$
        let spread_deviation = self.last_spread - self.mean_spread;
        if spread_deviation.abs() > 1e-6 {
            let instantaneous_theta = (-diff_spread / spread_deviation).clamp(0.01, 2.0);
            self.theta_reversion_speed =
                0.95 * self.theta_reversion_speed + 0.05 * instantaneous_theta;
        }
        self.last_spread = spread;

        // 4. Calcular Z-Score del spread
        let std_dev = self.var_spread.sqrt();
        let z_score = (spread - self.mean_spread) / std_dev;

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
        let sum_abs: f64 = self.weights.iter().map(|w| w.abs()).sum();
        let norm = if sum_abs > 1e-6 { sum_abs } else { 1.0 };
        for i in 0..MAX_ASSETS {
            alloc[i] = sign * (self.weights[i] / norm);
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
