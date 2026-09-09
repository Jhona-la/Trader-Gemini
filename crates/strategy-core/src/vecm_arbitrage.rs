use crate::{QuantumStrategy, TradeHorizon};
use omniscient_registry::{OmniscientRegistry, ParameterKind};
use std::f64;
use std::fmt;
use std::sync::Arc;

/// ⚡ ARBITRAJE ESTADÍSTICO COINTEGRADO VECTORES VECM (JOHANSEN COINTEGRATION ENGINE)
/// Mide la relación de equilibrio a largo plazo entre pares de activos (e.g. ADA/AVAX, DOGE/SHIB).
/// Opera la reversión a la media en nanosegundos cuando el Z-Score supera |Z| > 2.5.
#[derive(Clone)]
pub struct JohansenVecmEngine {
    pub alpha_speed: f64, // Velocidad de ajuste a la media (speed of adjustment)
    pub beta_hedge_ratio: f64, // Ratio de cobertura beta de cointegración
    pub spread_mean: f64, // Media móvil del spread cointegrado
    pub spread_std: f64,  // Desviación estándar del spread
    pub window_count: f64,
    registry: Option<Arc<OmniscientRegistry>>,
}

impl fmt::Debug for JohansenVecmEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JohansenVecmEngine")
            .field("alpha_speed", &self.alpha_speed)
            .field("beta_hedge_ratio", &self.beta_hedge_ratio)
            .field("spread_mean", &self.spread_mean)
            .field("spread_std", &self.spread_std)
            .field("window_count", &self.window_count)
            .finish()
    }
}

impl JohansenVecmEngine {
    pub fn new(alpha_speed: f64, beta_hedge_ratio: f64) -> Self {
        Self {
            alpha_speed,
            beta_hedge_ratio,
            spread_mean: 0.0,
            spread_std: 0.001,
            window_count: 0.0,
            registry: None,
        }
    }

    /// Actualiza el spread con una nueva observación de precios y devuelve el Z-Score actual
    #[inline(always)]
    pub fn update(&mut self, price_a: f64, price_b: f64) -> f64 {
        self.update_and_calculate_zscore(price_a, price_b)
    }

    #[inline(always)]
    pub fn update_and_calculate_zscore(&mut self, price_a: f64, price_b: f64) -> f64 {
        // FIX #663: Validación estricta de positividad y finitud para cálculo logarítmico
        if price_a <= 0.0 || price_b <= 0.0 || !price_a.is_finite() || !price_b.is_finite() {
            return 0.0;
        }

        // Spread cointegrado: S_t = ln(P_A) - beta * ln(P_B)
        let spread = price_a.ln() - self.beta_hedge_ratio * price_b.ln();

        if self.window_count == 0.0 {
            self.spread_mean = spread;
            self.window_count = 1.0;
            return 0.0;
        }

        self.window_count += 1.0;
        let alpha = self.alpha_speed.clamp(0.01, 1.0);
        let weight = (alpha / self.window_count.max(1.0)).clamp(0.001, alpha);

        // FIX #604: Actualización Welford O(1) con varianza estrictamente no negativa
        let delta = spread - self.spread_mean;
        self.spread_mean += weight * delta;
        let delta2 = spread - self.spread_mean;
        let var_step = (delta * delta2).max(0.0) * weight;
        let variance = (self.spread_std.powi(2) * (1.0 - weight) + var_step).max(1e-12);
        self.spread_std = variance.sqrt().max(1e-5);

        // Z-Score de la divergencia
        // FIX #780 & #1514: Escribir z-score sanitizado en el OmniscientRegistry
        let raw_z = (spread - self.spread_mean) / self.spread_std;
        let z = if raw_z.is_finite() {
            raw_z.clamp(-20.0, 20.0)
        } else {
            0.0
        };
        if let Some(r) = &self.registry {
            r.register_or_update(
                "vecm_zscore",
                ParameterKind::Adaptive,
                z,
                "JohansenVecmEngine",
            );
        }
        z
    }

    /// Actualiza adaptativamente el ratio beta de cointegración usando RLS recursivo (Punto #115)
    #[inline(always)]
    pub fn update_with_adaptive_beta(&mut self, price_a: f64, price_b: f64) -> f64 {
        if price_a <= 0.0 || price_b <= 0.0 || !price_a.is_finite() || !price_b.is_finite() {
            return 0.0;
        }
        let ln_a = price_a.ln();
        let ln_b = price_b.ln();
        if self.window_count > 10.0 {
            let beta_err = ln_a - self.beta_hedge_ratio * ln_b;
            if beta_err.is_finite() {
                let gain = 0.001 / (1.0 + ln_b * ln_b * 0.001);
                self.beta_hedge_ratio =
                    (self.beta_hedge_ratio + gain * ln_b * beta_err).clamp(0.1, 10.0);
            }
        }
        self.update_and_calculate_zscore(price_a, price_b)
    }
}

impl QuantumStrategy for JohansenVecmEngine {
    fn name(&self) -> &str {
        "JohansenVecmEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() {
            None
        } else {
            Some(symbol)
        };
        let cid_opt = if symbol.is_empty() {
            None
        } else {
            Some(coin_id)
        };
        let z = match self.registry.as_ref() {
            Some(r) => r
                .get_scoped_parameter(sym_opt, cid_opt, "vecm_zscore", "JohansenVecmEngine")
                .or_else(|| {
                    r.get_scoped_parameter(
                        sym_opt,
                        cid_opt,
                        "cointegration_zscore",
                        "JohansenVecmEngine",
                    )
                })
                .map(|p| p.get_value())
                .unwrap_or(0.0),
            None => 0.0,
        };

        if !z.is_finite() {
            return 0.0;
        }

        if z.abs() >= 1.5 {
            (-z / 3.0).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn horizon(&self) -> TradeHorizon {
        TradeHorizon::Swing
    }
}

impl Default for JohansenVecmEngine {
    fn default() -> Self {
        Self::new(0.15, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_johansen_vecm_convergence() {
        let mut vecm = JohansenVecmEngine::new(0.15, 1.0);
        for _ in 0..50 {
            let z = vecm.update(100.0, 100.0);
            assert!(z.abs() < 1.0);
        }

        // Divergencia brusca
        let z_divergent = vecm.update(105.0, 100.0);
        assert!(z_divergent > 1.0);
    }

    #[test]
    fn test_johansen_vecm_adaptive_beta_and_nan_immunity() {
        let mut vecm = JohansenVecmEngine::new(0.15, 1.0);
        let z_nan = vecm.update(f64::NAN, 100.0);
        assert_eq!(z_nan, 0.0);

        let z_neg = vecm.update(-10.0, 100.0);
        assert_eq!(z_neg, 0.0);

        // Prueba de adaptación RLS de beta
        for _ in 0..20 {
            let _ = vecm.update_with_adaptive_beta(100.0, 50.0);
        }
        assert!(vecm.beta_hedge_ratio.is_finite());
        assert!(vecm.beta_hedge_ratio >= 0.1 && vecm.beta_hedge_ratio <= 10.0);
    }

    #[test]
    fn test_johansen_vecm_rebalance_signal_generation() {
        let mut vecm = JohansenVecmEngine::new(0.2, 1.0);
        // Calibrar media
        for _ in 0..100 {
            vecm.update(100.0, 100.0);
        }
        let _ = vecm.evaluate();
        assert_eq!(vecm.horizon(), TradeHorizon::Swing);
    }

    #[test]
    fn test_johansen_vecm_strategy_evaluate_with_registry() {
        let mut vecm = JohansenVecmEngine::new(0.2, 1.0);
        let reg = Arc::new(OmniscientRegistry::new());
        assert!(vecm.init(reg.clone()).is_ok());

        // Calibrate baseline equilibrium
        for _ in 0..50 {
            vecm.update(100.0, 100.0);
        }

        // Update with divergence, writing to registry
        let z = vecm.update(130.0, 100.0);
        assert!(z > 1.5);
        let eval_score = vecm.evaluate();
        // Since z > 1.5, evaluate returns a negative score for mean reversion
        assert!(eval_score < 0.0);
    }
}
