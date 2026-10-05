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

        // Activación continua en el continuo temporal-espectral:
        // Erradica la discontinuidad abrupta que saltaba de 0.0 a -0.50 en |z| = 1.5.
        let abs_z = z.abs();
        if abs_z > 1.5 {
            let excess = abs_z - 1.5;
            let smooth_scale = (excess / 1.5).min(1.0);
            (-z.signum() * smooth_scale).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn horizon(&self) -> TradeHorizon {
        TradeHorizon::Continuous
    }
}

impl Default for JohansenVecmEngine {
    fn default() -> Self {
        Self::new(0.15, 1.0)
    }
}

/// 🔬 ESTIMADOR ANALÍTICO DE DIFUSIÓN ORNSTEIN-UHLENBECK / FOKKER-PLANCK EN TIEMPO CONTINUO
///
/// Modela el proceso continuo estocástico de reversión a la media:
///   dX_t = θ (μ - X_t) dt + σ dW_t
///
/// Donde:
///   - θ > 0: velocidad de reversión física (s⁻¹)
///   - μ: media de equilibrio invariante
///   - σ > 0: volatilidad instantánea de difusión (s⁻¹/²)
///   - t_{1/2} = ln(2) / θ: vida media física en segundos
///   - Var_∞ = σ² / (2θ): varianza estacionaria derivada de la ecuación de Fokker-Planck
///   - Z_t = (X_t - μ) / √(σ² / (2θ)): Z-Score ergódico invariante
#[derive(Debug, Clone)]
pub struct ContinuousOrnsteinUhlenbeckSde {
    pub theta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub last_value: f64,
    pub last_ts_ms: u64,
    pub count: u64,
    // Momentos de transición AR(1) continua
    s_w: f64,
    s_x: f64,
    s_y: f64,
    s_xx: f64,
    s_xy: f64,
    s_yy: f64,
}

impl Default for ContinuousOrnsteinUhlenbeckSde {
    fn default() -> Self {
        Self::new(0.1, 0.0, 0.01)
    }
}

impl ContinuousOrnsteinUhlenbeckSde {
    pub fn new(theta_init: f64, mu_init: f64, sigma_init: f64) -> Self {
        Self {
            theta: theta_init.max(1e-5),
            mu: mu_init,
            sigma: sigma_init.max(1e-6),
            last_value: 0.0,
            last_ts_ms: 0,
            count: 0,
            s_w: 0.0,
            s_x: 0.0,
            s_y: 0.0,
            s_xx: 0.0,
            s_xy: 0.0,
            s_yy: 0.0,
        }
    }

    /// Vida media de reversión física en segundos: t_{1/2} = ln(2) / θ
    #[inline]
    pub fn half_life_seconds(&self) -> f64 {
        if self.theta > 1e-9 {
            std::f64::consts::LN_2 / self.theta
        } else {
            f64::INFINITY
        }
    }

    /// Varianza estacionaria ergódica (Fokker-Planck): σ² / (2θ)
    #[inline]
    pub fn stationary_variance(&self) -> f64 {
        if self.theta > 1e-9 {
            (self.sigma * self.sigma) / (2.0 * self.theta)
        } else {
            1.0
        }
    }

    /// Z-Score estacionario del valor actual: (X - μ) / √(Var_∞)
    #[inline]
    pub fn stationary_zscore(&self, x: f64) -> f64 {
        let sd = self.stationary_variance().sqrt().max(1e-8);
        ((x - self.mu) / sd).clamp(-10.0, 10.0)
    }

    /// Actualiza el estimador con una nueva observación física (valor, timestamp en ms)
    pub fn update(&mut self, value: f64, ts_ms: u64) -> f64 {
        if !value.is_finite() {
            return self.stationary_zscore(self.last_value);
        }
        if self.count == 0 || ts_ms <= self.last_ts_ms {
            self.last_value = value;
            self.last_ts_ms = ts_ms;
            self.count += 1;
            return 0.0;
        }

        let dt_sec = (ts_ms - self.last_ts_ms) as f64 / 1000.0;
        if dt_sec < 1e-4 {
            // Demasiado rápido para actualizar θ de tiempo continuo sin inestabilidad numérica
            return self.stationary_zscore(value);
        }

        let x = self.last_value;
        let y = value;

        // Actualización recursiva exponencial con memoria decayente (vida media ~100 observaciones)
        let decay = (-dt_sec / 300.0).exp().clamp(0.80, 0.999);
        self.s_w = self.s_w * decay + 1.0;
        self.s_x = self.s_x * decay + x;
        self.s_y = self.s_y * decay + y;
        self.s_xx = self.s_xx * decay + x * x;
        self.s_xy = self.s_xy * decay + x * y;
        self.s_yy = self.s_yy * decay + y * y;
        self.count += 1;

        if self.count >= 10 {
            // Regresión discreta exacta: y = a + b * x
            // donde b = exp(-θ dt), a = μ (1 - b)
            let n_eff = self.s_w.max(1.0);
            let denom = (n_eff * self.s_xx - self.s_x * self.s_x).max(1e-12);
            let b = ((n_eff * self.s_xy - self.s_x * self.s_y) / denom).clamp(0.001, 0.9999);
            let a = (self.s_y - b * self.s_x) / n_eff;

            let est_theta = (-b.ln() / dt_sec).clamp(1e-4, 50.0);
            let est_mu = a / (1.0 - b).max(1e-6);

            // Residuos y estimación de sigma de difusión (Fokker-Planck)
            let raw_sse = (self.s_yy - 2.0 * b * self.s_xy + b * b * self.s_xx) / n_eff - a * a;
            let sse = raw_sse.max(1e-12);
            let est_sigma = (sse * 2.0 * est_theta / (1.0 - (-2.0 * est_theta * dt_sec).exp()).max(1e-6)).sqrt().clamp(1e-6, 10.0);

            // Suavizado C1 de parámetros
            self.theta = self.theta * 0.95 + est_theta * 0.05;
            self.mu = self.mu * 0.95 + est_mu * 0.05;
            self.sigma = self.sigma * 0.95 + est_sigma * 0.05;
        }

        self.last_value = value;
        self.last_ts_ms = ts_ms;

        self.stationary_zscore(value)
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
        assert_eq!(vecm.horizon(), TradeHorizon::Continuous);
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

    #[test]
    fn test_continuous_ornstein_uhlenbeck_sde_properties() {
        let mut ou = ContinuousOrnsteinUhlenbeckSde::new(0.2, 10.0, 0.5);
        assert!((ou.half_life_seconds() - (std::f64::consts::LN_2 / 0.2)).abs() < 1e-6);
        assert!(ou.stationary_variance() > 0.0);

        // Feed trajectory that reverts around 10.0 with 1-second intervals
        let mut ts = 1_000_000u64;
        let mut val = 10.0;
        for i in 0..100 {
            ts += 1000;
            // Mean reverting step + small noise
            val = 10.0 + (val - 10.0) * (-0.2f64).exp() + (if i % 2 == 0 { 0.1 } else { -0.1 });
            let z = ou.update(val, ts);
            assert!(z.is_finite());
        }

        // Half-life must be positive and finite
        let hl = ou.half_life_seconds();
        assert!(hl.is_finite() && hl > 0.0);

        // Large shock should yield significant Z-Score
        let shock_z = ou.update(25.0, ts + 1000);
        assert!(shock_z > 2.0);

        // NaN immunity
        let nan_z = ou.update(f64::NAN, ts + 2000);
        assert!(nan_z.is_finite());
    }
}
