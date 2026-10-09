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
    pub tau_mem_sec: f64,
    // Momentos de regresión de tiempo continuo WLS (R6-B3, R6-B4)
    s_w: f64,
    s_dt: f64,
    s_x_dt: f64,
    s_xx_dt: f64,
    s_dx: f64,
    s_xdx: f64,
    s_dxdx_dt: f64,
}

impl Default for ContinuousOrnsteinUhlenbeckSde {
    fn default() -> Self {
        Self::new(0.1, 0.0, 0.01)
    }
}

impl ContinuousOrnsteinUhlenbeckSde {
    pub fn new(theta_init: f64, mu_init: f64, sigma_init: f64) -> Self {
        Self::with_memory_seconds(theta_init, mu_init, sigma_init, 300.0)
    }

    /// Crea un nuevo estimador SDE OU con memoria temporal calibrada en segundos (R6-B4).
    pub fn with_memory_seconds(theta_init: f64, mu_init: f64, sigma_init: f64, tau_mem_sec: f64) -> Self {
        Self {
            theta: theta_init.max(1e-5),
            mu: mu_init,
            sigma: sigma_init.max(1e-6),
            last_value: 0.0,
            last_ts_ms: 0,
            count: 0,
            tau_mem_sec: tau_mem_sec.clamp(10.0, 7200.0),
            s_w: 0.0,
            s_dt: 0.0,
            s_x_dt: 0.0,
            s_xx_dt: 0.0,
            s_dx: 0.0,
            s_xdx: 0.0,
            s_dxdx_dt: 0.0,
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
        if self.count == 0 {
            self.last_value = value;
            self.last_ts_ms = ts_ms;
            self.count = 1;
            return 0.0;
        }

        if ts_ms <= self.last_ts_ms {
            // OU-R4-02: Monotonicidad temporal estricta de la física SDE.
            // Timestamps repetidos o retrógrados no mutan el estado, no avanzan count
            // ni retroceden el reloj físico.
            return self.stationary_zscore(self.last_value);
        }

        let dt_sec = (ts_ms - self.last_ts_ms) as f64 / 1000.0;
        if dt_sec < 1e-4 {
            // Demasiado rápido para actualizar θ de tiempo continuo sin inestabilidad numérica
            return self.stationary_zscore(value);
        }

        let x = self.last_value;
        let y = value;
        let dx = y - x;

        // R6-B4: Decaimiento continuo exponencial en tiempo físico real (segundos), sin clamp espurio a eventos
        let decay = (-dt_sec / self.tau_mem_sec).exp();
        self.s_w = self.s_w * decay + 1.0;
        self.s_dt = self.s_dt * decay + dt_sec;
        self.s_x_dt = self.s_x_dt * decay + x * dt_sec;
        self.s_xx_dt = self.s_xx_dt * decay + x * x * dt_sec;
        self.s_dx = self.s_dx * decay + dx;
        self.s_xdx = self.s_xdx * decay + x * dx;
        self.s_dxdx_dt = self.s_dxdx_dt * decay + (dx * dx) / dt_sec;
        self.count += 1;

        let z_prior = self.stationary_zscore(value);

        if self.count >= 10 && self.s_dt >= 1.0 {
            // R6-B3: Regresión de tiempo continuo homoscedástica WLS estratificada para Δt heterogéneo:
            //   ΔX_i / √Δt_i = α √Δt_i - θ (X_{t_{i-1}} √Δt_i) + σ ε_i,  donde α = θ μ
            // Matriz normal:
            //   [ s_dt     -s_x_dt  ] [ α ] = [  s_dx  ]
            //   [ -s_x_dt   s_xx_dt ] [ θ ] = [ -s_xdx ]
            let denom = (self.s_dt * self.s_xx_dt - self.s_x_dt * self.s_x_dt).max(1e-12);
            let alpha_num = self.s_xx_dt * self.s_dx - self.s_x_dt * self.s_xdx;
            let theta_num = -(self.s_dt * self.s_xdx - self.s_x_dt * self.s_dx);

            let est_theta = (theta_num / denom).clamp(1e-4, 50.0);
            let empirical_mean = self.s_x_dt / self.s_dt.max(1e-6);

            let est_mu = if est_theta > 1e-3 {
                let mu_ols = alpha_num / theta_num;
                if mu_ols.is_finite() {
                    mu_ols.clamp(empirical_mean - 5.0, empirical_mean + 5.0)
                } else {
                    empirical_mean
                }
            } else {
                empirical_mean
            };

            // Estimación de volatilidad de difusión continua σ (Fokker-Planck)
            // RSS = s_dxdx_dt - α s_dx + θ s_xdx
            let est_alpha = est_theta * est_mu;
            let raw_rss = self.s_dxdx_dt - est_alpha * self.s_dx + est_theta * self.s_xdx;
            let n_eff = self.s_w.max(1.0);
            let est_sigma = (raw_rss.max(1e-12) / n_eff).sqrt().clamp(1e-6, 10.0);

            // Suavizado C1 de parámetros para garantizar trayectorias Lipschitz continuas
            self.theta = self.theta * 0.95 + est_theta * 0.05;
            self.mu = self.mu * 0.95 + est_mu * 0.05;
            self.sigma = self.sigma * 0.95 + est_sigma * 0.05;
        }

        self.last_value = value;
        self.last_ts_ms = ts_ms;

        if self.count > 10 {
            z_prior
        } else {
            self.stationary_zscore(value)
        }
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

    #[test]
    fn test_ou_r4_02_retrograde_and_duplicate_timestamp_does_not_mutate_state_or_advance_count() {
        let mut ou = ContinuousOrnsteinUhlenbeckSde::new(0.2, 10.0, 0.5);
        // Primera observación: inicializa estado
        let z0 = ou.update(10.0, 1000);
        assert_eq!(z0, 0.0);
        assert_eq!(ou.count, 1);
        assert_eq!(ou.last_ts_ms, 1000);
        assert_eq!(ou.last_value, 10.0);

        // Intento retrógrado: ts=500 < 1000
        let z_retro = ou.update(25.0, 500);
        assert!(z_retro.is_finite());
        assert_eq!(ou.count, 1, "tick retrógrado NO debe incrementar count");
        assert_eq!(ou.last_ts_ms, 1000, "tick retrógrado NO debe retroceder last_ts_ms");
        assert_eq!(ou.last_value, 10.0, "tick retrógrado NO debe sobreescribir last_value");

        // Intento duplicado: ts=1000 == 1000
        let z_dup = ou.update(30.0, 1000);
        assert!(z_dup.is_finite());
        assert_eq!(ou.count, 1, "tick duplicado NO debe incrementar count");
        assert_eq!(ou.last_ts_ms, 1000, "tick duplicado NO debe alterar last_ts_ms");
        assert_eq!(ou.last_value, 10.0, "tick duplicado NO debe sobreescribir last_value");

        // Siguiente tick causal estrictamente creciente: ts=1500 > 1000
        let z_causal = ou.update(10.2, 1500);
        assert!(z_causal.is_finite());
        assert_eq!(ou.count, 2, "tick causal debe avanzar count");
        assert_eq!(ou.last_ts_ms, 1500, "tick causal debe avanzar reloj");
        assert_eq!(ou.last_value, 10.2, "tick causal debe actualizar last_value");
    }

    #[test]
    fn test_r6_b3_and_b4_heterogeneous_dt_wls_and_continuous_time_decay() {
        // R6-B4: Estimador configurado con memoria temporal en segundos (tau_mem = 120 s)
        let mut ou = ContinuousOrnsteinUhlenbeckSde::with_memory_seconds(0.15, 100.0, 0.5, 120.0);
        assert_eq!(ou.tau_mem_sec, 120.0);

        // R6-B3: Muestreo con dt altamente heterogéneo (de 200ms a 20s)
        let dt_sequence_ms = [200, 1500, 500, 10000, 300, 20000, 1000, 8000, 400, 12000];
        let mut current_ts = 1_000_000u64;
        let mut val = 100.0;
        let target_mu = 100.0;
        let target_theta = 0.15;

        // Trayectoria que revierte hacia target_mu bajo Euler-Maruyama con dt heterogéneo
        for iter in 0..150 {
            let dt_ms = dt_sequence_ms[iter % dt_sequence_ms.len()];
            current_ts += dt_ms;
            let dt_sec = dt_ms as f64 / 1000.0;
            let noise = if iter % 2 == 0 { 0.2 } else { -0.2 };
            val = val + target_theta * (target_mu - val) * dt_sec + noise * dt_sec.sqrt();

            let z = ou.update(val, current_ts);
            assert!(z.is_finite());
        }

        // Verificar convergencia ergódica sana bajo WLS heterogéneo
        assert!(ou.theta > 0.05 && ou.theta < 1.0, "theta estimador ({}) debe converger en rango físico", ou.theta);
        assert!((ou.mu - 100.0).abs() < 5.0, "mu estimador ({}) debe converger cerca del centro ergódico 100.0", ou.mu);
        assert!(ou.sigma > 0.05 && ou.sigma < 2.0, "sigma de difusión ({}) debe ser finito y positivo", ou.sigma);
        assert!(ou.half_life_seconds() > 0.5 && ou.half_life_seconds() < 30.0, "t_1/2 ({}) debe ser coherente", ou.half_life_seconds());

        // Verificar decaimiento continuo sin clamp espurio a 0.80
        let s_dt_before = ou.s_dt;
        let large_jump_ts = current_ts + 120_000; // Salto de 120 segundos (1 tau_mem)
        ou.update(100.0, large_jump_ts);
        // decay = exp(-120 / 120) = exp(-1) ≈ 0.367879 (anteriormente quedaba clamp en 0.80 forzando n_eff ≈ 5)
        let expected_decay = (-1.0f64).exp();
        assert!((ou.s_dt - (s_dt_before * expected_decay + 120.0)).abs() < 1e-4, "decay debe ser continuo exp(-dt/tau)");
    }
}
