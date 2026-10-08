/// Legacy weighted-log basket and event-index mean-reversion proxy (#83-#95).
/// Supplied weights are assumed, not estimated cointegration vectors. Neither
/// stationarity nor a physical-time OU model is certified by this helper.
use crate::types::{SignalIntent, SignalType, TradeHorizon};
use crate::vecm_arbitrage::ContinuousOrnsteinUhlenbeckSde;

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
    /// AGY-AUD-001: consecutive jump rejections counter.
    consecutive_rejections: u32,
    /// Optional limit of consecutive rejections before resetting on structural break.
    /// None by default to preserve legacy open-debt contracts, Some(N) for adaptive recovery.
    structural_break_limit: Option<u32>,
    /// Estimador SDE de tiempo continuo de Ornstein-Uhlenbeck / Fokker-Planck con reloj físico real.
    pub physical_sde: Option<ContinuousOrnsteinUhlenbeckSde>,
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
            consecutive_rejections: 0,
            structural_break_limit: None,
            physical_sde: None,
        }
    }

    /// Configura el límite de rechazos consecutivos para la recuperación adaptativa ante un cambio estructural.
    pub fn with_structural_break_recovery(mut self, limit: u32) -> Self {
        self.structural_break_limit = Some(limit.max(1));
        self
    }

    /// Activa el estimador continuo SDE de Ornstein-Uhlenbeck / Fokker-Planck con reloj físico real.
    pub fn with_continuous_ou(mut self) -> Self {
        self.physical_sde = Some(ContinuousOrnsteinUhlenbeckSde::new(0.1, 0.0, 0.01));
        self
    }

    /// Updates the multivariate cointegration estimator.
    /// When `physical_sde` is active, evaluates under the continuous-time SDE using physical timestamps.
    /// In legacy mode, preserves event-index statistics.
    pub fn update_and_evaluate(
        &mut self,
        prices: &[f64; MAX_ASSETS],
        timestamp_ms: u64,
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

        // Estimador SDE de tiempo continuo de Ornstein-Uhlenbeck / Fokker-Planck con reloj físico real.
        // Si está activo (modo opt-in exclusivo vía with_continuous_ou), opera la reversión continua
        // multiactivo sobre tiempo físico y emite una intención con expected_duration_ms calibrada
        // a la vida media física t_{1/2} en ms. Si no supera umbral o está frío, se abstiene (None)
        // con honestidad matemática estricta, sin caer al evaluador discreto legacy (OU-R4-01).
        if let Some(sde) = &mut self.physical_sde {
            let sde_z = sde.update(spread, timestamp_ms);
            let sde_hl_sec = sde.half_life_seconds();
            if sde.count >= 10 && sde_hl_sec.is_finite() && sde_hl_sec <= 3600.0 {
                let tau_rev_ms = (sde_hl_sec * 1000.0).clamp(500.0, 43_200_000.0) as u64;
                let sd = sde.stationary_variance().sqrt().max(1e-8);
                let expected_magnitude = (sde_z.abs() * sd).clamp(0.002, 0.20);
                if sde_z < -self.z_score_threshold {
                    let confidence = (-sde_z / 3.0).clamp(0.5, 0.99);
                    return Some(SignalIntent {
                        signal: SignalType::Long,
                        confidence,
                        horizon: TradeHorizon::Continuous,
                        expected_duration_ms: tau_rev_ms,
                        expected_magnitude,
                        ..Default::default()
                    });
                } else if sde_z > self.z_score_threshold {
                    let confidence = (sde_z / 3.0).clamp(0.5, 0.99);
                    return Some(SignalIntent {
                        signal: SignalType::Short,
                        confidence,
                        horizon: TradeHorizon::Continuous,
                        expected_duration_ms: tau_rev_ms,
                        expected_magnitude,
                        ..Default::default()
                    });
                }
            }
            // En modo SDE continuo, la ausencia de señal o la fase de maduración inicial
            // retorna abstención honesta (None). No cae al evaluador discreto legacy.
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
        // AGY-AUD-001: a legitimate permanent level shift (structural break)
        // used to permanently freeze this estimator because `last_spread` was
        // never updated on rejection — diff_spread stayed large forever.
        // If structural_break_limit is configured via `with_structural_break_recovery`,
        // we track consecutive rejections and RESET to the new level after N rejections.
        let max_jump = 10.0 * self.var_spread.sqrt();
        if next_count > 10 && diff_spread.abs() > max_jump.max(1.5) {
            self.consecutive_rejections += 1;
            if let Some(limit) = self.structural_break_limit {
                if self.consecutive_rejections >= limit {
                    // Structural break accepted: reset estimator to new level.
                    self.mean_spread = spread;
                    self.var_spread = diff_spread.abs().powi(2).max(1e-4);
                    self.last_spread = spread;
                    self.theta_reversion_speed = 0.1;
                    self.count = 2; // preserve warm state
                    self.consecutive_rejections = 0;
                }
            }
            return None;
        }
        self.consecutive_rejections = 0; // normal tick resets the counter

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
            // Magnitud esperada físicamente del retorno a la media: E[|ΔS|] = |z| · σ_spread
            let expected_magnitude = (z_score.abs() * std_dev).clamp(0.002, 0.20);
            if z_score < -self.z_score_threshold {
                // Spread subvaluado -> Comprar el cluster (Long)
                let confidence = (-z_score / 3.0).clamp(0.5, 0.99);
                return Some(SignalIntent {
                    signal: SignalType::Long,
                    confidence,
                    horizon: TradeHorizon::Continuous,
                    expected_magnitude,
                    ..Default::default()
                });
            } else if z_score > self.z_score_threshold {
                // Spread sobrevaluado -> Vender el cluster (Short)
                let confidence = (z_score / 3.0).clamp(0.5, 0.99);
                return Some(SignalIntent {
                    signal: SignalType::Short,
                    confidence,
                    horizon: TradeHorizon::Continuous,
                    expected_magnitude,
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

    /// AGY-AUD-001: after a structural break (persistent jump), the
    /// estimator must reset to the new level after MAX_CONSECUTIVE_REJECTIONS
    /// instead of bricking permanently.
    #[test]
    fn structural_break_resets_estimator_instead_of_bricking() {
        let weights = [1.0, -0.5, -0.3, -0.2];
        let mut engine = MultivariateCointegrationEngine::new(weights, 2.0)
            .with_structural_break_recovery(20);

        let base_prices = [100.0, 50.0, 30.0, 20.0];
        // Warm up at original level
        for i in 0..50 {
            let noise = ((i % 5) as f64 - 2.0) * 0.05;
            let prices = [
                base_prices[0] + noise,
                base_prices[1] + noise * 0.5,
                base_prices[2] + noise * 0.3,
                base_prices[3] + noise * 0.2,
            ];
            let _ = engine.update_and_evaluate(&prices, 1000 + i * 1000);
        }

        // Apply a permanent level shift in asset 0 (structural break with diff_spread = ln(6) ≈ 1.79 > 1.5)
        let new_level_prices = [600.0, 50.0, 30.0, 20.0];
        // Feed the new level for 25 ticks (more than MAX_CONSECUTIVE_REJECTIONS=20)
        for i in 0..25 {
            let noise = ((i % 5) as f64 - 2.0) * 0.05;
            let prices = [
                new_level_prices[0] + noise,
                new_level_prices[1] + noise * 0.5,
                new_level_prices[2] + noise * 0.3,
                new_level_prices[3] + noise * 0.2,
            ];
            let _ = engine.update_and_evaluate(&prices, 100_000 + i * 1000);
        }

        // The estimator should NOT be bricked: it should have reset to the
        // new level and its mean_spread should be near the new spread value,
        // not the old one.
        let old_spread: f64 = base_prices.iter()
            .zip(weights.iter())
            .map(|(p, w)| w * p.max(1e-12).ln())
            .sum();
        let new_spread: f64 = new_level_prices.iter()
            .zip(weights.iter())
            .map(|(p, w)| w * p.max(1e-12).ln())
            .sum();
        // mean_spread should be closer to the new level than the old one
        let dist_old = (engine.mean_spread - old_spread).abs();
        let dist_new = (engine.mean_spread - new_spread).abs();
        assert!(
            dist_new < dist_old,
            "estimator should have reset to new level: mean={}, old={old_spread}, new={new_spread}",
            engine.mean_spread
        );
    }

    #[test]
    fn test_multivariate_cointegration_measured_expected_magnitude() {
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

        // Shock para generar señal
        let shock_prices = [75.0, 50.0, 30.0, 20.0];
        let signal = engine.update_and_evaluate(&shock_prices, 200000).expect("señal activa");
        assert!(signal.expected_magnitude > 0.002, "expected magnitude no debe ser cero");
        assert!(signal.expected_magnitude <= 0.20, "expected magnitude acotada");
    }

    #[test]
    fn test_multivariate_cointegration_continuous_ou_physical_clock() {
        let weights = [1.0, -0.5, -0.3, -0.2];
        let mut engine = MultivariateCointegrationEngine::new(weights, 2.0).with_continuous_ou();

        assert!(engine.physical_sde.is_some(), "estimador SDE físico debe estar activo");

        let base_prices = [100.0, 50.0, 30.0, 20.0];
        // Calibrar el proceso continuo con 60 observaciones físicas cada 1000 ms (1 segundo físico)
        for i in 0..60 {
            let noise = ((i % 5) as f64 - 2.0) * 0.05;
            let prices = [
                base_prices[0] + noise,
                base_prices[1] + noise * 0.2,
                base_prices[2] - noise * 0.1,
                base_prices[3] + noise * 0.1,
            ];
            let _ = engine.update_and_evaluate(&prices, 10_000 + i * 1_000);
        }

        let sde = engine.physical_sde.as_ref().unwrap();
        assert!(sde.count >= 60, "todas las observaciones físicas deben contarse");
        assert!(sde.theta > 0.0, "velocidad física theta debe ser positiva");
        assert!(sde.half_life_seconds().is_finite(), "vida media física debe ser finita");

        // Shock positivo en el activo líder -> genera divergencia de spread y señal Short con duración espectral
        let shock_prices = [180.0, 50.0, 30.0, 20.0];
        let signal = engine
            .update_and_evaluate(&shock_prices, 70_000)
            .expect("debe emitir señal continua por shock de divergencia física");

        assert_eq!(signal.signal, SignalType::Short);
        assert_eq!(signal.horizon, TradeHorizon::Continuous);
        assert!(
            signal.expected_duration_ms >= 500 && signal.expected_duration_ms <= 43_200_000,
            "duración esperada ({}) debe estar acotada al espectro temporal continuo",
            signal.expected_duration_ms
        );
        assert!(signal.confidence >= 0.50, "confianza Bayesiana calibrada");
    }

    #[test]
    fn test_ou_r4_01_sde_mode_never_falls_back_to_legacy_with_zero_duration() {
        // OU-R4-01: En modo continuo SDE, el motor NUNCA debe caer silenciosamente al evaluador
        // legacy basado en eventos que emite señales con expected_duration_ms=0.
        let mut engine =
            MultivariateCointegrationEngine::new([1.0, 0.0, 0.0, 0.0], 2.0).with_continuous_ou();

        // 1. Inicialización en frío: primera observación devuelve None
        let res0 = engine.update_and_evaluate(&[1.0, 1.0, 1.0, 1.0], 1000);
        assert!(res0.is_none());

        // 2. Segunda observación con salto que en el evaluador legacy dispararía señal:
        // Pero en modo SDE (count < 10) está en fase de maduración física y DEBE abstenerse (None)
        let res1 = engine.update_and_evaluate(&[(0.1f64).exp(), 1.0, 1.0, 1.0], 2000);
        assert!(
            res1.is_none(),
            "en modo SDE frío debe abstenerse (None) sin caer al evaluador legacy"
        );

        // 3. Cuando emite señal tras maduración física, expected_duration_ms debe ser > 0 y finito
        for i in 2..20 {
            let _ = engine.update_and_evaluate(&[1.0, 1.0, 1.0, 1.0], 2000 + i * 1000);
        }
        let shock = engine.update_and_evaluate(&[1.5, 1.0, 1.0, 1.0], 25_000);
        if let Some(intent) = shock {
            assert_ne!(
                intent.expected_duration_ms, 0,
                "señal continua SDE jamás debe tener expected_duration_ms=0"
            );
            assert_eq!(intent.horizon, TradeHorizon::Continuous);
        }
    }
}
