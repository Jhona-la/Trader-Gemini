use crate::vecm_arbitrage::ContinuousOrnsteinUhlenbeckSde;
use crate::{QuantumStrategy, SignalIntent, SignalType, TradeHorizon};
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;

pub struct StatArbEngine {
    window_size: usize,
    history: Vec<f64>,
    index: usize,
    count: usize,
    sum: f64,
    z_score_threshold: f64,
    pub min_spread_profit_bps: f64,
    /// Coeficiente de cobertura beta de cointegración (P_a vs P_b)
    pub beta_hedge_ratio: f64,
    /// Si es true, adapta beta_hedge_ratio dinámicamente mediante RLS
    pub adaptive_beta: bool,
    /// Estimador analítico SDE continuo de Ornstein-Uhlenbeck / Fokker-Planck con reloj físico
    pub physical_sde: Option<ContinuousOrnsteinUhlenbeckSde>,
    /// Registro omnisciente para deliberación y lectura de parámetros continuos
    pub registry: Option<Arc<OmniscientRegistry>>,
}

impl StatArbEngine {
    pub fn new(window_size: usize, z_score_threshold: f64) -> Self {
        let safe_window = window_size.max(2);
        let safe_thresh = if z_score_threshold.is_finite() && z_score_threshold > 0.0 {
            z_score_threshold
        } else {
            1.5
        };
        Self {
            window_size: safe_window,
            history: vec![0.0; safe_window],
            index: 0,
            count: 0,
            sum: 0.0,
            z_score_threshold: safe_thresh,
            min_spread_profit_bps: 0.0020,
            beta_hedge_ratio: 1.0,
            adaptive_beta: false,
            physical_sde: None,
            registry: None,
        }
    }

    /// Configura el ratio de cobertura beta inicial.
    pub fn with_beta_hedge_ratio(mut self, beta: f64) -> Self {
        if beta.is_finite() && beta > 0.0 {
            self.beta_hedge_ratio = beta.clamp(0.01, 100.0);
        }
        self
    }

    /// Habilita o deshabilita la adaptación recursiva RLS del ratio beta.
    pub fn with_adaptive_beta(mut self, enabled: bool) -> Self {
        self.adaptive_beta = enabled;
        self
    }

    /// Activa el estimador continuo SDE de Ornstein-Uhlenbeck con reloj físico.
    pub fn with_continuous_ou_sde(mut self) -> Self {
        self.physical_sde = Some(ContinuousOrnsteinUhlenbeckSde::new(0.1, 0.0, 0.01));
        self
    }

    /// Vida media analítica de reversión a la media en segundos (ln(2) / θ).
    pub fn half_life_seconds(&self) -> f64 {
        if let Some(sde) = &self.physical_sde {
            sde.half_life_seconds()
        } else {
            f64::INFINITY
        }
    }

    /// Varianza estacionaria de Fokker-Planck del spread.
    pub fn stationary_variance(&self) -> f64 {
        if let Some(sde) = &self.physical_sde {
            sde.stationary_variance()
        } else {
            1.0
        }
    }

    /// Configura el umbral de spread mínimo para cubrir comisiones y fricción dinámicamente.
    pub fn with_min_spread_profit_bps(mut self, bps: f64) -> Self {
        if bps.is_finite() && bps >= 0.0 {
            self.min_spread_profit_bps = bps;
        }
        self
    }

    /// Toma los precios de dos activos correlacionados y devuelve la intención de arbitraje sobre el Activo A.
    /// (El Activo B debe operar en la dirección contraria).
    #[inline(always)]
    pub fn update(&mut self, price_a: f64, price_b: f64) -> SignalIntent {
        if price_a <= 0.0 || price_b <= 0.0 || !price_a.is_finite() || !price_b.is_finite() {
            return SignalIntent::flat();
        }

        let spread = price_a.ln() - self.beta_hedge_ratio * price_b.ln();

        let old_val = self.history[self.index];
        self.history[self.index] = spread;

        if self.count < self.window_size {
            self.count += 1;
            self.sum += spread;
        } else {
            self.sum += spread - old_val;
        }

        self.index = (self.index + 1) % self.window_size;

        if self.count < self.window_size {
            return SignalIntent::flat();
        }

        let n = self.count as f64;
        let mean = self.sum / n;
        let var_sum: f64 = self.history[..self.count]
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum();
        let variance = (var_sum / n).max(0.0);
        let stdev = if variance > 1e-12 {
            variance.sqrt()
        } else {
            1e-6
        };

        let z_score = if stdev > 0.0 {
            (spread - mean) / stdev
        } else {
            0.0
        };
        // FIX #664: Guarda de finitud estricta en Z-Score
        if !z_score.is_finite() {
            return SignalIntent::flat();
        }
        let expected_spread_edge = (spread - mean).abs();
        // AGY-AUD-P04: borde mínimo adaptativo (self.min_spread_profit_bps) para cubrir comisiones + fricción
        let min_spread_profit_bps = self.min_spread_profit_bps;

        // FIX #784: StatArb emite señales de forma puramente funcional/stateless
        // Previene estados fantasma si RiskEngine o Consejo vetan la orden downstream
        if z_score > self.z_score_threshold && expected_spread_edge > min_spread_profit_bps {
            let norm_conf = (z_score.abs() / self.z_score_threshold.max(0.1)).tanh();
            return SignalIntent {
                signal: SignalType::Short,
                confidence: norm_conf,
                horizon: crate::TradeHorizon::Continuous,
                ..Default::default()
            };
        } else if z_score < -self.z_score_threshold && expected_spread_edge > min_spread_profit_bps
        {
            let norm_conf = (z_score.abs() / self.z_score_threshold.max(0.1)).tanh();
            return SignalIntent {
                signal: SignalType::Long,
                confidence: norm_conf,
                horizon: crate::TradeHorizon::Continuous,
                ..Default::default()
            };
        } else if z_score.abs() < 0.1 {
            // Regresión a la media alcanzada (Señal de cierre / Neutral)
            return SignalIntent {
                signal: SignalType::Flat,
                confidence: 1.0,
                horizon: crate::TradeHorizon::Continuous,
                ..Default::default()
            };
        }

        SignalIntent::flat()
    }

    /// Actualiza el estimador considerando el reloj físico continuo (timestamp_ms) y el horizonte espectral dominante (tau_ms).
    ///
    /// ## Física Continua y Acoplamiento Espectral:
    /// 1. Si `adaptive_beta == true`, actualiza recursivamente el ratio de cobertura RLS $\beta_t$.
    /// 2. Si `physical_sde` está activo, calibra la SDE de Ornstein-Uhlenbeck $dS_t = \theta (\mu - S_t) dt + \sigma dW_t$.
    /// 3. **Acoplamiento Espectral**: Compara la vida media analítica $t_{1/2} = \ln(2)/\theta$ con la escala espectral dominante $\tau^*$.
    ///    Si la reversión requiere más del doble de tiempo que el ciclo dominante ($t_{1/2} > 2 \tau^*_{\text{sec}}$),
    ///    la operación se veta (Flat) para evitar trampas de deriva estructural.
    pub fn update_with_clock(
        &mut self,
        price_a: f64,
        price_b: f64,
        timestamp_ms: u64,
        dominant_tau_ms: f64,
    ) -> SignalIntent {
        if price_a <= 0.0 || price_b <= 0.0 || !price_a.is_finite() || !price_b.is_finite() {
            return SignalIntent::flat();
        }

        let ln_a = price_a.ln();
        let ln_b = price_b.ln();

        if self.adaptive_beta && ln_b.is_finite() {
            let beta_err = ln_a - self.beta_hedge_ratio * ln_b;
            if beta_err.is_finite() {
                let gain = 0.001 / (1.0 + ln_b * ln_b * 0.001);
                self.beta_hedge_ratio = (self.beta_hedge_ratio + gain * ln_b * beta_err).clamp(0.01, 100.0);
            }
        }

        let spread = ln_a - self.beta_hedge_ratio * ln_b;

        // Si tenemos estimador físico SDE activo:
        if let Some(sde) = &mut self.physical_sde {
            let z_score = sde.update(spread, timestamp_ms);
            let t_half = sde.half_life_seconds();
            let tau_sec = if dominant_tau_ms.is_finite() && dominant_tau_ms > 0.0 {
                dominant_tau_ms / 1000.0
            } else {
                30.0
            };

            // Guarda de acoplamiento espectral:
            // Si la reversión física es demasiado lenta comparada con la escala de operación,
            // abstenerse o modular la confianza.
            if t_half > 2.0 * tau_sec {
                return SignalIntent::flat();
            }

            let expected_edge = (spread - sde.mu).abs();
            if expected_edge < self.min_spread_profit_bps {
                return SignalIntent::flat();
            }

            // Factor de atenuación si t_half se acerca a tau_sec
            let spectral_damping = if t_half > tau_sec {
                (2.0 - t_half / tau_sec).clamp(0.1, 1.0)
            } else {
                1.0
            };

            if z_score > self.z_score_threshold {
                let norm_conf = ((z_score.abs() / self.z_score_threshold.max(0.1)).tanh() * spectral_damping).clamp(0.0, 1.0);
                return SignalIntent {
                    signal: SignalType::Short,
                    confidence: norm_conf,
                    horizon: crate::TradeHorizon::Continuous,
                    ..Default::default()
                };
            } else if z_score < -self.z_score_threshold {
                let norm_conf = ((z_score.abs() / self.z_score_threshold.max(0.1)).tanh() * spectral_damping).clamp(0.0, 1.0);
                return SignalIntent {
                    signal: SignalType::Long,
                    confidence: norm_conf,
                    horizon: crate::TradeHorizon::Continuous,
                    ..Default::default()
                };
            } else if z_score.abs() < 0.1 {
                return SignalIntent {
                    signal: SignalType::Flat,
                    confidence: 1.0,
                    horizon: crate::TradeHorizon::Continuous,
                    ..Default::default()
                };
            }

            return SignalIntent::flat();
        }

        // Si no hay SDE físico, delegar a la ventana discreta estándar
        self.update(price_a, price_b)
    }
}

impl QuantumStrategy for StatArbEngine {
    fn name(&self) -> &str {
        "StatArbEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() { None } else { Some(symbol) };
        let cid_opt = if symbol.is_empty() { None } else { Some(coin_id) };
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };

        // Leer z-score de cointegración / arbitraje estadístico publicado en el registro
        let z = r
            .get_scoped_parameter(sym_opt, cid_opt, "vecm_zscore", "StatArbEngine")
            .or_else(|| r.get_scoped_parameter(sym_opt, cid_opt, "cointegration_zscore", "StatArbEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        if !z.is_finite() {
            return 0.0;
        }

        // Acoplamiento espectral continuo (Ola Ω36):
        // Si el estimador continuo SDE OU tiene una vida media mayor al doble del horizonte
        // dominante tau*, no absorbe la deriva secular y se inhibe el voto.
        if let Some(sde) = &self.physical_sde {
            let half_life_s = sde.half_life_seconds();
            let dominant_tau_s = r
                .get_scoped_parameter(sym_opt, cid_opt, "dominant_tau_ms", "StatArbEngine")
                .map(|p| p.get_value() / 1000.0)
                .unwrap_or(1138.0);
            if half_life_s > 2.0 * dominant_tau_s {
                return 0.0;
            }
        }

        // Activación suave y continua:
        // z > thresh => Short (voto negativo hacia reversión), z < -thresh => Long (voto positivo)
        let thresh = self.z_score_threshold.max(0.5);
        if z.abs() > thresh {
            let excess = z.abs() - thresh;
            let norm_signal = -z.signum() * (excess / thresh).tanh();
            norm_signal.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn horizon(&self) -> TradeHorizon {
        TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stat_arb_spread_and_signal() {
        let mut engine = StatArbEngine::new(10, 1.5);
        for i in 0..9 {
            let res = engine.update(100.0 + (i as f64 * 0.1), 100.0);
            assert_eq!(res.signal, SignalType::Flat);
        }

        // Divergencia grande de precio A respecto a B
        let signal = engine.update(110.0, 100.0);
        assert_eq!(signal.signal, SignalType::Short);
    }

    #[test]
    fn test_stat_arb_zero_window_and_nan_immunity() {
        let mut engine = StatArbEngine::new(0, f64::NAN);
        assert_eq!(engine.window_size, 2);
        assert_eq!(engine.z_score_threshold, 1.5);

        let sig_nan = engine.update(f64::NAN, 100.0);
        assert_eq!(sig_nan.signal, SignalType::Flat);

        let sig_neg = engine.update(-10.0, 100.0);
        assert_eq!(sig_neg.signal, SignalType::Flat);
    }

    #[test]
    fn test_stat_arb_symmetric_long_and_flat_reversion() {
        let mut engine = StatArbEngine::new(10, 1.5);
        for i in 0..9 {
            let res = engine.update(100.0, 100.0 + (i as f64 * 0.1));
            assert_eq!(res.signal, SignalType::Flat);
        }

        // Divergencia donde A cae fuertemente respecto a B
        let long_signal = engine.update(90.0, 100.0);
        assert_eq!(long_signal.signal, SignalType::Long);
        assert_eq!(long_signal.horizon, crate::TradeHorizon::Continuous);

        // Reversión a la media
        for _ in 0..10 {
            let _ = engine.update(100.0, 100.0);
        }
        let flat_signal = engine.update(100.0, 100.0);
        assert_eq!(flat_signal.signal, SignalType::Flat);
    }

    #[test]
    fn test_stat_arb_numerical_stability_large_prices() {
        // Validación contra cancelación catastrófica (Punto #36)
        // Con precios del orden de $65,000 (BTC), la fórmula ingenua (E[x^2] - E[x]^2)
        // produce varianzas negativas o colapso por pérdida de precisión IEEE 754.
        // La formulación centrada de StatArbEngine garantiza varianza no negativa y z-score finito.
        let mut engine = StatArbEngine::new(20, 2.0);
        let base_btc = 65_432.10;
        let base_eth = 3_456.78;

        for i in 0..19 {
            let p_a = base_btc + (i as f64 * 1.5);
            let p_b = base_eth + (i as f64 * 0.1);
            let intent = engine.update(p_a, p_b);
            assert_eq!(intent.signal, SignalType::Flat);
        }

        // Pequeño desplazamiento que no rompe Z-Score
        let intent_normal = engine.update(base_btc + 20.0, base_eth + 1.0);
        assert_eq!(intent_normal.signal, SignalType::Flat);

        // Dislocación grande
        let intent_dislocated = engine.update(base_btc + 1500.0, base_eth);
        assert_eq!(intent_dislocated.signal, SignalType::Short);
        assert!(intent_dislocated.confidence > 0.0 && intent_dislocated.confidence <= 1.0);
    }

    #[test]
    fn test_stat_arb_configurable_min_spread_profit_bps() {
        let engine_default = StatArbEngine::new(20, 2.0);
        assert_eq!(engine_default.min_spread_profit_bps, 0.0020);

        let engine_custom = StatArbEngine::new(20, 2.0).with_min_spread_profit_bps(0.0008);
        assert_eq!(engine_custom.min_spread_profit_bps, 0.0008);
    }

    #[test]
    fn test_stat_arb_continuous_ou_sde_and_clock() {
        let mut engine = StatArbEngine::new(20, 1.8)
            .with_continuous_ou_sde()
            .with_min_spread_profit_bps(0.0005);

        assert!(engine.physical_sde.is_some());
        // Calibrar SDE con serie de precios en tiempo físico real (paso de 500 ms)
        let mut t = 1_000_000_u64;
        for i in 0..30 {
            t += 500;
            let intent = engine.update_with_clock(100.0 + (i as f64 * 0.01), 100.0, t, 60_000.0);
            assert_eq!(intent.signal, SignalType::Flat);
        }

        // Shock de precio positivo que dispara señal Short en tiempo continuo
        t += 500;
        let shock_intent = engine.update_with_clock(105.0, 100.0, t, 60_000.0);
        assert_eq!(shock_intent.signal, SignalType::Short);
        assert!(shock_intent.confidence > 0.0);
        assert_eq!(shock_intent.horizon, crate::TradeHorizon::Continuous);

        // Vida media física finita calculada
        let half_life = engine.half_life_seconds();
        assert!(half_life.is_finite() && half_life > 0.0);
    }

    #[test]
    fn test_stat_arb_spectral_coupling_damps_or_abstains() {
        let mut engine = StatArbEngine::new(20, 1.5)
            .with_continuous_ou_sde()
            .with_min_spread_profit_bps(0.0005);

        let mut t = 1_000_000_u64;
        for _ in 0..20 {
            t += 1000;
            let _ = engine.update_with_clock(100.0, 100.0, t, 10_000.0);
        }

        // Forzar un shock con horizonte espectral dominante extremadamente corto (ej. tau = 50 ms)
        // La reversión física de Ornstein-Uhlenbeck es de varios segundos, superando con creces 2 * tau
        t += 1000;
        let intent_blocked_by_spectral = engine.update_with_clock(110.0, 100.0, t, 50.0);
        // Debe abstenerse (Flat) porque t_half >> 2 * tau
        assert_eq!(intent_blocked_by_spectral.signal, SignalType::Flat);
    }

    #[test]
    fn test_stat_arb_adaptive_beta_rls() {
        let mut engine = StatArbEngine::new(20, 1.5)
            .with_adaptive_beta(true)
            .with_beta_hedge_ratio(1.0);

        // Actualizar donde el precio A sistemáticamente se mueve al doble que B (beta ~ 2.0)
        let mut t = 1_000_000_u64;
        for i in 1..40 {
            t += 500;
            let p_b = 50.0 + (i as f64 * 0.5);
            let p_a = p_b.powf(1.8);
            let _ = engine.update_with_clock(p_a, p_b, t, 60_000.0);
        }

        // El beta adaptativo debe haber evolucionado lejos de 1.0 hacia la verdadera elasticidad
        assert!(engine.beta_hedge_ratio > 1.05);
    }
}
