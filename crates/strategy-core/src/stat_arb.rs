use crate::vecm_arbitrage::ContinuousOrnsteinUhlenbeckSde;
use crate::{QuantumStrategy, SignalIntent, SignalType, TradeHorizon};
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;

/// Fallback canónico de τ* dominante en milisegundos: centro geométrico de la
/// banda operativa [30 s, 12 h] (√(30 000 · 43 200 000) ≈ 1 138 419,6 ms).
/// Único valor para TODOS los caminos del motor (R6-B13: antes 30.0 s en
/// `update_with_clock` vs 1138.0 s en `evaluate_for_coin`).
const DEFAULT_DOMINANT_TAU_MS: f64 = 1_138_419.6;

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
    /// Covarianza estimada escalar P_t para el filtro RLS recursivo (R6-B12)
    pub rls_p: f64,
    /// Factor de olvido exponencial lambda in [0.95, 0.9999] (R6-B12)
    pub rls_lambda: f64,
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
            rls_p: 1.0,
            rls_lambda: 0.998,
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

    /// R6-B12 (Ola Ω51): Configura el factor de olvido exponencial lambda in [0.95, 0.9999] del filtro RLS.
    pub fn with_rls_lambda(mut self, lambda: f64) -> Self {
        if lambda.is_finite() && (0.95..=0.9999).contains(&lambda) {
            self.rls_lambda = lambda;
        }
        self
    }

    /// R6-B12 (Ola Ω51): Configura la covarianza inicial P_0 del filtro RLS.
    pub fn with_rls_p(mut self, p: f64) -> Self {
        if p.is_finite() && p > 0.0 {
            self.rls_p = p.clamp(1e-4, 1000.0);
        }
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

    /// Z-score estacionario del spread bajo la SDE viva, si está madura
    /// (≥ 10 observaciones causales). `None` durante el calentamiento o sin SDE.
    /// Este es el estadístico que el core publica como `statarb_ou_zscore` y
    /// que `evaluate_for_coin` prefiere como física propia (R6-A3/B1).
    pub fn last_ou_zscore(&self) -> Option<f64> {
        let sde = self.physical_sde.as_ref()?;
        if sde.count < 10 {
            return None;
        }
        let z = sde.stationary_zscore(sde.last_value);
        if z.is_finite() {
            Some(z)
        } else {
            None
        }
    }

    /// Timestamp físico (ms) del último update causal aceptado por la SDE.
    /// 0 si la SDE nunca ha observado. Permite al publicador detectar
    /// staleness del feed de pares (R6-A3).
    pub fn last_ou_ts_ms(&self) -> u64 {
        self.physical_sde.as_ref().map(|s| s.last_ts_ms).unwrap_or(0)
    }

    /// Configura el umbral de spread mínimo para cubrir comisiones y fricción dinámicamente.
    pub fn with_min_spread_profit_bps(mut self, bps: f64) -> Self {
        if bps.is_finite() && bps >= 0.0 {
            self.min_spread_profit_bps = bps;
        }
        self
    }

    /// R6-B12 (Ola Ω51): Actualización recursiva estricta por mínimos cuadrados (RLS) del ratio beta.
    /// Modelo: ln(P_a) = beta_t * ln(P_b) + e_t.
    /// Ganancia de Kalman / RLS: K_t = P_{t-1} * x_t / (lambda + x_t^2 * P_{t-1})
    /// Parámetro: beta_t = beta_{t-1} + K_t * e_t
    /// Covarianza: P_t = (P_{t-1} - K_t * x_t * P_{t-1}) / lambda
    #[inline(always)]
    fn update_rls_beta(&mut self, ln_a: f64, ln_b: f64) {
        if !self.adaptive_beta || !ln_a.is_finite() || !ln_b.is_finite() {
            return;
        }
        let beta_err = ln_a - self.beta_hedge_ratio * ln_b;
        if !beta_err.is_finite() {
            return;
        }
        let x = ln_b;
        let p_prev = self.rls_p;
        let lambda = self.rls_lambda.clamp(0.95, 0.9999);
        let denom = lambda + x * x * p_prev;
        if denom > 1e-12 {
            let k = (p_prev * x) / denom;
            self.beta_hedge_ratio = (self.beta_hedge_ratio + k * beta_err).clamp(0.01, 100.0);
            self.rls_p = ((p_prev - k * x * p_prev) / lambda).clamp(1e-6, 1000.0);
        }
    }

    /// Toma los precios de dos activos correlacionados y devuelve la intención de arbitraje sobre el Activo A.
    /// (El Activo B debe operar en la dirección contraria).
    #[inline(always)]
    pub fn update(&mut self, price_a: f64, price_b: f64) -> SignalIntent {
        if price_a <= 0.0 || price_b <= 0.0 || !price_a.is_finite() || !price_b.is_finite() {
            return SignalIntent::flat();
        }

        let ln_a = price_a.ln();
        let ln_b = price_b.ln();
        self.update_rls_beta(ln_a, ln_b);

        let spread = ln_a - self.beta_hedge_ratio * ln_b;

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

        self.update_rls_beta(ln_a, ln_b);

        let spread = ln_a - self.beta_hedge_ratio * ln_b;

        // Si tenemos estimador físico SDE activo:
        if let Some(sde) = &mut self.physical_sde {
            let z_score = sde.update(spread, timestamp_ms);
            let t_half = sde.half_life_seconds();
            let tau_sec = if dominant_tau_ms.is_finite() && dominant_tau_ms > 0.0 {
                dominant_tau_ms / 1000.0
            } else {
                // R6-B13: fallback unificado al centro geométrico de la banda
                // operativa (antes 30.0 s aquí vs 1138.0 s en evaluate_for_coin).
                DEFAULT_DOMINANT_TAU_MS / 1000.0
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

        // OLA 73 (R6-A3/B1/A5): preferir la física PROPIA — z estacionario de
        // la SDE OU que el CORE calibra sobre la basis futuro-spot con reloj
        // físico real y β RLS (clave `statarb_ou_zscore`). El core sólo publica
        // cuando esa física EXISTE (SDE madura ≥ 10 pares); un 0.0 publicado es
        // por tanto abstención honesta (spot stale > TTL), no arranque frío.
        // Clave AUSENTE = el core aún no observa pares spot-futuro para esta
        // moneda ⇒ fallback bit a bit con el árbol previo.
        //
        // FALLBACK LEGACY (R6-A5, etiqueta honesta): `vecm_zscore` NO es
        // cointegración Johansen ni un spread entre dos activos. El core lo
        // publica como z de la basis futuro-spot / ATR cuando hay feed de spot
        // y, en su ausencia, como desviación del mid al EMA lento de klines en
        // unidades de ATR (god-engine-core/src/lib.rs, `vecm_basis_z`). Es un
        // estadístico de reversión a la media propia, no de paridad multiactivo.
        let own_z = r
            .get_scoped_parameter(sym_opt, cid_opt, "statarb_ou_zscore", "StatArbEngine")
            .map(|p| p.get_value());

        let z = match own_z {
            Some(z) if z.is_finite() => {
                // Guarda espectral con la MISMA física viva (R6-B2): t½ del
                // registro (θ calibrada en producción) contra τ* dominante del
                // espectro de la moneda. Si la reversión tarda más del doble
                // del ciclo dominante, el spread absorbe deriva secular: veto.
                let t_half_ms = r
                    .get_scoped_parameter(sym_opt, cid_opt, "statarb_half_life_ms", "StatArbEngine")
                    .map(|p| p.get_value())
                    .unwrap_or(f64::INFINITY);
                let tau_dom_ms = r
                    .get_scoped_parameter(sym_opt, cid_opt, "dominant_tau_ms", "StatArbEngine")
                    .map(|p| p.get_value())
                    .unwrap_or(DEFAULT_DOMINANT_TAU_MS);
                // Fail-closed: NaN en cualquiera de las dos cotas ⇒ veto.
                if !(t_half_ms <= 2.0 * tau_dom_ms) {
                    return 0.0;
                }
                z
            }
            // Clave presente pero venenosa (NaN/Inf): abstención honesta.
            Some(_) => return 0.0,
            None => r
                .get_scoped_parameter(sym_opt, cid_opt, "vecm_zscore", "StatArbEngine")
                .or_else(|| {
                    r.get_scoped_parameter(sym_opt, cid_opt, "cointegration_zscore", "StatArbEngine")
                })
                .map(|p| p.get_value())
                .unwrap_or(0.0),
        };

        if !z.is_finite() {
            return 0.0;
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

    #[test]
    fn test_r6_b12_rls_convergencia_exacta() {
        let mut engine = StatArbEngine::new(20, 1.5)
            .with_adaptive_beta(true)
            .with_beta_hedge_ratio(1.0)
            .with_rls_lambda(0.99)
            .with_rls_p(10.0);

        // Modelo exacto: ln(P_a) = 2.5 * ln(P_b)
        let mut t = 2_000_000_u64;
        for i in 1..=50 {
            t += 250;
            let p_b = 10.0 + (i as f64) * 0.2;
            let p_a = p_b.powf(2.5);
            engine.update_with_clock(p_a, p_b, t, 60_000.0);
        }

        // Con RLS estricto y covarianza P_t, beta converge directamente hacia 2.5
        assert!((engine.beta_hedge_ratio - 2.5).abs() < 0.15, "RLS debe converger hacia 2.5: got {}", engine.beta_hedge_ratio);
        // Covarianza P_t debe haberse contraído
        assert!(engine.rls_p < 1.0, "La covarianza P_t debe contraerse con la evidencia: got {}", engine.rls_p);
    }

    // ===== OLA 73 (R6-A3/A4/A5/B1/B2/B13): StatArb honesto =====

    #[test]
    fn test_stat_arb_ola73_last_ou_zscore_maturity_gates() {
        let mut engine = StatArbEngine::new(20, 1.5).with_continuous_ou_sde();
        // SDE fría: sin estadístico propio
        assert_eq!(engine.last_ou_ts_ms(), 0);
        assert!(engine.last_ou_zscore().is_none());

        let mut t = 1_000_000_u64;
        for i in 0..9 {
            t += 1000;
            let _ = engine.update_with_clock(100.0 + (i as f64 * 0.01), 100.0, t, 60_000.0);
        }
        // 9 observaciones causales: aún bajo el umbral de madurez (count < 10)
        assert!(engine.last_ou_ts_ms() > 0);
        assert!(engine.last_ou_zscore().is_none());

        t += 1000;
        let _ = engine.update_with_clock(101.0, 100.0, t, 60_000.0);
        // Madura: z estacionario finito
        let z = engine.last_ou_zscore().expect("z propio tras madurez");
        assert!(z.is_finite());
    }

    #[test]
    fn test_stat_arb_ola73_fallback_tau_unificado_centro_geometrico() {
        // R6-B13: un único fallback de τ* — centro geométrico de la banda
        // operativa [30 s, 12 h]: √(30 000 · 43 200 000) ≈ 1 138 419,6 ms.
        let centro = (30_000.0_f64 * 43_200_000.0).sqrt();
        assert!((DEFAULT_DOMINANT_TAU_MS - centro).abs() < 1.0);
    }

    #[test]
    fn test_stat_arb_ola73_evaluate_prefiere_fisica_propia_con_guarda_espectral() {
        let reg = Arc::new(OmniscientRegistry::new());
        let mut engine = StatArbEngine::new(30, 1.5);
        assert!(engine.init(reg.clone()).is_ok());

        // Física propia publicada: z = 2.5 (dislocación corta), t½ = 60 s,
        // τ* dominante = 60 s ⇒ t½ ≤ 2τ* ⇒ el voto vive y apunta a Short.
        reg.set("statarb_ou_zscore", 2.5);
        reg.set("statarb_half_life_ms", 60_000.0);
        reg.set("dominant_tau_ms", 60_000.0);
        // vecm_zscore legacy con signo OPUESTO: la física propia debe ganar.
        reg.set("vecm_zscore", -2.5);
        let voto = engine.evaluate();
        assert!(voto < -0.5, "voto Short por z propio positivo, got {voto}");

        // Guarda espectral: t½ = 250 s > 2 · 100 s ⇒ veto (deriva secular).
        reg.set("statarb_half_life_ms", 250_000.0);
        reg.set("dominant_tau_ms", 100_000.0);
        assert_eq!(engine.evaluate(), 0.0);

        // τ* envenenada (el registro sanea NaN → 0.0 al escribir): 2·0 = 0 ⇒
        // cualquier t½ finita viola la guarda ⇒ veto fail-closed.
        reg.set("dominant_tau_ms", f64::NAN);
        assert_eq!(engine.evaluate(), 0.0);

        // Half-life AUSENTE (SDE nunca observó) ⇒ cota INFINITA ⇒ veto honesto
        // aunque la z propia llegue publicada.
        let reg2 = Arc::new(OmniscientRegistry::new());
        let mut engine2 = StatArbEngine::new(30, 1.5);
        assert!(engine2.init(reg2.clone()).is_ok());
        reg2.set("statarb_ou_zscore", 2.5);
        reg2.set("dominant_tau_ms", 60_000.0);
        assert_eq!(engine2.evaluate(), 0.0);
    }

    #[test]
    fn test_stat_arb_ola73_evaluate_fallback_legacy_bit_a_bit() {
        let reg = Arc::new(OmniscientRegistry::new());
        let mut engine = StatArbEngine::new(30, 1.5);
        assert!(engine.init(reg.clone()).is_ok());

        // Sin clave propia: fallback legacy a vecm_zscore (basis/ATR),
        // conducta idéntica al árbol previo.
        reg.set("vecm_zscore", 2.0);
        let voto = engine.evaluate();
        let esperado = -(0.5_f64 / 1.5).tanh();
        assert!((voto - esperado).abs() < 1e-12);

        // z dentro del umbral: silencio.
        reg.set("vecm_zscore", 0.5);
        assert_eq!(engine.evaluate(), 0.0);
    }

    #[test]
    fn test_stat_arb_ola73_evaluate_zscore_cero_abstiene() {
        let reg = Arc::new(OmniscientRegistry::new());
        let mut engine = StatArbEngine::new(30, 1.5);
        assert!(engine.init(reg.clone()).is_ok());

        // El core publica 0.0 explícito cuando su SDE ya maduró y el feed de
        // spot quedó stale: abstención de la física propia, sin caer al
        // fallback legacy (la clave publicada es la autoridad del lector).
        reg.set("statarb_ou_zscore", 0.0);
        reg.set("vecm_zscore", 2.5);
        assert_eq!(engine.evaluate(), 0.0);
    }
}
