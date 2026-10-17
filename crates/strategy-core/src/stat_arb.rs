use crate::prueba_reversion::{PruebaReversion, Veredicto};
use crate::{QuantumStrategy, SignalIntent, SignalType, TradeHorizon};
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;

/// Fallback canónico de τ* dominante en milisegundos: centro geométrico de la
/// banda operativa [30 s, 12 h] (√(30 000 · 43 200 000) ≈ 1 138 419,6 ms).
/// Único valor para TODOS los caminos del motor (R6-B13: antes 30.0 s en
/// el camino de la SDE OU, retirado por CL-97, vs 1138.0 s en
/// `evaluate_for_coin`).
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
    /// CL-97 (F3.7-02, v2): prueba de reversión de la base a la escala de
    /// decisión, sobre pares síncronos (`observar_evento`). Sustituye a la
    /// SDE OU por evento, que confundía el ruido de tick y la asincronía con
    /// reversión.
    pub prueba: Option<PruebaReversion>,
    /// Mid de A tras su último evento: el vigente cuando llega un dato de B.
    mid_a_previo: Option<f64>,
    /// Marca de llegada del dato de B visto en el último evento.
    marca_b_previa: Option<u64>,
    /// Registro omnisciente para deliberación y lectura de parámetros continuos
    pub registry: Option<Arc<OmniscientRegistry>>,
}

/// CL-97: veredicto de StatArb a una τ de decisión.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VeredictoStatArb {
    /// z del último par contra la media y la dispersión de la base en la
    /// memoria del nodo; 0 sin reversión demostrada.
    pub z: f64,
    /// Cota superior de la vida media demostrada (2·τ_k del nodo), en ms;
    /// ∞ sin demostración. Es lo que el núcleo publica como
    /// `statarb_half_life_ms`: la guarda t½ ≤ 2τ* del lector pasa sólo con
    /// reversión demostrada en un nodo τ_k ≤ τ*.
    pub t_media_cota_ms: f64,
    pub demostrada: bool,
    /// Detalle de la prueba.
    pub prueba: Veredicto,
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
            prueba: None,
            mid_a_previo: None,
            marca_b_previa: None,
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

    /// CL-97 (F3.7-02, v2): activa la prueba de reversión de la base sobre
    /// pares síncronos. La base es ln a − β·ln b con β FIJA
    /// (`beta_hedge_ratio`, 1 por defecto): con β estimada sobre los mismos
    /// datos el residuo es paso-alto y revierte por construcción (CL-84), así
    /// que la RLS (`adaptive_beta`) sólo actúa en `update`, nunca aquí. Para
    /// futuro y spot del mismo activo el vector de cointegración es (1, −1).
    pub fn with_prueba_reversion(mut self) -> Self {
        self.prueba = Some(PruebaReversion::new());
        self
    }

    /// CL-97: observa un par síncrono (a y b vistos a la vez) de marca `ts_ms`.
    pub fn observar_par(&mut self, precio_a: f64, precio_b: f64, ts_ms: u64) {
        let valido = |p: f64| p.is_finite() && p > 0.0;
        if !valido(precio_a) || !valido(precio_b) {
            return;
        }
        let base = precio_a.ln() - self.beta_hedge_ratio * precio_b.ln();
        if let Some(prueba) = self.prueba.as_mut() {
            prueba.observar(ts_ms, base);
        }
    }

    /// CL-97: evento de A (el futuro) con el último dato de B (el spot) y su
    /// marca de llegada `marca_b_ms`.
    ///
    /// Pares síncronos: el dato de B que llegó entre el evento anterior de A
    /// y éste se empareja con el mid de A VIGENTE cuando llegó, el del evento
    /// anterior, y con su marca de llegada. Un dato de B que no cambia de
    /// marca no es un par nuevo: repetir el futuro contra un spot sin
    /// refrescar es la asincronía que fabricaba reversión. Hay que llamarlo
    /// en todos los eventos de A, también sin dato de B (`precio_b` ≤ 0) y
    /// con las entradas bloqueadas: si no, el mid previo sería de un evento
    /// viejo.
    pub fn observar_evento(&mut self, precio_a: f64, precio_b: f64, marca_b_ms: u64, ts_ms: u64) {
        let nueva_llegada = matches!(self.marca_b_previa, Some(m) if m != marca_b_ms);
        if nueva_llegada {
            if let Some(a0) = self.mid_a_previo {
                // La marca es el reloj de eventos cuando llegó B; si va por
                // delante del evento de A (reloj global de otra moneda), el
                // par vale como mucho hasta este evento.
                self.observar_par(a0, precio_b, marca_b_ms.min(ts_ms));
            }
        }
        self.marca_b_previa = Some(marca_b_ms);
        self.mid_a_previo = (precio_a.is_finite() && precio_a > 0.0).then_some(precio_a);
    }

    /// CL-97: veredicto a la τ de decisión `tau_decision_ms`, con `piso` como
    /// resolución de la base (≥ 0, la σ mínima del z). `None` sin prueba o
    /// sin ningún par.
    pub fn veredicto(&self, tau_decision_ms: f64, piso: f64) -> Option<VeredictoStatArb> {
        let v = self.prueba.as_ref()?.veredicto(tau_decision_ms)?;
        let piso = if piso.is_finite() { piso.max(0.0) } else { 0.0 };
        let sigma = v.sigma.max(piso);
        let z = (v.base - v.media) / sigma;
        let demostrada = v.demostrada && sigma > 0.0 && z.is_finite();
        Some(VeredictoStatArb {
            z: if demostrada { z } else { 0.0 },
            t_media_cota_ms: if demostrada { 2.0 * v.tau_nodo_ms } else { f64::INFINITY },
            demostrada,
            prueba: v,
        })
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

        // CL-97 (F3.7-02, v2): el voto sale SÓLO de la prueba de reversión
        // que el núcleo hace sobre la base futuro-spot (`statarb_ou_zscore`,
        // `statarb_half_life_ms`). Sin esas claves la moneda no tiene pares
        // spot-futuro y StatArb se abstiene: ya no cae a `vecm_zscore`, que
        // votaba reversión sin haberla demostrado y es el escalar que vota
        // VECM (F3.7-05). Un 0,0 publicado es abstención: sin reversión
        // demostrada o con el spot caducado.
        // CL-80 (F3.7-04, LC-F1-07): con símbolo, lectura ESTRICTA (clave de
        // la moneda o del símbolo, nunca la global). `set_reg` del núcleo
        // también escribe la global, que es la de la ÚLTIMA moneda que
        // escribió: una moneda sin prueba propia votaba con la z y la t½ de
        // otra. Sin símbolo (`evaluate()`), la lectura global se conserva.
        let leer = |k: &str| {
            if sym_opt.is_some() {
                r.get_scoped_parameter_strict(sym_opt, cid_opt, k, "StatArbEngine")
            } else {
                r.get_scoped_parameter(sym_opt, cid_opt, k, "StatArbEngine")
            }
        };
        let own_z = leer("statarb_ou_zscore").map(|p| p.get_value());

        let z = match own_z {
            Some(z) if z.is_finite() => {
                // Guarda espectral: t½ publicada (CL-97: la cota 2·τ_k de la
                // prueba, ∞ sin demostración) contra τ* dominante de la
                // moneda. Si la reversión no está demostrada por debajo del
                // doble del ciclo dominante, la base puede ser deriva: veto.
                let t_half_ms = leer("statarb_half_life_ms")
                    .map(|p| p.get_value())
                    .unwrap_or(f64::INFINITY);
                let tau_dom_ms = leer("dominant_tau_ms")
                    .map(|p| p.get_value())
                    .unwrap_or(DEFAULT_DOMINANT_TAU_MS);
                // Fail-closed: NaN en cualquiera de las dos cotas ⇒ veto.
                if !(t_half_ms <= 2.0 * tau_dom_ms) {
                    return 0.0;
                }
                z
            }
            // Clave presente pero venenosa (NaN/Inf), o ausente: abstención.
            _ => return 0.0,
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

    fn is_operable_for_coin(&self, coin_id: usize, symbol: &str) -> bool {
        let sym_opt = if symbol.is_empty() { None } else { Some(symbol) };
        let cid_opt = if symbol.is_empty() { None } else { Some(coin_id) };
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return false,
        };
        let leer = |k: &str| {
            if sym_opt.is_some() {
                r.get_scoped_parameter_strict(sym_opt, cid_opt, k, "StatArbEngine")
            } else {
                r.get_scoped_parameter(sym_opt, cid_opt, k, "StatArbEngine")
            }
        };
        leer("statarb_ou_zscore")
            .map(|p| p.get_value())
            .filter(|v| v.is_finite())
            .is_some()
            && leer("statarb_half_life_ms")
                .map(|p| p.get_value())
                .filter(|v| v.is_finite() && *v > 0.0 && !v.is_infinite())
                .is_some()
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

    /// LCG de 64 bits (Knuth/MMIX) con normal por Box-Muller.
    struct Lcg(u64);
    impl Lcg {
        fn u(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn n(&mut self) -> f64 {
            let (u1, u2) = (self.u(), self.u());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// CL-97. Verdad: un par nace sólo cuando cambia la marca de llegada de
    /// B, con el mid de A del evento anterior (el vigente cuando llegó B) y
    /// la marca de llegada. Un B que no se refresca no forma pares.
    #[test]
    fn cl97_los_pares_son_sincronos() {
        let mut e = StatArbEngine::new(30, 1.5).with_prueba_reversion();
        // B llega una vez (marca 0) y no se refresca: ningún par.
        for i in 0..50u64 {
            e.observar_evento(100.0 + i as f64, 100.0, 0, 1_000 + i * 1_000);
        }
        assert_eq!(e.prueba.as_ref().unwrap().pares(), 0);
        // B llega en 51 000 (marca del evento anterior, 50 000): el par usa
        // el mid de A del evento 50 000 (149), no el del 51 000 (500).
        e.observar_evento(500.0, 120.0, 50_000, 51_000);
        let p = e.prueba.as_ref().unwrap();
        assert_eq!(p.pares(), 1);
        let v = e.veredicto(60_000.0, 0.0).expect("hay un par");
        assert!((v.prueba.base - (149.0f64.ln() - 120.0f64.ln())).abs() < 1e-15, "{v:?}");
        // Sin dato de B válido no hay par, pero el mid de A se actualiza.
        e.observar_evento(200.0, 0.0, 52_000, 53_000);
        assert_eq!(e.prueba.as_ref().unwrap().pares(), 1);
    }

    /// CL-97. Verdad: la RLS de β no actúa sobre la prueba. Con
    /// `adaptive_beta`, `observar_evento` deja β en 1.
    #[test]
    fn cl97_la_prueba_usa_beta_fija() {
        let mut e = StatArbEngine::new(30, 1.5)
            .with_adaptive_beta(true)
            .with_prueba_reversion();
        for i in 0..200u64 {
            let b = 50.0 + i as f64 * 0.5;
            e.observar_evento(b.powf(1.8), b, i * 1_000, i * 1_000 + 500);
        }
        assert_eq!(e.beta_hedge_ratio.to_bits(), 1f64.to_bits());
    }

    /// CL-97. Verdad: una base OU con vida media de τ_17/5 (3,4 s), con
    /// pares cada 500 ms durante cuatro memorias del nodo 17, demuestra la
    /// reversión a τ* = 30 s (nodo 17): cota publicada 2·τ_17 y z del último
    /// par. Sin pares suficientes no hay demostración y la cota es ∞.
    #[test]
    fn cl97_una_reversion_rapida_se_demuestra() {
        use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
        let tau17 = SPECTRUM_SCALES_MS[17];
        let theta = std::f64::consts::LN_2 / (tau17 / 5.0);
        let mut g = Lcg(11);
        let mut e = StatArbEngine::new(30, 1.5).with_prueba_reversion();
        let (dt, sd) = (500.0, 1e-4);
        let rho = (-theta * dt).exp();
        let mut x = 0.0f64;
        let mut lf = 50_000f64.ln();
        let pasos = (4.0 * 16.0 * tau17 / dt) as u64;
        for i in 0..pasos {
            x = rho * x + sd * (1.0 - rho * rho).sqrt() * g.n();
            lf += 1e-4 * g.n();
            let ts = 1_000 + i * dt as u64;
            e.observar_par(lf.exp(), (lf - x - 3e-4).exp(), ts);
            if i == 100 {
                let v = e.veredicto(30_000.0, 0.0).expect("hay pares");
                assert!(!v.demostrada && v.t_media_cota_ms.is_infinite() && v.z == 0.0, "{v:?}");
            }
        }
        let v = e.veredicto(30_000.0, 0.0).expect("hay pares");
        assert!(v.demostrada, "{v:?}");
        assert_eq!(v.t_media_cota_ms, 2.0 * tau17);
        assert!(v.z.is_finite() && v.z != 0.0, "{v:?}");
        // La misma prueba a una τ bajo la malla no demuestra nada.
        assert!(!e.veredicto(10_000.0, 0.0).unwrap().demostrada);
    }

    #[test]
    fn test_stat_arb_adaptive_beta_rls() {
        let mut engine = StatArbEngine::new(20, 1.5)
            .with_adaptive_beta(true)
            .with_beta_hedge_ratio(1.0);

        // Actualizar donde el precio A sistemáticamente se mueve al doble que B (beta ~ 2.0).
        // CL-97: la RLS sólo actúa en `update` (la prueba de reversión usa β fija).
        for i in 1..40 {
            let p_b = 50.0 + (i as f64 * 0.5);
            let p_a = p_b.powf(1.8);
            let _ = engine.update(p_a, p_b);
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
        for i in 1..=50 {
            let p_b = 10.0 + (i as f64) * 0.2;
            let p_a = p_b.powf(2.5);
            engine.update(p_a, p_b);
        }

        // Con RLS estricto y covarianza P_t, beta converge directamente hacia 2.5
        assert!((engine.beta_hedge_ratio - 2.5).abs() < 0.15, "RLS debe converger hacia 2.5: got {}", engine.beta_hedge_ratio);
        // Covarianza P_t debe haberse contraído
        assert!(engine.rls_p < 1.0, "La covarianza P_t debe contraerse con la evidencia: got {}", engine.rls_p);
    }

    // ===== OLA 73 (R6-A3/A4/A5/B1/B2/B13): StatArb honesto =====

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

        // Física propia publicada: z = 2.5 (dislocación corta), cota de t½ =
        // 60 s, τ* dominante = 60 s ⇒ t½ ≤ 2τ* ⇒ el voto vive y apunta a Short.
        reg.set("statarb_ou_zscore", 2.5);
        reg.set("statarb_half_life_ms", 60_000.0);
        reg.set("dominant_tau_ms", 60_000.0);
        // vecm_zscore con signo OPUESTO: no lo lee (CL-97).
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

        // Half-life AUSENTE ⇒ cota INFINITA ⇒ veto honesto aunque la z propia
        // llegue publicada.
        let reg2 = Arc::new(OmniscientRegistry::new());
        let mut engine2 = StatArbEngine::new(30, 1.5);
        assert!(engine2.init(reg2.clone()).is_ok());
        reg2.set("statarb_ou_zscore", 2.5);
        reg2.set("dominant_tau_ms", 60_000.0);
        assert_eq!(engine2.evaluate(), 0.0);
    }

    /// CL-97 (F3.7-02 v2, F3.7-05): sin claves propias StatArb se abstiene;
    /// ya no cae a `vecm_zscore` (votaba reversión sin demostrarla, con el
    /// escalar que vota VECM).
    #[test]
    fn cl97_sin_prueba_propia_statarb_se_abstiene() {
        let reg = Arc::new(OmniscientRegistry::new());
        let mut engine = StatArbEngine::new(30, 1.5);
        assert!(engine.init(reg.clone()).is_ok());
        for z in [-3.0, -2.0, 0.5, 2.0, 3.0] {
            reg.set("vecm_zscore", z);
            reg.set("cointegration_zscore", z);
            assert_eq!(engine.evaluate(), 0.0, "vecm_zscore = {z}");
        }
    }

    #[test]
    fn test_stat_arb_ola73_evaluate_zscore_cero_abstiene() {
        let reg = Arc::new(OmniscientRegistry::new());
        let mut engine = StatArbEngine::new(30, 1.5);
        assert!(engine.init(reg.clone()).is_ok());

        // El núcleo publica 0.0 sin reversión demostrada o con el spot
        // caducado: abstención (la clave publicada es la autoridad del
        // lector; desde CL-97 no hay respaldo en vecm_zscore).
        reg.set("statarb_ou_zscore", 0.0);
        reg.set("vecm_zscore", 2.5);
        assert_eq!(engine.evaluate(), 0.0);
    }
}
