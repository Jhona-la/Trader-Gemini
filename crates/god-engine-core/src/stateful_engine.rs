use crate::math_kernels::{
    ContinuousVPIN, ExponentialDecayTensor, FundingRateElasticity, ObiAcceleration, RecursiveHurst,
    ShannonEntropy,
};
use feature_engine::OrderFlowTracker;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// C-02 — índices del vector 34D (`get_universal_features`) cuya fuente de datos
/// está MUERTA en producción: hoy sólo [9] (dark_alpha / dex_severity, sin
/// productor MEV-DEX en vivo: `update_macro_features` la recibe como 0.0
/// constante). Contrato del trainer: estos índices van a 0.0 TAMBIÉN en
/// entrenamiento (train_forest) — el modelo no puede aprender a depender de
/// una columna que en serve es constante. Si un productor dex revive la
/// fuente, actualizar este slice y re-entrenar en el mismo cambio.
pub const FEATURES_DEAD_IN_SERVE: &[usize] = &[4, 5, 9, 10];
// B3.35: [4][5][10] (obi_accel) añadidas — el OBI del trainer (aggTrades
// sintético con is_buyer_maker) tiene DISTRIBUCIÓN INCOMPATIBLE con el OBI
// del vivo (depth L2 real). El modelo entrenado con OBI sintético predice
// ~0.503 en vivo (verificado en v41: señal muerta, cero entradas). Zerificar
// en AMBOS lados restaura la transferencia del modelo. Si en el futuro se
// calibra un OBI sintético que matchee la distribución del libro, retirar
// estos índices y re-entrenar.

#[derive(Debug, PartialEq, Clone, Copy, Default)]
/// U-6 (MOTOR UNIVERSAL CONTINUO): variantes Scalping/Swing extirpadas —
/// el régimen del motor continuo es Continuous/Neutral (el régimen MACRO
/// mayor vive en risk_engine::regime::MarketRegime, ortogonal).
pub enum MarketRegime {
    #[default]
    Continuous,
    Neutral,
}

/// Internal recursive state using strictly f64 (Double Precision)
#[repr(C, align(64))]
pub struct StatefulEngine {
    pub order_flow: OrderFlowTracker,
    pub ofi_model: feature_engine::OFIModel,
    pub rsi_rs: f64,
    pub last_price: f64,
    pub v_t: f64,
    pub a_t: f64,
    /// MOD2/7-031 — EMA del centroide espectral (adimensional, actualizada
    /// cada 64 ticks). ANTES vivía en `a_t` y era sobrescrita al tick
    /// siguiente por la aceleración cinemática ($/tick): escritor doble con
    /// un solo sobreviviente. Separada: `a_t` es SÓLO la cinemática
    /// instantánea que alimentan las features (norm_at) y el registry.
    pub a_t_spectral: f64,
    pub last_inst_v: f64,
    pub dir_velocity: f64,
    pub tick_count: u64,
    pub hurst: RecursiveHurst,
    pub obi_accel: ObiAcceleration,
    /// D-688: media y varianza exponenciales del OBI (ruido del libro).
    pub obi_noise: ObiNoise,
    pub fr_elasticity: FundingRateElasticity,
    pub cvpin: ContinuousVPIN,
    pub entropy: ShannonEntropy,
    pub dark_alpha: ExponentialDecayTensor,
    pub last_entropy: f64,
    // Add compatibility properties so swing engine isn't completely broken
    pub ema_fast: f64,
    pub ema_slow: f64,
    pub omni: feature_engine::OmniStrategyEngine,
    pub spectral: feature_engine::SpectralCycleEngine,
    pub multifractal: feature_engine::MultiScaleHurstConfluence,
    pub lead_lag: feature_engine::LeadLagAlphaEngine,
    pub regime: MarketRegime,
    // Native Kline Aggregator
    pub kline_start_ms: u64,
    pub kline_open: f64,
    pub kline_high: f64,
    pub kline_low: f64,
    pub kline_volume: f64,
    pub kline_ema_fast: f64,
    pub kline_ema_slow: f64,
    pub kline_ema_trend: f64,
    pub kline_ema_macro: f64,
    /// B2.3 — bloque espectral publicado (antes calculado y descartado):
    /// frecuencia dominante, potencia y centroide del FFT de 64 retornos;
    /// Hurst multifractal en 3 escalas. + momentum multiescala vía las EMAs
    /// de kline. Es el vector que el F8 dice que DECIDE, ahora alcanzable
    /// por el entrenamiento (train_forest) y por la inferencia, idéntico.
    pub spectral_bin: f32,
    pub spectral_power: f32,
    pub spectral_centroid: f32,
    pub hurst_micro: f32,
    pub hurst_meso: f32,
    pub hurst_macro: f32,
    pub last_scalp_exit_tick: u64,
    pub last_scalp_exit_ts: u64,
    pub last_exit_tau_ms: u64,
    pub current_ts: u64,
    pub last_scalp_was_loss: bool,
    pub scalp_loss_streak: u32,
    pub scalp_short_loss_streak: u32,
    pub scalp_long_loss_streak: u32,
    pub spectral_loss_streaks: [u32; 3], // [0: Micro (<60s), 1: Meso (60s..30m), 2: Macro (>=30m)]
    pub spectral_directional_loss_streaks: [[u32; 2]; 3], // [band][0: Short, 1: Long]
    pub spectral_exit_ts: [u64; 3],
    pub last_trade_is_sell: bool,
    /// #18: Filtro de Kalman 1D para estimar el micro-precio justo en O(1)
    pub kalman: feature_engine::KalmanFilter1D,
    pub fair_price: f64,
    /// #21: Motor de cuantiles adaptativos P^2 para estimación de percentiles sin alocar
    pub quantiles: quantum_arena::AdaptiveQuantileEngine,
    /// #19: Anillo tensorial de precios para cálculo O(1) de derivadas cinemáticas multiescala
    pub price_ring: feature_engine::TensorRing<16>,
    /// #19: Jerk cinemático instantáneo (3ra derivada del precio: d(a_t)/dt)
    pub jerk_t: f64,
    /// #20: Red neuronal SIMD MLP ultraligera para inferencia vectorial AVX2 en L1 cache
    pub simd_nn: feature_engine::SimdNeuralNet,
    /// #16: Motor estocástico Hawkes de auto-excitación y clustering de flujo de órdenes
    pub hawkes: feature_engine::HawkesProcessEngine,
    pub last_hawkes_ratio: f64,
}

impl Default for StatefulEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl StatefulEngine {
    pub fn new() -> Self {
        DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            order_flow: OrderFlowTracker::new(),
            ofi_model: feature_engine::OFIModel::new(),
            rsi_rs: 0.0,
            last_price: 0.0,
            v_t: 0.0,
            a_t: 0.0,
            a_t_spectral: 0.0,
            last_inst_v: 0.0,
            dir_velocity: 0.0,
            tick_count: 0,
            hurst: RecursiveHurst::new(),
            obi_accel: ObiAcceleration::new(),
            obi_noise: ObiNoise::new(),
            fr_elasticity: FundingRateElasticity::new(),
            cvpin: ContinuousVPIN::new(10_000.0), // $10,000 USD rolling bucket size
            entropy: ShannonEntropy::new(),
            dark_alpha: ExponentialDecayTensor::new(10000.0), // 10s half-life
            last_entropy: 0.0,
            ema_fast: 0.0,
            ema_slow: 0.0,
            omni: feature_engine::OmniStrategyEngine::new(),
            spectral: feature_engine::SpectralCycleEngine::new(),
            multifractal: feature_engine::MultiScaleHurstConfluence::new(),
            lead_lag: feature_engine::LeadLagAlphaEngine::new(50),
            regime: MarketRegime::Neutral,
            kline_start_ms: 0,
            kline_open: 0.0,
            kline_high: 0.0,
            kline_low: 0.0,
            kline_volume: 0.0,
            kline_ema_fast: 0.0,
            kline_ema_slow: 0.0,
            kline_ema_trend: 0.0,
            kline_ema_macro: 0.0,
            spectral_bin: 0.0,
            spectral_power: 0.0,
            spectral_centroid: 0.0,
            hurst_micro: 0.5,
            hurst_meso: 0.5,
            hurst_macro: 0.5,
            last_scalp_exit_tick: 0,
            last_scalp_exit_ts: 0,
            last_exit_tau_ms: 0,
            current_ts: 0,
            last_scalp_was_loss: false,
            scalp_loss_streak: 0,
            scalp_short_loss_streak: 0,
            scalp_long_loss_streak: 0,
            spectral_loss_streaks: [0; 3],
            spectral_directional_loss_streaks: [[0; 2]; 3],
            spectral_exit_ts: [0; 3],
            last_trade_is_sell: false,
            kalman: feature_engine::KalmanFilter1D::new(0.0, 1.0, 1e-4, 0.1),
            fair_price: 0.0,
            quantiles: quantum_arena::AdaptiveQuantileEngine::new(),
            price_ring: feature_engine::TensorRing::new(),
            jerk_t: 0.0,
            simd_nn: feature_engine::SimdNeuralNet::default(),
            hawkes: feature_engine::HawkesProcessEngine::new(0.05, 0.35, 1.5),
            last_hawkes_ratio: 0.0,
        }
    }

    #[inline(always)]
    pub fn spectral_band_index(tau_ms: f64) -> usize {
        if tau_ms < 60_000.0 {
            0
        } else if tau_ms < 1_800_000.0 {
            1
        } else {
            2
        }
    }

    /// Registra el resultado de un trade desacoplado en la banda espectral correspondiente
    #[inline(always)]
    pub fn record_trade_outcome(
        &mut self,
        tau_ms: f64,
        is_directional_loss: bool,
        is_long: bool,
        tick: u64,
        ts: u64,
    ) {
        let band = Self::spectral_band_index(tau_ms);
        self.last_exit_tau_ms = (tau_ms.max(10.0)).round() as u64;
        self.spectral_exit_ts[band] = ts;
        self.last_scalp_exit_tick = tick;
        self.last_scalp_exit_ts = ts;
        self.last_scalp_was_loss = is_directional_loss;

        let dir_idx = if is_long { 1 } else { 0 };
        if is_directional_loss {
            self.spectral_loss_streaks[band] += 1;
            self.spectral_directional_loss_streaks[band][dir_idx] += 1;
            self.scalp_loss_streak += 1;
            if is_long {
                self.scalp_long_loss_streak += 1;
            } else {
                self.scalp_short_loss_streak += 1;
            }
        } else {
            self.spectral_loss_streaks[band] = 0;
            self.spectral_directional_loss_streaks[band][dir_idx] = 0;
            self.scalp_loss_streak = 0;
            if is_long {
                self.scalp_long_loss_streak = 0;
            } else {
                self.scalp_short_loss_streak = 0;
            }
        }
    }

    /// Smart cooldown per asset con decaimiento temporal en milisegundos: evita parálisis eterna por rachas pasadas
    #[inline(always)]
    pub fn can_open_position(&self, min_cooldown_ms: u64) -> bool {
        self.can_open_at_tau(30_000.0, min_cooldown_ms)
    }

    /// Smart cooldown espectral continuo por escala armónica tau_candidate_ms.
    /// Si la racha de pérdidas activa ocurrió en una escala ortogonal (|Δ ln τ| >= 0.60),
    /// la nueva escala candidata NO hereda la penalización de una banda diferente.
    /// Además, el tiempo de cooldown se modula armónicamente por tau_candidate_ms para
    /// evitar que pérdidas en microsegundos congelen el motor durante 1 hora.
    #[inline(always)]
    pub fn can_open_at_tau(&self, tau_candidate_ms: f64, min_cooldown_ms: u64) -> bool {
        let band = Self::spectral_band_index(tau_candidate_ms);
        let same_spectral_band = if self.last_exit_tau_ms > 0 && tau_candidate_ms > 10.0 {
            let ln_cand = tau_candidate_ms.ln();
            let ln_last = (self.last_exit_tau_ms as f64).ln();
            (ln_cand - ln_last).abs() < 0.60
        } else {
            true
        };

        let last_band_ts = if self.spectral_exit_ts[band] > 0 {
            self.spectral_exit_ts[band]
        } else {
            self.last_scalp_exit_ts
        };
        let elapsed_ms = if self.current_ts > 0 && last_band_ts > 0 {
            self.current_ts.saturating_sub(last_band_ts)
        } else {
            self.tick_count.saturating_sub(self.last_scalp_exit_tick) * 100
        };

        let raw_streak = if same_spectral_band {
            self.spectral_loss_streaks[band].max(self.scalp_loss_streak)
        } else {
            self.spectral_loss_streaks[band]
        };
        let active_streak = if elapsed_ms > 3_600_000 {
            0
        } else if elapsed_ms > 1_800_000 {
            raw_streak.saturating_sub(1)
        } else {
            raw_streak
        };

        // Modulación armónica del cooldown por escala temporal tau:
        // En microescalas (tau ~ 5-15s), el cooldown requerido se relaja armónicamente.
        // En macroescalas (tau ~ 1-4h), el cooldown respira con el ciclo macro.
        let safe_tau = if tau_candidate_ms.is_finite() && tau_candidate_ms > 10.0 {
            tau_candidate_ms
        } else {
            30_000.0
        };
        let scale_factor = (safe_tau / 30_000.0).clamp(0.20, 5.0);

        let required_ms = match active_streak {
            0 => min_cooldown_ms,
            1 => {
                let base = if self.v_t > 0.0015 {
                    min_cooldown_ms * 4
                } else {
                    min_cooldown_ms * 2
                };
                ((base as f64) * scale_factor).round() as u64
            }
            2 => ((900_000.0 * scale_factor).clamp(60_000.0, 900_000.0)).round() as u64,
            3 => ((1_800_000.0 * scale_factor).clamp(120_000.0, 1_800_000.0)).round() as u64,
            _ => ((3_600_000.0 * scale_factor).clamp(300_000.0, 3_600_000.0)).round() as u64,
        };
        elapsed_ms >= required_ms
    }

    /// Obtiene la racha total de pérdidas activa considerando el decaimiento temporal en milisegundos
    #[inline(always)]
    pub fn get_active_total_loss_streak(&self) -> u32 {
        self.get_active_total_loss_streak_at_tau(30_000.0)
    }

    /// Obtiene la racha total de pérdidas activa desacoplada por banda espectral considerando el decaimiento temporal
    #[inline(always)]
    pub fn get_active_total_loss_streak_at_tau(&self, tau_ms: f64) -> u32 {
        let band = Self::spectral_band_index(tau_ms);
        let band_exit_ts = self.spectral_exit_ts[band];
        let elapsed_ms = if self.current_ts > 0 && band_exit_ts > 0 {
            self.current_ts.saturating_sub(band_exit_ts)
        } else {
            self.tick_count.saturating_sub(self.last_scalp_exit_tick) * 100
        };
        let raw = self.spectral_loss_streaks[band];
        // Decaimiento analítico continuo proporcional a la escala física tau:
        // Una perturbación a escala tau se disipa naturalmente en ~4 periodos de su frecuencia fundamental,
        // acotada entre 60 segundos (piso físico micro) y 2 horas (techo macro).
        let decay_window_ms = (4.0 * tau_ms.max(10.0)).clamp(60_000.0, 7_200_000.0) as u64;
        if elapsed_ms > decay_window_ms {
            0
        } else if elapsed_ms > decay_window_ms / 2 {
            raw.saturating_sub(1)
        } else {
            raw
        }
    }

    /// Obtiene la racha de pérdidas activa para una dirección (long/short), considerando el decaimiento temporal en milisegundos
    #[inline(always)]
    pub fn get_active_directional_streak(&self, is_long: bool) -> u32 {
        self.get_active_directional_streak_at_tau(is_long, 30_000.0)
    }

    /// Obtiene la racha direccional de pérdidas activa desacoplada por banda espectral considerando el decaimiento temporal
    #[inline(always)]
    pub fn get_active_directional_streak_at_tau(&self, is_long: bool, tau_ms: f64) -> u32 {
        let band = Self::spectral_band_index(tau_ms);
        let dir_idx = if is_long { 1 } else { 0 };
        let band_exit_ts = self.spectral_exit_ts[band];
        let elapsed_ms = if self.current_ts > 0 && band_exit_ts > 0 {
            self.current_ts.saturating_sub(band_exit_ts)
        } else {
            self.tick_count.saturating_sub(self.last_scalp_exit_tick) * 100
        };
        let raw = self.spectral_directional_loss_streaks[band][dir_idx];
        let decay_window_ms = (4.0 * tau_ms.max(10.0)).clamp(60_000.0, 7_200_000.0) as u64;
        if elapsed_ms > decay_window_ms {
            0
        } else if elapsed_ms > decay_window_ms / 2 {
            raw.saturating_sub(1)
        } else {
            raw
        }
    }

    /// Centra las predicciones ML en 0.50 con rango [-1.0, 1.0] en O(1).
    /// MOD2/7-002 (INFORME DECIMOCUARTO): los campos `ml_prob_ewma`/`ml_prob_var`
    /// (la supuesta "normalización adaptativa") se eliminaron — declarados,
    /// inicializados y jamás leídos: estado fantasma con contrato falsamente
    /// documentado. Este mapeo es estático por diseño.
    #[inline(always)]
    pub fn update_ml_prediction(&mut self, ml_prob: f64) -> f64 {
        if !ml_prob.is_finite() || ml_prob < 0.0 || ml_prob > 1.0 {
            return 0.0;
        }
        ((ml_prob - 0.50) * 2.0).clamp(-1.0, 1.0)
    }

    /// Flushes all internal buffers. Used to auto-heal time-series glitches after network disconnects.
    pub fn reset(&mut self) {
        self.order_flow = OrderFlowTracker::new();
        self.ofi_model = feature_engine::OFIModel::new();
        self.rsi_rs = 0.0;
        self.last_price = 0.0;
        self.v_t = 0.0;
        self.a_t = 0.0;
        self.a_t_spectral = 0.0;
        self.tick_count = 0;
        self.last_scalp_exit_tick = 0;
        self.last_scalp_exit_ts = 0;
        self.last_exit_tau_ms = 0;
        self.current_ts = 0;
        self.last_scalp_was_loss = false;
        self.scalp_loss_streak = 0;
        self.scalp_short_loss_streak = 0;
        self.scalp_long_loss_streak = 0;
        self.spectral_loss_streaks = [0; 3];
        self.spectral_directional_loss_streaks = [[0; 2]; 3];
        self.spectral_exit_ts = [0; 3];
        self.hurst = RecursiveHurst::new();
        self.obi_accel = ObiAcceleration::new();
        self.obi_noise = ObiNoise::new();
        self.fr_elasticity = FundingRateElasticity::new();
        self.cvpin = ContinuousVPIN::new(10_000.0);
        self.entropy = ShannonEntropy::new();
        self.dark_alpha = ExponentialDecayTensor::new(10000.0);
        self.last_entropy = 0.0;
        self.ema_fast = 0.0;
        self.ema_slow = 0.0;
        self.omni = feature_engine::OmniStrategyEngine::new();
        self.spectral = feature_engine::SpectralCycleEngine::new();
        self.multifractal = feature_engine::MultiScaleHurstConfluence::new();
        self.lead_lag = feature_engine::LeadLagAlphaEngine::new(50);
        self.regime = MarketRegime::Neutral;
        self.kline_start_ms = 0;
        self.kline_open = 0.0;
        self.kline_high = 0.0;
        self.kline_low = 0.0;
        self.kline_volume = 0.0;
        self.kline_ema_fast = 0.0;
        self.kline_ema_slow = 0.0;
        self.kline_ema_trend = 0.0;
        self.kline_ema_macro = 0.0;
        // D7 / MOD6/8-004: el bloque espectral publicado también se limpia —
        // tras un reset (reconexión WS) los buffers del espectro quedaron
        // vacíos y servir los valores del período anterior es servir historia
        // congelada. Neutro hasta que 64 retornos nuevos llenen el anillo.
        self.spectral_bin = 0.0;
        self.spectral_power = 0.0;
        self.spectral_centroid = 0.0;
        self.hurst_micro = 0.5;
        self.hurst_meso = 0.5;
        self.hurst_macro = 0.5;
        self.kalman = feature_engine::KalmanFilter1D::new(0.0, 1.0, 1e-4, 0.1);
        self.fair_price = 0.0;
        self.quantiles = quantum_arena::AdaptiveQuantileEngine::new();
        self.price_ring = feature_engine::TensorRing::new();
        self.jerk_t = 0.0;
        self.simd_nn = feature_engine::SimdNeuralNet::default();
        self.hawkes = feature_engine::HawkesProcessEngine::new(0.05, 0.35, 1.5);
        self.last_hawkes_ratio = 0.0;
    }

    /// Processes a new tick internally in f64
    pub fn process_tick(&mut self, price: f64, _volume: f64, event_time_ms: u64) {
        if price <= 0.0 || !price.is_finite() {
            return;
        }
        self.current_ts = event_time_ms;
        if self.last_price == 0.0 {
            self.ema_fast = price;
            self.ema_slow = price;
            self.kalman = feature_engine::KalmanFilter1D::new(price, 1.0, 1e-4, 0.1);
            self.fair_price = price;
        } else {
            // #18: Filtro de Kalman 1D actualizando el precio justo suavizado con R dinámico
            self.fair_price = self.kalman.update_with_dynamic_r(price, (price * 0.0005).max(1e-6));
            // S-8 — DECISIÓN DOCUMENTADA: los genes ema_fast_period /
            // ema_slow_period (~12.5/~25.1) NO se cablean aquí aunque
            // existan. Estos 20/200 alimentan la feature [0] del contrato
            // 34D del vector universal: cambiarlos SOLO en vivo rompería la
            // paridad train/serve que B3.30 y B3.35 pagaron caro por
            // restaurar (distribución de la feature distinta ⇒ el modelo
            // sirve ruido). Condición para activarlos: trainer leyendo el
            // MISMO genoma del símbolo + retrain completo del roster en el
            // MISMO cambio. Sustituir el ladder por signal_at(τ) del
            // espectro exige lo mismo.
            let alpha_fast = 2.0 / (20.0 + 1.0);
            let alpha_slow = 2.0 / (200.0 + 1.0);

            self.ema_fast = (price - self.ema_fast) * alpha_fast + self.ema_fast;
            self.ema_slow = (price - self.ema_slow) * alpha_slow + self.ema_slow;

            let diff = (price - self.last_price).abs();
            let norm_return = (price - self.last_price) / self.last_price;
            self.last_entropy = self.entropy.update(norm_return);
            // D7 / MOD6/8-004 (INFORME DECIMOCUARTO) — ESPECTRO VIVO EN
            // process_tick: en producción ESTE es el camino que corre
            // (process_event → process_tick_dual → process_tick);
            // process_kline sólo alimenta el warmup REST de arranque. El
            // SpectralCycleEngine acumula AQUÍ el RETORNO de cada tick y el
            // FFT Radix-2 se re-analiza cada 64 retornos (contador
            // tick_count); el multifractal se actualiza con el PRECIO de
            // cada tick. Las 6 features espectrales [0..6] del vector ML
            // (bin/potencia/centroide FFT + Hurst micro/meso/macro) viven
            // por esta vía — no dependen de klines cerrados.
            self.spectral.push(norm_return);
            // D-434: Invocar análisis espectral FFT Radix-2 periódicamente cada 64 ticks
            if self.tick_count % 64 == 0 {
                let (dominant_bin, max_power, centroid) = self.spectral.analyze_spectrum();
                if max_power > 0.0 && centroid.is_finite() {
                    // MOD2/7-031: el EMA del centroide espectral vive en su
                    // PROPIO campo — antes escribía `a_t` y la cinemática del
                    // tick siguiente lo borraba (escritor doble, un solo
                    // sobreviviente). Observabilidad del espectro; no toca el
                    // contrato 48D (que consume spectral_centroid directo).
                    self.a_t_spectral = self.a_t_spectral * 0.95 + (centroid * 0.001) * 0.05;
                }
                // B2.3: el espectro ya se calculaba aquí y se DESCARTABA
                // (capacidad fantasma). Ahora se publica para el vector ML
                // — mismas features en vivo y en entrenamiento (paridad 1:1).
                self.spectral_bin = dominant_bin as f32;
                self.spectral_power = max_power as f32;
                self.spectral_centroid = centroid as f32;
            }
            let (h_mic, h_mes, h_mac, _score, _micro_p, _macro_p) = self.multifractal.update(price);
            self.hurst_micro = h_mic as f32;
            self.hurst_meso = h_mes as f32;
            self.hurst_macro = h_mac as f32;
            self.regime = MarketRegime::Continuous;

            // Tick-level instantaneous velocity & acceleration
            let inst_v = diff;
            // FIX D-52: Aceleración cinemática dimensionalmente correcta a = (v_t - v_{t-1}) / dt con filtro EMA
            let raw_a = inst_v - self.last_inst_v;
            self.a_t = self.a_t * 0.70 + raw_a * 0.30;
            self.last_inst_v = inst_v;
            // Velocidad direccional suavizada (EMA de 10 ticks)
            self.dir_velocity = self.dir_velocity * 0.85 + inst_v * 0.15;
            // #19: Actualización del anillo tensorial de precios y extracción de derivadas cinemáticas
            self.price_ring.push(self.fair_price);
            self.jerk_t = self.price_ring.jerk();
            // #21: Actualización de cuantiles adaptativos P^2 en O(1)
            self.quantiles.update(self.ofi_model.ema_ofi, self.obi_accel.prev_obi, self.a_t, self.v_t / price.max(1e-6));
        }

        // D-615b: el Hurst YA NO se alimenta por evento (ver el cierre de la
        // vela interna de 1 minuto, más abajo).
        let notional_usd = if _volume > 0.0 && price > 0.0 {
            _volume * price
        } else {
            _volume
        };
        // D-110: Regla canónica Lee-Ready (1991): en ticks planos (price == last_price),
        // propagar la dirección del tick previo en lugar de falsear hacia compra sistemática
        let is_sell = if self.last_price > 0.0 {
            if price < self.last_price {
                self.last_trade_is_sell = true;
                true
            } else if price > self.last_price {
                self.last_trade_is_sell = false;
                false
            } else {
                self.last_trade_is_sell
            }
        } else {
            false
        };
        self.cvpin.update(notional_usd, is_sell);

        if self.kline_start_ms == 0 {
            self.kline_start_ms = event_time_ms;
            self.kline_open = price;
            self.kline_high = price;
            self.kline_low = price;
            self.kline_volume = _volume;
            if self.v_t == 0.0 && price > 0.0 {
                self.v_t = price * 0.005; // Fallback inicial 50 bps True Range
            }
        } else {
            self.kline_high = self.kline_high.max(price);
            self.kline_low = self.kline_low.min(price);
            self.kline_volume += _volume;

            // FIX #1206: Actualizar features Omni en tiempo real en cada tick para eliminar desfase de 59s en la inferencia HFT
            self.omni.update(price, self.kline_high, self.kline_low);

            // D-435 & D-437: Actualización continua intra-vela del True Range (captura expansiones de volatilidad sin colapso a spread de tick)
            let intra_candle_tr = (self.kline_high - self.kline_low).max(price * 0.0010);
            if intra_candle_tr > self.v_t || self.v_t == 0.0 || !self.v_t.is_finite() {
                self.v_t = intra_candle_tr;
            }

            // Generate 1-minute Kline internally (60,000 ms) and update True Range EMA & Trend EMAs
            if event_time_ms.saturating_sub(self.kline_start_ms) >= 60000 {
                // D-615b (DÉCIMA OLA) — HURST MUESTREADO POR RELOJ.
                //
                // Se alimentaba en cada evento. Un estimador de memoria sobre
                // retornos por evento mide la microestructura del FEED, no la
                // del activo: en producción llegan eventos cada ~100 ms y en el
                // backtest forense barras de ~15 s, así que el mismo mercado
                // producía exponentes distintos en cada entorno (0,22 en el
                // forense). Y ese exponente gobierna la ley de escala del TP/SL,
                // el régimen, el Kelly, el apalancamiento, la duración y siete
                // umbrales de decisión.
                //
                // Se muestrea al cierre de la vela interna de 1 minuto: la
                // escala en la que se mide el ATR (v_t) y la de las velas REST
                // del calentamiento (`interval=1m`, vía `process_kline`). Así
                // el exponente describe la difusión entre 1 minuto y el
                // horizonte, que es exactamente lo que `tp_sl` necesita, y es
                // idéntico en producción y en backtest.
                self.hurst.update(price);
                // FIX #608: True Range robusto y no nulo para evitar distorsiones en SL dinámico
                let tr = (self.kline_high - self.kline_low).max(price * 0.0010);
                self.v_t = if self.v_t == 0.0 || !self.v_t.is_finite() {
                    tr
                } else {
                    (self.v_t * 0.85 + tr * 0.15).max(price * 0.0010)
                };

                // Actualizar EMAs de tendencia macro de 1 minuto (EMA 9 y EMA 21), tendencia intermedia (EMA 120 ~ 2 horas) y tendencia secular (EMA 720 ~ 12 horas)
                let alpha_k_fast = 2.0 / (9.0 + 1.0);
                let alpha_k_slow = 2.0 / (21.0 + 1.0);
                let alpha_k_trend = 2.0 / (120.0 + 1.0);
                let alpha_k_macro = 2.0 / (720.0 + 1.0);
                if self.kline_ema_fast == 0.0 {
                    self.kline_ema_fast = price;
                    self.kline_ema_slow = price;
                    self.kline_ema_trend = price;
                    self.kline_ema_macro = price;
                } else {
                    self.kline_ema_fast =
                        (price - self.kline_ema_fast) * alpha_k_fast + self.kline_ema_fast;
                    self.kline_ema_slow =
                        (price - self.kline_ema_slow) * alpha_k_slow + self.kline_ema_slow;
                    self.kline_ema_trend =
                        (price - self.kline_ema_trend) * alpha_k_trend + self.kline_ema_trend;
                    self.kline_ema_macro =
                        (price - self.kline_ema_macro) * alpha_k_macro + self.kline_ema_macro;
                }

                self.kline_start_ms = event_time_ms;
                self.kline_open = price;
                self.kline_high = price;
                self.kline_low = price;
                self.kline_volume = _volume;
            }
        }

        // #16: Auto-excitación estocástica de Hawkes (clustering de flujo)
        let vol_usd = _volume.max(0.0) * price;
        let (_, _, hawkes_r) = self.hawkes.update(
            event_time_ms,
            self.ofi_model.ema_ofi,
            vol_usd,
            10_000.0,
        );
        self.last_hawkes_ratio = hawkes_r;

        self.last_price = price;
        self.tick_count += 1;
    }

    /// #16: Actualiza el proceso de auto-excitación de Hawkes con timestamps y OFI
    #[inline(always)]
    pub fn update_hawkes(
        &mut self,
        timestamp_ms: u64,
        delta_ofi: f64,
        volume_usd: f64,
        volume_norm: f64,
    ) -> (f64, f64, f64) {
        let res = self.hawkes.update(timestamp_ms, delta_ofi, volume_usd, volume_norm);
        self.last_hawkes_ratio = res.2;
        res
    }

    pub fn update_trade_flow(&mut self, volume: f64, is_buyer_maker: bool) {
        self.order_flow.update(volume, is_buyer_maker);
    }

    /// Updates the Order Flow Imbalance (OFI) predictive model
    pub fn update_ofi(
        &mut self,
        bid_price: f64,
        ask_price: f64,
        bid_qty: f64,
        ask_qty: f64,
    ) -> f64 {
        self.ofi_model
            .update(bid_price, ask_price, bid_qty, ask_qty)
    }

    pub fn process_kline(&mut self, _open: f64, high: f64, low: f64, close: f64, _volume: f64) {
        // FIX #665: Descarte preventivo de klines con precios corruptos o no finitos
        // FIX: Erradicación del Feature Leakage (Ceguera Causal)
        // Usar los altos y bajos de la vela ANTERIOR para el cálculo actual de features de IA.
        // Si el modelo ve el high/low de esta misma vela, el backtest hace trampa leyendo el futuro.
        let prev_high = if self.kline_high > 0.0 {
            self.kline_high
        } else {
            high
        };
        let prev_low = if self.kline_low > 0.0 {
            self.kline_low
        } else {
            low
        };

        self.omni.update(close, prev_high, prev_low);

        // Guardar estado futuro para la próxima evaluación causal
        self.kline_high = high;
        self.kline_low = low;
        self.hurst.update(close);
        if self.last_price > 0.0 {
            self.spectral
                .push((close - self.last_price) / self.last_price);
        }
        let (_h_mic, _h_mes, _h_mac, _score, _micro_p, _macro_p) = self.multifractal.update(close);
        self.regime = MarketRegime::Continuous;
        // MOD2/7-032: `ema_fast`/`ema_slow` ya NO se escriben aquí. Tenían
        // DOS kernels: 20/200 (process_tick, el camino del trainer y del
        // backtest) y 12/26 (esta función, warmup REST 1m del vivo). El
        // estado era una quimera de dos escalas y una ruptura de paridad
        // train/serve en la feature (ema_fast−ema_slow)/ema_slow del vector
        // 34D: el vivo arrancaba con EMA de velas que el entrenamiento jamás
        // vio. Kernel ÚNICO 20/200 por ticks — el primer tick vivo siembra
        // (ema_fast==0 ⇒ =precio) y converge solo, idéntico a train.
        self.last_price = close;

        // FIX #624: True Range robusto y no nulo en kline processing
        let tr = (high - low).max(close * 0.0005);
        self.v_t = if self.v_t == 0.0 || !self.v_t.is_finite() {
            tr
        } else {
            (self.v_t * 0.8 + tr * 0.2).max(close * 0.0005)
        };
    }

    pub fn update_macro_features(
        &mut self,
        obi: f64,
        funding_rate: f64,
        dex_severity: f64,
        ts_ms: u64,
    ) {
        // FIX #665: Sanitizar macro features
        if !obi.is_finite() || !funding_rate.is_finite() || !dex_severity.is_finite() {
            return;
        }

        self.obi_accel.update(obi);
        self.obi_noise.update(obi);
        self.fr_elasticity.update(funding_rate, self.last_price);
        self.dark_alpha.apply_event(dex_severity, ts_ms);
    }

    pub fn update_macro_flow(
        &mut self,
        funding_rate: f64,
        dex_severity: f64,
        ts_ms: u64,
    ) {
        self.fr_elasticity.update(funding_rate, self.last_price);
        self.dark_alpha.apply_event(dex_severity, ts_ms);
    }

    pub fn get_market_regime(&self) -> MarketRegime {
        MarketRegime::Continuous
    }

    pub fn get_features(&self) -> [f32; 12] {
        let price_change = if self.last_price != 0.0 && self.ema_slow != 0.0 {
            (self.ema_fast - self.ema_slow) / self.ema_slow
        } else {
            0.0
        };

        let hurst = self.hurst.current();
        let ofi = self.ofi_model.ema_ofi;
        let vol_delta = self.order_flow.get_volume_delta_ratio();
        let norm_vt = if self.last_price > 0.0 {
            (self.v_t / self.last_price).clamp(0.0, 1.0)
        } else {
            0.005
        };
        let norm_at = if self.last_price > 0.0 {
            (self.a_t / self.last_price).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        [
            price_change as f32,
            hurst as f32,
            ofi as f32,
            norm_vt as f32,
            self.obi_accel.prev_obi_velocity as f32,
            self.obi_accel.prev_obi as f32,
            vol_delta as f32,
            (((self.cvpin.buy_volume - self.cvpin.sell_volume)
                / (self.cvpin.buy_volume + self.cvpin.sell_volume).max(1e-6))
            .clamp(-1.0, 1.0)) as f32,
            self.last_entropy as f32,
            self.dark_alpha.current_severity as f32,
            self.obi_accel.accel as f32,
            norm_at as f32,
        ]
    }

    /// C-02 (INFORME DECIMOCUARTO) — mapa vivo/muerto del contrato 34D:
    /// VIVAS en train y serve (microestructura/precio del PROPIO símbolo):
    /// [0..4], [6..9), [10..12) y las 22 omni [12..34] (RSI/MACD/BB/ATR/Fib
    /// de precio — NO del tensor 54D del omni_multiplexer).
    /// MUERTA en serve y en train: [9] dark_alpha (dex_severity=0.0, sin
    /// productor). VIVAS en serve (libro real vía update_macro_features),
    /// muertas en train salvo que el trainer llame update_macro_features:
    /// [4], [5], [10] (obi_accel). Ver FEATURES_DEAD_IN_SERVE.
    pub fn get_universal_features(&self) -> [f32; 34] {
        let micro = self.get_features();
        let omni_feats = self.omni.extract_features();

        // CERT-M2-H02: zerificar las dims MUERTAS en SERVE también (no
        // sólo en el trainer). B3.35 comentaba "zerificadas en AMBOS
        // lados" pero el serve seguía alimentando obi_accel vivo en dims
        // [4][5][10] mientras el trainer las zerificaba — cualquier
        // modelo pre-B3.35 con splits en esas dims servía una
        // distribución que nunca vio en entrenamiento. FEATURES_DEAD_IN_
        // SERVE es la fuente única de verdad del mapa vivo/muerto.
        let mut raw: [f32; 34] = [
            micro[0],
            micro[1],
            micro[2],
            micro[3],
            micro[4],
            micro[5],
            micro[6],
            micro[7],
            micro[8],
            micro[9],
            micro[10],
            micro[11],
            // Omni Features (22 slots)
            omni_feats[0],
            omni_feats[1],
            omni_feats[2],
            omni_feats[3],
            omni_feats[4],
            omni_feats[5],
            omni_feats[6],
            omni_feats[7],
            omni_feats[8],
            omni_feats[9],
            omni_feats[10],
            omni_feats[11],
            omni_feats[12],
            omni_feats[13],
            omni_feats[14],
            omni_feats[15],
            omni_feats[16],
            omni_feats[17],
            omni_feats[18],
            omni_feats[19],
            omni_feats[20],
            omni_feats[21],
        ];
        // CERT-M2-H02: aplicar el zerificado del contrato EN SERVE.
        for &d in FEATURES_DEAD_IN_SERVE {
            if d < 34 {
                raw[d] = 0.0;
            }
        }
        raw
    }

    /// Returns ATR as a percentage of last price for Stop Loss scaling
    pub fn get_atr_pct(&self) -> f64 {
        if self.last_price > 0.0 {
            self.v_t / self.last_price
        } else {
            0.0
        }
    }

    /// B2.3 — bloque espectral (10D) para el vector ML. Se concatena a las
    /// 34 swing features TANTO en inferencia como en train_forest: la
    /// directriz F8 (el espectro DECIDE) entra así al aprendizaje. Los
    /// árboles existentes no se rompen: sus splits viven en índices <34 y
    /// los nuevos árboles pueden usar 34..44.
    ///
    /// [0..3] FFT de 64 retornos: bin dominante /32, ln(1+potencia),
    /// centroide /32 — frecuencia y energía del ciclo vivo.
    /// [3..6] Hurst multifractal micro(10)/meso(25)/macro(50) — persistencia
    /// por escala: >0.55 tendencial, <0.45 mean-reverting.
    /// [6..10] momentum multiescala: ln(p/EMA_kline)×100 en 4 escalas
    /// crecientes (posición del precio dentro de su tendencia por escala).
    pub fn get_spectral_ml_features(&self) -> [f32; 10] {
        let p = self.last_price;
        let dev = |ema: f64| -> f32 {
            if p > 0.0 && ema > 0.0 {
                (((p / ema).ln() * 100.0) as f32).clamp(-20.0, 20.0)
            } else {
                0.0
            }
        };
        [
            (self.spectral_bin / 32.0).clamp(0.0, 1.0),
            (1.0 + self.spectral_power as f64).ln() as f32,
            (self.spectral_centroid / 32.0).clamp(0.0, 1.0),
            self.hurst_micro.clamp(0.0, 1.0),
            self.hurst_meso.clamp(0.0, 1.0),
            self.hurst_macro.clamp(0.0, 1.0),
            dev(self.kline_ema_fast),
            dev(self.kline_ema_slow),
            dev(self.kline_ema_trend),
            dev(self.kline_ema_macro),
        ]
    }

    /// Retorna la pendiente relativa del micro-trend (Tick-level EMA 12 vs 26)
    #[inline(always)]
    pub fn get_micro_trend(&self) -> f64 {
        if self.ema_slow > 0.0 {
            (self.ema_fast - self.ema_slow) / self.ema_slow
        } else {
            0.0
        }
    }

    /// Retorna la pendiente relativa del macro-trend (1-Minute Kline EMA 9 vs 21)
    #[inline(always)]
    pub fn get_macro_trend(&self) -> f64 {
        if self.kline_ema_slow > 0.0 {
            (self.kline_ema_fast - self.kline_ema_slow) / self.kline_ema_slow
        } else if self.ema_slow > 0.0 {
            (self.ema_fast - self.ema_slow) / self.ema_slow
        } else {
            0.0
        }
    }

    /// Retorna la pendiente del macro-trend de orden superior (Price vs 2-Hour EMA 120)
    #[inline(always)]
    pub fn get_higher_trend(&self) -> f64 {
        if self.kline_ema_trend > 0.0 {
            (self.last_price - self.kline_ema_trend) / self.kline_ema_trend
        } else if self.kline_ema_slow > 0.0 {
            (self.last_price - self.kline_ema_slow) / self.kline_ema_slow
        } else {
            0.0
        }
    }

    /// Retorna la pendiente del macro-trend secular (Price vs 12-Hour EMA 720)
    #[inline(always)]
    pub fn get_secular_trend(&self) -> f64 {
        if self.kline_ema_macro > 0.0 {
            (self.last_price - self.kline_ema_macro) / self.kline_ema_macro
        } else if self.kline_ema_trend > 0.0 {
            (self.last_price - self.kline_ema_trend) / self.kline_ema_trend
        } else {
            0.0
        }
    }

    /// Determina si la estructura de mercado es inequívocamente bajista (Death Cross de orden superior y precio bajo EMA de tendencia)
    #[inline(always)]
    pub fn is_macro_bear(&self) -> bool {
        if self.kline_ema_fast > 0.0 && self.kline_ema_slow > 0.0 && self.kline_ema_fast > self.kline_ema_slow {
            return false; // El momentum rápido (EMA 9 > EMA 21) es alcista: régimen no bajista
        }
        if self.kline_ema_macro > 0.0 && self.kline_ema_trend > 0.0 {
            self.last_price < self.kline_ema_trend
                && self.kline_ema_slow < self.kline_ema_trend
                && (self.last_price < self.kline_ema_macro || self.kline_ema_trend < self.kline_ema_macro)
        } else if self.kline_ema_trend > 0.0 && self.kline_ema_slow > 0.0 {
            self.last_price < self.kline_ema_trend && self.kline_ema_slow < self.kline_ema_trend
        } else if self.kline_ema_slow > 0.0 {
            self.last_price < self.kline_ema_slow
        } else {
            false
        }
    }

    /// Determina si la estructura de mercado es inequívocamente alcista (Golden Cross y Precio sobre EMA 120 y 720)
    #[inline(always)]
    pub fn is_macro_bull(&self) -> bool {
        if self.kline_ema_fast > 0.0 && self.kline_ema_slow > 0.0 && self.kline_ema_fast < self.kline_ema_slow {
            return false; // El momentum rápido (EMA 9 < EMA 21) es bajista: régimen no alcista
        }
        if self.kline_ema_macro > 0.0 && self.kline_ema_trend > 0.0 {
            self.last_price > self.kline_ema_trend
                && self.kline_ema_slow > self.kline_ema_trend
                && (self.last_price > self.kline_ema_macro || self.kline_ema_trend > self.kline_ema_macro)
        } else if self.kline_ema_trend > 0.0 && self.kline_ema_slow > 0.0 {
            self.last_price > self.kline_ema_trend && self.kline_ema_slow > self.kline_ema_trend
        } else if self.kline_ema_slow > 0.0 {
            self.last_price > self.kline_ema_slow
        } else {
            false
        }
    }

    /// Projects the state out to the f32 barrier (576 bytes / 144 floats)
    pub fn export_f32(&mut self, out: &mut [f32; 144]) {
        out[0] = self.ema_fast as f32;
        out[1] = self.ema_slow as f32;
        out[2] = self.rsi_rs as f32;
        out[3] = self.last_price as f32;
        // Read Hurst for external observability — NO mutation (use current(), not update())
        out[4] = self.hurst.current() as f32;
    }

    /// #20: Inferencia SIMD ultraligera sobre el vector de características universales 34D en registros AVX2
    #[inline(always)]
    pub fn infer_simd_alpha(&self) -> [f64; 2] {
        let f32_feats = self.get_universal_features();
        let mut f64_feats = [0.0; 34];
        for (dst, &src) in f64_feats.iter_mut().zip(f32_feats.iter()) {
            *dst = src as f64;
        }
        self.simd_nn.infer(&f64_feats)
    }

    /// #20: Aprendizaje online SIMD en microsegundos con clipping de gradientes y decaimiento L2
    #[inline(always)]
    pub fn train_simd_step(&mut self, target_idx: usize, lr: f64) {
        let f32_feats = self.get_universal_features();
        let mut f64_feats = [0.0; 34];
        for (dst, &src) in f64_feats.iter_mut().zip(f32_feats.iter()) {
            *dst = src as f64;
        }
        self.simd_nn.train_step(&f64_feats, target_idx, lr);
    }
}

/// D-688 (DÉCIMA OLA) — RUIDO DEL DESEQUILIBRIO DEL LIBRO.
///
/// El escudo de libro L2 vetaba con `|OBI| > 0,10`: un umbral absoluto que
/// significa cosas distintas en un libro profundo y estable y en uno fino y
/// ruidoso. La desviación típica del OBI se estima con una media y una varianza
/// exponenciales sobre `OBI_NOISE_EVENTS` eventos —la misma escala que la EMA
/// lenta de ticks del motor (α = 2/201)— y el escudo exige que la presión sea
/// significativa frente a ese ruido.
#[derive(Debug, Clone, Copy)]
pub struct ObiNoise {
    mean: f64,
    var: f64,
    count: u32,
}

/// Eventos de la ventana exponencial y mínimo para considerar la estimación.
pub const OBI_NOISE_EVENTS: u32 = 200;

impl Default for ObiNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl ObiNoise {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            var: 0.0,
            count: 0,
        }
    }

    #[inline]
    pub fn update(&mut self, obi: f64) {
        if !obi.is_finite() {
            return;
        }
        if self.count == 0 {
            self.mean = obi;
            self.var = 0.0;
        } else {
            let alpha = 2.0 / (OBI_NOISE_EVENTS as f64 + 1.0);
            let delta = obi - self.mean;
            self.mean += alpha * delta;
            self.var = (1.0 - alpha) * (self.var + alpha * delta * delta);
        }
        self.count = self.count.saturating_add(1);
    }

    /// Desviación típica del OBI, o `None` durante el calentamiento.
    #[inline]
    pub fn sd(&self) -> Option<f64> {
        if self.count < OBI_NOISE_EVENTS {
            None
        } else {
            Some(self.var.max(0.0).sqrt())
        }
    }
}

impl Drop for StatefulEngine {
    fn drop(&mut self) {
        DROP_COUNTER.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B2.3: el bloque espectral debe ser finito, estar en rango y VIVIR
    /// (cambiar con ticks con estructura cíclica) — la paranoia nace del
    /// diagnóstico "34 features congeladas sin edge".
    #[test]
    fn test_spectral_ml_features_finite_and_alive() {
        let mut engine = StatefulEngine::new();
        // Onda lenta + ruido: 8 minutos de ticks de 100ms con ciclo de 64s
        let mut ts: u64 = 1_789_000_000_000;
        for i in 0..4800 {
            let phase = (i as f64) * 0.1; // ~6.28 rad cada 63 ticks
            let price = 100.0 + (phase).sin() * 2.0;
            engine.process_tick(price, 1.0, ts);
            ts += 100;
        }
        let f = engine.get_spectral_ml_features();
        assert_eq!(f.len(), 10);
        for v in &f {
            assert!(v.is_finite(), "feature espectral no finita: {:?}", f);
        }
        assert!((f[3] >= 0.0) && (f[3] <= 1.0), "hurst micro fuera de rango");
        // El momentum multiescala debe haberse movido (EMAs de kline vivas):
        // en el pico de la onda el precio supera su EMA macro.
        assert!(
            f[6].abs() + f[7].abs() + f[8].abs() + f[9].abs() > 1e-3,
            "momentum multiescala muerto: {:?}",
            f
        );
    }

    /// D7 / MOD6/8-004 (INFORME DECIMOCUARTO): el espectro debe seguir VIVO
    /// tras el arranque con SOLO ticks — en vivo `process_kline` corre
    /// únicamente en el warmup REST inicial; todo lo demás es process_tick.
    /// Warmup con klines → tramo de ticks con ciclo rápido → las features
    /// espectrales deben REPUBLICARSE (moverse de sus valores post-warmup),
    /// y un cambio de régimen del ciclo bajo ticks debe volver a moverlas.
    #[test]
    fn d7_espectro_vivo_tras_warmup_solo_con_ticks() {
        let mut engine = StatefulEngine::new();
        // Warmup REST de arranque: 120 klines de 1m con ciclo lento (~51 velas).
        let mut ts: u64 = 1_789_000_000_000;
        for i in 0..120u64 {
            let o = 100.0 + ((i % 51) as f64).sin() * 3.0;
            let c = 100.0 + (((i + 1) % 51) as f64).sin() * 3.0;
            let h = o.max(c) + 0.4;
            let l = o.min(c) - 0.4;
            engine.process_kline(o, h, l, c, 50.0);
        }
        let post_warmup = engine.get_spectral_ml_features();

        // VIVO: SOLO process_tick. Ciclo rápido de 8 ticks (~2 ventanas de
        // FFT completas: 128 retornos nuevos).
        for i in 0..128 {
            let phase = (i as f64) * (std::f64::consts::TAU / 8.0);
            let price = 100.0 + phase.sin() * 2.0;
            engine.process_tick(price, 1.0, ts);
            ts += 100;
        }
        let post_live = engine.get_spectral_ml_features();

        // El bloque FFT [0..3] debe haberse republicado con el ciclo nuevo.
        let fft_changed = (0..3).any(|i| {
            (post_warmup[i] - post_live[i]).abs() > 1e-6
        });
        assert!(
            fft_changed,
            "FFT congelado post-arranque: {:?} vs {:?}",
            &post_warmup[0..3],
            &post_live[0..3]
        );
        // El bloque multifractal [3..6] debe estar en rango y vivir (alguna
        // escala movida respecto al warmup por tick, no por kline).
        let hurst_changed = (3..6).any(|i| {
            (post_warmup[i] - post_live[i]).abs() > 1e-6
        });
        assert!(
            hurst_changed,
            "Hurst multifractal congelado post-arranque: {:?} vs {:?}",
            &post_warmup[3..6],
            &post_live[3..6]
        );

        // Cambio de régimen EN VIVO (solo ticks): ciclo lento de 64 ticks —
        // el FFT debe volver a moverse, probando republicación continua.
        for i in 0..128 {
            let phase = (i as f64) * (std::f64::consts::TAU / 48.0);
            let price = 100.0 + phase.sin() * 2.0;
            engine.process_tick(price, 1.0, ts);
            ts += 100;
        }
        let post_regime = engine.get_spectral_ml_features();
        let fft_changed_again = (0..3).any(|i| {
            (post_live[i] - post_regime[i]).abs() > 1e-6
        });
        assert!(
            fft_changed_again,
            "FFT no se republica ante cambio de régimen: {:?} vs {:?}",
            &post_live[0..3],
            &post_regime[0..3]
        );
    }

    /// C-02 (INFORME DECIMOCUARTO): el mapa vivo/muerto del contrato 34D es
    /// verificable, no documentación muerta. [9] (dark_alpha) sirve 0.0
    /// constante en producción (dex_severity sin productor) — y el trainer
    /// la fuerza a 0 vía FEATURES_DEAD_IN_SERVE. Las dims de obi_accel
    /// [4],[5],[10] VIVEN en serve: se mueven con el obi real del libro que
    /// `update_macro_features` recibe por evento.
    #[test]
    fn c02_mapa_vivo_muerto_del_vector_34d() {
        // Contrato del slice: índices dentro de las 34, únicos y ordenados.
        let mut sorted = FEATURES_DEAD_IN_SERVE.to_vec();
        sorted.sort_unstable();
        assert!(
            sorted.windows(2).all(|w| w[0] < w[1]),
            "FEATURES_DEAD_IN_SERVE con índices repetidos: {:?}",
            sorted
        );
        assert!(
            sorted.iter().all(|&i| i < 34),
            "índice fuera del contrato 34D: {:?}",
            sorted
        );

        // Serve: sin productor dex, dim [9] sirve exactamente 0.0.
        let mut e = StatefulEngine::new();
        let mut ts: u64 = 1_789_000_000_000;
        for i in 0..600u64 {
            let p = 100.0 + ((i % 37) as f64).sin();
            e.process_tick(p, 1.0, ts);
            let obi = 0.3 * (((i % 11) as f64) - 5.0) / 5.0;
            e.update_macro_features(obi, 0.0, 0.0, ts);
            ts += 100;
        }
        // Determinismo: dos obis distintos y no nulos al final.
        e.update_macro_features(0.25, 0.0, 0.0, ts);
        e.update_macro_features(0.35, 0.0, 0.0, ts + 100);
        let f = e.get_universal_features();
        assert_eq!(f.len(), 34);
        // CERT-M2-H02: dims [4][5][9][10] AHORA zerificadas en AMBOS lados
        // (serve incluido). El test anterior verificaba que obi_accel
        // vivía en serve — eso era EXACTAMENTE el defecto: el trainer
        // zerificaba pero el serve no, rompiendo la paridad.
        assert_eq!(f[9], 0.0, "dark_alpha debe servir 0.0 sin productor dex");
        assert_eq!(f[4], 0.0, "obi_accel [4] zerificado en serve (paridad B3.35)");
        assert_eq!(f[5], 0.0, "obi_accel [5] zerificado en serve (paridad B3.35)");
        assert_eq!(f[10], 0.0, "obi_accel [10] zerificado en serve (paridad B3.35)");
    }

    #[test]
    fn test_stateful_engine_reset_and_feature_extraction() {
        let mut engine = StatefulEngine::new();
        engine.reset();

        let micro_feats = engine.get_features();
        assert_eq!(micro_feats.len(), 12);
        for f in &micro_feats {
            assert!(f.is_finite(), "Micro feature debe ser finita");
        }

        let swing_feats = engine.get_universal_features();
        assert_eq!(swing_feats.len(), 34);
        for f in &swing_feats {
            assert!(f.is_finite(), "Swing feature debe ser finita");
        }
    }

    #[test]
    fn test_stateful_engine_market_regime_classification() {
        let engine = StatefulEngine::new();
        let regime = engine.get_market_regime();
        // U-6: el continuo no tiene variantes Scalping/Swing.
        assert!(matches!(regime, MarketRegime::Continuous | MarketRegime::Neutral));

        let atr_pct = engine.get_atr_pct();
        assert!(atr_pct.is_finite());
    }

    /// D-615b: el Hurst depende del RELOJ, no de la cadencia del feed. La
    /// misma ventana de 3 min 5 s alimentada a 100 ms (producción) y a 15 s
    /// (backtest forense) debe entregar al estimador las mismas muestras.
    #[test]
    fn d615b_hurst_se_muestrea_por_reloj_no_por_evento() {
        let t0: u64 = 1_700_000_000_000;
        let precio = |i: u64| 50_000.0 + ((i % 7) as f64 - 3.0);

        let mut rapido = StatefulEngine::new();
        for i in 0..1_850u64 {
            rapido.process_tick(precio(i), 1.0, t0 + i * 100);
        }
        let mut lento = StatefulEngine::new();
        for i in 0..13u64 {
            lento.process_tick(precio(i), 1.0, t0 + i * 15_000);
        }

        assert_eq!(
            rapido.hurst.samples(),
            lento.hurst.samples(),
            "la cadencia del feed no debe cambiar la muestra del estimador"
        );
        assert!(
            rapido.hurst.samples() <= 3,
            "3 cierres de minuto no pueden producir {} retornos",
            rapido.hurst.samples()
        );
    }

    /// D-688: la desviación estimada del OBI converge a la del ruido real y
    /// no se publica antes del calentamiento.
    #[test]
    fn d688_ruido_del_obi_se_estima_y_espera_al_calentamiento() {
        let mut n = ObiNoise::new();
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        for i in 0..20_000u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            // Uniforme en [−0,3; 0,3]: σ = 0,6/√12 ≈ 0,1732.
            n.update(-0.3 + 0.6 * u);
            if i + 1 < OBI_NOISE_EVENTS {
                assert!(n.sd().is_none());
            }
        }
        let sd = n.sd().expect("calentado");
        assert!((sd - 0.1732).abs() < 0.03, "σ estimada {sd}");
    }

    #[test]
    fn test_stateful_engine_export_f32_stability() {
        let mut engine = StatefulEngine::new();
        let mut buffer = [0.0f32; 144];
        engine.export_f32(&mut buffer);

        for val in &buffer[0..5] {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_hawkes_integration_in_stateful_engine() {
        let mut engine = StatefulEngine::new();
        assert_eq!(engine.last_hawkes_ratio, 0.0);

        // Actualizar con un tick normal
        engine.process_tick(50000.0, 0.5, 1000);
        assert!(engine.last_hawkes_ratio.is_finite());

        // Inyectar ráfagas alcistas intensas con OFI positivo
        for i in 1..=10 {
            let ts = 1000 + i * 50;
            engine.update_hawkes(ts, 0.8, 25000.0, 10000.0);
        }
        assert!(engine.last_hawkes_ratio > 0.0, "La ráfaga alcista debe inducir Hawkes ratio positivo: {}", engine.last_hawkes_ratio);
        assert!(engine.hawkes.intensity_bull > engine.hawkes.intensity_bear);

        // Reset
        engine.reset();
        assert_eq!(engine.last_hawkes_ratio, 0.0);
        assert_eq!(engine.hawkes.last_update_ms, 0);
    }

    #[test]
    fn test_can_open_at_tau_spectral_decoupling() {
        let mut engine = StatefulEngine::new();
        engine.current_ts = 1_000_000;
        engine.last_scalp_exit_ts = 1_000_000;
        engine.last_exit_tau_ms = 5_000; // Micro-scalp de 5 segundos
        engine.scalp_loss_streak = 3;    // Racha severa de 3 pérdidas en micro-escala

        // 1. A los 70 segundos (70_000 ms):
        // - El cooldown base min_cooldown_ms (60_000 ms) ya transcurrió.
        // - Para la escala micro (5s), la racha 3 impone un cooldown de 360_000 ms (> 70_000 ms) -> BLOQUEADO
        engine.current_ts = 1_000_000 + 70_000;
        assert!(!engine.can_open_at_tau(5_000.0, 60_000));
        assert!(!engine.can_open_at_tau(7_000.0, 60_000));

        // 2. Para la escala macro (τ = 1 hora = 3_600_000 ms, |Δ ln τ| = 6.57 >= 0.60):
        // Por desacoplamiento espectral, active_streak = 0 (no hereda la racha micro).
        // Cooldown requerido = min_cooldown_ms (60_000 ms).
        // 70_000 ms >= 60_000 ms -> DESBLOQUEADO (captura la onda macro sin parálisis)
        assert!(engine.can_open_at_tau(3_600_000.0, 60_000));

        // 3. A los 400 segundos (400_000 ms > 360_000 ms), el cooldown micro expira y se reabre
        engine.current_ts = 1_000_000 + 400_000;
        assert!(engine.can_open_at_tau(5_000.0, 60_000));
    }
}
