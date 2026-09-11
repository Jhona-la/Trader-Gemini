use crate::math_kernels::{
    ContinuousVPIN, ExponentialDecayTensor, FundingRateElasticity, ObiAcceleration, RecursiveHurst,
    ShannonEntropy,
};
use feature_engine::OrderFlowTracker;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub enum MarketRegime {
    #[default]
    Continuous,
    Scalping,
    Swing,
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
    pub ml_prob_ewma: f64,
    pub ml_prob_var: f64,
    pub last_scalp_exit_tick: u64,
    pub last_scalp_was_loss: bool,
    pub scalp_loss_streak: u32,
    pub scalp_short_loss_streak: u32,
    pub scalp_long_loss_streak: u32,
    pub last_trade_is_sell: bool,
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
            ml_prob_ewma: 0.0,
            ml_prob_var: 0.01,
            last_scalp_exit_tick: 0,
            last_scalp_was_loss: false,
            scalp_loss_streak: 0,
            scalp_short_loss_streak: 0,
            scalp_long_loss_streak: 0,
            last_trade_is_sell: false,
        }
    }

    /// Smart cooldown per asset con decaimiento temporal: evita parálisis eterna por rachas pasadas
    #[inline(always)]
    pub fn can_open_scalp(&self, min_cooldown: u64) -> bool {
        let elapsed = self.tick_count.saturating_sub(self.last_scalp_exit_tick);
        let active_streak = if elapsed > 18_000 {
            0
        } else if elapsed > 7_200 {
            self.scalp_loss_streak.saturating_sub(1)
        } else {
            self.scalp_loss_streak
        };
        let required = match active_streak {
            0 => min_cooldown,
            1 => {
                if self.v_t > 0.0015 {
                    min_cooldown * 4
                } else {
                    min_cooldown * 2
                }
            }
            2 => min_cooldown * 6,   // ~3,600 ticks (~15-20 min)
            3 => min_cooldown * 15,  // ~9,000 ticks (~40 min)
            _ => min_cooldown * 30,  // ~18,000 ticks (~1.5 horas)
        };
        elapsed >= required
    }

    /// Smart cooldown universal para posiciones en el espectro continuo
    #[inline(always)]
    pub fn can_open_position(&self, min_cooldown: u64) -> bool {
        self.can_open_scalp(min_cooldown)
    }

    /// Obtiene la racha de pérdidas activa para una dirección (long/short), considerando el decaimiento temporal
    #[inline(always)]
    pub fn get_active_directional_streak(&self, is_long: bool) -> u32 {
        let elapsed = self.tick_count.saturating_sub(self.last_scalp_exit_tick);
        let raw = if is_long {
            self.scalp_long_loss_streak
        } else {
            self.scalp_short_loss_streak
        };
        if elapsed > 18_000 {
            0
        } else if elapsed > 7_200 {
            raw.saturating_sub(1)
        } else {
            raw
        }
    }

    /// Normaliza adaptativamente las predicciones ML en O(1) centradas en 0.50 con rango [-1.0, 1.0]
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
        self.tick_count = 0;
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
    }

    /// Processes a new tick internally in f64
    pub fn process_tick(&mut self, price: f64, _volume: f64, event_time_ms: u64) {
        if price <= 0.0 || !price.is_finite() {
            return;
        }
        if self.last_price == 0.0 {
            self.ema_fast = price;
            self.ema_slow = price;
        } else {
            let alpha_fast = 2.0 / (20.0 + 1.0);
            let alpha_slow = 2.0 / (200.0 + 1.0);

            self.ema_fast = (price - self.ema_fast) * alpha_fast + self.ema_fast;
            self.ema_slow = (price - self.ema_slow) * alpha_slow + self.ema_slow;

            let diff = (price - self.last_price).abs();
            let norm_return = (price - self.last_price) / self.last_price;
            self.last_entropy = self.entropy.update(norm_return);
            self.spectral.push(norm_return);
            // D-434: Invocar análisis espectral FFT Radix-2 periódicamente cada 64 ticks
            if self.tick_count % 64 == 0 {
                let (_dominant_bin, max_power, centroid) = self.spectral.analyze_spectrum();
                if max_power > 0.0 && centroid.is_finite() {
                    self.a_t = self.a_t * 0.95 + (centroid * 0.001) * 0.05;
                }
            }
            let (_h_mic, _h_mes, _h_mac, _score, _micro_p, _macro_p) = self.multifractal.update(price);
            self.regime = MarketRegime::Continuous;

            // Tick-level instantaneous velocity & acceleration
            let inst_v = diff;
            // FIX D-52: Aceleración cinemática dimensionalmente correcta a = (v_t - v_{t-1}) / dt con filtro EMA
            let raw_a = inst_v - self.last_inst_v;
            self.a_t = self.a_t * 0.70 + raw_a * 0.30;
            self.last_inst_v = inst_v;
            // Velocidad direccional suavizada (EMA de 10 ticks)
            self.dir_velocity = self.dir_velocity * 0.85 + inst_v * 0.15;
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

        self.last_price = price;
        self.tick_count += 1;
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
        self.last_price = close;

        if self.ema_fast == 0.0 {
            self.ema_fast = close;
            self.ema_slow = close;
        } else {
            let alpha_fast = 2.0 / (12.0 + 1.0);
            let alpha_slow = 2.0 / (26.0 + 1.0);
            self.ema_fast = (close - self.ema_fast) * alpha_fast + self.ema_fast;
            self.ema_slow = (close - self.ema_slow) * alpha_slow + self.ema_slow;
        }

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

    /// Extracts Omni ML Features (SWING - 34D Macro+Micro)
    pub fn get_swing_features(&self) -> [f32; 34] {
        let micro = self.get_features();
        let omni_feats = self.omni.extract_features();

        [
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
        ]
    }

    /// Returns ATR as a percentage of last price for Stop Loss scaling
    pub fn get_atr_pct(&self) -> f64 {
        if self.last_price > 0.0 {
            self.v_t / self.last_price
        } else {
            0.0
        }
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

    #[test]
    fn test_stateful_engine_reset_and_feature_extraction() {
        let mut engine = StatefulEngine::new();
        engine.reset();

        let micro_feats = engine.get_features();
        assert_eq!(micro_feats.len(), 12);
        for f in &micro_feats {
            assert!(f.is_finite(), "Micro feature debe ser finita");
        }

        let swing_feats = engine.get_swing_features();
        assert_eq!(swing_feats.len(), 34);
        for f in &swing_feats {
            assert!(f.is_finite(), "Swing feature debe ser finita");
        }
    }

    #[test]
    fn test_stateful_engine_market_regime_classification() {
        let engine = StatefulEngine::new();
        let regime = engine.get_market_regime();
        assert!(matches!(
            regime,
            MarketRegime::Continuous
                | MarketRegime::Scalping
                | MarketRegime::Swing
                | MarketRegime::Neutral
        ));

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
}
