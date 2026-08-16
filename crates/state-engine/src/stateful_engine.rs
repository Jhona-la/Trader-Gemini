use std::sync::atomic::{AtomicUsize, Ordering};
use crate::math_kernels::{RecursiveHurst, FundingRateElasticity, ContinuousVPIN, ShannonEntropy, ExponentialDecayTensor};
use feature_engine::{OrderFlowTracker, MultifractalSpectrumEngine, LeadLagAlphaEngine, WelfordOnline};
use quantum_arena::AdaptiveQuantileEngine;
use strategy_core::{JohansenVecmEngine, ConformalPredictor};
use risk_engine::EpigeneticFitnessLandscapeEngine;

pub static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum MarketRegime {
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
    pub tick_count: u64,
    pub hurst: RecursiveHurst,
    pub fr_elasticity: FundingRateElasticity,
    pub cvpin: ContinuousVPIN,
    pub entropy: ShannonEntropy,
    pub dark_alpha: ExponentialDecayTensor,
    pub last_entropy: f64,
    // Add compatibility properties so swing engine isn't completely broken
    pub ema_fast: f64,
    pub ema_slow: f64,
    pub omni: feature_engine::OmniStrategyEngine,
    pub tick_high: f64,
    pub tick_low: f64,
    pub last_reset_ms: u64,
    // Motores Predictivos Fundacionales Integrados
    pub multifractal: MultifractalSpectrumEngine,
    pub welford: WelfordOnline,
    pub adaptive_quantile: AdaptiveQuantileEngine,
    pub lead_lag: LeadLagAlphaEngine,
    pub johansen: JohansenVecmEngine,
    pub conformal: ConformalPredictor,
    pub holder_exponent: f64,
    pub johansen_z: f64,
    pub cvd_tensor: feature_engine::TensorRing<4>,
    pub obi_tensor: feature_engine::TensorRing<4>,
    pub price_tensor: feature_engine::TensorRing<4>,
}



impl StatefulEngine {
    pub fn get_default_omni_macro() -> [f64; 54] {
        let mut macro_feats = [0.0; 54];
        macro_feats[12] = 100.0;  // open_interest
        macro_feats[13] = 1.05;   // long_short_ratio
        macro_feats[14] = 50.0;   // fear_greed_index
        macro_feats[15] = 55.0;   // altcoin_dominance (former M2 Supply comment)
        macro_feats[16] = 0.0;    // mempool_congestion (former CPI comment)
        macro_feats[17] = 0.0;    // usdt_mint_alert (former Interest comment)
        macro_feats[18] = 0.0;    // exchange_inflows (former GDP comment)
        macro_feats[19] = 0.0;    // exchange_outflows (former Funding comment)
        macro_feats[21] = 104.0;  // dxy
        macro_feats[22] = 5100.0; // sp500
        macro_feats[23] = 18000.0;// nasdaq
        macro_feats[24] = 15.0;   // vix
        macro_feats[25] = 4.2;    // us10y
        macro_feats[26] = 2300.0; // gold
        macro_feats[27] = 80.0;   // oil_wti
        macro_feats[29] = 5.5;    // fed_interest_rate
        macro_feats[30] = 0.5;    // spot_cvd
        macro_feats[31] = 0.5;    // futures_cvd
        macro_feats[32] = 1.0;    // taker_buy_sell_ratio
        macro_feats[33] = 0.001;  // futures_basis_premium
        macro_feats
    }

    pub fn new(config: &quantum_arena::QuantumConfig) -> Self {
        DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            order_flow: OrderFlowTracker::new(),
            ofi_model: feature_engine::OFIModel::new(),
            rsi_rs: 0.0,
            last_price: 0.0,
            v_t: 0.0,
            a_t: 0.0,
            tick_count: 0,
            hurst: RecursiveHurst::new(),
            fr_elasticity: FundingRateElasticity::new(),
            cvpin: ContinuousVPIN::new(config.cvpin_bucket_size.load(Ordering::Relaxed).max(10.0)),
            entropy: ShannonEntropy::new(),
            dark_alpha: ExponentialDecayTensor::new(config.dark_alpha_half_life.load(Ordering::Relaxed).max(100.0)),
            last_entropy: 0.0,
            ema_fast: 0.0,
            ema_slow: 0.0,
            omni: feature_engine::OmniStrategyEngine::new_with_params(
                config.volatility_window.load(Ordering::Relaxed),
                config.ema_fast_window.load(Ordering::Relaxed),
                config.ema_slow_window.load(Ordering::Relaxed),
                9.0,
                config.volatility_window.load(Ordering::Relaxed),
            ),
            tick_high: 0.0,
            tick_low: f64::MAX,
            last_reset_ms: 0,
            multifractal: MultifractalSpectrumEngine::default(),
            welford: WelfordOnline::new(),
            adaptive_quantile: AdaptiveQuantileEngine::new(),
            lead_lag: LeadLagAlphaEngine::default(),
            johansen: JohansenVecmEngine::new(
                config.vecm_alpha_speed.load(Ordering::Relaxed),
                config.vecm_beta_hedge.load(Ordering::Relaxed)
            ),
            conformal: ConformalPredictor::new(config.conformal_alpha.load(Ordering::Relaxed), 1000),
            holder_exponent: 0.5,
            johansen_z: 0.0,
            cvd_tensor: feature_engine::TensorRing::new(),
            obi_tensor: feature_engine::TensorRing::new(),
            price_tensor: feature_engine::TensorRing::new(),
        }
    }

    /// Flushes all internal buffers. Used to auto-heal time-series glitches after network disconnects.
    pub fn reset(&mut self, config: &quantum_arena::QuantumConfig) {
        self.order_flow = OrderFlowTracker::new();
        self.ofi_model = feature_engine::OFIModel::new();
        self.rsi_rs = 0.0;
        self.last_price = 0.0;
        self.v_t = 0.0;
        self.a_t = 0.0;
        self.tick_count = 0;
        self.hurst = RecursiveHurst::new();

        self.fr_elasticity = FundingRateElasticity::new();
        self.cvpin = ContinuousVPIN::new(config.cvpin_bucket_size.load(Ordering::Relaxed).max(10.0));
        self.entropy = ShannonEntropy::new();
        self.dark_alpha = ExponentialDecayTensor::new(config.dark_alpha_half_life.load(Ordering::Relaxed).max(100.0));
        self.last_entropy = 0.0;
        self.ema_fast = 0.0;
        self.ema_slow = 0.0;
        self.omni = feature_engine::OmniStrategyEngine::new_with_params(
            config.volatility_window.load(Ordering::Relaxed),
            config.ema_fast_window.load(Ordering::Relaxed),
            config.ema_slow_window.load(Ordering::Relaxed),
            9.0,
            config.volatility_window.load(Ordering::Relaxed),
        );
        self.tick_high = 0.0;
        self.tick_low = f64::MAX;
        self.last_reset_ms = 0;
        self.multifractal = MultifractalSpectrumEngine::default();
        self.welford = WelfordOnline::new();
        self.adaptive_quantile = AdaptiveQuantileEngine::new();
        self.lead_lag = LeadLagAlphaEngine::default();
        self.johansen = JohansenVecmEngine::new(
            config.vecm_alpha_speed.load(Ordering::Relaxed),
            config.vecm_beta_hedge.load(Ordering::Relaxed)
        );
        self.conformal = ConformalPredictor::new(config.conformal_alpha.load(Ordering::Relaxed), 1000);
        self.holder_exponent = 0.50;
        self.johansen_z = 0.0;
    }

    /// Processes a new tick internally in f64
    pub fn process_tick(&mut self, price: f64, _volume: f64, config: &quantum_arena::QuantumConfig) {
        if self.last_price == 0.0 {
            self.ema_fast = price;
            self.ema_slow = price;
            self.tick_high = price;
            self.tick_low = price;
        } else {
            // --- TIME DILATION (PHASE 46) ---
            let h_val = self.hurst.current();
            let mut dilation_factor = 1.0;
            if h_val > 0.65 {
                dilation_factor = 0.5; // Compress window (Faster reaction)
            } else if h_val < 0.45 {
                dilation_factor = 2.0; // Expand window (Smooth noise)
            }
            if self.last_entropy > 1.5 {
                dilation_factor *= 1.5; // Chaos requires more smoothing
            }
            
            let fast_window = (config.ema_fast_window.load(Ordering::Relaxed) * dilation_factor).max(2.0);
            let slow_window = (config.ema_slow_window.load(Ordering::Relaxed) * dilation_factor).max(2.0);
            let alpha_fast = 2.0 / (fast_window + 1.0);
            let alpha_slow = 2.0 / (slow_window + 1.0);
            
            self.ema_fast = (price - self.ema_fast) * alpha_fast + self.ema_fast;
            self.ema_slow = (price - self.ema_slow) * alpha_slow + self.ema_slow;
            
            let vol_window = (config.volatility_window.load(Ordering::Relaxed) * dilation_factor).max(2.0);
            let alpha_v = 2.0 / (vol_window + 1.0);
            
            // Fix: Use the mean absolute deviation from the slow EMA as the macro-volatility proxy,
            // rather than the microscopic tick-by-tick absolute difference.
            let ema_deviation = (price - self.ema_slow).abs();
            let new_v_t = self.v_t * (1.0 - alpha_v) + ema_deviation * alpha_v;
            
            self.a_t = new_v_t - self.v_t;
            self.v_t = new_v_t;
            
            let norm_return = if self.last_price > 0.0 {
                (price - self.last_price) / self.last_price
            } else {
                0.0
            };
            if norm_return.is_finite() {
                let base_entropy = self.entropy.update(norm_return);
                // Integración Epigenética: Mapeo de Paisaje de Fitness O(1)
                let fit_density = EpigeneticFitnessLandscapeEngine::compute_fitness_density(2.5, base_entropy, 1.5);
                self.last_entropy = base_entropy * fit_density;
            } else {
                self.last_entropy = 0.0;
            }
        }
        
        self.tick_high = self.tick_high.max(price);
        self.tick_low = self.tick_low.min(price);
        
        self.hurst.update(price);
        self.cvpin.update(_volume, price < self.last_price); // Approximation: tick down = seller initiated
        
        // Update Omni Strategy Engine for macro features
        self.omni.update(price, self.tick_high, self.tick_low);
        
        // Update pure O(1) foundational engines
        self.welford.update(price);
        let (holder, _) = self.multifractal.update(price);
        self.holder_exponent = holder;
        self.adaptive_quantile.update(self.v_t, self.a_t, _volume, 0.005);
        self.johansen_z = self.johansen.update(price, self.ema_slow);
        self.conformal.update(self.welford.std_dev(), self.a_t);
        
        // Finally, update the tracking variables
        self.last_price = price;
        self.tick_count += 1;
        self.price_tensor.push(price);
    }
    
    pub fn update_trade_flow(&mut self, volume: f64, is_buyer_maker: bool) {
        self.order_flow.update(volume, is_buyer_maker);
        self.cvd_tensor.push(self.order_flow.get_volume_delta_ratio());
    }
    
    /// Updates the Order Flow Imbalance (OFI) predictive model
    pub fn update_ofi(&mut self, bid_price: f64, ask_price: f64, bid_qty: f64, ask_qty: f64) -> f64 {
        self.ofi_model.update(bid_price, ask_price, bid_qty, ask_qty)
    }
    
    pub fn update_macro_features(&mut self, obi: f64, funding_rate: f64, dex_severity: f64, ts_ms: u64) {
        self.obi_tensor.push(obi);
        self.fr_elasticity.update(funding_rate, self.last_price);
        self.dark_alpha.apply_event(dex_severity, ts_ms);

        // Reset high/low every 10 seconds based on actual market timestamp instead of arbitrary tick count
        if ts_ms > 0 && (self.last_reset_ms == 0 || ts_ms >= self.last_reset_ms + 10_000) {
            self.tick_high = self.last_price;
            self.tick_low = self.last_price;
            self.last_reset_ms = ts_ms;
        }
    }
    
    pub fn get_market_regime(&self, config: &quantum_arena::QuantumConfig) -> MarketRegime {
        self.get_scalp_regime(config)
    }

    pub fn get_scalp_regime(&self, config: &quantum_arena::QuantumConfig) -> MarketRegime {
        let h = self.hurst.current(); // Read last computed Hurst
        let trend_thr = config.trend_threshold.load(Ordering::Relaxed);
        let range_thr = config.range_threshold.load(Ordering::Relaxed);
        if h < range_thr {
            MarketRegime::Scalping // Mean reverting micro-structure
        } else if h > trend_thr {
            MarketRegime::Swing // Micro trend momentum
        } else {
            MarketRegime::Neutral // Random walk
        }
    }

    pub fn get_swing_regime(&self, config: &quantum_arena::QuantumConfig) -> MarketRegime {
        let h = self.hurst.current();
        let swing_hurst_thr = config.swing_hurst_threshold.load(Ordering::Relaxed).max(0.55);
        if h > swing_hurst_thr {
            MarketRegime::Swing // Macro persistent trend
        } else if h < 0.40 {
            MarketRegime::Scalping // Deep mean reversion
        } else {
            MarketRegime::Neutral // Undefined macro regime
        }
    }

    pub fn get_features(&self) -> [f64; 12] {
        let price_change = if self.last_price > 0.0 && self.ema_slow > 0.0 {
            (self.ema_fast - self.ema_slow) / self.ema_slow
        } else {
            0.0
        };
        
        let sanitize = |val: f64| -> f64 {
            if val.is_finite() { val } else { 0.0 }
        };

        let hurst = sanitize(self.hurst.current());
        let cvd_velocity = sanitize(self.cvd_tensor.velocity());
        let vol_delta = sanitize(self.order_flow.get_volume_delta_ratio());

        [
            sanitize(price_change),
            hurst,
            cvd_velocity,
            sanitize(self.v_t),
            sanitize(self.obi_tensor.velocity()),
            sanitize(self.obi_tensor.get_current()),
            vol_delta,
            sanitize(self.cvpin.buy_volume - self.cvpin.sell_volume),
            sanitize(self.last_entropy),
            sanitize(self.dark_alpha.current_severity),
            sanitize(self.obi_tensor.acceleration()),
            sanitize(self.a_t),
        ]
    }

    /// Extracts Omni ML Features (SWING - 34D Macro+Micro)
    pub fn get_swing_features(&self, omni_macro: &[f64; 54]) -> [f64; 54] {
        let micro = self.get_features();

        // Normalization (approximated Softsign)
        let norm = |x: f32, scale: f32| -> f64 {
            let scaled = x * scale;
            (scaled / (1.0 + scaled.abs())) as f64
        };
        
        let p = self.last_price.max(1.0) as f32;

        [
            norm(micro[0] as f32, 1000.0), // price_change
            (micro[1] as f64 - 0.5) * 2.0, // hurst [0,1] -> [-1,1]
            norm(micro[2] as f32, 0.05), // ofi normalized
            norm(micro[3] as f32 / p, 1000.0), // v_t as percentage (e.g. 0.1% = 1.0)
            norm(micro[4] as f32, 10.0), // obi_accel prev_vel
            micro[5] as f64, // prev_obi [-1, 1]
            norm(micro[6] as f32, 1.0), // vol_delta
            norm(micro[7] as f32, 0.01), // cvpin delta
            norm(micro[8] as f32, 1.0), // entropy
            micro[9] as f64, // dark alpha severity
            norm(micro[10] as f32, 10.0), // obi accel
            norm(micro[11] as f32 / p, 1000.0), // a_t as percentage
            
            // Real Omni Features (22 slots) - Indices 12..34 represent macro data
            norm(omni_macro[12] as f32, 0.01), // open_interest
            norm(omni_macro[13] as f32, 1.0),  // long_short_ratio
            norm(omni_macro[14] as f32, 0.01), // fear_greed_index
            norm(omni_macro[15] as f32, 0.00005), // altcoin_dominance (scaled for compatibility)
            norm(omni_macro[16] as f32, 0.3),  // mempool_congestion (scaled for compatibility)
            norm(omni_macro[17] as f32, 0.4),  // usdt_mint_alert (scaled for compatibility)
            norm(omni_macro[18] as f32, 0.4),  // exchange_inflows (scaled for compatibility)
            norm(omni_macro[19] as f32, 100.0), // exchange_outflows (scaled for compatibility)
            norm(omni_macro[20] as f32, 200.0), // whale_alert_proxy (bybit_spread)
            norm(omni_macro[21] as f32, 0.01),  // dxy
            norm(omni_macro[22] as f32, 0.0001), // sp500
            norm(omni_macro[23] as f32, 0.0001), // nasdaq
            norm(omni_macro[24] as f32, 0.01),  // vix
            norm(omni_macro[25] as f32, 0.1),   // us10y
            norm(omni_macro[26] as f32, 0.001), // gold
            
            // Fase 19: Tensores Evolutivos (Momentum / Correlación)
            norm(omni_macro[27] as f32, 10.0), // oi_momentum
            norm(omni_macro[28] as f32, 0.001), // funding_momentum
            omni_macro[29],  // corr_dxy_ndx (Pearson [-1, 1])
            omni_macro[30],  // corr_dxy_spy (Pearson [-1, 1])
            omni_macro[31],  // corr_ndx_spy (Pearson [-1, 1])
            
            norm(omni_macro[32] as f32, 0.01),  // spot_cvd
            norm(omni_macro[33] as f32, 0.01),  // futures_basis_premium
            
            // Extended features for [f64; 54] (34..54): Tensor Kinematics (Vel, Accel, Jerk)
            norm(self.price_tensor.velocity() as f32, 100.0),      // 34
            norm(self.price_tensor.acceleration() as f32, 10.0),   // 35
            norm(self.price_tensor.jerk() as f32, 1.0),            // 36
            norm(self.obi_tensor.jerk() as f32, 1.0),              // 37
            norm(self.cvd_tensor.acceleration() as f32, 10.0),     // 38
            norm(self.cvd_tensor.jerk() as f32, 1.0),              // 39
            omni_macro[40], omni_macro[41], omni_macro[42], omni_macro[43],
            omni_macro[44], omni_macro[45], omni_macro[46], omni_macro[47], omni_macro[48],
            omni_macro[49], omni_macro[50], omni_macro[51], omni_macro[52], omni_macro[53],
        ]
    }

    /// Extracts Omni ML Features (SCALP - 54D Hyper-Micro / Velocity)
    pub fn get_scalp_features(&self, _omni_macro: &[f64; 54]) -> [f64; 54] {
        let micro = self.get_features();

        // Agresividad de normalización (tanh) para latencias ultracortas
        let norm_tanh = |x: f32, scale: f32| -> f64 {
            let scaled = x * scale;
            (scaled.exp() - (-scaled).exp()) as f64 / (scaled.exp() + (-scaled).exp() + 1e-9) as f64
        };
        
        let p = self.last_price.max(1.0) as f32;

        [
            // Top 12 micro features con multiplicadores 10x
            norm_tanh(micro[0] as f32, 10000.0), // price_change 
            (micro[1] as f64 - 0.5) * 4.0,       // hurst amplificado
            norm_tanh(micro[2] as f32, 0.5),     // ofi super-sensible
            norm_tanh(micro[3] as f32 / p, 10000.0), // v_t
            norm_tanh(micro[4] as f32, 100.0),   // obi_accel
            (micro[5] as f64) * 2.0,             // prev_obi
            norm_tanh(micro[6] as f32, 10.0),    // vol_delta
            norm_tanh(micro[7] as f32, 0.1),     // cvpin delta
            norm_tanh(micro[8] as f32, 10.0),    // entropy
            micro[9] as f64 * 3.0,               // dark alpha
            norm_tanh(micro[10] as f32, 100.0),  // obi accel cur
            norm_tanh(micro[11] as f32 / p, 10000.0), // a_t
            
            // Macro silenciado o reemplazado por métricas de velocidad puras (Indices 12..34)
            norm_tanh(self.price_tensor.velocity() as f32, 1000.0),    
            norm_tanh(self.price_tensor.acceleration() as f32, 100.0), 
            norm_tanh(self.price_tensor.jerk() as f32, 10.0),          
            norm_tanh(self.obi_tensor.velocity() as f32, 100.0),       
            norm_tanh(self.obi_tensor.acceleration() as f32, 10.0),    
            norm_tanh(self.obi_tensor.jerk() as f32, 1.0),             
            norm_tanh(self.cvd_tensor.velocity() as f32, 100.0),       
            norm_tanh(self.cvd_tensor.acceleration() as f32, 10.0),    
            norm_tanh(self.cvd_tensor.jerk() as f32, 1.0),             
            
            // Padding nulos para mantener tamaño del tensor 54D (para simplificar la arquitectura actual)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Rellenar hasta índice 34
            
            // Más padding a 0 (indices 34..54) para obviar ruido de largo plazo en Scalp.
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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

    /// [ALGORITMO #8] Dynamic EWMA Volatility Leverage Adaptation.
    /// Dampens leverage automatically during instantaneous volatility spikes to protect account capital,
    /// while unlocking full leverage (up to 30x) during smooth low-noise micro-trends.
    pub fn get_adaptive_leverage(&self, base_leverage: f64) -> f64 {
        let atr_pct = self.get_atr_pct();
        let volatility_factor = 1.0 + (atr_pct * 50.0);
        let adaptive_lev = base_leverage / volatility_factor;
        adaptive_lev.clamp(10.0, 30.0)
    }

    /// [ALGORITMO #8] Dynamic Volatility-Scaled TP/SL Multiplier.
    /// Expands TP and SL proportionally with local ATR to avoid noise stop-outs.
    pub fn get_dynamic_tp_sl(&self, base_tp: f64, base_sl: f64) -> (f64, f64) {
        let atr_pct = self.get_atr_pct();
        let scale = (1.0 + atr_pct * 10.0).clamp(0.8, 3.0);
        (base_tp * scale, base_sl * scale)
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

impl Drop for StatefulEngine {
    fn drop(&mut self) {
        DROP_COUNTER.fetch_sub(1, Ordering::SeqCst);
    }
}
