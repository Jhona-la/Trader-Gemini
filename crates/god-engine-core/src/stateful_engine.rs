use std::sync::atomic::{AtomicUsize, Ordering};
use crate::math_kernels::{RecursiveHurst, ObiAcceleration, FundingRateElasticity, ContinuousVPIN, ShannonEntropy, ExponentialDecayTensor};
use feature_engine::OrderFlowTracker;

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
    pub obi_accel: ObiAcceleration,
    pub fr_elasticity: FundingRateElasticity,
    pub cvpin: ContinuousVPIN,
    pub entropy: ShannonEntropy,
    pub dark_alpha: ExponentialDecayTensor,
    pub last_entropy: f64,
    // Add compatibility properties so swing engine isn't completely broken
    pub ema_fast: f64,
    pub ema_slow: f64,
    pub omni: feature_engine::OmniStrategyEngine,
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
            tick_count: 0,
            hurst: RecursiveHurst::new(),
            obi_accel: ObiAcceleration::new(),
            fr_elasticity: FundingRateElasticity::new(),
            cvpin: ContinuousVPIN::new(100.0), // 100 volume bucket size
            entropy: ShannonEntropy::new(),
            dark_alpha: ExponentialDecayTensor::new(10000.0), // 10s half-life
            last_entropy: 0.0,
            ema_fast: 0.0,
            ema_slow: 0.0,
            omni: feature_engine::OmniStrategyEngine::new(),
        }
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
        self.fr_elasticity = FundingRateElasticity::new();
        self.cvpin = ContinuousVPIN::new(100.0);
        self.entropy = ShannonEntropy::new();
        self.dark_alpha = ExponentialDecayTensor::new(10000.0);
        self.last_entropy = 0.0;
        self.ema_fast = 0.0;
        self.ema_slow = 0.0;
        self.omni = feature_engine::OmniStrategyEngine::new();
    }

    /// Processes a new tick internally in f64
    pub fn process_tick(&mut self, price: f64, _volume: f64) {
        if self.last_price == 0.0 {
            self.ema_fast = price;
            self.ema_slow = price;
        } else {
            let alpha_fast = 2.0 / (12.0 + 1.0);
            let alpha_slow = 2.0 / (26.0 + 1.0);
            
            self.ema_fast = (price - self.ema_fast) * alpha_fast + self.ema_fast;
            self.ema_slow = (price - self.ema_slow) * alpha_slow + self.ema_slow;
            
            let diff = (price - self.last_price).abs();
            let new_v_t = (self.v_t * 0.9) + (diff * 0.1);
            self.a_t = new_v_t - self.v_t;
            self.v_t = new_v_t;
            
            let norm_return = (price - self.last_price) / self.last_price;
            self.last_entropy = self.entropy.update(norm_return);
        }
        
        self.hurst.update(price);
        self.cvpin.update(_volume, price < self.last_price); // Approximation: tick down = seller initiated
        
        self.omni.update(price, price, price); // For pure tick data, H=L=C=price

        self.last_price = price;
        self.tick_count += 1;
    }
    
    pub fn update_trade_flow(&mut self, volume: f64, is_buyer_maker: bool) {
        self.order_flow.update(volume, is_buyer_maker);
    }
    
    /// Updates the Order Flow Imbalance (OFI) predictive model
    pub fn update_ofi(&mut self, bid_price: f64, ask_price: f64, bid_qty: f64, ask_qty: f64) -> f64 {
        self.ofi_model.update(bid_price, ask_price, bid_qty, ask_qty)
    }
    
    pub fn update_macro_features(&mut self, obi: f64, funding_rate: f64, dex_severity: f64, ts_ms: u64) {
        self.obi_accel.update(obi);
        self.fr_elasticity.update(funding_rate, self.last_price);
        self.dark_alpha.apply_event(dex_severity, ts_ms);
    }
    
    pub fn get_market_regime(&self) -> MarketRegime {
        let h = self.hurst.current(); // Read last computed Hurst — NO double-update
        
        if h < 0.45 {
            MarketRegime::Scalping // Mean reverting
        } else if h > 0.55 {
            MarketRegime::Swing // Trending
        } else {
            MarketRegime::Neutral // Random walk
        }
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

        [
            price_change as f32,
            hurst as f32,
            ofi as f32,
            self.v_t as f32,
            self.obi_accel.prev_obi_velocity as f32,
            self.obi_accel.prev_obi as f32,
            vol_delta as f32,
            (self.cvpin.buy_volume - self.cvpin.sell_volume) as f32,
            self.last_entropy as f32,
            self.dark_alpha.current_severity as f32,
            self.obi_accel.accel as f32,
            self.a_t as f32,
        ]
    }

    /// Extracts Omni ML Features (SWING - 34D Macro+Micro)
    pub fn get_swing_features(&self) -> [f32; 34] {
        let micro = self.get_features();
        let omni_feats = self.omni.extract_features();

        [
            micro[0], micro[1], micro[2], micro[3], micro[4],
            micro[5], micro[6], micro[7], micro[8], micro[9],
            micro[10], micro[11],
            // Omni Features (22 slots)
            omni_feats[0], omni_feats[1], omni_feats[2], omni_feats[3], omni_feats[4],
            omni_feats[5], omni_feats[6], omni_feats[7], omni_feats[8], omni_feats[9],
            omni_feats[10], omni_feats[11], omni_feats[12], omni_feats[13], omni_feats[14],
            omni_feats[15], omni_feats[16], omni_feats[17], omni_feats[18], omni_feats[19],
            omni_feats[20], omni_feats[21],
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
