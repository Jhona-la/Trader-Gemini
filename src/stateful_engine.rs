use std::sync::atomic::{AtomicUsize, Ordering};
use crate::math_kernels::{RecursiveHurst, ObiAcceleration, FundingRateElasticity, ContinuousVPIN, ShannonEntropy, ExponentialDecayTensor};

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
    pub ema_fast: f64,
    pub ema_slow: f64,
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
}

impl StatefulEngine {
    pub fn new() -> Self {
        DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            ema_fast: 0.0,
            ema_slow: 0.0,
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
        }
    }

    /// Flushes all internal buffers. Used to auto-heal time-series glitches after network disconnects.
    pub fn reset(&mut self) {
        self.ema_fast = 0.0;
        self.ema_slow = 0.0;
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
            self.v_t = (self.v_t * 0.9) + (diff * 0.1);
            
            let norm_return = (price - self.last_price) / self.last_price;
            self.last_entropy = self.entropy.update(norm_return);
        }
        
        self.hurst.update(price);
        
        // Note: ObiAcceleration and FundingRateElasticity are typically updated in a separate method or ingested from DarkAlphaRouter/Binance. 
        // We leave them as 0.0 updates here if purely price-volume ticks are passed, but expose methods for them.
        self.cvpin.update(_volume, price < self.last_price); // Approximation: tick down = seller initiated
        
        self.last_price = price;
        self.tick_count += 1;
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

    /// Extracts NanoForest ML Features
    pub fn get_features(&self) -> [f32; 10] {
        let price_change = if self.last_price != 0.0 && self.ema_slow != 0.0 {
            (self.ema_fast - self.ema_slow) / self.ema_slow
        } else {
            0.0
        };
        
        let ema_fast_dist = if self.ema_fast != 0.0 {
            (self.last_price - self.ema_fast) / self.ema_fast
        } else {
            0.0
        };
        
        let ema_slow_dist = if self.ema_slow != 0.0 {
            (self.last_price - self.ema_slow) / self.ema_slow
        } else {
            0.0
        };

        [
            price_change as f32,
            ema_fast_dist as f32,
            ema_slow_dist as f32,
            self.v_t as f32,
            self.obi_accel.prev_obi_velocity as f32,
            self.obi_accel.prev_obi as f32,
            self.fr_elasticity.prev_funding_rate as f32,
            self.cvpin.buy_volume as f32 - self.cvpin.sell_volume as f32,
            self.last_entropy as f32,
            self.dark_alpha.current_severity as f32,
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
        // Add Hurst for external observability
        out[4] = self.hurst.update(self.last_price) as f32;
    }
}

impl Drop for StatefulEngine {
    fn drop(&mut self) {
        DROP_COUNTER.fetch_sub(1, Ordering::SeqCst);
    }
}
