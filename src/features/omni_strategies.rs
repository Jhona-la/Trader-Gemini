use crate::features::ewma::Ewma;
use crate::features::welford::WelfordOnline;

/// O(1) Recursive Mathematical Omni-Strategy Engine
#[derive(Clone, Debug)]
#[repr(C)]
pub struct OmniStrategyEngine {
    // Momentum
    rsi_up_ewma: Ewma,
    rsi_down_ewma: Ewma,
    macd_fast: Ewma,
    macd_slow: Ewma,
    macd_signal: Ewma,
    
    // Volatility (Bollinger & Parkinson)
    pub bb_stats: WelfordOnline,
    pub atr_ewma: Ewma,
    pub parkinson_ewma: Ewma,
    
    // Geometry & Trend
    pub local_max: f64,
    pub local_min: f64,
    
    last_price: f64,
}

impl Default for OmniStrategyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OmniStrategyEngine {
    pub fn new() -> Self {
        Self {
            rsi_up_ewma: Ewma::new(14.0),
            rsi_down_ewma: Ewma::new(14.0),
            macd_fast: Ewma::new(12.0),
            macd_slow: Ewma::new(26.0),
            macd_signal: Ewma::new(9.0),
            bb_stats: WelfordOnline::new(),
            atr_ewma: Ewma::new(14.0),
            parkinson_ewma: Ewma::new(14.0),
            local_max: 0.0,
            local_min: f64::MAX,
            last_price: 0.0,
        }
    }
    
    #[inline(always)]
    pub fn update(&mut self, price: f64, high: f64, low: f64) {
        if self.last_price == 0.0 {
            self.last_price = price;
            self.local_max = high;
            self.local_min = low;
            self.macd_fast.update(price);
            self.macd_slow.update(price);
            return;
        }
        
        let diff = price - self.last_price;
        if diff > 0.0 {
            self.rsi_up_ewma.update(diff);
            self.rsi_down_ewma.update(0.0);
        } else {
            self.rsi_up_ewma.update(0.0);
            self.rsi_down_ewma.update(-diff);
        }
        
        // MACD
        self.macd_fast.update(price);
        self.macd_slow.update(price);
        let macd_line = self.macd_fast.get() - self.macd_slow.get();
        self.macd_signal.update(macd_line);
        
        // Bollinger
        self.bb_stats.update(price);
        
        // ATR Approximation
        let tr = (high - low).max((high - self.last_price).abs()).max((low - self.last_price).abs());
        self.atr_ewma.update(tr);
        
        // Parkinson Volatility Proxy O(1)
        let hl_ln = (high / low.max(1e-8)).ln();
        self.parkinson_ewma.update(hl_ln * hl_ln);
        
        // Local Extrema Decay (Fibonacci proxy)
        self.local_max = self.local_max.max(high);
        self.local_min = self.local_min.min(low);
        // Slowly decay extrema towards current price to forget old history
        self.local_max -= (self.local_max - price) * 0.0001;
        self.local_min += (price - self.local_min) * 0.0001;
        
        self.last_price = price;
    }
    
    #[inline(always)]
    pub fn extract_features(&self) -> [f32; 22] {
        let rsi_up = self.rsi_up_ewma.get();
        let rsi_down = self.rsi_down_ewma.get();
        let rsi = if rsi_down == 0.0 { 100.0 } else { 100.0 - (100.0 / (1.0 + rsi_up / rsi_down)) };
        let rsi_norm = ((rsi - 50.0) / 50.0) as f32; // Natively bound [-1.0, 1.0]
        
        let p = self.last_price.max(1.0) as f32;
        let macd_line = (self.macd_fast.get() - self.macd_slow.get()) as f32 / p;
        let macd_hist = macd_line - (self.macd_signal.get() as f32 / p);
        
        let bb_mean = self.bb_stats.mean();
        let bb_std = self.bb_stats.std_dev();
        let bb_zscore = if bb_std > 0.0 { ((self.last_price - bb_mean) / bb_std) as f32 } else { 0.0 };
        // Z-score mapped: usually falls within -3.0 to 3.0
        let bb_norm = bb_zscore / 3.0;
        
        let atr_pct = (self.atr_ewma.get() as f32) / p;
        let parkinson_vol = ((self.parkinson_ewma.get() / (4.0 * std::f64::consts::LN_2)).sqrt()) as f32;
        
        // Fib levels proxy
        let range = self.local_max - self.local_min;
        let pos_in_range = if range > 0.0 { ((self.last_price - self.local_min) / range) as f32 } else { 0.5 };
        let pos_norm = (pos_in_range - 0.5) * 2.0; // [-1.0, 1.0]
        
        let local_max_dist = ((self.local_max - self.last_price) / p as f64) as f32;
        let local_min_dist = ((self.last_price - self.local_min) / p as f64) as f32;
        
        // Normalize features instead of scaling by arbitrary values (100.0, 1000.0)
        // Volatility is used as the normalization baseline to make the inputs scale-invariant
        let baseline_vol = atr_pct.max(1e-5); 
        
        [
            rsi_norm,
            macd_line / baseline_vol, // Scale-invariant MACD
            macd_hist / baseline_vol, // Scale-invariant Histogram
            bb_norm,
            atr_pct / 0.01, // Normalized to typical 1% movement for ML stability
            pos_norm,
            rsi_norm * bb_norm, // Oscillator synthesis [-1, 1]
            macd_hist / baseline_vol, 
            (((self.last_price - self.macd_fast.get()) as f32 / p) / baseline_vol),
            (((self.last_price - self.macd_slow.get()) as f32 / p) / baseline_vol),
            local_max_dist / baseline_vol, // Distance to high normalized by vol
            local_min_dist / baseline_vol, // Distance to low normalized by vol
            parkinson_vol / baseline_vol, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    }
}
