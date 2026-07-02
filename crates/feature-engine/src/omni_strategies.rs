use crate::ewma::Ewma;
use crate::welford::WelfordOnline;

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
        
        let macd_line = self.macd_fast.get() - self.macd_slow.get();
        let macd_hist = macd_line - self.macd_signal.get();
        
        let bb_mean = self.bb_stats.mean();
        let bb_std = self.bb_stats.std_dev();
        let bb_zscore = if bb_std > 0.0 { (self.last_price - bb_mean) / bb_std } else { 0.0 };
        
        let atr = self.atr_ewma.get();
        let parkinson_vol = (self.parkinson_ewma.get() / (4.0 * std::f64::consts::LN_2)).sqrt();
        
        // Fib levels proxy
        let range = self.local_max - self.local_min;
        let pos_in_range = if range > 0.0 { (self.last_price - self.local_min) / range } else { 0.5 };
        
        [
            rsi as f32,
            macd_line as f32,
            macd_hist as f32,
            bb_zscore as f32,
            (atr / self.last_price.max(1.0)) as f32,
            pos_in_range as f32,
            (rsi / 100.0 * bb_zscore) as f32,
            (macd_hist * atr) as f32,
            (self.last_price - self.macd_fast.get()) as f32,
            (self.last_price - self.macd_slow.get()) as f32,
            ((self.local_max - self.last_price) / self.last_price.max(1.0)) as f32,
            ((self.last_price - self.local_min) / self.last_price.max(1.0)) as f32,
            parkinson_vol as f32, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    }
}
