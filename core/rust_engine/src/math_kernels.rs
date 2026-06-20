use std::f64;

/// Welford's Online Algorithm for computing variance and standard deviation in O(1).
/// Used for Z-Scores, Bollinger Bands, and running volatility.
#[derive(Debug, Clone, Default)]
pub struct WelfordVariance {
    pub count: f64,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordVariance {
    #[inline(always)]
    pub fn new() -> Self {
        Self { count: 0.0, mean: 0.0, m2: 0.0 }
    }

    #[inline(always)]
    pub fn update(&mut self, new_value: f64) {
        self.count += 1.0;
        let delta = new_value - self.mean;
        self.mean += delta / self.count;
        let delta2 = new_value - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        if self.count < 2.0 {
            return 0.0;
        }
        self.m2 / (self.count - 1.0)
    }

    #[inline(always)]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    #[inline(always)]
    pub fn remove(&mut self, old_value: f64) {
        if self.count <= 1.0 {
            self.count = 0.0;
            self.mean = 0.0;
            self.m2 = 0.0;
            return;
        }
        let delta = old_value - self.mean;
        self.count -= 1.0;
        self.mean -= delta / self.count;
        let delta2 = old_value - self.mean;
        self.m2 -= delta * delta2;
    }
}

/// Kahan summation algorithm to reduce floating-point error accumulation in O(1).
/// Used for Volume Delta, cumulative PnL, and other large running sums.
#[derive(Debug, Clone, Default)]
pub struct KahanSummation {
    pub sum: f64,
    pub c: f64,
}

impl KahanSummation {
    #[inline(always)]
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    #[inline(always)]
    pub fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline(always)]
    pub fn get_sum(&self) -> f64 {
        self.sum
    }
}

/// Kyle's Lambda (Market Impact): ∂P / ∂V.
/// Computed online. Returns log(1 + lambda) to domesticate fat tails.
#[derive(Debug, Clone, Default)]
pub struct KylesLambda {
    pub delta_p_kahan: KahanSummation,
    pub delta_v_kahan: KahanSummation,
    pub last_price: f64,
}

impl KylesLambda {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            delta_p_kahan: KahanSummation::new(),
            delta_v_kahan: KahanSummation::new(),
            last_price: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, current_price: f64, volume: f64) -> f64 {
        if self.last_price != 0.0 {
            let delta_p = (current_price - self.last_price).abs();
            self.delta_p_kahan.add(delta_p);
            self.delta_v_kahan.add(volume);
        }
        self.last_price = current_price;
        
        let dv = self.delta_v_kahan.get_sum();
        if dv > 0.0 {
            let lambda = self.delta_p_kahan.get_sum() / dv;
            (1.0 + lambda).ln()
        } else {
            0.0
        }
    }
}

/// Continuous VPIN (Volume-Synchronized Probability of Informed Trading) in O(1)
#[derive(Debug, Clone, Default)]
pub struct ContinuousVPIN {
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub bucket_size: f64,
}

impl ContinuousVPIN {
    #[inline(always)]
    pub fn new(bucket_size: f64) -> Self {
        Self {
            buy_volume: 0.0,
            sell_volume: 0.0,
            bucket_size,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, volume: f64, is_buyer_maker: bool) -> f64 {
        if is_buyer_maker {
            self.sell_volume += volume;
        } else {
            self.buy_volume += volume;
        }
        
        // Decay to keep within bucket context (EWMA style decay for O(1) rolling VPIN)
        let total_vol = self.buy_volume + self.sell_volume;
        if total_vol > self.bucket_size {
            let ratio = self.bucket_size / total_vol;
            self.buy_volume *= ratio;
            self.sell_volume *= ratio;
        }
        
        if total_vol > 0.0 {
            (self.buy_volume - self.sell_volume).abs() / total_vol
        } else {
            0.0
        }
    }
}

/// Recursive O(1) SMA
#[derive(Debug, Clone, Default)]
pub struct RecursiveSMA {
    pub sum: f64,
    pub count: usize,
    pub window: usize,
}

impl RecursiveSMA {
    #[inline(always)]
    pub fn new(window: usize) -> Self {
        Self { sum: 0.0, count: 0, window }
    }
    
    #[inline(always)]
    pub fn update(&mut self, new_val: f64, old_val: f64) -> f64 {
        if self.count < self.window {
            self.count += 1;
            self.sum += new_val;
        } else {
            self.sum = self.sum + new_val - old_val;
        }
        self.sum / (self.count as f64)
    }
}

/// Recursive O(1) EMA
#[derive(Debug, Clone, Default)]
pub struct RecursiveEMA {
    pub ema: f64,
    pub alpha: f64,
    pub initialized: bool,
}

impl RecursiveEMA {
    #[inline(always)]
    pub fn new(window: usize) -> Self {
        Self {
            ema: 0.0,
            alpha: 2.0 / (window as f64 + 1.0),
            initialized: false,
        }
    }
    
    #[inline(always)]
    pub fn update(&mut self, new_val: f64) -> f64 {
        if !self.initialized {
            self.ema = new_val;
            self.initialized = true;
        } else {
            self.ema = (new_val - self.ema) * self.alpha + self.ema;
        }
        self.ema
    }
}

/// Adaptive Quarter-Kelly Sizing for aggressive exponential compounding
#[derive(Debug, Clone)]
pub struct QuarterKelly {
    pub win_rate_welford: WelfordVariance,
    pub pnl_welford: WelfordVariance,
}

impl QuarterKelly {
    pub fn new() -> Self {
        Self {
            win_rate_welford: WelfordVariance::new(),
            pnl_welford: WelfordVariance::new(),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, is_win: bool, pnl_pct: f64) {
        self.win_rate_welford.update(if is_win { 1.0 } else { 0.0 });
        self.pnl_welford.update(pnl_pct);
    }

    #[inline(always)]
    pub fn sizing_fraction(&self) -> f64 {
        let wr = self.win_rate_welford.mean;
        let avg_win = self.pnl_welford.mean.max(0.0001); // Avoid div zero
        // In this aggressive strategy we assume loss size = stop_loss ~= avg_win * 1.5 typically.
        // For standard Kelly: K = W - ( (1 - W) / R )
        // Using an approximated R of 1.0
        let kelly = wr - (1.0 - wr); 
        let quarter_kelly = kelly * 0.25;
        quarter_kelly.clamp(0.0, 1.0)
    }
}

/// Shannon Entropy O(1) Approximation for market noise measurement
#[derive(Debug, Clone)]
pub struct ShannonEntropy {
    bins: [f64; 10], // Simple 10-bin histogram approximation
    total_count: f64,
}

impl ShannonEntropy {
    pub fn new() -> Self {
        Self { bins: [0.0; 10], total_count: 0.0 }
    }
    
    #[inline(always)]
    pub fn update(&mut self, norm_return: f64) -> f64 {
        // Map norm_return (-0.05 to 0.05) to bin 0-9
        let bin_idx = ((norm_return * 100.0 + 5.0).clamp(0.0, 9.99) as usize);
        self.bins[bin_idx] += 1.0;
        self.total_count += 1.0;
        
        let mut entropy = 0.0;
        for &count in self.bins.iter() {
            if count > 0.0 {
                let p = count / self.total_count;
                entropy -= p * p.ln();
            }
        }
        entropy
    }
}

/// Hurst Exponent via Rescaled Range (O(1) Recursive Approximation)
#[derive(Debug, Clone)]
pub struct RecursiveHurst {
    pub max_p: f64,
    pub min_p: f64,
    pub std_dev: WelfordVariance,
    pub n: f64,
}

impl RecursiveHurst {
    pub fn new() -> Self {
        Self { max_p: f64::MIN, min_p: f64::MAX, std_dev: WelfordVariance::new(), n: 0.0 }
    }
    
    #[inline(always)]
    pub fn update(&mut self, price: f64) -> f64 {
        self.n += 1.0;
        if price > self.max_p { self.max_p = price; }
        if price < self.min_p { self.min_p = price; }
        self.std_dev.update(price);
        
        let range = self.max_p - self.min_p;
        let std = self.std_dev.std_dev();
        
        if std > 0.0 && self.n > 2.0 {
            let rs = range / std;
            (rs.ln() / self.n.ln()).clamp(0.0, 1.0)
        } else {
            0.5 // Random walk
        }
    }
}
