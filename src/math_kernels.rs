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

/// True Dynamic Kelly Sizing for exponential compounding
#[derive(Debug, Clone)]
pub struct DynamicKelly {
    pub win_rate_welford: WelfordVariance,
    pub win_size_welford: WelfordVariance,
    pub loss_size_welford: WelfordVariance,
    pub kelly_multiplier: f64,
}

impl DynamicKelly {
    pub fn new(multiplier: f64) -> Self {
        Self {
            win_rate_welford: WelfordVariance::new(),
            win_size_welford: WelfordVariance::new(),
            loss_size_welford: WelfordVariance::new(),
            kelly_multiplier: multiplier,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, is_win: bool, pnl_pct: f64) {
        self.win_rate_welford.update(if is_win { 1.0 } else { 0.0 });
        if is_win {
            self.win_size_welford.update(pnl_pct.abs());
        } else {
            self.loss_size_welford.update(pnl_pct.abs());
        }
    }

    #[inline(always)]
    pub fn sizing_fraction(&self) -> f64 {
        let wr = self.win_rate_welford.mean;
        let avg_win = self.win_size_welford.mean;
        let avg_loss = self.loss_size_welford.mean;
        
        // If not enough data, return a safe base default
        if self.win_rate_welford.count < 5.0 || avg_loss == 0.0 {
            return 0.10;
        }

        let r = avg_win / avg_loss;
        // Kelly Formula: K = W - ((1 - W) / R)
        let kelly = wr - ((1.0 - wr) / r); 
        let adjusted_kelly = kelly * self.kelly_multiplier;
        
        adjusted_kelly.clamp(0.01, 1.0)
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

    /// Returns the current Hurst exponent WITHOUT updating state.
    #[inline(always)]
    pub fn current(&self) -> f64 {
        let range = self.max_p - self.min_p;
        let std = self.std_dev.std_dev();
        if std > 0.0 && self.n > 2.0 {
            let rs = range / std;
            (rs.ln() / self.n.ln()).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}


// FFI Kelly Fraction & Stats Calculation

#[inline(always)]
pub fn compute_kelly_fraction(p: f64, b: f64, apply_mult: bool, kelly_mult: f64, stress_score: f64, max_exposure: f64) -> f64 {
    if b <= 0.0 {
        return 0.0;
    }
    let q = 1.0 - p;
    let kelly = (p * b - q) / b;
    if !apply_mult {
        return kelly;
    }
    let mut mult = kelly_mult;
    if stress_score < 90.0 {
        mult = 0.125;
    }
    let mut fractional_kelly = kelly * mult;
    if fractional_kelly < 0.0 {
        fractional_kelly = 0.0;
    }
    if fractional_kelly > max_exposure {
        fractional_kelly = max_exposure;
    }
    fractional_kelly
}

#[inline(always)]
pub fn extract_kelly_stats(pnl_array: &[f64], is_win_array: &[bool]) -> (f64, f64) {
    let n = pnl_array.len() as f64;
    if n == 0.0 {
        return (0.5, 1.0);
    }
    let mut wins = 0.0;
    let mut losses = 0.0;
    let mut sum_wins = 0.0;
    let mut sum_losses = 0.0;
    for i in 0..pnl_array.len() {
        if is_win_array[i] {
            wins += 1.0;
            sum_wins += pnl_array[i];
        } else {
            losses += 1.0;
            sum_losses += pnl_array[i].abs();
        }
    }
    let p = if wins > 0.0 { wins / n } else { 0.5 };
    let avg_win = if wins > 0.0 { sum_wins / wins } else { 0.01 };
    let avg_loss = if losses > 0.0 { sum_losses / losses } else { 0.01 };
    let b = if avg_loss > 0.0 { avg_win / avg_loss } else { 1.0 };
    (p, b)
}

#[inline(always)]
pub fn compute_cvar(loss_history: &[f64], confidence_level: f64) -> f64 {
    if loss_history.is_empty() {
        return 0.0;
    }
    let mut sorted_losses = loss_history.to_vec();
    // Sort in descending order (largest losses first)
    sorted_losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    let n = sorted_losses.len();
    let cutoff_idx = ((1.0 - confidence_level) * n as f64).floor() as usize;
    let cutoff_idx = cutoff_idx.max(1);
    
    let mut sum = 0.0;
    for i in 0..cutoff_idx {
        sum += sorted_losses[i];
    }
    sum / (cutoff_idx as f64)
}

// =========================================================
// VECTORIZED TECHNICAL INDICATORS
// =========================================================

#[inline(always)]
pub fn compute_ema_vectorized(data: &[f64], period: usize, out: &mut [f64]) {
    let n = data.len();
    if n == 0 || period == 0 || out.len() != n {
        return;
    }
    let k = 2.0 / (period as f64 + 1.0);
    out[0] = data[0];
    for i in 1..n {
        out[i] = data[i] * k + out[i - 1] * (1.0 - k);
    }
}

#[inline(always)]
pub fn compute_rsi_vectorized(data: &[f64], period: usize, out: &mut [f64]) {
    let n = data.len();
    if n < period || period == 0 || out.len() != n {
        for i in 0..n { out[i] = 50.0; } // Default safe value
        return;
    }
    
    let mut gain = 0.0;
    let mut loss = 0.0;
    
    // Seed first window
    for i in 1..period {
        let diff = data[i] - data[i - 1];
        if diff > 0.0 {
            gain += diff;
        } else {
            loss -= diff;
        }
    }
    
    gain /= period as f64;
    loss /= period as f64;
    
    // Fill until period with 50.0 to prevent artifacting
    for i in 0..period {
        out[i] = 50.0;
    }
    
    if loss == 0.0 {
        out[period - 1] = 100.0;
    } else {
        let rs = gain / loss;
        out[period - 1] = 100.0 - (100.0 / (1.0 + rs));
    }
    
    // Smoothed Wilders moving average
    for i in period..n {
        let diff = data[i] - data[i - 1];
        if diff > 0.0 {
            gain = (gain * (period as f64 - 1.0) + diff) / period as f64;
            loss = (loss * (period as f64 - 1.0)) / period as f64;
        } else {
            gain = (gain * (period as f64 - 1.0)) / period as f64;
            loss = (loss * (period as f64 - 1.0) - diff) / period as f64;
        }
        if loss == 0.0 {
            out[i] = 100.0;
        } else {
            let rs = gain / loss;
            out[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
}

#[inline(always)]
pub fn compute_bollinger_bands(data: &[f64], period: usize, std_dev_mult: f64, out_up: &mut [f64], out_mid: &mut [f64], out_low: &mut [f64]) {
    let n = data.len();
    if n < period || period == 0 {
        for i in 0..n {
            out_mid[i] = data[i];
            out_up[i] = data[i];
            out_low[i] = data[i];
        }
        return;
    }
    
    for i in 0..period-1 {
        out_mid[i] = data[i];
        out_up[i] = data[i];
        out_low[i] = data[i];
    }
    
    let window = period as f64;
    for i in (period - 1)..n {
        let mut sum = 0.0;
        for j in 0..period {
            sum += data[i - j];
        }
        let mean = sum / window;
        
        let mut variance = 0.0;
        for j in 0..period {
            let diff = data[i - j] - mean;
            variance += diff * diff;
        }
        let std_dev = (variance / window).sqrt();
        
        out_mid[i] = mean;
        out_up[i] = mean + std_dev_mult * std_dev;
        out_low[i] = mean - std_dev_mult * std_dev;
    }
}

#[inline(always)]
pub fn compute_macd(data: &[f64], fast_period: usize, slow_period: usize, signal_period: usize, out_macd: &mut [f64], out_signal: &mut [f64], out_hist: &mut [f64]) {
    let n = data.len();
    if n == 0 { return; }
    
    let mut fast_ema = vec![0.0; n];
    let mut slow_ema = vec![0.0; n];
    
    compute_ema_vectorized(data, fast_period, &mut fast_ema);
    compute_ema_vectorized(data, slow_period, &mut slow_ema);
    
    for i in 0..n {
        out_macd[i] = fast_ema[i] - slow_ema[i];
    }
    
    compute_ema_vectorized(&out_macd, signal_period, out_signal);
    
    for i in 0..n {
        out_hist[i] = out_macd[i] - out_signal[i];
    }
}

// =====================================================================
// MACHINE LEARNING INFERENCE KERNELS (Nano-Latency)
// =====================================================================

pub fn predict_rf(
    x: &[f64],
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    value: &[f64],
    tree_offsets: &[i64],
) -> f64 {
    let n_trees = tree_offsets.len().saturating_sub(1);
    if n_trees == 0 { return 0.0; }
    let mut total_prob = 0.0;

    for i in 0..n_trees {
        let mut node = tree_offsets[i] as usize;
        while children_left[node] != -1 {
            let f_idx = feature[node] as usize;
            if x[f_idx] <= threshold[node] {
                node = children_left[node] as usize;
            } else {
                node = children_right[node] as usize;
            }
        }
        total_prob += value[node];
    }
    total_prob / (n_trees as f64)
}

pub fn predict_gb(
    x: &[f64],
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    value: &[f64],
    tree_offsets: &[i64],
    init_score: f64,
    learning_rate: f64,
) -> f64 {
    let n_trees = tree_offsets.len().saturating_sub(1);
    let mut score = init_score;

    for i in 0..n_trees {
        let mut node = tree_offsets[i] as usize;
        while children_left[node] != -1 {
            let f_idx = feature[node] as usize;
            if x[f_idx] <= threshold[node] {
                node = children_left[node] as usize;
            } else {
                node = children_right[node] as usize;
            }
        }
        score += learning_rate * value[node];
    }

    // Sigmoid
    if score >= 0.0 {
        1.0 / (1.0 + (-score).exp())
    } else {
        let exp_s = score.exp();
        exp_s / (1.0 + exp_s)
    }
}

pub fn fused_compute_step(
    closes: &[f64],
    volumes: &[f64],
    portfolio_state: &[f64; 3], // [has_pos, pnl_norm, dur_norm]
    gene_params: &[f64; 2],     // [sl_norm, tp_norm]
    brain_weights: &[f64; 100], // 25 * 4 = 100 flattened
    l2_state: &[f64; 2],        // [ofi, microprice_divergence]
    window: usize,
    out_scores: &mut [f64; 4]
) {
    let n = closes.len();
    if n < 30 {
        out_scores.fill(0.0);
        return;
    }

    let mut state_tensor = [0.0f64; 25];

    // 1A. Market Data (20 Features)
    // Returns (5)
    for i in 0..window {
        let idx = n - window + i;
        let val = (closes[idx] - closes[idx - 1]) / closes[idx - 1];
        state_tensor[i] = val;
    }

    // Volatility (5)
    let mut vol_sum = 0.0;
    for i in (n - 20)..n {
        vol_sum += volumes[i];
    }
    let mut mean_vol = vol_sum / 20.0;
    if mean_vol < 1e-8 {
        mean_vol = 1.0;
    }

    for i in 0..window {
        let idx = n - window + i;
        state_tensor[5 + i] = volumes[idx] / mean_vol;
    }

    // Momentum / Custom (5)
    for i in 0..window {
        let idx = n - window + i;
        let mom = if idx >= 2 {
            (closes[idx] / closes[idx - 2]) - 1.0
        } else {
            0.0
        };
        state_tensor[10 + i] = mom;
        state_tensor[15 + i] = 0.0;
    }

    // Inject L2 Data
    state_tensor[18] = l2_state[0];
    state_tensor[19] = l2_state[1];

    // 2. Add Portfolio & Gene (5 Features)
    state_tensor[20] = portfolio_state[0];
    state_tensor[21] = portfolio_state[1];
    state_tensor[22] = portfolio_state[2];
    state_tensor[23] = gene_params[0];
    state_tensor[24] = gene_params[1];

    // 3. Neural Inference Dot Product
    for act in 0..4 {
        let mut score = 0.0;
        let base_idx = act * 25;
        for j in 0..25 {
            score += state_tensor[j] * brain_weights[base_idx + j];
        }
        out_scores[act] = score;
    }
}

/// O(1) Second Derivative of Order Book Imbalance (Liquidity Acceleration)
#[derive(Debug, Clone, Default)]
pub struct ObiAcceleration {
    pub prev_obi: f64,
    pub prev_obi_velocity: f64,
}

impl ObiAcceleration {
    #[inline(always)]
    pub fn new() -> Self {
        Self { prev_obi: 0.0, prev_obi_velocity: 0.0 }
    }

    #[inline(always)]
    pub fn update(&mut self, current_obi: f64) -> f64 {
        let current_velocity = current_obi - self.prev_obi;
        let acceleration = current_velocity - self.prev_obi_velocity;
        self.prev_obi = current_obi;
        self.prev_obi_velocity = current_velocity;
        acceleration
    }
}

/// O(1) Funding Rate Elasticity (∂FundingRate / ∂Price)
#[derive(Debug, Clone, Default)]
pub struct FundingRateElasticity {
    pub prev_funding_rate: f64,
    pub prev_price: f64,
}

impl FundingRateElasticity {
    #[inline(always)]
    pub fn new() -> Self {
        Self { prev_funding_rate: 0.0, prev_price: 0.0 }
    }

    #[inline(always)]
    pub fn update(&mut self, funding_rate: f64, price: f64) -> f64 {
        let delta_fr = funding_rate - self.prev_funding_rate;
        let delta_p = price - self.prev_price;
        
        self.prev_funding_rate = funding_rate;
        self.prev_price = price;

        if delta_p.abs() > f64::EPSILON && self.prev_price > 0.0 {
            let pct_delta_p = delta_p / self.prev_price;
            if pct_delta_p.abs() > f64::EPSILON {
                return delta_fr / pct_delta_p;
            }
        }
        0.0
    }
}

/// O(1) Exponential Decay Tensor for MEV/RBF Severity (Dark Alpha)
#[derive(Debug, Clone)]
pub struct ExponentialDecayTensor {
    pub current_severity: f64,
    pub decay_lambda: f64,
    pub last_timestamp_ms: u64,
}

impl ExponentialDecayTensor {
    #[inline(always)]
    pub fn new(half_life_ms: f64) -> Self {
        let decay_lambda = std::f64::consts::LN_2 / half_life_ms;
        Self {
            current_severity: 0.0,
            decay_lambda,
            last_timestamp_ms: 0,
        }
    }

    #[inline(always)]
    pub fn apply_event(&mut self, event_severity: f64, timestamp_ms: u64) {
        self.decay_to(timestamp_ms);
        self.current_severity += event_severity;
        self.last_timestamp_ms = timestamp_ms;
    }

    #[inline(always)]
    pub fn decay_to(&mut self, current_timestamp_ms: u64) -> f64 {
        if current_timestamp_ms > self.last_timestamp_ms {
            let dt = (current_timestamp_ms - self.last_timestamp_ms) as f64;
            let decay_factor = (-self.decay_lambda * dt).exp();
            self.current_severity *= decay_factor;
            self.last_timestamp_ms = current_timestamp_ms;
        }
        self.current_severity
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_variance() {
        let mut welford = WelfordVariance::new();
        welford.update(10.0);
        welford.update(12.0);
        welford.update(14.0);
        welford.update(16.0);
        welford.update(18.0);
        
        let mean = welford.mean;
        let std_dev = welford.std_dev();
        
        // Mean of 10, 12, 14, 16, 18 is 14
        assert!((mean - 14.0).abs() < 1e-6);
        
        // Variance of sample is sum((x - mean)^2) / (n - 1)
        // 16 + 4 + 0 + 4 + 16 = 40. 40 / 4 = 10
        // Std Dev = sqrt(10) = 3.162277...
        assert!((std_dev - 10.0_f64.sqrt()).abs() < 1e-6);
        
        // Test rolling removal
        welford.remove(10.0);
        assert!((welford.mean - 15.0).abs() < 1e-6); // Mean of 12, 14, 16, 18
    }

    #[test]
    fn test_kahan_summation() {
        let mut kahan = KahanSummation::new();
        // Add 1.0 ten million times
        for _ in 0..10_000_000 {
            kahan.add(1.0);
        }
        // Then add a very small number
        kahan.add(1e-10);
        
        assert!((kahan.get_sum() - 10_000_000.0000000001).abs() < 1e-10);
    }
}
