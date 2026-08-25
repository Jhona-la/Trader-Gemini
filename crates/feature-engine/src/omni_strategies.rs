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
            rsi_up_ewma: Ewma::from_period(14.0),
            rsi_down_ewma: Ewma::from_period(14.0),
            macd_fast: Ewma::from_period(12.0),
            macd_slow: Ewma::from_period(26.0),
            macd_signal: Ewma::from_period(9.0),
            bb_stats: WelfordOnline::new(),
            atr_ewma: Ewma::from_period(14.0),
            parkinson_ewma: Ewma::from_period(14.0),
            local_max: 0.0,
            local_min: 0.0,
            last_price: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64, high: f64, low: f64) {
        if self.last_price == 0.0 {
            self.last_price = price;
            self.local_max = high.max(price);
            self.local_min = low.min(price);
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
        let tr = (high - low)
            .max((high - self.last_price).abs())
            .max((low - self.last_price).abs());
        self.atr_ewma.update(tr);

        // Parkinson Volatility Proxy O(1)
        let hl_ln = (high / low.max(1e-8)).ln();
        self.parkinson_ewma.update(hl_ln * hl_ln);

        // Local Extrema Decay (Fibonacci proxy)
        self.local_max = self.local_max.max(high);
        self.local_min = if self.local_min == 0.0 { low } else { self.local_min.min(low) };
        // FIX #901 & #1204: Decaimiento ultrasuave (1e-6) para mantener los soportes/resistencias Fibonacci estables en HFT
        if price < self.local_max {
            self.local_max -= (self.local_max - price) * 1e-6;
        }
        if price > self.local_min {
            self.local_min += (price - self.local_min) * 1e-6;
        }

        self.last_price = price;
    }

    #[inline(always)]
    pub fn extract_features(&self) -> [f32; 22] {
        let rsi_up = self.rsi_up_ewma.get();
        let rsi_down = self.rsi_down_ewma.get();
        // FIX #1205: Retornar 50.0 (neutral) cuando el mercado no tiene movimiento (rsi_up y rsi_down == 0)
        let rsi = if rsi_up <= 1e-9 && rsi_down <= 1e-9 {
            50.0
        } else if rsi_down <= 1e-9 {
            100.0
        } else {
            100.0 - (100.0 / (1.0 + rsi_up / rsi_down))
        };

        let macd_line = self.macd_fast.get() - self.macd_slow.get();
        let macd_hist = macd_line - self.macd_signal.get();

        let bb_mean = self.bb_stats.mean();
        let bb_std = self.bb_stats.std_dev();
        let bb_zscore = if bb_std > 0.0 {
            (self.last_price - bb_mean) / bb_std
        } else {
            0.0
        };

        let atr = self.atr_ewma.get();
        let parkinson_vol = (self.parkinson_ewma.get() / (4.0 * std::f64::consts::LN_2)).sqrt();

        // Fib levels proxy
        let (eff_max, eff_min) = if self.local_min > 0.0 && self.local_max >= self.local_min {
            (self.local_max, self.local_min)
        } else if self.last_price > 0.0 {
            (self.last_price, self.last_price)
        } else {
            (0.0, 0.0)
        };

        let range = eff_max - eff_min;
        let pos_in_range = if range > 0.0 {
            ((self.last_price - eff_min) / range).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let safe_last_price = self.last_price.max(1.0);
        let dist_max = if eff_max >= self.last_price && self.last_price > 0.0 {
            (eff_max - self.last_price) / safe_last_price
        } else {
            0.0
        };
        let dist_min = if self.last_price >= eff_min && self.last_price > 0.0 {
            (self.last_price - eff_min) / safe_last_price
        } else {
            0.0
        };
        let norm_range = if self.last_price > 0.0 {
            range / safe_last_price
        } else {
            0.0
        };

        [
            (rsi / 100.0) as f32,
            (macd_line / safe_last_price) as f32,
            (macd_hist / safe_last_price) as f32,
            (bb_zscore.clamp(-4.0, 4.0)) as f32,
            ((atr / safe_last_price).clamp(0.0, 0.1)) as f32,
            pos_in_range as f32,
            ((rsi / 100.0) * bb_zscore.clamp(-4.0, 4.0)) as f32,
            ((macd_hist / safe_last_price) * (atr / safe_last_price)) as f32,
            ((self.last_price - self.macd_fast.get()) / safe_last_price) as f32,
            ((self.last_price - self.macd_slow.get()) / safe_last_price) as f32,
            dist_max as f32,
            dist_min as f32,
            (parkinson_vol.clamp(0.0, 0.1)) as f32,
            (((rsi - 50.0) / 50.0).clamp(-1.0, 1.0)) as f32,
            (((self.macd_fast.get() - self.macd_slow.get()) / safe_last_price).clamp(-0.1, 0.1)) as f32,
            (((self.last_price - self.macd_signal.get()) / safe_last_price).clamp(-0.1, 0.1)) as f32,
            ((self.bb_stats.variance().sqrt() / safe_last_price).clamp(0.0, 0.1)) as f32,
            (norm_range.clamp(0.0, 0.2)) as f32,
            (((pos_in_range - 0.5) * 2.0).clamp(-1.0, 1.0)) as f32,
            ((parkinson_vol / (atr / safe_last_price).max(1e-6)).clamp(0.0, 10.0)) as f32,
            (((macd_hist / safe_last_price) * bb_zscore).clamp(-0.1, 0.1)) as f32,
            (((rsi / 100.0) * pos_in_range).clamp(0.0, 1.0)) as f32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omni_strategies_uninitialized_and_nan_immunity() {
        let engine = OmniStrategyEngine::new();
        let feats = engine.extract_features();
        assert_eq!(feats.len(), 22);
        for f in &feats {
            assert!(f.is_finite(), "Feature sin inicializar debe ser finita");
        }
    }
}
