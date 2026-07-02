#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalType {
    Long,
    Short,
    Flat,
}

#[derive(Debug, Clone, Copy)]
pub struct SignalIntent {
    pub signal: SignalType,
    pub confidence: f64,
}

impl SignalIntent {
    pub fn flat() -> Self {
        Self {
            signal: SignalType::Flat,
            confidence: 0.0,
        }
    }
}

pub struct ScalpML;
impl ScalpML {
    pub fn new() -> Self { Self }
    pub fn infer(&self, _obi: f64, _accel: f64, _spread: f64) -> f64 { 0.0 }
}

pub struct SwingML;
impl SwingML {
    pub fn new() -> Self { Self }
    pub fn infer(&self, _macd: f64, _z_score: f64, _hurst: f64) -> f64 { 0.0 }
}

// Stubs for missing math functions previously in feature_engine
pub fn calculate_atr(_high: f64, _low: f64, _close: f64, _prev_close: f64) -> f64 { 0.0 }
pub fn calculate_hurst_exponent(_prices: &[f64]) -> f64 { 0.5 }
pub fn calculate_z_score_welford(_val: f64, _mean: f64, _std_dev: f64) -> f64 { 0.0 }
