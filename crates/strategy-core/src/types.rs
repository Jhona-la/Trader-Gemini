#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SignalType {
    Long,
    Short,
    #[default]
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TradeHorizon {
    #[default]
    Scalp,
    Swing,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SignalIntent {
    pub signal: SignalType,
    pub confidence: f64,
    pub expected_duration_ms: u64,
    pub expected_volume_usd: f64,
    pub volume_flow_rate: f64,
    pub drift: f64,
    pub expected_magnitude: f64,
    pub tp_price_target: f64,
    pub sl_price_target: f64,
    pub trajectory_volatility: f64,
    pub horizon: TradeHorizon,
}

impl SignalIntent {
    pub fn flat() -> Self {
        Self::default()
    }
}
