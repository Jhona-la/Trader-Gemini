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
    Continuous,
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
    /// D-690: probabilidad de ganar calibrada con resultados reales, sólo para
    /// el dimensionado. `0.0` significa «sin calibrar»: el consumidor usa
    /// `confidence`. La selección nunca la lee (etiquetas selectivas).
    pub win_probability: f64,
}

impl SignalIntent {
    pub fn flat() -> Self {
        Self::default()
    }
}
