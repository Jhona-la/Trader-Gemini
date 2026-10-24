#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SignalType {
    Long,
    Short,
    #[default]
    Flat,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SignalIntent {
    pub signal: SignalType,
    pub confidence: f64,
    pub expected_duration_ms: u64,
    pub expected_volume_usd: f64,
    pub volume_flow_rate: f64,
    pub expected_magnitude: f64,
    pub tp_price_target: f64,
    pub sl_price_target: f64,
    /// D-690 / LC-F1-37 / SI: Probabilidad condicional de ganar neto bajo el lado y
    /// la geometría TP/SL de esta orden concreta: P(ganar neto | lado, geometría, τ).
    /// Reservada exclusivamente para el dimensionamiento (Kelly/Riesgo).
    /// `0.0` significa «sin calibrar»: el consumidor recurre a `confidence`.
    /// Valores no finitos o fuera de [0.0, 1.0] son rechazados por `risk-engine` con `REJ_INVALID_INPUT`.
    /// La selección de señales nunca la lee para evitar sesgos de filtrado.
    pub win_probability: f64,
}

impl SignalIntent {
    pub fn flat() -> Self {
        Self::default()
    }
}
