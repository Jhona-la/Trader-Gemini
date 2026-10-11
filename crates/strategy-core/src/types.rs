#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SignalType {
    Long,
    Short,
    #[default]
    Flat,
}

/// U-6 (MOTOR UNIVERSAL CONTINUO): el horizonte de un trade NO es una
/// etiqueta binaria — es una posición τ en el continuo espectral. Este enum
/// conserva sólo `Continuous` como marcador de "el eje temporal vive en
/// `expected_duration_ms` + `temporal_scale`"; las variantes Scalp/Swing
/// fueron extirpadas (con sus mapas duplicados a 0.0/1.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TradeHorizon {
    #[default]
    Continuous,
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
    /// D-690 / LC-F1-37: Probabilidad condicional de ganar neto bajo el lado y
    /// la geometría TP/SL de esta orden concreta: P(ganar neto | lado, geometría, τ).
    /// Reservada exclusivamente para el dimensionamiento (Kelly/Riesgo).
    /// `0.0` o no finita significa «sin calibrar»: el consumidor recurre a `confidence`.
    /// La selección de señales nunca la lee para evitar sesgos de filtrado.
    pub win_probability: f64,
}

impl SignalIntent {
    pub fn flat() -> Self {
        Self::default()
    }
}
