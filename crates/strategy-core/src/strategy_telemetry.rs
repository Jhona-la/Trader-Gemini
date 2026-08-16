/// 🔬 TELEMETRÍA ZERO-LATENCY PARA ESTRATEGIAS (V9)
/// 
/// Define constantes y funciones de empaquetado para telemetría de estrategias.
/// El bus de telemetría real se inyecta desde god-engine-core, evitando
/// dependencias circulares entre crates.

/// Subsistemas de telemetría de estrategias
pub const SUBSYSTEM_SCALP_STRATEGY: u8 = 10;
pub const SUBSYSTEM_SWING_STRATEGY: u8 = 11;
pub const SUBSYSTEM_TENSOR_PREDICTOR: u8 = 12;
pub const SUBSYSTEM_LEVERAGE_MATRIX: u8 = 13;
pub const SUBSYSTEM_EVOLUTION_ENGINE: u8 = 14;

/// Tipos de frame para estrategias
pub const FRAME_SIGNAL_GENERATED: u8 = 20;
pub const FRAME_SIGNAL_REJECTED: u8 = 21;
pub const FRAME_CONFIDENCE_SNAPSHOT: u8 = 22;
pub const FRAME_LEVERAGE_COMPUTED: u8 = 23;
pub const FRAME_EVOLUTION_MUTATION: u8 = 24;

/// Frame de telemetría de estrategia (64 bytes: subsystem + type + 6×f64)
#[derive(Debug, Clone, Copy)]
pub struct StrategyTelemetryFrame {
    pub subsystem: u8,
    pub frame_type: u8,
    pub payload: [f64; 6],
}

impl StrategyTelemetryFrame {
    /// Crea un frame de señal Scalp generada.
    /// payload: [confidence, z_score, obi, spread_bps, coin_id, direction]
    #[inline(always)]
    pub fn scalp_signal(
        confidence: f64, z_score: f64, obi: f64, spread_bps: f64, 
        coin_id: usize, direction: f64,
    ) -> Self {
        Self {
            subsystem: SUBSYSTEM_SCALP_STRATEGY,
            frame_type: FRAME_SIGNAL_GENERATED,
            payload: [confidence, z_score, obi, spread_bps, coin_id as f64, direction],
        }
    }

    /// Crea un frame de señal Swing generada.
    /// payload: [confidence, hurst, macd_hist, atr_pct, coin_id, direction]
    #[inline(always)]
    pub fn swing_signal(
        confidence: f64, hurst: f64, macd_hist: f64, atr_pct: f64, 
        coin_id: usize, direction: f64,
    ) -> Self {
        Self {
            subsystem: SUBSYSTEM_SWING_STRATEGY,
            frame_type: FRAME_SIGNAL_GENERATED,
            payload: [confidence, hurst, macd_hist, atr_pct, coin_id as f64, direction],
        }
    }

    /// Crea un frame de cálculo de leverage.
    /// payload: [raw_leverage, final_leverage, capital, volatility, confidence, coin_id]
    #[inline(always)]
    pub fn leverage_computed(
        raw: f64, final_lev: f64, capital: f64, vol: f64, conf: f64, coin_id: usize,
    ) -> Self {
        Self {
            subsystem: SUBSYSTEM_LEVERAGE_MATRIX,
            frame_type: FRAME_LEVERAGE_COMPUTED,
            payload: [raw, final_lev, capital, vol, conf, coin_id as f64],
        }
    }

    /// Crea un frame de mutación evolutiva.
    /// payload: [dsr, win_rate, profit_factor, trade_count, drawdown, accepted]
    #[inline(always)]
    pub fn evolution_mutation(
        dsr: f64, wr: f64, pf: f64, trades: usize, dd: f64, accepted: bool,
    ) -> Self {
        Self {
            subsystem: SUBSYSTEM_EVOLUTION_ENGINE,
            frame_type: FRAME_EVOLUTION_MUTATION,
            payload: [dsr, wr, pf, trades as f64, dd, if accepted { 1.0 } else { 0.0 }],
        }
    }
}
