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

/// Frame de telemetría de estrategia (Exactamente 64 bytes alineado a L1 cache line: 8 + 1 + 1 + 6 + 48)
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct StrategyTelemetryFrame {
    pub timestamp_ns: u64,
    pub subsystem: u8,
    pub frame_type: u8,
    pub _reserved: [u8; 6],
    pub payload: [f64; 6],
}

impl StrategyTelemetryFrame {
    #[inline(always)]
    fn current_time_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Crea un frame de señal Scalp generada.
    /// payload: [confidence, z_score, obi, spread_bps, coin_id, direction]
    #[inline(always)]
    pub fn scalp_signal(
        confidence: f64, z_score: f64, obi: f64, spread_bps: f64, 
        coin_id: usize, direction: f64,
    ) -> Self {
        Self {
            timestamp_ns: Self::current_time_ns(),
            subsystem: SUBSYSTEM_SCALP_STRATEGY,
            frame_type: FRAME_SIGNAL_GENERATED,
            _reserved: [0; 6],
            payload: [
                if confidence.is_finite() { confidence } else { 0.0 },
                if z_score.is_finite() { z_score } else { 0.0 },
                if obi.is_finite() { obi } else { 0.0 },
                if spread_bps.is_finite() { spread_bps } else { 0.0 },
                coin_id as f64,
                if direction.is_finite() { direction } else { 0.0 },
            ],
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
            timestamp_ns: Self::current_time_ns(),
            subsystem: SUBSYSTEM_SWING_STRATEGY,
            frame_type: FRAME_SIGNAL_GENERATED,
            _reserved: [0; 6],
            payload: [
                if confidence.is_finite() { confidence } else { 0.0 },
                if hurst.is_finite() { hurst } else { 0.5 },
                if macd_hist.is_finite() { macd_hist } else { 0.0 },
                if atr_pct.is_finite() { atr_pct } else { 0.01 },
                coin_id as f64,
                if direction.is_finite() { direction } else { 0.0 },
            ],
        }
    }

    /// Crea un frame de cálculo de leverage.
    /// payload: [raw_leverage, final_leverage, capital, volatility, confidence, coin_id]
    #[inline(always)]
    pub fn leverage_computed(
        raw: f64, final_lev: f64, capital: f64, vol: f64, conf: f64, coin_id: usize,
    ) -> Self {
        Self {
            timestamp_ns: Self::current_time_ns(),
            subsystem: SUBSYSTEM_LEVERAGE_MATRIX,
            frame_type: FRAME_LEVERAGE_COMPUTED,
            _reserved: [0; 6],
            payload: [
                if raw.is_finite() { raw } else { 1.0 },
                if final_lev.is_finite() { final_lev } else { 1.0 },
                if capital.is_finite() { capital } else { 13.0 },
                if vol.is_finite() { vol } else { 0.01 },
                if conf.is_finite() { conf } else { 0.5 },
                coin_id as f64,
            ],
        }
    }

    /// Crea un frame de mutación evolutiva.
    /// payload: [dsr, win_rate, profit_factor, trade_count, drawdown, accepted]
    #[inline(always)]
    pub fn evolution_mutation(
        dsr: f64, wr: f64, pf: f64, trades: usize, dd: f64, accepted: bool,
    ) -> Self {
        Self {
            timestamp_ns: Self::current_time_ns(),
            subsystem: SUBSYSTEM_EVOLUTION_ENGINE,
            frame_type: FRAME_EVOLUTION_MUTATION,
            _reserved: [0; 6],
            payload: [
                if dsr.is_finite() { dsr } else { 0.0 },
                if wr.is_finite() { wr } else { 0.5 },
                if pf.is_finite() { pf } else { 1.0 },
                trades as f64,
                if dd.is_finite() { dd } else { 0.0 },
                if accepted { 1.0 } else { 0.0 },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_telemetry_frame_creation_and_alignment() {
        assert_eq!(std::mem::size_of::<StrategyTelemetryFrame>(), 64);

        let scalp = StrategyTelemetryFrame::scalp_signal(0.85, 2.5, 0.4, 1.5, 0, 1.0);
        assert_eq!(scalp.subsystem, SUBSYSTEM_SCALP_STRATEGY);
        assert_eq!(scalp.frame_type, FRAME_SIGNAL_GENERATED);
        assert_eq!(scalp.payload[0], 0.85);

        let swing = StrategyTelemetryFrame::swing_signal(0.90, 0.75, 0.05, 0.02, 1, -1.0);
        assert_eq!(swing.subsystem, SUBSYSTEM_SWING_STRATEGY);
        assert_eq!(swing.payload[1], 0.75);

        let lev = StrategyTelemetryFrame::leverage_computed(10.0, 10.0, 13.0, 0.02, 0.8, 0);
        assert_eq!(lev.subsystem, SUBSYSTEM_LEVERAGE_MATRIX);

        let evo = StrategyTelemetryFrame::evolution_mutation(1.5, 0.70, 2.0, 100, 0.05, true);
        assert_eq!(evo.subsystem, SUBSYSTEM_EVOLUTION_ENGINE);
        assert_eq!(evo.payload[5], 1.0);
    }

    #[test]
    fn test_strategy_telemetry_nan_sanitization() {
        let scalp_nan = StrategyTelemetryFrame::scalp_signal(f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0, f64::NAN);
        for &p in &scalp_nan.payload {
            assert!(p.is_finite());
        }

        let swing_nan = StrategyTelemetryFrame::swing_signal(f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0, f64::NAN);
        for &p in &swing_nan.payload {
            assert!(p.is_finite());
        }
    }
}

