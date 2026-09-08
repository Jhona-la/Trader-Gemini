use crossbeam_queue::ArrayQueue;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TradeStats {
    pub gross_pnl: f64,
    pub net_pnl: f64,
    pub fees_paid: f64,
    pub roi_pct: f64,
    pub timeframe_ms: u64,
    pub active_leverage: f64,
}

pub struct StackBuffer {
    pub buf: [u8; 512],
    pub len: usize,
}

impl fmt::Write for StackBuffer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let remaining = 512 - self.len;
        let to_copy = std::cmp::min(bytes.len(), remaining);
        self.buf[self.len..self.len + to_copy].copy_from_slice(&bytes[..to_copy]);
        self.len += to_copy;
        Ok(())
    }
}

/// Global Lock-Free Ring Buffer for Binary Telemetry Events
static BINARY_QUEUE: OnceLock<ArrayQueue<([u8; 512], usize)>> = OnceLock::new();

pub fn init_binary_telemetry(capacity: usize) {
    // FIX #1447 / D-213: Clamping defensivo de capacidad para no sobrecargar RAM (max 64K)
    let safe_capacity = capacity.clamp(16, 65_536);
    let _ = BINARY_QUEUE.set(ArrayQueue::new(safe_capacity));
}

pub fn push_binary_telemetry(data: &[u8]) {
    let queue = BINARY_QUEUE.get_or_init(|| ArrayQueue::new(16_384));
    let mut fixed_buf = [0u8; 512];
    let len = std::cmp::min(data.len(), 512);
    fixed_buf[..len].copy_from_slice(&data[..len]);
    let _ = queue.force_push((fixed_buf, len)); // O(1) wait-free push with oldest drop on full
}

pub fn pop_binary_telemetry() -> Option<([u8; 512], usize)> {
    let queue = BINARY_QUEUE.get_or_init(|| ArrayQueue::new(16_384));
    queue.pop()
}

/// Global Lock-Free Queue for Structured Trade Stats
static STATS_QUEUE: OnceLock<ArrayQueue<TradeStats>> = OnceLock::new();

pub fn init_stats_telemetry(capacity: usize) {
    // FIX #1447 / D-213: Clamping defensivo de capacidad (max 64K)
    let safe_capacity = capacity.clamp(16, 65_536);
    let _ = STATS_QUEUE.set(ArrayQueue::new(safe_capacity));
}

pub fn push_trade_stats(mut stats: TradeStats) {
    // FIX #1447: Sanitización estricta de métricas estadísticas
    if !stats.gross_pnl.is_finite() { stats.gross_pnl = 0.0; }
    if !stats.net_pnl.is_finite() { stats.net_pnl = 0.0; }
    if !stats.fees_paid.is_finite() { stats.fees_paid = 0.0; }
    if !stats.roi_pct.is_finite() { stats.roi_pct = 0.0; }
    if !stats.active_leverage.is_finite() || stats.active_leverage < 1.0 { stats.active_leverage = 1.0; }

    let queue = STATS_QUEUE.get_or_init(|| ArrayQueue::new(16_384));
    let _ = queue.force_push(stats);
}

pub fn pop_trade_stats() -> Option<TradeStats> {
    let queue = STATS_QUEUE.get_or_init(|| ArrayQueue::new(16_384));
    queue.pop()
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn test_stack_buffer_formatting() {
        let mut buf = StackBuffer { buf: [0u8; 512], len: 0 };
        write!(buf, "PING: {}", 123).unwrap();
        assert_eq!(buf.len, 9);
        assert_eq!(&buf.buf[..9], b"PING: 123");
    }

    #[test]
    fn test_trade_stats_queue() {
        init_stats_telemetry(16);
        while pop_trade_stats().is_some() {}
        let stats = TradeStats {
            gross_pnl: 1.50,
            net_pnl: 1.48,
            fees_paid: 0.02,
            roi_pct: 11.38,
            timeframe_ms: 1000,
            active_leverage: 10.0,
        };
        push_trade_stats(stats.clone());
        let popped = pop_trade_stats();
        assert!(popped.is_some());
        let p = popped.unwrap();
        assert_eq!(p.gross_pnl, 1.50);
        assert_eq!(p.active_leverage, 10.0);
    }

    #[test]
    fn test_binary_telemetry_push_and_nan_sanitization() {
        init_binary_telemetry(32);
        while pop_binary_telemetry().is_some() {}
        let sample = b"TELEMETRY_SAMPLE_OCTET_STREAM";
        push_binary_telemetry(sample);
        let popped = pop_binary_telemetry();
        assert!(popped.is_some());
        let (buf, len) = popped.unwrap();
        assert_eq!(len, sample.len());
        assert_eq!(&buf[..len], sample);

        // Verify TradeStats NaN sanitization
        while pop_trade_stats().is_some() {}
        push_trade_stats(TradeStats {
            gross_pnl: f64::NAN,
            net_pnl: f64::NAN,
            fees_paid: f64::NAN,
            roi_pct: f64::NAN,
            timeframe_ms: 500,
            active_leverage: 0.5, // < 1.0 -> should clamp to 1.0
        });
        let p_nan = pop_trade_stats().unwrap();
        assert_eq!(p_nan.gross_pnl, 0.0);
        assert_eq!(p_nan.net_pnl, 0.0);
        assert_eq!(p_nan.fees_paid, 0.0);
        assert_eq!(p_nan.roi_pct, 0.0);
        assert_eq!(p_nan.active_leverage, 1.0);
    }
}

