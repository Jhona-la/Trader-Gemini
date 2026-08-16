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
    let _ = BINARY_QUEUE.set(ArrayQueue::new(capacity));
}

pub fn push_binary_telemetry(data: &[u8]) {
    if let Some(queue) = BINARY_QUEUE.get() {
        let mut fixed_buf = [0u8; 512];
        let len = std::cmp::min(data.len(), 512);
        fixed_buf[..len].copy_from_slice(&data[..len]);
        let _ = queue.force_push((fixed_buf, len)); // O(1) wait-free push
    } else {
        // Fallback
        if let Ok(s) = std::str::from_utf8(data) {
            println!("{}", s);
        }
    }
}

pub fn pop_binary_telemetry() -> Option<([u8; 512], usize)> {
    if let Some(queue) = BINARY_QUEUE.get() {
        queue.pop()
    } else {
        None
    }
}

/// Global Lock-Free Queue for Structured Trade Stats
static STATS_QUEUE: OnceLock<ArrayQueue<TradeStats>> = OnceLock::new();

pub fn init_stats_telemetry(capacity: usize) {
    let _ = STATS_QUEUE.set(ArrayQueue::new(capacity));
}

pub fn push_trade_stats(stats: TradeStats) {
    if let Some(queue) = STATS_QUEUE.get() {
        let _ = queue.force_push(stats);
    }
}

pub fn pop_trade_stats() -> Option<TradeStats> {
    if let Some(queue) = STATS_QUEUE.get() {
        queue.pop()
    } else {
        None
    }
}
