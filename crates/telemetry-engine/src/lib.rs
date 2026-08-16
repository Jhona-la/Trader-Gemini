pub mod atomic_telemetry;
pub mod dns_optimizer;

/// Initialize the telemetry engine with a fixed capacity ring buffer
pub fn init_telemetry(capacity: usize) {
    atomic_telemetry::init_binary_telemetry(capacity);
}

pub fn pop_telemetry() -> Option<String> {
    atomic_telemetry::pop_binary_telemetry().map(|(buf, len)| {
        String::from_utf8_lossy(&buf[..len]).into_owned()
    })
}

pub use atomic_telemetry::TradeStats;

pub fn init_stats_telemetry(capacity: usize) {
    atomic_telemetry::init_stats_telemetry(capacity);
}

pub fn push_trade_stats(stats: TradeStats) {
    atomic_telemetry::push_trade_stats(stats);
}

pub fn pop_trade_stats() -> Option<TradeStats> {
    atomic_telemetry::pop_trade_stats()
}

/// A zero-latency replacement for `println!` that writes directly to a stack buffer
/// and pushes to the lock-free ring buffer without heap allocations.
#[macro_export]
macro_rules! telemetry {
    ($($arg:tt)*) => {
        {
            use core::fmt::Write;
            let mut buf = $crate::atomic_telemetry::StackBuffer { buf: [0; 512], len: 0 };
            let _ = write!(&mut buf, $($arg)*);
            $crate::atomic_telemetry::push_binary_telemetry(&buf.buf[..buf.len]);
        }
    };
}

#[macro_export]
macro_rules! telemetry_err {
    ($($arg:tt)*) => {
        {
            use core::fmt::Write;
            let mut buf = $crate::atomic_telemetry::StackBuffer { buf: [0; 512], len: 0 };
            let _ = write!(&mut buf, "⚠️ ERROR: ");
            let _ = write!(&mut buf, $($arg)*);
            $crate::atomic_telemetry::push_binary_telemetry(&buf.buf[..buf.len]);
        }
    };
}

use crossbeam::channel::{bounded, Sender, Receiver};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct TensorRecord {
    pub timestamp_ns: u64,
    pub symbol_id: usize,
    pub net_confidence: f64,
    pub expected_volatility: f64,
    pub ml_long_thresh: f64,
    pub ml_short_thresh: f64,
    pub long_votes: f64,
    pub short_votes: f64,
    pub z_score: f64,
    pub pnl_gross: f64,
}

static TENSOR_TX: OnceLock<Sender<TensorRecord>> = OnceLock::new();

pub fn init_tensor_telemetry(capacity: usize) -> Receiver<TensorRecord> {
    let (tx, rx) = bounded(capacity);
    let _ = TENSOR_TX.set(tx);
    rx
}

#[inline(always)]
pub fn push_tensor_record(record: TensorRecord) {
    if let Some(tx) = TENSOR_TX.get() {
        let _ = tx.try_send(record); // Non-blocking, drops if full
    }
}
