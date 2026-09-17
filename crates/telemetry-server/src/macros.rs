//! Wait-Free Telemetry Macros
//! FASE 16: Pervasive Wait-Free Telemetry
//! These macros allow printing information and warnings without blocking the Hot Path.
//! They submit logs into a global lock-free ArrayQueue.

use std::fmt::{self, Write};
use std::sync::{Arc, OnceLock};

/// Limit the max string length to avoid large dynamic allocations in hot paths.
pub const MAX_LOG_LEN: usize = 512;

#[derive(Clone, Copy)]
pub struct FixedLogBuffer {
    pub data: [u8; MAX_LOG_LEN],
    pub len: usize,
}

impl FixedLogBuffer {
    pub fn new() -> Self {
        Self {
            data: [0; MAX_LOG_LEN],
            len: 0,
        }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.data[..self.len]).unwrap_or("UTF8_ERR")
    }
}

impl Write for FixedLogBuffer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let remaining = MAX_LOG_LEN - self.len;
        let to_copy = std::cmp::min(bytes.len(), remaining);
        if to_copy > 0 {
            self.data[self.len..self.len + to_copy].copy_from_slice(&bytes[..to_copy]);
            self.len += to_copy;
        }
        Ok(())
    }
}

// FASE VII: Zero-Latency RingBuffer (Lock-Free)
// Usamos ArrayQueue en lugar de channels MPMC para garantizar latencia O(1) determinística.
pub static LOG_QUEUE: OnceLock<Arc<crossbeam::queue::ArrayQueue<FixedLogBuffer>>> = OnceLock::new();

/// Inicializa la cola global de logs asíncronos en modo RingBuffer Lock-Free
pub fn init_telemetry_logger() {
    let queue = Arc::new(crossbeam::queue::ArrayQueue::new(1_000_000));
    let _ = LOG_QUEUE.set(queue.clone());

    // Hilo de Ghost Flusher exclusivo para impresiones a stdout.
    // Extrae del RingBuffer sin bloquear al GodEngine.
    std::thread::Builder::new()
        .name("GhostLogger".to_string())
        .spawn(move || {
            loop {
                if let Some(msg) = queue.pop() {
                    println!("{}", msg.as_str());
                } else {
                    // Backoff nanosegundos para evitar spinlock CPU al 100%
                    std::thread::yield_now();
                }
            }
        })
        .expect("Fallo al crear hilo GhostLogger");
}

#[macro_export]
macro_rules! telemetry_log {
    ($($arg:tt)*) => {{
        if let Some(queue) = $crate::macros::LOG_QUEUE.get() {
            use std::fmt::Write;
            let mut buf = $crate::macros::FixedLogBuffer::new();
            let _ = write!(&mut buf, $($arg)*);
            let _ = queue.force_push(buf);
        } else {
            println!($($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! telemetry_err {
    ($($arg:tt)*) => {{
        if let Some(queue) = $crate::macros::LOG_QUEUE.get() {
            use std::fmt::Write;
            let mut buf = $crate::macros::FixedLogBuffer::new();
            let _ = write!(&mut buf, "⚠️ ERROR: ");
            let _ = write!(&mut buf, $($arg)*);
            let _ = queue.force_push(buf);
        } else {
            eprintln!($($arg)*);
        }
    }};
}
