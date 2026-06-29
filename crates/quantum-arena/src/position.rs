use crate::atomic_float::AtomicF64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Lock-free Position tracking for the Hot Path
#[repr(C, align(64))]
pub struct Position {
    pub is_open: AtomicBool,
    pub is_long: AtomicBool,
    pub entry_price: AtomicF64,
    pub quantity: AtomicF64,
    pub entry_time_ms: AtomicU64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            is_open: AtomicBool::new(false),
            is_long: AtomicBool::new(true),
            entry_price: AtomicF64::new(0.0),
            quantity: AtomicF64::new(0.0),
            entry_time_ms: AtomicU64::new(0),
        }
    }
}

impl Position {
    #[inline(always)]
    pub fn open(&self, is_long: bool, price: f64, qty: f64) {
        self.is_long.store(is_long, Ordering::Relaxed);
        self.entry_price.store(price, Ordering::Relaxed);
        self.quantity.store(qty, Ordering::Relaxed);
        
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        self.entry_time_ms.store(now, Ordering::Relaxed);
        
        self.is_open.store(true, Ordering::Release);
    }

    #[inline(always)]
    pub fn close(&self) -> (bool, f64, f64) {
        self.is_open.store(false, Ordering::Release);
        
        let is_long = self.is_long.load(Ordering::Relaxed);
        let price = self.entry_price.load(Ordering::Relaxed);
        let qty = self.quantity.load(Ordering::Relaxed);
        
        (is_long, price, qty)
    }

    #[inline(always)]
    pub fn is_open(&self) -> bool {
        self.is_open.load(Ordering::Acquire)
    }
}

#[repr(C, align(64))]
pub struct PositionManager {
    pub scalp_position: Position,
    pub swing_position: Position,
}

impl Default for PositionManager {
    fn default() -> Self {
        Self {
            scalp_position: Position::default(),
            swing_position: Position::default(),
        }
    }
}
