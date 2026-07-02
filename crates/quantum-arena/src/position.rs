use crate::atomic_float::AtomicF64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};


/// Lock-free Position tracking for the Hot Path
#[repr(C, align(64))]
pub struct Position {
    pub is_open: AtomicBool,
    pub is_long: AtomicBool,
    pub entry_price: AtomicF64,
    pub quantity: AtomicF64,
    pub margin_used: AtomicF64,
    pub entry_time_ms: AtomicU64,
    pub trailing_phase: std::sync::atomic::AtomicU8,
    pub mfe_atr: AtomicF64,
    pub max_pnl_pct: AtomicF64,
    pub trail_stop: AtomicF64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            is_open: AtomicBool::new(false),
            is_long: AtomicBool::new(true),
            entry_price: AtomicF64::new(0.0),
            quantity: AtomicF64::new(0.0),
            margin_used: AtomicF64::new(0.0),
            entry_time_ms: AtomicU64::new(0),
            trailing_phase: std::sync::atomic::AtomicU8::new(0),
            mfe_atr: AtomicF64::new(0.0),
            max_pnl_pct: AtomicF64::new(0.0),
            trail_stop: AtomicF64::new(0.0),
        }
    }
}

impl Position {
    #[inline(always)]
    pub fn open(&self, is_long: bool, price: f64, qty: f64, margin: f64, current_time_ms: u64) {
        self.is_long.store(is_long, Ordering::Relaxed);
        self.entry_price.store(price, Ordering::Relaxed);
        self.quantity.store(qty, Ordering::Relaxed);
        self.margin_used.store(margin, Ordering::Relaxed);
        self.entry_time_ms.store(current_time_ms, Ordering::Relaxed);
        self.trailing_phase.store(0, Ordering::Relaxed);
        self.mfe_atr.store(0.0, Ordering::Relaxed);
        self.max_pnl_pct.store(0.0, Ordering::Relaxed);
        self.trail_stop.store(0.0, Ordering::Relaxed);
        self.is_open.store(true, Ordering::Release);
    }

    pub fn close(&self) -> (bool, f64, f64, f64) {
        self.is_open.store(false, Ordering::Release);
        let is_long = self.is_long.load(Ordering::Relaxed);
        let price = self.entry_price.swap(0.0, Ordering::Relaxed);
        let qty = self.quantity.swap(0.0, Ordering::Relaxed);
        let margin = self.margin_used.swap(0.0, Ordering::Relaxed);
        self.entry_time_ms.store(0, Ordering::Relaxed);
        self.trailing_phase.store(0, Ordering::Relaxed);
        self.mfe_atr.store(0.0, Ordering::Relaxed);
        self.max_pnl_pct.store(0.0, Ordering::Relaxed);
        self.trail_stop.store(0.0, Ordering::Relaxed);
        (is_long, price, qty, margin)
    }

    #[inline(always)]
    pub fn is_open(&self) -> bool {
        self.is_open.load(Ordering::Acquire)
    }
}

#[repr(C, align(64))]
#[derive(Default)]
pub struct PositionManager {
    pub scalp_position: Position,
    pub swing_position: Position,
}

