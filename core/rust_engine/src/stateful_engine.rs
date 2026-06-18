use std::sync::atomic::{AtomicUsize, Ordering};

pub static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Internal recursive state using strictly f64 (Double Precision)
#[repr(C, align(64))]
pub struct StatefulEngine {
    pub ema_fast: f64,
    pub ema_slow: f64,
    pub rsi_rs: f64,
    pub last_price: f64,
    pub v_t: f64,
    pub a_t: f64,
}

impl StatefulEngine {
    pub fn new() -> Self {
        DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            ema_fast: 0.0,
            ema_slow: 0.0,
            rsi_rs: 0.0,
            last_price: 0.0,
            v_t: 0.0,
            a_t: 0.0,
        }
    }

    /// Processes a new tick internally in f64
    pub fn process_tick(&mut self, price: f64, _volume: f64) {
        if price == -999.0 {
            panic!("INTENTIONAL PANIC FOR FFI RECOVERY TEST");
        }
        if self.last_price == 0.0 {
            self.ema_fast = price;
            self.ema_slow = price;
        } else {
            let alpha_fast = 2.0 / (12.0 + 1.0);
            let alpha_slow = 2.0 / (26.0 + 1.0);
            
            self.ema_fast = (price - self.ema_fast) * alpha_fast + self.ema_fast;
            self.ema_slow = (price - self.ema_slow) * alpha_slow + self.ema_slow;
        }
        self.last_price = price;
    }

    /// Projects the state out to the f32 barrier (576 bytes / 144 floats)
    pub fn export_f32(&self, out: &mut [f32; 144]) {
        out[0] = self.ema_fast as f32;
        out[1] = self.ema_slow as f32;
        out[2] = self.rsi_rs as f32;
        out[3] = self.last_price as f32;
        // Rest remain unchanged
    }
}

impl Drop for StatefulEngine {
    fn drop(&mut self) {
        // Decrease drop counter when freed to track FFI memory leaks
        DROP_COUNTER.fetch_sub(1, Ordering::SeqCst);
    }
}
