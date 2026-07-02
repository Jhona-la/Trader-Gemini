pub mod world_bank;
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct TokenBucket {
    capacity: u64,
    fill_rate: f64, // tokens per millisecond
    tokens: AtomicU64,
    last_update: AtomicU64,
}

impl TokenBucket {
    pub fn new(capacity: u64, fill_rate: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            capacity,
            fill_rate,
            tokens: AtomicU64::new(capacity),
            last_update: AtomicU64::new(now),
        }
    }

    /// Try to consume 1 token. Lock-free logic.
    pub fn try_consume(&self) -> bool {
        loop {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
                
            let last = self.last_update.load(Ordering::SeqCst);
            let current_tokens = self.tokens.load(Ordering::SeqCst);
            
            let elapsed_ms = now.saturating_sub(last);
            let added_tokens = (elapsed_ms as f64 * self.fill_rate) as u64;
            
            let mut new_tokens = std::cmp::min(self.capacity, current_tokens + added_tokens);
            
            if new_tokens == 0 {
                return false;
            }
            
            new_tokens -= 1;
            
            // Try to update time and tokens atomically via CAS-like loop approach
            // In a strict high-concurrency setting, we might use a spin-loop with compare_exchange
            if self.tokens.compare_exchange(current_tokens, new_tokens, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                if added_tokens > 0 {
                    let _ = self.last_update.compare_exchange(last, now, Ordering::SeqCst, Ordering::Relaxed);
                }
                return true;
            }
        }
    }
}

pub struct ZeroCopyReader {
    mmap: memmap2::Mmap,
}

impl ZeroCopyReader {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}
