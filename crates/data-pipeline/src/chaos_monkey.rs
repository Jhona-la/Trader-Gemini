use std::sync::atomic::{AtomicU64, Ordering};

pub struct ChaosMonkey {
    enabled: bool,
    disconnect_probability: f64, // Probability per tick
    seed: AtomicU64,
}

impl Default for ChaosMonkey {
    fn default() -> Self {
        Self::new()
    }
}

impl ChaosMonkey {
    pub fn new() -> Self {
        let val = std::env::var("BINANCE_CHAOS_MODE").unwrap_or_default().trim().to_lowercase();
        let enabled = val == "true" || val == "1";
        Self {
            enabled,
            disconnect_probability: 0.0001, // 0.01% chance per tick to disconnect
            seed: AtomicU64::new(0x853c49e6748fea9b),
        }
    }

    /// Determines if the stream should artificially drop the connection
    #[inline(always)]
    pub fn should_disconnect(&self) -> bool {
        if !self.enabled {
            return false;
        }
        
        // FIX #696: SplitMix64 $O(1)$ lock-free sin locks ni TLS en hot path
        let prev = self.seed.fetch_add(0x9E3779B97F4A7C15, Ordering::Relaxed);
        let mut x = prev ^ (prev >> 30);
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x = x ^ (x >> 27);
        x = x.wrapping_mul(0x94d049bb133111eb);
        x = x ^ (x >> 31);
        let rand_float = (x >> 11) as f64 / (1u64 << 53) as f64;
        rand_float < self.disconnect_probability
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chaos_monkey_disabled_by_default() {
        let monkey = ChaosMonkey::default();
        assert!(!monkey.enabled);
        assert!(!monkey.should_disconnect());
    }

    #[test]
    fn test_chaos_monkey_enabled_splitmix64() {
        let monkey = ChaosMonkey {
            enabled: true,
            disconnect_probability: 1.0, // 100% chance for test
            seed: AtomicU64::new(12345),
        };
        assert!(monkey.should_disconnect());
    }
}
