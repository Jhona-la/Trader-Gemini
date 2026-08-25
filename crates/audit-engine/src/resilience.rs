
use std::time::Duration;

/// Simulador de Caos para el ApiCircuitBreaker
pub struct ChaosMonkey {
    pub drop_rate: f64,
    pub latency_spike: Duration,
    pub rate_limit_hit: bool,
}

impl ChaosMonkey {
    pub fn new() -> Self {
        Self {
            drop_rate: 0.0,
            latency_spike: Duration::from_millis(0),
            rate_limit_hit: false,
        }
    }

    /// Simula un fallo de red o rechazo de Binance
    pub fn inject_chaos(&self) -> Result<(), String> {
        if self.rate_limit_hit {
            return Err("HTTP 429 Too Many Requests (Simulado)".to_string());
        }
        
        // FIX #1493: Entropía determinista y multi-arquitectura (x86_64 / ARM / WASM)
        #[cfg(target_arch = "x86_64")]
        let r = unsafe { core::arch::x86_64::_rdtsc() };
        #[cfg(not(target_arch = "x86_64"))]
        let r = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345);
        let rand_val = (r % 100) as f64 / 100.0;
        if rand_val < self.drop_rate {
            return Err("Network Drop (Simulado)".to_string());
        }

        if self.latency_spike.as_millis() > 0 {
            std::thread::sleep(self.latency_spike);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chaos_injection() {
        let mut monkey = ChaosMonkey::new();
        monkey.rate_limit_hit = true;
        
        let res = monkey.inject_chaos();
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "HTTP 429 Too Many Requests (Simulado)");
    }
}
