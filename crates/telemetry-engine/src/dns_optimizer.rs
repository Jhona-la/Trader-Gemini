use std::net::ToSocketAddrs;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct DnsOptimizer {
    pub best_endpoint: Arc<std::sync::RwLock<String>>,
    pub lowest_latency_ms: AtomicU64,
}

impl DnsOptimizer {
    pub fn new() -> Self {
        Self {
            best_endpoint: Arc::new(std::sync::RwLock::new("stream.binance.com".to_string())),
            lowest_latency_ms: AtomicU64::new(9999),
        }
    }

    /// Run this in a background Tokio task
    pub async fn run_continuous_monitoring(&self) {
        let endpoints = vec![
            "stream.binance.com:9443",
            "fstream.binance.com:443",
            "dstream.binance.com:443",
        ];

        loop {
            let mut best_local = String::new();
            let mut best_lat = 9999;

            for ep in &endpoints {
                let start = Instant::now();
                if let Ok(mut addrs) = ep.to_socket_addrs() {
                    if let Some(_addr) = addrs.next() {
                        // In a real scenario we'd do a TCP handshake to measure TTFB
                        let lat = start.elapsed().as_millis() as u64;
                        if lat < best_lat {
                            best_lat = lat;
                            best_local = ep.to_string();
                        }
                    }
                }
            }

            if best_lat < self.lowest_latency_ms.load(Ordering::Relaxed) || best_lat < 50 {
                self.lowest_latency_ms.store(best_lat, Ordering::Relaxed);
                if let Ok(mut write_lock) = self.best_endpoint.write() {
                    *write_lock = best_local.replace(":9443", "").replace(":443", "");
                }
            }

            // Test DNS every 5 minutes
            tokio::time::sleep(Duration::from_secs(300)).await;
        }
    }

    pub fn get_optimal_endpoint(&self) -> String {
        self.best_endpoint.read().unwrap().clone()
    }
}

impl Default for DnsOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dns_optimizer_defaults_and_endpoint_resolution() {
        let optimizer = DnsOptimizer::default();
        let ep = optimizer.get_optimal_endpoint();
        assert_eq!(ep, "stream.binance.com");
        assert_eq!(optimizer.lowest_latency_ms.load(Ordering::Relaxed), 9999);
    }

    #[test]
    fn test_dns_optimizer_manual_endpoint_update() {
        let optimizer = DnsOptimizer::new();
        optimizer.lowest_latency_ms.store(12, Ordering::Relaxed);
        if let Ok(mut write_lock) = optimizer.best_endpoint.write() {
            *write_lock = "fstream.binance.com".to_string();
        }

        assert_eq!(optimizer.get_optimal_endpoint(), "fstream.binance.com");
        assert_eq!(optimizer.lowest_latency_ms.load(Ordering::Relaxed), 12);
    }
}

