use reqwest::{Client, ClientBuilder};
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use std::time::Duration;


/// Buffer stack de 512 bytes para cero-alocación.
pub struct ZeroAllocBuffer {
    bytes: [u8; 512],
    len: usize,
}

impl ZeroAllocBuffer {
    #[inline(always)]
    pub fn new() -> Self {
        Self { bytes: [0; 512], len: 0 }
    }

    #[inline(always)]
    pub fn push_str(&mut self, s: &str) {
        let b = s.as_bytes();
        self.bytes[self.len..self.len+b.len()].copy_from_slice(b);
        self.len += b.len();
    }

    #[inline(always)]
    pub fn push_f64(&mut self, v: f64) {
        let mut b = ryu::Buffer::new();
        self.push_str(b.format(v));
    }

    #[inline(always)]
    pub fn push_u64(&mut self, v: u64) {
        let mut b = itoa::Buffer::new();
        self.push_str(b.format(v));
    }

    #[inline(always)]
    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len]) }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}


pub const BINANCE_BASE_URL: &str = "https://fapi.binance.com";

#[derive(Debug, Serialize, Deserialize)]
pub struct BinanceErrorResponse {
    pub code: i64,
    pub msg: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BinanceRateLimits {
    pub weight_1m: Option<usize>,
    pub orders_10s: Option<usize>,
    pub orders_1m: Option<usize>,
}

#[inline(always)]
fn extract_limits(headers: &HeaderMap) -> BinanceRateLimits {
    let mut limits = BinanceRateLimits::default();
    if let Some(val) = headers.get("X-MBX-USED-WEIGHT-1M") {
        if let Ok(s) = val.to_str() {
            limits.weight_1m = s.parse().ok();
        }
    }
    if let Some(val) = headers.get("X-MBX-ORDER-COUNT-10S") {
        if let Ok(s) = val.to_str() {
            limits.orders_10s = s.parse().ok();
        }
    }
    if let Some(val) = headers.get("X-MBX-ORDER-COUNT-1M") {
        if let Ok(s) = val.to_str() {
            limits.orders_1m = s.parse().ok();
        }
    }
    limits
}

pub struct BinanceClient {
    http: Client,
    api_key: String,
}

impl BinanceClient {
    pub fn new(api_key: String) -> Self {
        // HFT Connection Pooling with TCP_NODELAY (Disabling Nagle's algorithm)
        let http = ClientBuilder::new()
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(300))
            .tcp_nodelay(true)
            .timeout(Duration::from_millis(1500))
            .build()
            .expect("Failed to build hyper-optimized reqwest client");

        Self { http, api_key }
    }

    /// Ejecuta una orden firmada enviando el payload HTTP de forma asíncrona O(1).
    #[inline(always)]
    pub async fn execute_order_payload(&self, full_url: &str) -> Result<BinanceRateLimits, String> {
        let response = self
            .http
            .post(full_url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let limits = extract_limits(resp.headers());
                    Ok(limits)
                } else {
                    let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }

    /// Cancela una orden enviando el payload HTTP (DELETE).
    #[inline(always)]
    pub async fn cancel_order_payload(&self, full_url: &str) -> Result<BinanceRateLimits, String> {
        let response = self
            .http
            .delete(full_url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let limits = extract_limits(resp.headers());
                    Ok(limits)
                } else {
                    let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }
}
