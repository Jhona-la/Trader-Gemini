use arc_swap::ArcSwap;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Buffer stack de 512 bytes para cero-alocación.
pub struct ZeroAllocBuffer {
    bytes: [u8; 512],
    len: usize,
}

impl ZeroAllocBuffer {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            bytes: [0; 512],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push_str(&mut self, s: &str) {
        let b = s.as_bytes();
        self.bytes[self.len..self.len + b.len()].copy_from_slice(b);
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
pub const BINANCE_TESTNET_URL: &str = "https://testnet.binancefuture.com";

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

use std::sync::atomic::{AtomicBool, Ordering};

pub struct BinanceClient {
    http: Client,
    pub api_key: ArcSwap<HeaderValue>,
    pub is_testnet: AtomicBool,
}

impl BinanceClient {
    pub fn new(api_key: String, is_testnet: bool) -> Self {
        // HFT Connection Pooling with TCP_NODELAY (Disabling Nagle's algorithm)
        let http = ClientBuilder::new()
            .pool_max_idle_per_host(25) // Maximize concurrent keep-alives for Binance
            .pool_idle_timeout(None) // NEVER drop idle connections to avoid TLS handshake latency
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30))) // Evita que los firewalls/OS dropeen la conexión silenciosamente
            .hickory_dns(true) // Fast DNS
            .timeout(Duration::from_millis(5000)) // 5 seconds timeout for init / testnet
            .build()
            .expect("Failed to build hyper-optimized reqwest client");

        let header_val =
            HeaderValue::from_str(&api_key).unwrap_or_else(|_| HeaderValue::from_static(""));
        Self {
            http,
            api_key: ArcSwap::from_pointee(header_val),
            is_testnet: AtomicBool::new(is_testnet),
        }
    }

    pub fn get_base_url(&self) -> &str {
        if self.is_testnet.load(Ordering::Relaxed) {
            BINANCE_TESTNET_URL
        } else {
            BINANCE_BASE_URL
        }
    }

    pub fn hot_swap_credentials(&self, new_key: String, is_testnet: bool) {
        let header_val =
            HeaderValue::from_str(&new_key).unwrap_or_else(|_| HeaderValue::from_static(""));
        self.api_key.store(Arc::new(header_val));
        self.is_testnet.store(is_testnet, Ordering::Relaxed);
    }

    /// Ejecuta una orden firmada enviando el payload HTTP de forma asíncrona O(1).
    #[inline(always)]
    pub async fn execute_order_payload(&self, full_url: &str) -> Result<BinanceRateLimits, String> {
        let api_key = self.api_key.load();
        let response = self
            .http
            .post(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let limits = extract_limits(resp.headers());
                    Ok(limits)
                } else if resp.status().as_u16() == 429 || resp.status().as_u16() == 418 {
                    Err("HTTP_429_TOO_MANY_REQUESTS_OR_BANNED".to_string())
                } else {
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }

    /// Cancela una orden enviando el payload HTTP (DELETE).
    #[inline(always)]
    pub async fn cancel_order_payload(&self, full_url: &str) -> Result<BinanceRateLimits, String> {
        let api_key = self.api_key.load();
        let response = self
            .http
            .delete(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let limits = extract_limits(resp.headers());
                    Ok(limits)
                } else {
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }
    /// GET payload (used for fetching balances, etc)
    #[inline(always)]
    pub async fn get_payload(&self, full_url: &str) -> Result<(BinanceRateLimits, String), String> {
        let api_key = self.api_key.load();
        let response = self
            .http
            .get(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await;

        match response {
            Ok(resp) => {
                let status = resp.status();
                let limits = extract_limits(resp.headers());
                let raw_text = resp
                    .text()
                    .await
                    .unwrap_or_else(|e| format!("TEXT_ERR: {}", e));
                if status.is_success() {
                    Ok((limits, raw_text))
                } else {
                    // Nunca loguear `full_url`: contiene la firma HMAC del request.
                    Err(format!(
                        "Binance API Error: status={} body={}",
                        status, raw_text
                    ))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }

    /// POST payload (used for changing leverage, margin type, etc)
    #[inline(always)]
    pub async fn post_payload(
        &self,
        full_url: &str,
    ) -> Result<(BinanceRateLimits, String), String> {
        let api_key_arc = self.api_key.load();
        let api_key = (*api_key_arc).clone();
        let response = self
            .http
            .post(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref())
            .send()
            .await;

        match response {
            Ok(resp) => {
                let limits = extract_limits(resp.headers());
                if resp.status().is_success() {
                    let text = resp.text().await.unwrap_or_else(|_| "[]".to_string());
                    Ok((limits, text))
                } else {
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }
}
