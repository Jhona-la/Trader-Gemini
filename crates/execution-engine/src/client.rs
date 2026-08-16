use arc_swap::ArcSwap;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Buffer stack para cero-alocación en el hot-path.
/// F1.2: 1024 bytes (un clientOrderId UUIDv7 + firma excedían el margen del
/// original de 512) y detección de overflow: jamás pánico ni truncamiento
/// silencioso — el caller consulta `is_overflow()` antes de firmar/enviar.
pub struct ZeroAllocBuffer {
    bytes: [u8; 1024],
    len: usize,
    overflow: bool,
}

impl ZeroAllocBuffer {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            bytes: [0; 1024],
            len: 0,
            overflow: false,
        }
    }

    #[inline(always)]
    pub fn push_str(&mut self, s: &str) {
        let b = s.as_bytes();
        if self.len + b.len() > self.bytes.len() {
            self.overflow = true;
            return;
        }
        self.bytes[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
    }

    #[inline(always)]
    pub fn is_overflow(&self) -> bool {
        self.overflow
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
        self.overflow = false;
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

#[derive(Clone)]
pub struct BinanceClient {
    http: Client,
    /// Arc: ArcSwap no es Clone por diseño; los clones comparten el swap
    /// (hot-swap de credenciales coherente entre réplica del stream F1.6).
    pub api_key: std::sync::Arc<ArcSwap<HeaderValue>>,
    /// Arc interno: los clones comparten el flag (hot-swap coherente, F1.6).
    pub is_testnet: std::sync::Arc<AtomicBool>,
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
            api_key: std::sync::Arc::new(ArcSwap::from_pointee(header_val)),
            is_testnet: std::sync::Arc::new(AtomicBool::new(is_testnet)),
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

    /// F1.6: POST /fapi/v1/listenKey — crea la clave del user-data stream.
    /// NO requiere firma: solo el header X-MBX-APIKEY.
    pub async fn create_listen_key(&self) -> Result<String, String> {
        let api_key = self.api_key.load();
        let url = format!("{}/fapi/v1/listenKey", self.get_base_url());
        let response = self
            .http
            .post(&url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await
            .map_err(|e| format!("Network Error: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::order_types::parse_reject_body(
                &body,
                status.as_u16(),
            ));
        }
        let body = response
            .text()
            .await
            .map_err(|e| format!("TEXT_ERR: {}", e))?;
        #[derive(serde::Deserialize)]
        struct ListenKeyResp {
            #[serde(rename = "listenKey")]
            listen_key: String,
        }
        let parsed: ListenKeyResp =
            serde_json::from_str(&body).map_err(|e| format!("LISTENKEY_PARSE: {}", e))?;
        Ok(parsed.listen_key)
    }

    /// F1.6: PUT /fapi/v1/listenKey — keepalive (expira a los 60 min; refrescar a los 30).
    pub async fn keep_alive_listen_key(&self) -> Result<(), String> {
        let api_key = self.api_key.load();
        let url = format!("{}/fapi/v1/listenKey", self.get_base_url());
        let response = self
            .http
            .put(&url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await
            .map_err(|e| format!("Network Error: {}", e))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(crate::order_types::parse_reject_body(
                &body,
                status.as_u16(),
            ))
        }
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

    /// F1.1: POST de orden con respuesta TIPADA — lee el body completo y lo
    /// parsea a OrderAck (orderId, status, executedQty, avgPrice, fills).
    /// Distingue tres clases de fallo para idempotencia (F1.2):
    ///   - Err empezando con "AMBIGUOUS:" → timeout/red: la orden PUEDE existir
    ///     en Binance; el caller DEBE consultar por clientOrderId antes de reintentar.
    ///   - Err con code negativo → rechazo definitivo, seguro reintentar.
    ///   - Ok(ack) → respuesta conocida.
    #[inline(always)]
    pub async fn execute_order_payload_typed(
        &self,
        full_url: &str,
    ) -> Result<(BinanceRateLimits, crate::order_types::OrderAck), String> {
        use crate::order_types::{parse_order_body, parse_reject_body, truncate, OrderAck};

        let api_key = self.api_key.load();
        let response = self
            .http
            .post(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await;

        match response {
            Ok(resp) => {
                let limits = extract_limits(resp.headers());
                let status = resp.status();
                let body = resp
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                if status.is_success() {
                    let ack: OrderAck = parse_order_body(&body)?;
                    Ok((limits, ack))
                } else if status.as_u16() == 429 || status.as_u16() == 418 {
                    Err("HTTP_429_TOO_MANY_REQUESTS_OR_BANNED".to_string())
                } else if status.is_client_error() {
                    // 4xx: Binance procesó el request y lo rechazó — la orden NO existe.
                    Err(parse_reject_body(&body, status.as_u16()))
                } else {
                    // 5xx: estado desconocido — tratar como ambiguo.
                    Err(format!(
                        "AMBIGUOUS: HTTP {} body={}",
                        status.as_u16(),
                        truncate(&body, 200)
                    ))
                }
            }
            Err(e) => Err(format!("AMBIGUOUS: Network Error: {}", e)),
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
