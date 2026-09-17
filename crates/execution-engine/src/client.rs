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
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
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

/// F1.8: Retry-After del 429 (segundos). Default 60 si Binance no lo envía:
/// conservador y documentado — cubre la ventana típica de reinicio de peso.
#[inline(always)]
fn extract_retry_after(headers: &HeaderMap) -> u64 {
    headers
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(60)
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
        // Timeout HFT calibrado a 1200ms para failover rápido y evitar slippage tardío
        Self::with_timeout(api_key, is_testnet, Duration::from_millis(1200))
    }

    pub fn with_timeout(api_key: String, is_testnet: bool, timeout: Duration) -> Self {
        // FIX #1497: Construcción resiliente de cliente HTTP reqwest HFT sin expect
        let http = ClientBuilder::new()
            .pool_max_idle_per_host(25) // Maximize concurrent keep-alives for Binance
            .pool_idle_timeout(None) // NEVER drop idle connections to avoid TLS handshake latency
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(15))) // Latidos cada 15s para evitar drops fantasma de socket
            .hickory_dns(true) // Fast DNS
            .timeout(timeout) // Timeout optimizado para HFT
            .build()
            .unwrap_or_else(|_| Client::new());

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

    pub fn set_testnet(&self, testnet: bool) {
        self.is_testnet.store(testnet, Ordering::Relaxed);
    }

    pub fn update_api_key(&self, new_key: String) {
        let header_val =
            HeaderValue::from_str(&new_key).unwrap_or_else(|_| HeaderValue::from_static(""));
        self.api_key.store(std::sync::Arc::new(header_val));
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

    /// Fetches server time directly from Binance endpoint GET /fapi/v1/time
    pub async fn fetch_server_time(&self) -> Result<u64, String> {
        let url = format!("{}/fapi/v1/time", self.get_base_url());
        let response = self
            .http
            .get(&url)
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

        #[derive(serde::Deserialize)]
        struct ServerTimeResp {
            #[serde(rename = "serverTime")]
            server_time: u64,
        }

        let resp: ServerTimeResp = response
            .json()
            .await
            .map_err(|e| format!("JSON Parse Error: {}", e))?;

        Ok(resp.server_time)
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
                } else if resp.status().as_u16() == 429 {
                    let retry_after = extract_retry_after(resp.headers());
                    Err(format!("HTTP_429_RATE_LIMITED retry_after={}", retry_after))
                } else if resp.status().as_u16() == 418 {
                    Err("HTTP_418_IP_BANNED".to_string())
                } else {
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("AMBIGUOUS: Network Error: {}", e)),
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
                let retry_after = extract_retry_after(resp.headers());
                let status = resp.status();
                let body = resp
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                if status.is_success() {
                    let ack: OrderAck = parse_order_body(&body)?;
                    Ok((limits, ack))
                } else if status.as_u16() == 429 {
                    // F1.8: rate limit = cooldown temporal, no kill-switch.
                    Err(format!("HTTP_429_RATE_LIMITED retry_after={}", retry_after))
                } else if status.as_u16() == 418 {
                    // F1.8: 418 = IP baneada por Binance → kill-switch legítimo.
                    Err("HTTP_418_IP_BANNED".to_string())
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
        self.get_payload_with_timeout(full_url, None).await
    }

    /// GET con presupuesto de tiempo POR REQUEST. El timeout global HFT
    /// (1.2s) incluye la lectura del body: exchangeInfo (~MBs) no cabe y
    /// falla intermitentemente según la latencia del momento. Los endpoints
    /// masivos usan esta variante con presupuesto explícito; los de orden
    /// siguen con el global (latencia es parte del contrato de ejecución).
    pub async fn get_payload_with_timeout(
        &self,
        full_url: &str,
        timeout: Option<std::time::Duration>,
    ) -> Result<(BinanceRateLimits, String), String> {
        let api_key = self.api_key.load();
        let mut req = self
            .http
            .get(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone());
        if let Some(t) = timeout {
            req = req.timeout(t);
        }
        let response = req.send().await;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_alloc_buffer_push_str_f64_u64_and_clear() {
        let mut buf = ZeroAllocBuffer::new();
        assert_eq!(buf.as_str(), "");
        assert!(!buf.is_overflow());

        buf.push_str("symbol=BTCUSDT&price=");
        buf.push_f64(50000.5);
        buf.push_str("&timestamp=");
        buf.push_u64(1700000000000);

        assert!(!buf.is_overflow());
        let s = buf.as_str();
        assert!(s.starts_with("symbol=BTCUSDT&price=50000.5&timestamp=1700000000000"));

        buf.clear();
        assert_eq!(buf.as_str(), "");
        assert!(!buf.is_overflow());
    }

    #[test]
    fn test_zero_alloc_buffer_overflow_detection() {
        let mut buf = ZeroAllocBuffer::new();
        let large_str = "A".repeat(1025);
        buf.push_str(&large_str);
        assert!(buf.is_overflow());
    }

    #[test]
    fn test_binance_rate_limits_and_retry_after_extraction() {
        let mut headers = HeaderMap::new();
        headers.insert("X-MBX-USED-WEIGHT-1M", HeaderValue::from_static("350"));
        headers.insert("X-MBX-ORDER-COUNT-10S", HeaderValue::from_static("25"));
        headers.insert("X-MBX-ORDER-COUNT-1M", HeaderValue::from_static("120"));
        headers.insert("Retry-After", HeaderValue::from_static("45"));

        let limits = extract_limits(&headers);
        assert_eq!(limits.weight_1m, Some(350));
        assert_eq!(limits.orders_10s, Some(25));
        assert_eq!(limits.orders_1m, Some(120));

        let retry_after = extract_retry_after(&headers);
        assert_eq!(retry_after, 45);

        let empty_headers = HeaderMap::new();
        let default_retry = extract_retry_after(&empty_headers);
        assert_eq!(default_retry, 60);
    }

    #[test]
    fn test_binance_client_hot_swap_api_key_and_testnet() {
        let client = BinanceClient::new("key1".to_string(), true);
        assert_eq!(client.get_base_url(), BINANCE_TESTNET_URL);

        client.hot_swap_credentials("key2".to_string(), false);
        assert_eq!(client.get_base_url(), BINANCE_BASE_URL);

        let current_key = client.api_key.load();
        assert_eq!(current_key.to_str().unwrap(), "key2");
    }
}
