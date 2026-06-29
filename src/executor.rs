use reqwest::{Client, header};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use simd_json::prelude::ValueObjectAccess;
use simd_json::prelude::ValueAsScalar;

type HmacSha256 = Hmac<Sha256>;

pub struct CircuitBreaker {
    failures: AtomicUsize,
    last_failure_ts: AtomicU64,
    threshold: usize,
    reset_timeout_ms: u64,
}

impl CircuitBreaker {
    pub fn new(threshold: usize, reset_timeout_ms: u64) -> Self {
        Self {
            failures: AtomicUsize::new(0),
            last_failure_ts: AtomicU64::new(0),
            threshold,
            reset_timeout_ms,
        }
    }

    pub fn check(&self) -> bool {
        let fails = self.failures.load(Ordering::Acquire);
        if fails >= self.threshold {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
            let last = self.last_failure_ts.load(Ordering::Acquire);
            if now - last > self.reset_timeout_ms {
                self.failures.store(self.threshold - 1, Ordering::Release);
                return true;
            }
            return false;
        }
        true
    }

    pub fn record_failure(&self) {
        let fails = self.failures.fetch_add(1, Ordering::SeqCst);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        self.last_failure_ts.store(now, Ordering::SeqCst);
        if fails + 1 == self.threshold {
            println!("🛑 [CIRCUIT BREAKER] TRIPPED! API requests paused for {}ms", self.reset_timeout_ms);
        }
    }

    pub fn record_success(&self) {
        let current = self.failures.load(Ordering::Acquire);
        if current > 0 {
            self.failures.store(0, Ordering::Release);
            println!("✅ [CIRCUIT BREAKER] RECOVERED. Normal operation resumed.");
        }
    }
}

pub struct BinanceRestExecutor {
    _api_key: String,
    secret_key: String,
    base_url: String,
    client: Client,
    circuit_breaker: CircuitBreaker,
}

impl BinanceRestExecutor {
    pub fn new(api_key: String, secret_key: String, is_testnet: bool) -> Self {
        let base_url = if is_testnet {
            "https://testnet.binancefuture.com".to_string()
        } else {
            "https://fapi.binance.com".to_string()
        };

        let mut headers = header::HeaderMap::new();
        headers.insert("X-MBX-APIKEY", header::HeaderValue::from_str(&api_key).unwrap());

        let client = Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_millis(5000))
            .build()
            .unwrap();

        Self {
            _api_key: api_key,
            secret_key,
            base_url,
            client,
            circuit_breaker: CircuitBreaker::new(5, 10000), // 5 failures trip it for 10s
        }
    }

    fn generate_signature(&self, query_string: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query_string.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    pub fn get_timestamp() -> u128 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()
    }

    pub async fn create_order(
        &self,
        symbol: &str,
        side: &str,
        order_type: &str,
        quantity: f64,
        price: Option<f64>,
    ) -> Result<Value, String> {
        if !self.circuit_breaker.check() {
            return Err("Circuit Breaker Open: Request Rejected".to_string());
        }

        let timestamp = Self::get_timestamp();
        let mut query = format!(
            "symbol={}&side={}&type={}&quantity={}&timestamp={}",
            symbol, side, order_type, quantity, timestamp
        );

        if let Some(p) = price {
            query.push_str(&format!("&price={}&timeInForce=GTC", p));
        }

        let signature = self.generate_signature(&query);
        let url = format!("{}/fapi/v1/order?{}&signature={}", self.base_url, query, signature);

        match self.client.post(&url).send().await {
            Ok(res) => {
                let status = res.status();
                if status.is_success() {
                    self.circuit_breaker.record_success();
                    Ok(res.json::<Value>().await.unwrap_or(Value::Null))
                } else if status.as_u16() == 429 {
                    self.circuit_breaker.record_failure();
                    Err("Rate Limit 429".to_string())
                } else {
                    self.circuit_breaker.record_failure();
                    Err(format!("HTTP Error: {}", status))
                }
            },
            Err(e) => {
                self.circuit_breaker.record_failure();
                Err(e.to_string())
            }
        }
    }

    pub async fn enable_hedge_mode(&self) -> Result<Value, String> {
        let timestamp = Self::get_timestamp();
        let query = format!("dualSidePosition=true&timestamp={}", timestamp);
        let signature = self.generate_signature(&query);
        let url = format!("{}/fapi/v1/positionSide/dual?{}&signature={}", self.base_url, query, signature);

        match self.client.post(&url).send().await {
            Ok(res) => {
                let status = res.status();
                if status.is_success() || status.as_u16() == 400 {
                    // 400 often means it's already in hedge mode
                    self.circuit_breaker.record_success();
                    Ok(res.json::<Value>().await.unwrap_or(Value::Null))
                } else {
                    self.circuit_breaker.record_failure();
                    Err(format!("HTTP Error: {}", status))
                }
            },
            Err(e) => {
                self.circuit_breaker.record_failure();
                Err(e.to_string())
            }
        }
    }
    pub async fn fetch_open_positions(&self) -> Result<Vec<Value>, String> {
        let timestamp = Self::get_timestamp();
        let query = format!("timestamp={}", timestamp);
        let signature = self.generate_signature(&query);
        let url = format!("{}/fapi/v2/positionRisk?{}&signature={}", self.base_url, query, signature);

        match self.client.get(&url).send().await {
            Ok(res) => {
                let status = res.status();
                if status.is_success() {
                    self.circuit_breaker.record_success();
                    Ok(res.json::<Vec<Value>>().await.unwrap_or_else(|_| vec![]))
                } else {
                    self.circuit_breaker.record_failure();
                    Err(format!("HTTP Error: {}", status))
                }
            },
            Err(e) => {
                self.circuit_breaker.record_failure();
                Err(e.to_string())
            }
        }
    }
}

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tokio::sync::mpsc;

pub struct BinanceWSFuturesExecutor {
    api_key: String,
    secret_key: String,
    tx_cmd: mpsc::Sender<String>,
}

impl BinanceWSFuturesExecutor {
    pub async fn new(api_key: String, secret_key: String, is_testnet: bool) -> Self {
        let ws_url = if is_testnet {
            "wss://testnet.binancefuture.com/ws-fapi/v1"
        } else {
            "wss://ws-fapi.binance.com/ws-fapi/v1"
        };

        // Backpressure safe lock-free channel (1000 orders max backlog)
        let (tx_cmd, mut rx_cmd) = mpsc::channel::<String>(1000); 
        let url = ws_url.to_string();
        
        tokio::spawn(async move {
            let mut retry_delay = 1;
            loop {
                if let Ok((ws_stream, _)) = connect_async(&url).await {
                    retry_delay = 1; 
                    let (mut write, mut read) = ws_stream.split();
                    
                    loop {
                        tokio::select! {
                            cmd = rx_cmd.recv() => {
                                if let Some(msg_str) = cmd {
                                    let _ = write.send(Message::Text(msg_str)).await;
                                } else {
                                    return;
                                }
                            }
                            msg = read.next() => {
                                if msg.is_none() {
                                    break; 
                                }
                                // Minimal parse for ACKs to track zero-copy latency internally here if needed
                            }
                        }
                    }
                }
                println!("⚠️ [WS EXECUTOR] Disconnected. Reconnecting in {}s...", retry_delay);
                tokio::time::sleep(tokio::time::Duration::from_secs(retry_delay)).await;
                retry_delay = std::cmp::min(retry_delay * 2, 60);
            }
        });

        Self {
            api_key,
            secret_key,
            tx_cmd,
        }
    }

    fn generate_signature(&self, query_string: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query_string.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    pub fn place_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        position_side: &str,
        req_id: &str,
    ) {
        let timestamp = BinanceRestExecutor::get_timestamp();
        
        let mut query = String::with_capacity(256);
        query.push_str("apiKey=");
        query.push_str(&self.api_key);
        query.push_str("&positionSide=");
        query.push_str(position_side);
        query.push_str("&quantity=");
        
        let mut ryu_buf = ryu::Buffer::new();
        let qty_str = ryu_buf.format(quantity);
        query.push_str(qty_str);
        
        query.push_str("&side=");
        query.push_str(side);
        query.push_str("&symbol=");
        query.push_str(symbol);
        query.push_str("&timestamp=");
        
        let mut itoa_buf = itoa::Buffer::new();
        let ts_str = itoa_buf.format(timestamp);
        query.push_str(ts_str);
        
        query.push_str("&type=MARKET");

        let signature = self.generate_signature(&query);

        // Zero-Alloc-Style payload formulation (avoids serde_json DOM and format! trait overhead)
        let mut payload = String::with_capacity(512);
        payload.push_str(r#"{"id":""#);
        payload.push_str(req_id);
        payload.push_str(r#"","method":"order.place","params":{"apiKey":""#);
        payload.push_str(&self.api_key);
        payload.push_str(r#"","positionSide":""#);
        payload.push_str(position_side);
        payload.push_str(r#"","quantity":""#);
        payload.push_str(qty_str);
        payload.push_str(r#"","side":""#);
        payload.push_str(side);
        payload.push_str(r#"","symbol":""#);
        payload.push_str(symbol);
        payload.push_str(r#"","timestamp":"#);
        payload.push_str(ts_str);
        payload.push_str(r#","type":"MARKET","signature":""#);
        payload.push_str(&signature);
        payload.push_str(r#""}}"#);

        // Non-blocking fire-and-forget for nanosecond latency in God Engine
        let _ = self.tx_cmd.try_send(payload);
    }
}

pub struct BinanceUserDataStream;

impl BinanceUserDataStream {
    pub async fn start(api_key: String, is_testnet: bool) {
        let base_url = if is_testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        };
        let ws_base_url = if is_testnet {
            "wss://stream.binancefuture.com/ws"
        } else {
            "wss://fstream.binance.com/ws"
        };

        let mut headers = header::HeaderMap::new();
        headers.insert("X-MBX-APIKEY", header::HeaderValue::from_str(&api_key).unwrap());

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_millis(5000))
            .build()
            .unwrap();
            
        let url = format!("{}/fapi/v1/listenKey", base_url);
        
        loop {
            let res = client.post(&url).send().await;
            if let Ok(response) = res {
                if let Ok(json) = response.json::<Value>().await {
                    if let Some(listen_key) = json["listenKey"].as_str() {
                        let lk = listen_key.to_string();
                        let ws_url = format!("{}/{}", ws_base_url, lk);
                        
                        let url_clone = url.clone();
                        let client_clone = client.clone();
                        let keep_alive_handle = tokio::spawn(async move {
                            loop {
                                tokio::time::sleep(tokio::time::Duration::from_secs(30 * 60)).await;
                                let res = client_clone.put(&url_clone).send().await;
                                let success = res.map(|r| r.status().is_success()).unwrap_or(false);
                                if !success {
                                    break; // Force reconnect
                                }
                            }
                        });
                        
                        println!("🔗 [USER DATA STREAM] Connecting to Binance Execution Stream...");
                        let mut retry_delay = 1;
                        loop {
                            if let Ok((ws_stream, _)) = connect_async(&ws_url).await {
                                retry_delay = 1;
                                let (_, mut read) = ws_stream.split();
                                while let Some(msg_result) = read.next().await {
                                    if let Ok(Message::Text(text)) = msg_result {
                                        let mut bytes = text.into_bytes();
                                        if let Ok(data) = simd_json::to_borrowed_value(&mut bytes) {
                                            if data.get("e").and_then(|v| v.as_str()) == Some("ORDER_TRADE_UPDATE") {
                                                if let Some(order) = data.get("o") {
                                                    if order.get("X").and_then(|v| v.as_str()) == Some("FILLED") {
                                                        let rp_str = order.get("rp").and_then(|v| v.as_str()).unwrap_or("0");
                                                        let realized_pnl: f64 = rp_str.parse().unwrap_or(0.0);
                                                        let symbol = order.get("s").and_then(|v| v.as_str()).unwrap_or("UNKNOWN");
                                                    
                                                        if realized_pnl != 0.0 {
                                                            let is_win = realized_pnl > 0.0;
                                                            let qty: f64 = order.get("q").and_then(|v| v.as_str()).unwrap_or("0").parse().unwrap_or(0.0);
                                                            let price: f64 = order.get("p").and_then(|v| v.as_str()).unwrap_or("0").parse().unwrap_or(0.0);
                                                            let pnl_pct = if qty > 0.0 && price > 0.0 {
                                                                (realized_pnl / (qty * price)).abs()
                                                            } else {
                                                                if is_win { 0.01 } else { -0.01 }
                                                            };
                                                            crate::risk::RiskManager::report_trade_result(symbol, is_win, pnl_pct);
                                                            println!("💰 [REAL PNL] Trade Closed! Symbol: {}, PnL: {:.4} USD (Win: {}).", symbol, realized_pnl, is_win);
                                                        }
                                                    }
                                                }
                                            }
                                        }; // Semi-colon here!
                                    }
                                }
                            }
                            println!("⚠️ [USER DATA STREAM] Disconnected. Reconnecting in {}s...", retry_delay);
                            tokio::time::sleep(tokio::time::Duration::from_secs(retry_delay)).await;
                            retry_delay = std::cmp::min(retry_delay * 2, 60);
                            if retry_delay >= 60 {
                                break; // Break out to re-fetch listenKey
                            }
                        }
                        keep_alive_handle.abort();
                    } else {
                        println!("❌ [USER DATA STREAM] Failed to get listenKey. Check API key permissions.");
                        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    }
                }
            } else {
                println!("❌ [USER DATA STREAM] HTTP Error fetching listenKey. Retrying in 5s.");
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    }
}
