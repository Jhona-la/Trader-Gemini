use reqwest::{Client, header};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

pub struct BinanceRestExecutor {
    api_key: String,
    secret_key: String,
    base_url: String,
    client: Client,
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
            .build()
            .unwrap();

        Self {
            api_key,
            secret_key,
            base_url,
            client,
        }
    }

    fn generate_signature(&self, query_string: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query_string.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    fn get_timestamp() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    }

    pub async fn create_order(
        &self,
        symbol: &str,
        side: &str,
        order_type: &str,
        quantity: f64,
        price: Option<f64>,
    ) -> Result<Value, reqwest::Error> {
        let timestamp = Self::get_timestamp();
        
        let mut query = format!(
            "symbol={}&side={}&type={}&quantity={}&timestamp={}",
            symbol, side, order_type, quantity, timestamp
        );

        if let Some(p) = price {
            query.push_str(&format!("&price={}&timeInForce=GTC", p));
        }

        let signature = self.generate_signature(&query);
        let url = format!("{}/api/v3/order?{}&signature={}", self.base_url, query, signature);

        let res = self.client.post(&url).send().await?;
        let json = res.json::<Value>().await?;
        Ok(json)
    }
}

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use std::collections::HashMap;
use serde_json::json;
use tokio::sync::mpsc;
use std::sync::atomic::{AtomicUsize, Ordering};

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

        let (tx_cmd, mut rx_cmd) = mpsc::channel::<String>(100);
        let url = ws_url.to_string();
        
        tokio::spawn(async move {
            loop {
                if let Ok((ws_stream, _)) = connect_async(&url).await {
                    let (mut write, mut read) = ws_stream.split();
                    
                    loop {
                        tokio::select! {
                            cmd = rx_cmd.recv() => {
                                if let Some(msg_str) = cmd {
                                    let _ = write.send(Message::Text(msg_str)).await;
                                } else {
                                    break;
                                }
                            }
                            msg = read.next() => {
                                if msg.is_none() {
                                    break; // Disconnected
                                }
                                // In a real HFT engine, we parse ACK here to measure latency
                            }
                        }
                    }
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
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

    pub async fn place_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        position_side: &str,
        req_id: &str,
    ) {
        let timestamp = BinanceRestExecutor::get_timestamp();
        
        // Construct query string for signature (alphabetical order)
        let query_string = format!(
            "apiKey={}&positionSide={}&quantity={}&side={}&symbol={}&timestamp={}&type=MARKET",
            self.api_key, position_side, quantity, side, symbol, timestamp
        );

        let signature = self.generate_signature(&query_string);

        let payload = json!({
            "id": req_id,
            "method": "order.place",
            "params": {
                "apiKey": self.api_key,
                "positionSide": position_side,
                "quantity": quantity.to_string(),
                "side": side,
                "symbol": symbol,
                "timestamp": timestamp,
                "type": "MARKET",
                "signature": signature
            }
        });

        let _ = self.tx_cmd.send(payload.to_string()).await;
    }
}

pub struct BinanceUserDataStream;

impl BinanceUserDataStream {
    pub async fn start(api_key: String, is_testnet: bool) {
        let base_url = if is_testnet {
            "https://testnet.binancefuture.com" // FAPI testnet
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
            .build()
            .unwrap();
            
        // POST to get listenKey
        let url = format!("{}/fapi/v1/listenKey", base_url);
        let res = client.post(&url).send().await;
        
        if let Ok(response) = res {
            if let Ok(json) = response.json::<Value>().await {
                if let Some(listen_key) = json["listenKey"].as_str() {
                    let lk = listen_key.to_string();
                    let ws_url = format!("{}/{}", ws_base_url, lk);
                    
                    // Spawn Keep-Alive Task
                    let lk_clone = lk.clone();
                    let url_clone = url.clone();
                    let client_clone = client.clone();
                    tokio::spawn(async move {
                        loop {
                            tokio::time::sleep(tokio::time::Duration::from_secs(30 * 60)).await;
                            let _ = client_clone.put(&url_clone).send().await;
                        }
                    });
                    
                    // Spawn WS Connection
                    tokio::spawn(async move {
                        println!("🔗 [USER DATA STREAM] Connecting to Binance Execution Stream...");
                        loop {
                            if let Ok((ws_stream, _)) = connect_async(&ws_url).await {
                                let (_, mut read) = ws_stream.split();
                                while let Some(msg_result) = read.next().await {
                                    if let Ok(Message::Text(text)) = msg_result {
                                        if let Ok(data) = serde_json::from_str::<Value>(&text) {
                                            if data["e"] == "ORDER_TRADE_UPDATE" {
                                                let order = &data["o"];
                                                if order["X"] == "FILLED" {
                                                    let rp_str = order["rp"].as_str().unwrap_or("0");
                                                    let realized_pnl: f64 = rp_str.parse().unwrap_or(0.0);
                                                    let symbol = order["s"].as_str().unwrap_or("UNKNOWN");
                                                    
                                                    if realized_pnl != 0.0 {
                                                        let is_win = realized_pnl > 0.0;
                                                        // Approximated 1% default PnL percentage for QuarterKelly math feeding
                                                        let pnl_pct = 0.01; 
                                                        crate::risk::RiskManager::report_trade_result(symbol, is_win, pnl_pct);
                                                        println!("💰 [REAL PNL] Trade Closed! Symbol: {}, PnL: {:.4} USD (Win: {}). Kelly updated.", symbol, realized_pnl, is_win);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                        }
                    });
                } else {
                    println!("❌ [USER DATA STREAM] Failed to get listenKey. Check API key permissions.");
                }
            }
        }
    }
}
