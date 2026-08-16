use futures_util::StreamExt;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_tungstenite::connect_async;
use url::Url;

use crate::client::BinanceClient;

pub struct UserDataStreamer {
    pub arena: Arc<quantum_arena::GlobalArena>,
    pub client: BinanceClient,
}

impl UserDataStreamer {
    pub fn new(arena: Arc<quantum_arena::GlobalArena>, api_key: String, _api_secret: String, is_testnet: bool) -> Self {
        let client = BinanceClient::new(api_key, is_testnet);
        Self { arena, client }
    }

    pub async fn start(&self, is_testnet: bool) {
        let base_ws_url = if is_testnet {
            "wss://stream.binancefuture.com/ws/"
        } else {
            "wss://fstream.binance.com/ws/"
        };

        let mut backoff_ms = 100;

        loop {
            // FASE XVIII: Keep-alive y recreación dinámica de ListenKey
            let listen_key = match self.client.create_listen_key().await {
                Ok(key) => key,
                Err(e) => {
                    println!("❌ [UserDataWS] Error al obtener ListenKey: {}. Reintentando en {}ms...", e, backoff_ms);
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(10000);
                    continue;
                }
            };
            
            println!("✅ [UserDataWS] ListenKey obtenido: {}", listen_key);

            let stream_url = format!("{}{}", base_ws_url, listen_key);
            let url = Url::parse(&stream_url).expect("Bad UserData WS URL");

            match connect_async(url.as_str()).await {
                Ok((ws_stream, _)) => {
                    let connected_at = std::time::Instant::now();
                    println!("🚀 [UserDataWS] Conectado a Binance User Data Stream");
                    
                    let (_, mut read) = ws_stream.split();
                    
                    // Task para enviar el KeepAlive cada 30 minutos
                    let keep_alive_client = self.client.clone();
                    let ping_task = tokio::spawn(async move {
                        loop {
                            tokio::time::sleep(tokio::time::Duration::from_secs(30 * 60)).await;
                            let _ = keep_alive_client.keep_alive_listen_key().await;
                        }
                    });

                    loop {
                        let timeout_res = tokio::time::timeout(
                            std::time::Duration::from_secs(300), // Binance manda pings regularmente
                            read.next()
                        ).await;

                        let msg = match timeout_res {
                            Ok(Some(m)) => m,
                            Ok(None) => break, // Stream cerrado
                            Err(_) => {
                                // Timeout de 300s sin datos ni ping = reconexión forzada
                                break;
                            }
                        };

                        if connected_at.elapsed().as_secs() > 10 {
                            backoff_ms = 100;
                        }

                        match msg {
                            Ok(msg) => {
                                let bytes = msg.into_data();
                                // Procesamiento a velocidad pico-segundo
                                if let Some(event_type_idx) = memchr::memmem::find(&bytes, b"\"e\":\"") {
                                    if bytes.len() > event_type_idx + 18 { // Seguro
                                        // Detectamos ACCOUNT_UPDATE o ORDER_TRADE_UPDATE
                                        if memchr::memmem::find(&bytes, b"ACCOUNT_UPDATE").is_some() {
                                            self.parse_account_update(&bytes);
                                        } else if memchr::memmem::find(&bytes, b"ORDER_TRADE_UPDATE").is_some() {
                                            self.parse_order_trade_update(&bytes);
                                        }
                                    }
                                }
                            }
                            Err(_) => break, // WS Error
                        }
                    }
                    
                    ping_task.abort();
                }
                Err(e) => {
                    println!("❌ [UserDataWS] Error de conexión WS: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(5000);
                }
            }
        }
    }

    #[inline(always)]
    fn parse_account_update(&self, json: &[u8]) {
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(json) {
            let account = &val["a"];
            
            // Extract balances
            if let Some(balances) = account["B"].as_array() {
                for b in balances {
                    if b["a"].as_str().unwrap_or("") == "USDT" {
                        let wb = b["wb"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                        // Zero latency update al motor
                        self.arena.unified_capital.store(wb, Ordering::Relaxed);
                        // Peak HW tracking update en picosegundos
                        let peak = self.arena.peak_unified_capital.load(Ordering::Relaxed);
                        if wb > peak {
                            self.arena.peak_unified_capital.store(wb, Ordering::Relaxed);
                        }
                    }
                }
            }
            
            // Extract positions
            let mut total_used_margin = 0.0;
            if let Some(positions) = account["P"].as_array() {
                for p in positions {
                    let _symbol = p["s"].as_str().unwrap_or("");
                    let pos_amount = p["pa"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                    let isolated_wallet = p["iw"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                    let _unrealized_pnl = p["up"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                    let entry_price = p["ep"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                    
                    if pos_amount.abs() > 0.0 {
                        // Approximate used margin for this position (Assuming Cross margin, so iw might be 0)
                        // This updates the global used margin state dynamically.
                        total_used_margin += (pos_amount.abs() * entry_price) * 0.01; // Rough 1% margin assumption if not isolated, can be refined.
                        total_used_margin += isolated_wallet; // If isolated, it takes this.
                    }
                }
                self.arena.used_margin.store(total_used_margin, Ordering::Relaxed);
                
                let capital = self.arena.unified_capital.load(Ordering::Relaxed);
                self.arena.available_margin.store((capital - total_used_margin).max(0.0), Ordering::Relaxed);
            }
        }
    }

    #[inline(always)]
    fn parse_order_trade_update(&self, _json: &[u8]) {
        // En un futuro cercano, tracking de fill quantities 
        // directo en la arena si el motor requiere micro-estado
    }
}
