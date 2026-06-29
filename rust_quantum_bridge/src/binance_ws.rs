use pyo3::prelude::*;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{StreamExt, SinkExt};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

// Shared memory for prices
lazy_static::lazy_static! {
    pub static ref ARENA_PRICES: Arc<RwLock<HashMap<String, f64>>> = Arc::new(RwLock::new(HashMap::new()));
}

#[pyfunction]
pub fn start_binance_websocket(symbols: Vec<String>) -> PyResult<()> {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let streams: Vec<String> = symbols.iter()
                .map(|s| format!("{}@depth5@100ms", s.replace("/", "").to_lowercase()))
                .collect();
            
            let url = format!("wss://stream.binance.com:9443/stream?streams={}", streams.join("/"));
            
            println!("🚀 [RUST] Connecting to Binance Quantum Stream: {}", url);
            
            match connect_async(url).await {
                Ok((ws_stream, _)) => {
                    println!("🚀 [RUST] Connected! Zero-Copy parsing active.");
                    let (mut _write, mut read) = ws_stream.split();
                    
                    while let Some(msg) = read.next().await {
                        if let Ok(Message::Text(text)) = msg {
                            // Parse JSON as fast as possible
                            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                                if let Some(data) = json.get("data") {
                                    if let (Some(bids), Some(asks)) = (data.get("bids"), data.get("asks")) {
                                        if let (Some(bid1), Some(ask1)) = (bids.get(0), asks.get(0)) {
                                            if let (Some(bp), Some(ap)) = (bid1.get(0), ask1.get(0)) {
                                                if let (Some(b_str), Some(a_str)) = (bp.as_str(), ap.as_str()) {
                                                    if let (Ok(b_f), Ok(a_f)) = (b_str.parse::<f64>(), a_str.parse::<f64>()) {
                                                        let mid_price = (b_f + a_f) / 2.0;
                                                        if let Some(stream) = json.get("stream").and_then(|s| s.as_str()) {
                                                            let symbol = stream.split('@').next().unwrap_or("").to_uppercase();
                                                            let mut arena = ARENA_PRICES.write();
                                                            arena.insert(symbol, mid_price);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("❌ [RUST] Failed to connect: {}", e);
                }
            }
        });
    });
    
    Ok(())
}

#[pyfunction]
pub fn get_arena_price(symbol: String) -> PyResult<f64> {
    let arena = ARENA_PRICES.read();
    Ok(*arena.get(&symbol).unwrap_or(&0.0))
}
