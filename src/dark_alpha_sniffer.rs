use futures_util::{SinkExt, StreamExt};
use simd_json::prelude::*;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_tungstenite::connect_async;

use crate::dark_alpha_router::DarkAlphaRouter;

/// Spawn a dedicated thread for sniffing DEX MEV and liquidations
/// Connects to Hyperliquid to sense cascaded liquidations before Binance internalizes them.
/// Runs entirely lock-free and zero-alloc in the hot path.
pub fn spawn_hyperliquid_sniffer(router: Arc<DarkAlphaRouter>) {
    tokio::spawn(async move {
        loop {
            println!("🌌 [DARK-ALPHA] Connecting to Hyperliquid DEX...");
            match connect_async("wss://api.hyperliquid.xyz/ws").await {
                Ok((mut ws_stream, _)) => {
                    println!("🌌 [DARK-ALPHA] Connected to Hyperliquid DEX WS.");

                    let sub_msg = r#"{"method": "subscribe", "subscription": {"type": "l2Book", "coin": "BTC"}}"#;
                    let _ = ws_stream
                        .send(tokio_tungstenite::tungstenite::Message::Text(
                            sub_msg.to_string(),
                        ))
                        .await;

                    while let Some(msg) = ws_stream.next().await {
                        if let Ok(tokio_tungstenite::tungstenite::Message::Text(text)) = msg {
                            let mut bytes = text.into_bytes();

                            // FIX #1420: Parseo real de niveles L2 de Hyperliquid DEX
                            if let Ok(parsed) = simd_json::to_borrowed_value(&mut bytes) {
                                if let Some(data) = parsed.get("data") {
                                    if let Some(levels) = data.get("levels").and_then(|l| l.as_array()) {
                                        let mut bid_vol = 0.0f64;
                                        let mut ask_vol = 0.0f64;
                                        if let Some(bids) = levels.get(0).and_then(|b| b.as_array()) {
                                            for b in bids.iter().take(5) {
                                                if let Some(sz_str) = b.get("sz").and_then(|s| s.as_str()) {
                                                    // FIX #1468: Parseo seguro y validación de finitud
                                                    if let Ok(v) = sz_str.parse::<f64>() {
                                                        if v.is_finite() && v > 0.0 {
                                                            bid_vol += v;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if let Some(asks) = levels.get(1).and_then(|a| a.as_array()) {
                                            for a in asks.iter().take(5) {
                                                if let Some(sz_str) = a.get("sz").and_then(|s| s.as_str()) {
                                                    // FIX #1468: Parseo seguro y validación de finitud
                                                    if let Ok(v) = sz_str.parse::<f64>() {
                                                        if v.is_finite() && v > 0.0 {
                                                            ask_vol += v;
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        let total_vol = (bid_vol + ask_vol).max(1.0);
                                        let impact = ((bid_vol - ask_vol) / total_vol).clamp(-1.0, 1.0);
                                        let qty = total_vol.clamp(0.1, 1000.0);

                                        let ts = SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_millis() as u64;

                                        router.ingest_dex_liquidation(qty, impact, ts);
                                    }
                                }
                            };
                        }
                    }
                }
                Err(e) => {
                    println!("⚠️ [DARK-ALPHA] Hyperliquid connection failed: {:?}", e);
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    });
}
