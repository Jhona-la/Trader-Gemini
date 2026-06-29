use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::task;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{StreamExt, SinkExt};
use serde_json::Value;
use std::sync::Arc;

pub struct BinanceWsClient {
    symbols: Vec<String>,
    sender: mpsc::Sender<Value>,
}

impl BinanceWsClient {
    pub fn new(symbols: Vec<String>, sender: mpsc::Sender<Value>) -> Self {
        Self {
            symbols,
            sender,
        }
    }

    pub async fn start(&self) {
        if self.symbols.is_empty() {
            return;
        }

        // Build the stream URL for multi-stream
        // E.g. wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m
        let mut streams = Vec::new();
        for sym in &self.symbols {
            let lower = sym.to_lowercase().replace("/", "");
            streams.push(format!("{}@kline_1m", lower));
            streams.push(format!("{}@kline_5m", lower));
            streams.push(format!("{}@kline_15m", lower));
            streams.push(format!("{}@kline_1h", lower));
            streams.push(format!("{}@bookTicker", lower)); // BBO (Best Bid/Offer) for OrderBook
        }
        let streams_str = streams.join("/");
        let url = format!("wss://fstream.binance.com/stream?streams={}", streams_str);

        let url = url::Url::parse(&url).unwrap();
        
        match connect_async(url).await {
            Ok((ws_stream, _)) => {
                let (_, mut read) = ws_stream.split();
                
                let tx = self.sender.clone();
                task::spawn(async move {
                    while let Some(msg) = read.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                if let Ok(parsed) = serde_json::from_str::<Value>(&text) {
                                    let _ = tx.send(parsed).await;
                                }
                            }
                            Ok(Message::Ping(_)) => {
                                // Tungstenite handles pong automatically
                            }
                            _ => {}
                        }
                    }
                });
            }
            Err(e) => {
                println!("Failed to connect to Binance WS: {}", e);
            }
        }
    }
}
