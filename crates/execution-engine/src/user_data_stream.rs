//! USER-DATA STREAM (F1.6) — WebSocket privado de Binance: fills en tiempo real.
//!
//! QUÉ: conexión a wss://fstream.binance.com/ws/<listenKey> con keepalive (30 min)
//!      y reconexión con backoff exponencial.
//! POR QUÉ: ORDER_TRADE_UPDATE entrega cada fill en el instante (sin polling REST);
//!          ACCOUNT_UPDATE entrega balances/posiciones — la verdad del exchange.
//! CÓMO: los eventos convergen en OrderRegistry (F1.5) y en un AccountSink
//!      implementado por el host (god_engine) para actualizar capital del arena.
//!      Este crate NO depende de quantum-arena: separación ejecución/motor
//!      (el archivo original era un fantasma que jamás compiló por esa dependencia).

use crate::client::BinanceClient;
use crate::order_registry::{OrderRegistry, OrderStatus, TradeUpdate};
use futures_util::StreamExt;
use serde::Deserialize;
use std::sync::Arc;
use tokio_tungstenite::connect_async;
use url::Url;

/// Posición reportada por el exchange (ACCOUNT_UPDATE / positionRisk).
#[derive(Debug, Clone, Default)]
pub struct RemotePosition {
    pub symbol: String,
    /// Con signo: positivo long, negativo short.
    pub position_amt: f64,
    pub entry_price: f64,
    pub unrealized_pnl: f64,
    pub isolated_wallet: f64,
}

/// Interfaz del host para eventos de cuenta. El motor implementa esto para
/// actualizar su estado (capital, posiciones) con la verdad del exchange.
pub trait AccountSink: Send + Sync {
    fn on_capital(&self, usdt_wallet_balance: f64);
    fn on_positions(&self, positions: &[RemotePosition]);
}

struct NoopSink;
impl AccountSink for NoopSink {
    fn on_capital(&self, _usdt: f64) {}
    fn on_positions(&self, _p: &[RemotePosition]) {}
}

/// F1.6: stream privado. `start()` es infinito (diseñado para tokio::spawn).
pub struct UserDataStreamer {
    client: BinanceClient,
    registry: Arc<OrderRegistry>,
    sink: Arc<dyn AccountSink>,
}

impl UserDataStreamer {
    pub fn new(client: BinanceClient, registry: Arc<OrderRegistry>) -> Self {
        Self {
            client,
            registry,
            sink: Arc::new(NoopSink),
        }
    }

    pub fn with_sink(mut self, sink: Arc<dyn AccountSink>) -> Self {
        self.sink = sink;
        self
    }

    pub async fn start(&self) {
        let base_ws_url = if self
            .client
            .is_testnet
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            "wss://stream.binancefuture.com/ws/"
        } else {
            "wss://fstream.binance.com/ws/"
        };
        let mut backoff_ms: u64 = 100;

        loop {
            // listenKey + keepalive periódico (expira a los 60 min en Binance).
            let listen_key = match self.client.create_listen_key().await {
                Ok(key) => key,
                Err(e) => {
                    println!(
                        "❌ [UserDataWS] ListenKey error: {}. Reintento en {}ms",
                        e, backoff_ms
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(10_000);
                    continue;
                }
            };
            println!("✅ [UserDataWS] ListenKey activo (stream privado listo)");

            let stream_url = format!("{}{}", base_ws_url, listen_key);
            let Ok(url) = Url::parse(&stream_url) else {
                println!("❌ [UserDataWS] URL inválida: {}", stream_url);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            };

            match connect_async(url.as_str()).await {
                Ok((ws_stream, _)) => {
                    println!("🚀 [UserDataWS] Conectado — fills en tiempo real");
                    backoff_ms = 100;
                    let (_, mut read) = ws_stream.split();

                    // Keepalive cada 30 min mientras viva esta conexión.
                    let keep_client = self.client.clone();
                    let ping_task = tokio::spawn(async move {
                        loop {
                            tokio::time::sleep(tokio::time::Duration::from_secs(30 * 60)).await;
                            let _ = keep_client.keep_alive_listen_key().await;
                        }
                    });

                    // Watchdog: Binance hace ping < 3 min; 5 min sin nada = reconectar.
                    loop {
                        let next =
                            tokio::time::timeout(std::time::Duration::from_secs(300), read.next())
                                .await;
                        let msg = match next {
                            Ok(Some(Ok(m))) => m,
                            Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
                        };
                        let bytes = msg.into_data();
                        if !bytes.is_empty() {
                            self.dispatch(&bytes);
                        }
                    }
                    ping_task.abort();
                    println!("⚠️ [UserDataWS] Conexión perdida. Reconectando...");
                }
                Err(e) => {
                    println!("❌ [UserDataWS] Error de conexión: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(5_000);
                }
            }
        }
    }

    fn dispatch(&self, bytes: &[u8]) {
        // Ruteo barato por tipo de evento antes del parse completo.
        let text = std::str::from_utf8(bytes).unwrap_or("");
        if text.contains("\"ORDER_TRADE_UPDATE\"") {
            self.on_order_trade_update(text);
        } else if text.contains("\"ACCOUNT_UPDATE\"") {
            self.on_account_update(text);
        }
    }

    fn on_order_trade_update(&self, text: &str) {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(rename = "c")]
            client_order_id: String,
            #[serde(rename = "s")]
            symbol: String,
            #[serde(rename = "S")]
            side: String,
            #[serde(rename = "o")]
            order_type: String,
            #[serde(rename = "X")]
            order_status: String,
            #[serde(rename = "i")]
            order_id: u64,
            #[serde(rename = "q")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            orig_qty: f64,
            #[serde(rename = "z")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            cumulative_filled_qty: f64,
            #[serde(rename = "l")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            last_filled_qty: f64,
            #[serde(rename = "L")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            last_filled_price: f64,
            #[serde(rename = "ap")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            avg_price: f64,
            #[serde(rename = "n")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            commission: f64,
            #[serde(rename = "N", default)]
            commission_asset: String,
            #[serde(rename = "T", default)]
            trade_time_ms: u64,
        }
        #[derive(Deserialize)]
        struct Event {
            o: Raw,
        }
        let Ok(ev) = serde_json::from_str::<Event>(text) else {
            println!(
                "⚠️ [UserDataWS] ORDER_TRADE_UPDATE ilegible: {}",
                crate::order_types::truncate(text, 150)
            );
            return;
        };
        let o = ev.o;
        let update = TradeUpdate {
            client_order_id: o.client_order_id.clone(),
            symbol: o.symbol.clone(),
            side: o.side,
            order_type: o.order_type,
            order_id: o.order_id,
            status: OrderStatus::parse(&o.order_status),
            orig_qty: o.orig_qty,
            cumulative_filled_qty: o.cumulative_filled_qty,
            last_filled_qty: o.last_filled_qty,
            last_filled_price: o.last_filled_price,
            avg_price: o.avg_price,
            commission: o.commission,
            commission_asset: o.commission_asset,
            trade_time_ms: o.trade_time_ms,
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(o.trade_time_ms);
        if update.last_filled_qty > 0.0 {
            println!(
                "💧 [FILL] {} {} {} @ {} (acum {}/{} status {:?})",
                update.symbol,
                update.side,
                update.last_filled_qty,
                update.last_filled_price,
                update.cumulative_filled_qty,
                update.orig_qty,
                update.status
            );
        }
        self.registry.apply_trade_update(&update, now);
    }

    fn on_account_update(&self, text: &str) {
        #[derive(Deserialize)]
        struct Balance {
            #[serde(rename = "a")]
            asset: String,
            #[serde(rename = "wb")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            wallet_balance: f64,
        }
        #[derive(Deserialize)]
        struct Position {
            #[serde(rename = "s")]
            symbol: String,
            #[serde(rename = "pa")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            position_amt: f64,
            #[serde(rename = "ep")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            entry_price: f64,
            #[serde(rename = "up")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            unrealized_pnl: f64,
            #[serde(rename = "iw")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            isolated_wallet: f64,
        }
        #[derive(Deserialize)]
        struct AccountData {
            #[serde(default, rename = "B")]
            balances: Vec<Balance>,
            #[serde(default, rename = "P")]
            positions: Vec<Position>,
        }
        #[derive(Deserialize)]
        struct Event {
            a: AccountData,
        }
        let Ok(ev) = serde_json::from_str::<Event>(text) else {
            return;
        };
        for b in &ev.a.balances {
            if b.asset == "USDT" && b.wallet_balance > 0.0 {
                self.sink.on_capital(b.wallet_balance);
            }
        }
        if !ev.a.positions.is_empty() {
            let positions: Vec<RemotePosition> =
                ev.a.positions
                    .iter()
                    .filter(|p| p.position_amt.abs() > 0.0)
                    .map(|p| RemotePosition {
                        symbol: p.symbol.clone(),
                        position_amt: p.position_amt,
                        entry_price: p.entry_price,
                        unrealized_pnl: p.unrealized_pnl,
                        isolated_wallet: p.isolated_wallet,
                    })
                    .collect();
            self.sink.on_positions(&positions);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_order_trade_update_event() {
        // Estructura real de Binance (campos como string).
        let raw = r#"{"e":"ORDER_TRADE_UPDATE","E":1700000000000,"o":{"s":"BTCUSDT","c":"myId123","S":"BUY","o":"LIMIT","f":"GTX","q":"2.0","p":"60000","ap":"60050","sp":"0","X":"PARTIALLY_FILLED","i":99,"l":"0.5","z":"0.5","L":"60050","n":"0.0001","N":"BNB","T":1700000000000,"t":7,"m":false}}"#;
        #[derive(Deserialize)]
        struct Raw {
            #[serde(rename = "c")]
            client_order_id: String,
            #[serde(rename = "s")]
            symbol: String,
            #[serde(rename = "S")]
            side: String,
            #[serde(rename = "o")]
            order_type: String,
            #[serde(rename = "X")]
            order_status: String,
            #[serde(rename = "i")]
            order_id: u64,
            #[serde(rename = "q")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            orig_qty: f64,
            #[serde(rename = "z")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            cumulative_filled_qty: f64,
            #[serde(rename = "l")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            last_filled_qty: f64,
        }
        // El evento real envuelve los datos en "o": {...} (igual que el parser de producción).
        #[derive(Deserialize)]
        struct Event {
            o: Raw,
        }
        let v: Raw = serde_json::from_str::<Event>(raw)
            .expect("parse WS event")
            .o;
        assert_eq!(v.symbol, "BTCUSDT");
        assert_eq!(v.order_status, "PARTIALLY_FILLED");
        assert!((v.cumulative_filled_qty - 0.5).abs() < 1e-12);
        assert!((v.last_filled_qty - 0.5).abs() < 1e-12);
        assert_eq!(v.order_id, 99);
    }

    #[test]
    fn account_update_routes_to_sink() {
        use std::sync::atomic::AtomicU64;
        let raw = r#"{"e":"ACCOUNT_UPDATE","a":{"B":[{"a":"USDT","wb":"105.25"}],"P":[{"s":"BTCUSDT","pa":"0.002","ep":"60000","up":"1.5","iw":"0"}]}}"#;
        struct Cap(AtomicU64);
        impl AccountSink for Cap {
            fn on_capital(&self, usdt: f64) {
                self.0
                    .store((usdt * 100.0) as u64, std::sync::atomic::Ordering::Relaxed);
            }
            fn on_positions(&self, _p: &[RemotePosition]) {}
        }
        let sink = Arc::new(Cap(AtomicU64::new(0)));
        let streamer = UserDataStreamer::new(
            BinanceClient::new("k".into(), true),
            Arc::new(OrderRegistry::new()),
        )
        .with_sink(sink.clone());
        streamer.dispatch(raw.as_bytes());
        assert_eq!(sink.0.load(std::sync::atomic::Ordering::Relaxed), 10525);
    }
}
