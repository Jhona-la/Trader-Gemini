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
    pub position_side: String,
}

/// Interfaz del host para eventos de cuenta. El motor implementa esto para
/// actualizar su estado (capital, posiciones) con la verdad del exchange.
pub trait AccountSink: Send + Sync {
    fn on_capital(&self, usdt_total_equity: f64);
    fn on_positions(&self, positions: &[RemotePosition]);
}

struct NoopSink;
impl AccountSink for NoopSink {
    fn on_capital(&self, _usdt: f64) {}
    fn on_positions(&self, _p: &[RemotePosition]) {}
}

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// F1.6: stream privado. `start()` es infinito (diseñado para tokio::spawn).
pub struct UserDataStreamer {
    client: BinanceClient,
    registry: Arc<OrderRegistry>,
    sink: Arc<dyn AccountSink>,
    cached_positions: Mutex<HashMap<(String, String), f64>>,
    api_secret: Option<String>,
    expired_flag: Arc<AtomicBool>,
    arena: Option<Arc<quantum_arena::GlobalArena>>,
    /// X-011 (REHAB-4): bandera de apagado — sin ella, el streamer de la era
    /// demo seguía vivo tras la transición a mainnet y sus ACCOUNT_UPDATE de
    /// TESTNET pisaban el capital real del plano compartido.
    shutdown: Option<Arc<AtomicBool>>,
}

impl UserDataStreamer {
    pub fn new(client: BinanceClient, registry: Arc<OrderRegistry>) -> Self {
        Self {
            client,
            registry,
            sink: Arc::new(NoopSink),
            cached_positions: Mutex::new(HashMap::new()),
            api_secret: None,
            expired_flag: Arc::new(AtomicBool::new(false)),
            arena: None,
            shutdown: None,
        }
    }

    /// X-011: registra la bandera de apagado (el spawn devuelve control; el
    /// llamador conserva el Arc y lo activa al reemplazar este streamer).
    pub fn with_shutdown(mut self, flag: Arc<AtomicBool>) -> Self {
        self.shutdown = Some(flag);
        self
    }

    pub fn with_api_secret(mut self, api_secret: impl Into<String>) -> Self {
        self.api_secret = Some(api_secret.into());
        self
    }

    pub fn with_sink(mut self, sink: Arc<dyn AccountSink>) -> Self {
        self.sink = sink;
        self
    }

    pub fn with_arena(mut self, arena: Arc<quantum_arena::GlobalArena>) -> Self {
        self.arena = Some(arena);
        self
    }

    /// Loop principal: listenKey → connect WebSocket → lee frames → reintenta
    /// en desconexión. Diseñado para ejecutarse en `tokio::spawn`.
    pub async fn start(&self) {
        let is_testnet = self
            .client
            .is_testnet
            .load(std::sync::atomic::Ordering::Relaxed);
        let base_ws = if is_testnet {
            "wss://stream.binancefuture.com/ws"
        } else {
            "wss://fstream.binance.com/ws"
        };

        let mut backoff_ms = 500u64;
        loop {
            // X-011: apagado cooperativo — el streamer reemplazado muere en la
            // PRÓXIMA iteración (nunca spawnea listenKeys de credenciales viejas).
            if let Some(flag) = &self.shutdown {
                if flag.load(std::sync::atomic::Ordering::Acquire) {
                    println!("🔌 [USER-STREAM] Apagado cooperativo (reemplazado por transición).");
                    return;
                }
            }
            let listen_key = match self.client.create_listen_key().await {
                Ok(k) if !k.is_empty() => {
                    backoff_ms = 500;
                    k
                }
                Ok(_) => {
                    let jitter_ms = (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.subsec_nanos() as u64)
                        .unwrap_or(42)
                        % 200)
                        + 50;
                    println!(
                        "⚠️ [USER-STREAM] listenKey vacío; reintentando en {}ms...",
                        backoff_ms + jitter_ms
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms + jitter_ms))
                        .await;
                    backoff_ms = (backoff_ms * 2).min(30_000);
                    continue;
                }
                Err(e) => {
                    let jitter_ms = (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.subsec_nanos() as u64)
                        .unwrap_or(42)
                        % 200)
                        + 50;
                    println!(
                        "⚠️ [USER-STREAM] create_listen_key falló: {}; reintentando en {}ms...",
                        e,
                        backoff_ms + jitter_ms
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms + jitter_ms))
                        .await;
                    backoff_ms = (backoff_ms * 2).min(30_000);
                    continue;
                }
            };

            let ws_url = format!("{}/{}", base_ws, listen_key);
            let url = match Url::parse(&ws_url) {
                Ok(u) => u,
                Err(e) => {
                    println!("❌ [USER-STREAM] URL inválida {}: {}", ws_url, e);
                    break;
                }
            };

            println!("🔌 [USER-STREAM] Conectando a {}", ws_url);
            let ws_stream = match connect_async(url.as_str()).await {
                Ok((stream, _)) => {
                    println!("✅ [USER-STREAM] Conectado. Escuchando fills y updates de cuenta.");
                    stream
                }
                Err(e) => {
                    println!(
                        "⚠️ [USER-STREAM] Conexión falló: {}; reintentando en {}ms...",
                        e, backoff_ms
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(30_000);
                    continue;
                }
            };

            let (mut _write, mut read) = ws_stream.split();

            // Tarea paralela de keepalive (PUT listenKey cada 25 min).
            let client_clone = self.client.clone();
            let keepalive_handle = tokio::spawn(async move {
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(25 * 60)).await;
                    if let Err(e) = client_clone.keep_alive_listen_key().await {
                        println!("⚠️ [USER-STREAM] keep_alive_listen_key falló: {}", e);
                    } else {
                        println!("💓 [USER-STREAM] listenKey refrescado.");
                    }
                }
            });

            while let Some(msg) = read.next().await {
                // X-011: muerte inmediata (no en la próxima reconexión) — un
                // frame del streamer viejo puede pisar el capital del nuevo.
                if let Some(flag) = &self.shutdown {
                    if flag.load(Ordering::Acquire) {
                        println!("🔌 [USER-STREAM] Apagado inmediato por reemplazo.");
                        let _ = keepalive_handle.abort();
                        return;
                    }
                }
                if self.expired_flag.swap(false, Ordering::Relaxed) {
                    println!("🔄 [USER-STREAM] listenKeyExpired: cerrando sesión para renovación inmediata.");
                    break;
                }
                match msg {
                    Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                        self.route_event(&text);
                        // D-198: Salida inmediata tras procesar listenKeyExpired para no colgar en read.next()
                        if self.expired_flag.swap(false, Ordering::Relaxed) {
                            println!("🔄 [USER-STREAM] listenKeyExpired recibido y procesado: cerrando WebSocket inmediatamente para renovación.");
                            break;
                        }
                    }
                    Ok(tokio_tungstenite::tungstenite::Message::Ping(_)) => {
                        // Tungstenite responde auto-PONG.
                    }
                    Ok(tokio_tungstenite::tungstenite::Message::Close(frame)) => {
                        println!("⚠️ [USER-STREAM] Servidor cerró conexión: {:?}", frame);
                        break;
                    }
                    Err(e) => {
                        println!("⚠️ [USER-STREAM] Error de lectura: {}", e);
                        break;
                    }
                    _ => {}
                }
            }

            keepalive_handle.abort();
            println!("🔌 [USER-STREAM] Desconectado. Reanudando en 1s...");
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }

    /// Rutea según bytes crudos del WebSocket.
    pub fn dispatch(&self, bytes: &[u8]) {
        let text = std::str::from_utf8(bytes).unwrap_or("");
        self.route_event(text);
    }

    /// Rutea según el campo `e` (tipo de evento).
    pub(crate) fn route_event(&self, text: &str) {
        #[derive(Deserialize)]
        struct EventType {
            #[serde(default)]
            e: String,
        }
        let Ok(ev) = serde_json::from_str::<EventType>(text) else {
            return;
        };
        match ev.e.as_str() {
            "ORDER_TRADE_UPDATE" => self.on_order_trade_update(text),
            "ACCOUNT_UPDATE" => self.on_account_update(text),
            "ALGO_UPDATE" => self.on_algo_update(text),
            "listenKeyExpired" => self.on_listen_key_expired(),
            _ => {}
        }
    }

    /// B1.1: ciclo de vida de los brackets TP/SL (órdenes ALGO, migración
    /// 2025-12-09). Hasta ahora este evento caía en `_ => {}`: el motor era
    /// ciego a cancelaciones/expiraciones/rechazos de sus propias
    /// protecciones y la posición quedaba desnuda sin ninguna señal local.
    ///
    /// Estados: NEW | TRIGGERING | TRIGGERED | FINISHED | CANCELED |
    /// REJECTED | EXPIRED. Los TERMINALES marcan `protection_dirty` para que
    /// el watchdog del motor audite y re-bracketee en su próximo ciclo.
    ///
    /// Parseo defensivo por Value con alias corto/largo: el payload exacto
    /// del WS no está documentado inline y no se puede asumir el naming.
    fn on_algo_update(&self, text: &str) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
            return;
        };
        let pick = |keys: &[&str]| -> String {
            for k in keys {
                if let Some(s) = v.get(k).and_then(|x| x.as_str()) {
                    return s.to_string();
                }
            }
            String::new()
        };
        let symbol = pick(&["s", "symbol", "S"]);
        let client_algo_id = pick(&["clientAlgoId", "c", "clientOrderId"]);
        let order_type = pick(&["orderType", "algoOrderType", "o", "type"]);
        let algo_status = pick(&["algoStatus", "X", "status"]);
        if symbol.is_empty() && algo_status.is_empty() && client_algo_id.is_empty() {
            // Nada reconocible: registrar crudo para el forense del esquema.
            println!(
                "📮 [ALGO-UPDATE] payload no reconocido: {}",
                &text[..text.len().min(240)]
            );
            return;
        }
        match algo_status.as_str() {
            "CANCELED" | "EXPIRED" | "REJECTED" => {
                println!(
                    "⚠️ [ALGO-UPDATE] {} {} {} TERMINAL ({}) — protección posiblemente revocada",
                    symbol, order_type, client_algo_id, algo_status
                );
                quantum_arena::protection_health::mark_dirty();
            }
            "FINISHED" => {
                // La pierna disparó y llenó/canceló en el matching engine:
                // la posición correspondiente debió cerrarse (o quedó
                // parcial). Auditar igual — el fill llegó por ORDER_TRADE_UPDATE.
                println!(
                    "✅ [ALGO-UPDATE] {} {} {} FINISHED — disparo completado",
                    symbol, order_type, client_algo_id
                );
                quantum_arena::protection_health::mark_dirty();
            }
            "TRIGGERING" | "TRIGGERED" => {
                println!(
                    "🔥 [ALGO-UPDATE] {} {} {} {} — salida en curso",
                    symbol, order_type, client_algo_id, algo_status
                );
            }
            _ => {} // NEW y transiciones internas: silencio (spam por bracket)
        }
    }

    fn on_listen_key_expired(&self) {
        println!(
            "🚨 [USER-STREAM] listenKeyExpired recibido de Binance! Forzando renovación inmediata."
        );
        self.expired_flag.store(true, Ordering::Relaxed);
    }

    fn on_order_trade_update(&self, text: &str) {
        #[derive(Deserialize)]
        struct OrderPayload {
            #[serde(rename = "s")]
            symbol: String,
            #[serde(rename = "c")]
            client_order_id: String,
            #[serde(rename = "S")]
            side: String,
            #[serde(rename = "o")]
            order_type: String,
            #[serde(rename = "X")]
            order_status: String,
            #[serde(rename = "x", default)]
            execution_type: String,
            #[serde(rename = "i")]
            order_id: u64,
            #[serde(rename = "l")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            last_filled_qty: f64,
            #[serde(rename = "z")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            cumulative_filled_qty: f64,
            #[serde(rename = "L")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            last_filled_price: f64,
            #[serde(rename = "ap", default)]
            #[serde(deserialize_with = "crate::order_types::string_or_f64")]
            avg_price: f64,
            #[serde(rename = "n")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            commission: f64,
            #[serde(rename = "N")]
            commission_asset: Option<String>,
            #[serde(rename = "T")]
            trade_time_ms: u64,
            #[serde(rename = "ps", default)]
            position_side: String,
            #[serde(rename = "q")]
            #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
            orig_qty: f64,
        }
        #[derive(Deserialize)]
        struct Event {
            #[allow(dead_code)]
            #[serde(rename = "E")]
            event_time_ms: u64,
            o: OrderPayload,
        }
        let Ok(ev) = serde_json::from_str::<Event>(text) else {
            return;
        };
        let o = ev.o;
        let update = TradeUpdate {
            client_order_id: o.client_order_id,
            symbol: o.symbol,
            side: o.side,
            position_side: o.position_side,
            order_type: o.order_type,
            execution_type: o.execution_type,
            order_id: o.order_id,
            status: OrderStatus::parse(&o.order_status),
            orig_qty: o.orig_qty,
            cumulative_filled_qty: o.cumulative_filled_qty,
            last_filled_qty: o.last_filled_qty,
            last_filled_price: o.last_filled_price,
            avg_price: if o.avg_price > 0.0 {
                o.avg_price
            } else {
                o.last_filled_price
            },
            commission: o.commission,
            commission_asset: o.commission_asset.unwrap_or_default(),
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

        // K-04 / R3.1 / D-179: Motor de cancelación automática de pierna hermana OCO
        // Soporta tanto identificadores estándar (_TP, _SL) como variantes con retry (_TPR, _SLR).
        // Cancela todas las variantes de la pierna hermana para evitar dobles ejecuciones u órdenes huérfanas.
        if update.status == OrderStatus::Filled {
            let sister_candidates = if let Some(base) = update.client_order_id.strip_suffix("_TPR")
            {
                vec![format!("{}_SL", base), format!("{}_SLR", base)]
            } else if let Some(base) = update.client_order_id.strip_suffix("_TP") {
                vec![format!("{}_SL", base), format!("{}_SLR", base)]
            } else if let Some(base) = update.client_order_id.strip_suffix("_SLR") {
                vec![format!("{}_TP", base), format!("{}_TPR", base)]
            } else if let Some(base) = update.client_order_id.strip_suffix("_SL") {
                vec![format!("{}_TP", base), format!("{}_TPR", base)]
            } else {
                Vec::new()
            };

            if !sister_candidates.is_empty() {
                if let Some(ref secret) = self.api_secret {
                    let client = self.client.clone();
                    let symbol = update.symbol.clone();
                    let secret = secret.clone();
                    let filled_id = update.client_order_id.clone();
                    let arena_clone = self.arena.clone();
                    tokio::spawn(async move {
                        for sister_id in sister_candidates {
                            let ts = crate::executor::current_synced_timestamp_ms(
                                arena_clone.as_deref(),
                            );
                            let mut buf = crate::client::ZeroAllocBuffer::new();
                            buf.push_str(
                                if client.is_testnet.load(std::sync::atomic::Ordering::Relaxed) {
                                    "https://testnet.binancefuture.com/fapi/v1/order?"
                                } else {
                                    "https://fapi.binance.com/fapi/v1/order?"
                                },
                            );
                            let payload_start = buf.as_str().len();
                            buf.push_str("symbol=");
                            buf.push_str(&symbol);
                            buf.push_str("&origClientOrderId=");
                            buf.push_str(&sister_id);
                            buf.push_str("&timestamp=");
                            buf.push_u64(ts);

                            let mut sig_buf = [0u8; 64];
                            let payload = &buf.as_str()[payload_start..];
                            crate::binance_api::sign_payload_to_buffer(
                                payload,
                                &secret,
                                &mut sig_buf,
                            );
                            let sig = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
                            buf.push_str("&signature=");
                            buf.push_str(sig);

                            match client.cancel_order_payload(buf.as_str()).await {
                                Ok(_) => println!("🎯 [OCO MOTOR] Pierna hermana {} cancelada exitosamente tras fill de {}.", sister_id, filled_id),
                                Err(e) => println!("ℹ️ [OCO MOTOR] Pierna hermana {} ya resuelta o cancelada: {}", sister_id, e),
                            }
                        }
                    });
                }
            }
        }
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
            #[serde(rename = "ps", default)]
            position_side: String,
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

        // F1.6: Equidad total = Wallet Balance + Sum(Unrealized PnL acumulado de todas las posiciones)
        // FIX #744: Binance envía deltas. Mantener cache de todas las posiciones para sumar PnL total real.
        let total_unrealized_pnl: f64 = if let Ok(mut cache) = self.cached_positions.lock() {
            for p in &ev.a.positions {
                let key = (p.symbol.clone(), p.position_side.clone());
                let pnl = if p.unrealized_pnl.is_finite() {
                    p.unrealized_pnl
                } else {
                    0.0
                };
                if p.position_amt.abs() > 1e-8 {
                    cache.insert(key, pnl);
                } else {
                    cache.remove(&key);
                }
            }
            cache.values().sum()
        } else {
            ev.a.positions
                .iter()
                .map(|p| {
                    if p.unrealized_pnl.is_finite() {
                        p.unrealized_pnl
                    } else {
                        0.0
                    }
                })
                .sum()
        };

        for b in &ev.a.balances {
            if b.asset == "USDT" && b.wallet_balance > 0.0 && b.wallet_balance.is_finite() {
                let total_equity = if (b.wallet_balance + total_unrealized_pnl).is_finite() {
                    (b.wallet_balance + total_unrealized_pnl).max(0.0)
                } else {
                    b.wallet_balance
                };
                self.sink.on_capital(total_equity);
            }
        }

        if !ev.a.positions.is_empty() {
            let positions: Vec<RemotePosition> = ev
                .a
                .positions
                .iter()
                .map(|p| RemotePosition {
                    symbol: p.symbol.clone(),
                    position_amt: if p.position_amt.is_finite() {
                        p.position_amt
                    } else {
                        0.0
                    },
                    entry_price: if p.entry_price.is_finite() && p.entry_price >= 0.0 {
                        p.entry_price
                    } else {
                        0.0
                    },
                    unrealized_pnl: if p.unrealized_pnl.is_finite() {
                        p.unrealized_pnl
                    } else {
                        0.0
                    },
                    isolated_wallet: if p.isolated_wallet.is_finite() && p.isolated_wallet >= 0.0 {
                        p.isolated_wallet
                    } else {
                        0.0
                    },
                    position_side: p.position_side.clone(),
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
        assert_eq!(v.client_order_id, "myId123");
        assert_eq!(v.symbol, "BTCUSDT");
        assert_eq!(v.side, "BUY");
        assert_eq!(v.order_type, "LIMIT");
        assert_eq!(v.orig_qty, 2.0);
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
        assert_eq!(sink.0.load(std::sync::atomic::Ordering::Relaxed), 10675);
    }

    #[test]
    fn test_dispatch_order_trade_update_to_registry() {
        let registry = Arc::new(OrderRegistry::new());
        registry.register_intent("ORD_101", "BTCUSDT", "BUY", "LONG", "LIMIT", 1.0, 1000);

        let streamer =
            UserDataStreamer::new(BinanceClient::new("key".into(), true), registry.clone());

        let trade_event = r#"{"e":"ORDER_TRADE_UPDATE","E":1700000000000,"o":{"s":"BTCUSDT","c":"ORD_101","S":"BUY","ps":"LONG","o":"LIMIT","f":"GTC","q":"1.0","p":"50000","ap":"50000","sp":"0","X":"FILLED","i":10101,"l":"1.0","z":"1.0","L":"50000","n":"0.0001","N":"USDT","T":1700000000000,"t":1,"m":false}}"#;
        streamer.dispatch(trade_event.as_bytes());

        let order = registry.get("ORD_101");
        assert_eq!(order.map(|o| o.status), Some(OrderStatus::Filled));
    }

    #[test]
    fn test_listen_key_expired_triggers_flag() {
        let registry = Arc::new(OrderRegistry::new());
        let streamer = UserDataStreamer::new(BinanceClient::new("key".into(), true), registry);
        assert!(!streamer.expired_flag.load(Ordering::Relaxed));
        let expired_event = r#"{"e":"listenKeyExpired","E":1700000000000}"#;
        streamer.dispatch(expired_event.as_bytes());
        assert!(streamer.expired_flag.load(Ordering::Relaxed));
    }

    /// B1.1: un ALGO_UPDATE terminal debe marcar protection_dirty (posición
    /// posiblemente desnuda) y uno de disparo NO debe marcarlo.
    #[test]
    fn test_algo_update_terminal_marks_protection_dirty() {
        quantum_arena::protection_health::clear_dirty();
        let registry = Arc::new(OrderRegistry::new());
        let streamer = UserDataStreamer::new(BinanceClient::new("key".into(), true), registry);

        // Disparo en curso: no marca (la salida ya está corriendo).
        let triggering = r#"{"e":"ALGO_UPDATE","E":1700000000000,"symbol":"BTCUSDT","clientAlgoId":"wdTP_1","orderType":"TAKE_PROFIT_MARKET","algoStatus":"TRIGGERING"}"#;
        streamer.dispatch(triggering.as_bytes());
        assert!(!quantum_arena::protection_health::is_dirty());

        // Cancelación de pierna: marca — la posición puede haber quedado desnuda.
        let canceled = r#"{"e":"ALGO_UPDATE","E":1700000000001,"symbol":"BTCUSDT","clientAlgoId":"wdTP_1","orderType":"TAKE_PROFIT_MARKET","algoStatus":"CANCELED"}"#;
        streamer.dispatch(canceled.as_bytes());
        assert!(quantum_arena::protection_health::is_dirty());
        assert_eq!(quantum_arena::protection_health::terminal_events_seen(), 1);

        quantum_arena::protection_health::clear_dirty();
    }
}
