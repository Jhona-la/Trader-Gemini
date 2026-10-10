//! Feed Spot-Futuro en Tiempo Real para Binance Spot (Defecto C-02, Ola Ω57).
//!
//! Alimenta en tiempo de ejecución:
//! 1. `GlobalArena::update_spot_data` (spot_bid, spot_ask, spot_bid_qty, spot_ask_qty por moneda)
//! 2. `OmniscientRegistry` con claves `{SYM}_spot_bid`, `{SYM}_spot_ask`, `{SYM}_spot_mid`
//! 3. `OmniState::binance_spot` y `OmniState::spot_futures_arb_spread`
//!
//! Permite que el motor `statarb_ou_engines[coin_id]` en `god-engine-core` observe pares
//! spot-futuro en tiempo físico real, madure ($\ge 10$ pares) y publique `statarb_ou_zscore`,
//! `statarb_half_life_ms` y `statarb_beta`.

use fast_float::parse;
use futures_util::StreamExt;
use quantum_arena::GlobalArena;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tokio_tungstenite::connect_async;

/// Intervalo canónico de sondeo REST como red de seguridad (1000 ms).
pub const SPOT_REST_POLL_INTERVAL_MS: u64 = 1_000;

/// URL base de la API REST de Binance Spot (pública, sin autenticación).
pub const BINANCE_SPOT_REST_BASE: &str = "https://api.binance.com";

/// URL base del WebSocket de Binance Spot (pública, sin autenticación).
pub const BINANCE_SPOT_WS_BASE: &str = "wss://stream.binance.com:9443";

/// Parsea el payload JSON de `/api/v3/ticker/bookTicker` (array o objeto único).
pub fn parse_rest_book_tickers(json_str: &str) -> Vec<(String, f64, f64, f64, f64)> {
    let mut results = Vec::new();
    let Ok(val) = serde_json::from_str::<Value>(json_str) else {
        return results;
    };

    if let Some(arr) = val.as_array() {
        for item in arr {
            if let Some(entry) = extract_ticker_entry(item) {
                results.push(entry);
            }
        }
    } else if let Some(entry) = extract_ticker_entry(&val) {
        results.push(entry);
    }

    results
}

fn extract_ticker_entry(item: &Value) -> Option<(String, f64, f64, f64, f64)> {
    let sym = item.get("symbol")?.as_str()?.to_ascii_uppercase();
    let b_str = item.get("bidPrice")?.as_str()?;
    let a_str = item.get("askPrice")?.as_str()?;
    let bq_str = item.get("bidQty")?.as_str()?;
    let aq_str = item.get("askQty")?.as_str()?;

    let bid = parse::<f64, _>(b_str).ok()?;
    let ask = parse::<f64, _>(a_str).ok()?;
    let bid_qty = parse::<f64, _>(bq_str).ok().unwrap_or(0.0);
    let ask_qty = parse::<f64, _>(aq_str).ok().unwrap_or(0.0);

    if bid > 0.0 && ask > 0.0 && bid.is_finite() && ask.is_finite() {
        Some((sym, bid, ask, bid_qty, ask_qty))
    } else {
        None
    }
}

/// Parsea el payload de WebSocket de Binance Spot `@bookTicker`.
/// Admite tanto stream directo como combined stream (`{"stream":"...","data":{...}}`).
pub fn parse_ws_book_ticker(json_str: &str) -> Option<(String, f64, f64, f64, f64)> {
    let val: Value = serde_json::from_str(json_str).ok()?;
    let data = if let Some(d) = val.get("data") {
        d
    } else {
        &val
    };

    let sym = data.get("s")?.as_str()?.to_ascii_uppercase();
    let b_str = data.get("b")?.as_str()?;
    let a_str = data.get("a")?.as_str()?;
    let bq_str = data.get("B")?.as_str()?;
    let aq_str = data.get("A")?.as_str()?;

    let bid = parse::<f64, _>(b_str).ok()?;
    let ask = parse::<f64, _>(a_str).ok()?;
    let bid_qty = parse::<f64, _>(bq_str).ok().unwrap_or(0.0);
    let ask_qty = parse::<f64, _>(aq_str).ok().unwrap_or(0.0);

    if bid > 0.0 && ask > 0.0 && bid.is_finite() && ask.is_finite() {
        Some((sym, bid, ask, bid_qty, ask_qty))
    } else {
        None
    }
}

/// Aplica una actualización de precio spot al Arena, Registry y OmniState.
pub fn apply_spot_tick(
    arena: &GlobalArena,
    omni_state: &crate::omni_multiplexer::OmniState,
    coin_id: usize,
    symbol: &str,
    bid: f64,
    ask: f64,
    bid_qty: f64,
    ask_qty: f64,
) {
    if !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= 0.0 {
        return;
    }

    // 1. Arena principal y tensor
    arena.update_spot_data(coin_id, bid, ask, bid_qty, ask_qty);

    // 2. Publicación con ámbito en el registro
    let spot_mid = (bid + ask) * 0.5;
    arena.registry.set_scoped(symbol, "spot_bid", bid);
    arena.registry.set_scoped(symbol, "spot_ask", ask);
    arena.registry.set_scoped(symbol, "spot_mid", spot_mid);

    // 3. Si es BTC (o coin_id == 0), actualizar OmniState
    if coin_id == 0 || symbol == "BTCUSDT" {
        omni_state
            .binance_spot
            .store(spot_mid.to_bits(), Ordering::Relaxed);
        let fut_bits = omni_state.binance_futures.load(Ordering::Relaxed);
        let fut_mid = f64::from_bits(fut_bits);
        if fut_mid > 0.0 && fut_mid.is_finite() {
            let basis = (fut_mid - spot_mid) / spot_mid;
            omni_state
                .spot_futures_arb_spread
                .store(basis.to_bits(), Ordering::Relaxed);
        }
    }
}

/// Inicia la sincronización en vivo del feed spot de Binance usando un bucle híbrido
/// (WebSocket de baja latencia + Sondeo REST de alta fiabilidad).
pub fn start_spot_feed_sync(
    rt_handle: &tokio::runtime::Handle,
    arena: Arc<GlobalArena>,
    omni_state: Arc<crate::omni_multiplexer::OmniState>,
    symbols: Vec<String>,
) {
    let mut symbol_to_coin_id = HashMap::new();
    let mut valid_spot_symbols = Vec::new();

    for (idx, sym) in symbols.iter().enumerate() {
        let sym_upper = sym.to_ascii_uppercase();
        symbol_to_coin_id.insert(sym_upper.clone(), idx);
        valid_spot_symbols.push(sym_upper);
    }

    if valid_spot_symbols.is_empty() {
        return;
    }

    let symbol_map = Arc::new(symbol_to_coin_id);
    let spot_symbols = Arc::new(valid_spot_symbols);

    // 1. Tarea REST de fondo: Sondeo garantizado cada 1000 ms
    {
        let arena_cloned = Arc::clone(&arena);
        let omni_cloned = Arc::clone(&omni_state);
        let sym_map = Arc::clone(&symbol_map);
        let sym_list = Arc::clone(&spot_symbols);

        rt_handle.spawn(async move {
            run_spot_rest_poller(arena_cloned, omni_cloned, sym_map, sym_list).await;
        });
    }

    // 2. Tarea WebSocket de fondo: Streaming push de sub-milisegundo
    {
        let arena_cloned = Arc::clone(&arena);
        let omni_cloned = Arc::clone(&omni_state);
        let sym_map = Arc::clone(&symbol_map);
        let sym_list = Arc::clone(&spot_symbols);

        rt_handle.spawn(async move {
            run_spot_ws_streamer(arena_cloned, omni_cloned, sym_map, sym_list).await;
        });
    }

    println!(
        "📡 [SPOT-FEED] Sincronización Spot-Futuro activada para {} símbolos (WS + REST fallback)",
        spot_symbols.len()
    );
}

async fn run_spot_rest_poller(
    arena: Arc<GlobalArena>,
    omni_state: Arc<crate::omni_multiplexer::OmniState>,
    symbol_map: Arc<HashMap<String, usize>>,
    symbols: Arc<Vec<String>>,
) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(1500))
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    // Construir la URL con parámetros si son pocos símbolos para reducir carga
    let syms_json = serde_json::to_string(&*symbols).unwrap_or_default();
    let query_url = format!("{}/api/v3/ticker/bookTicker?symbols={}", BINANCE_SPOT_REST_BASE, urlencoding_simple(&syms_json));

    let mut tick_timer = interval(Duration::from_millis(SPOT_REST_POLL_INTERVAL_MS));
    tick_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tick_timer.tick().await;

        match client.get(&query_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(body) = resp.text().await {
                    let entries = parse_rest_book_tickers(&body);
                    for (sym, bid, ask, bid_qty, ask_qty) in entries {
                        if let Some(&coin_id) = symbol_map.get(&sym) {
                            apply_spot_tick(&arena, &omni_state, coin_id, &sym, bid, ask, bid_qty, ask_qty);
                        }
                    }
                }
            }
            Ok(_) | Err(_) => {
                // Si la consulta agrupada falla (ej. algún símbolo no es par spot), intentar individualmente
                for sym in symbols.iter() {
                    let single_url = format!("{}/api/v3/ticker/bookTicker?symbol={}", BINANCE_SPOT_REST_BASE, sym);
                    if let Ok(resp) = client.get(&single_url).send().await {
                        if resp.status().is_success() {
                            if let Ok(body) = resp.text().await {
                                let entries = parse_rest_book_tickers(&body);
                                for (s, bid, ask, bid_qty, ask_qty) in entries {
                                    if let Some(&coin_id) = symbol_map.get(&s) {
                                        apply_spot_tick(&arena, &omni_state, coin_id, &s, bid, ask, bid_qty, ask_qty);
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

async fn run_spot_ws_streamer(
    arena: Arc<GlobalArena>,
    omni_state: Arc<crate::omni_multiplexer::OmniState>,
    symbol_map: Arc<HashMap<String, usize>>,
    symbols: Arc<Vec<String>>,
) {
    let mut streams_vec = Vec::new();
    for s in symbols.iter() {
        streams_vec.push(format!("{}@bookTicker", s.to_ascii_lowercase()));
    }
    let streams_param = streams_vec.join("/");
    let ws_url = format!("{}/stream?streams={}", BINANCE_SPOT_WS_BASE, streams_param);

    let mut backoff_ms = 500u64;

    loop {
        let url_parsed = match url::Url::parse(&ws_url) {
            Ok(u) => u,
            Err(_) => {
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        match connect_async(url_parsed).await {
            Ok((mut ws_stream, _)) => {
                backoff_ms = 500;
                while let Some(msg_result) = ws_stream.next().await {
                    match msg_result {
                        Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                            if let Some((sym, bid, ask, bid_qty, ask_qty)) = parse_ws_book_ticker(&text) {
                                if let Some(&coin_id) = symbol_map.get(&sym) {
                                    apply_spot_tick(&arena, &omni_state, coin_id, &sym, bid, ask, bid_qty, ask_qty);
                                }
                            }
                        }
                        Ok(tokio_tungstenite::tungstenite::Message::Ping(payload)) => {
                            use futures_util::SinkExt;
                            let _ = ws_stream.send(tokio_tungstenite::tungstenite::Message::Pong(payload)).await;
                        }
                        Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => break,
                        Err(_) => break,
                        _ => {}
                    }
                }
            }
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(10_000);
            }
        }
    }
}

fn urlencoding_simple(s: &str) -> String {
    let mut res = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'-' | b'_' | b'.' | b'~' => {
                res.push(b as char);
            }
            b'"' => res.push_str("%22"),
            b'[' => res.push_str("%5B"),
            b']' => res.push_str("%5D"),
            b',' => res.push_str("%2C"),
            _ => {
                res.push_str(&format!("%{:02X}", b));
            }
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rest_book_tickers_array() {
        let json = r#"[
            {"symbol":"BTCUSDT","bidPrice":"82647.87","bidQty":"11.72","askPrice":"82647.88","askQty":"5.44"},
            {"symbol":"ETHUSDT","bidPrice":"2493.94","bidQty":"91.48","askPrice":"2493.95","askQty":"33.51"}
        ]"#;

        let res = parse_rest_book_tickers(json);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0, "BTCUSDT");
        assert!((res[0].1 - 82647.87).abs() < 1e-4);
        assert!((res[0].2 - 82647.88).abs() < 1e-4);
        assert_eq!(res[1].0, "ETHUSDT");
        assert!((res[1].1 - 2493.94).abs() < 1e-4);
    }

    #[test]
    fn test_parse_ws_book_ticker_combined() {
        let json = r#"{
            "stream": "btcusdt@bookTicker",
            "data": {
                "u": 400900217,
                "s": "BTCUSDT",
                "b": "82650.10",
                "B": "1.500",
                "a": "82650.20",
                "A": "2.300"
            }
        }"#;

        let parsed = parse_ws_book_ticker(json);
        assert!(parsed.is_some());
        let (sym, bid, ask, bq, aq) = parsed.unwrap();
        assert_eq!(sym, "BTCUSDT");
        assert!((bid - 82650.10).abs() < 1e-4);
        assert!((ask - 82650.20).abs() < 1e-4);
        assert!((bq - 1.5).abs() < 1e-4);
        assert!((aq - 2.3).abs() < 1e-4);
    }

    #[test]
    fn test_apply_spot_tick_updates_arena_and_omni() {
        let arena = GlobalArena::build_in_own_stack(100.0);
        let omni = crate::omni_multiplexer::OmniState::new();

        apply_spot_tick(&arena, &omni, 0, "BTCUSDT", 50_000.0, 50_002.0, 1.5, 2.5);

        assert_eq!(arena.coins[0].spot_bid.load(Ordering::Relaxed), 50_000.0);
        assert_eq!(arena.coins[0].spot_ask.load(Ordering::Relaxed), 50_002.0);
        assert_eq!(arena.coins[0].spot_bid_qty.load(Ordering::Relaxed), 1.5);
        assert_eq!(arena.coins[0].spot_ask_qty.load(Ordering::Relaxed), 2.5);

        let reg_mid = arena.registry.get_scoped_parameter(Some("BTCUSDT"), Some(0), "spot_mid", "test").unwrap().get_value();
        assert!((reg_mid - 50_001.0).abs() < 1e-6);

        let omni_spot = f64::from_bits(omni.binance_spot.load(Ordering::Relaxed));
        assert!((omni_spot - 50_001.0).abs() < 1e-6);
    }
}
