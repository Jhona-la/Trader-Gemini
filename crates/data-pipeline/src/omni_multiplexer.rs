use futures_util::StreamExt;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration};

pub struct OmniState {
    pub binance_spot: AtomicU64,
    pub binance_futures: AtomicU64,
    pub bybit_linear: AtomicU64,
    pub okx_swap: AtomicU64,
    pub bitget_futures: AtomicU64,
    pub coinbase_spot: AtomicU64,
    pub kraken_spot: AtomicU64,
    pub htx_spot: AtomicU64,
    pub deribit_options: AtomicU64,
    pub bitfinex_spot: AtomicU64,
    pub binance_liquidations: AtomicU64,
    pub agg_funding_rate: AtomicU64,
    pub agg_open_interest: AtomicU64,
    pub long_short_ratio: AtomicU64,
    pub fear_greed_index: AtomicU64,
    pub altcoin_dominance: AtomicU64,
    pub mempool_congestion: AtomicU64,
    pub usdt_mint_alert: AtomicU64,
    pub exchange_inflows: AtomicU64,
    pub exchange_outflows: AtomicU64,
    pub whale_alert_proxy: AtomicU64,
    pub dxy: AtomicU64,
    pub sp500: AtomicU64,
    pub nasdaq: AtomicU64,
    pub vix: AtomicU64,
    pub us10y: AtomicU64,
    pub gold: AtomicU64,
    pub oil_wti: AtomicU64,
    pub econ_calendar_impact: AtomicU64,
    pub fed_interest_rate: AtomicU64,
    pub spot_cvd: AtomicU64,
    pub futures_cvd: AtomicU64,
    pub taker_buy_sell_ratio: AtomicU64,
    pub futures_basis_premium: AtomicU64,
    pub liq_cluster_shorts: AtomicU64,
    pub liq_cluster_longs: AtomicU64,
    pub cme_futures_premium: AtomicU64,
    pub cme_gap_proximity: AtomicU64,
    pub spot_futures_arb_spread: AtomicU64,
    pub order_flow_imbalance: AtomicU64,
    pub dvol_index: AtomicU64,
    pub options_25_delta_skew: AtomicU64,
    pub put_call_ratio: AtomicU64,
    pub max_pain_price: AtomicU64,
    pub top_traders_pos_accounts: AtomicU64,
    pub top_traders_pos_volume: AtomicU64,
    pub stablecoin_supply_ratio: AtomicU64,
    pub margin_debt_ratio: AtomicU64,
    pub etf_net_inflows: AtomicU64,
    pub micro_volatility: AtomicU64,
    pub wb_us_m2_supply: AtomicU64,
    pub wb_us_cpi_inflation: AtomicU64,
    pub wb_us_real_interest: AtomicU64,
    pub wb_global_gdp_growth: AtomicU64,
    /// F2.4: timestamp del último fetch macro exitoso. Un valor viejo ⇒ las
    /// features macro están CONGELADAS y deben exponerse como staleness,
    /// jamás usarse en silencio como si estuvieran vivas.
    pub macro_last_success_ms: AtomicU64,
}

impl Default for OmniState {
    fn default() -> Self {
        Self::new()
    }
}

impl OmniState {
    pub fn new() -> Self {
        Self {
            binance_spot: AtomicU64::new(0.0_f64.to_bits()),
            binance_futures: AtomicU64::new(0.0_f64.to_bits()),
            bybit_linear: AtomicU64::new(0.0_f64.to_bits()),
            okx_swap: AtomicU64::new(0.0_f64.to_bits()),
            bitget_futures: AtomicU64::new(0.0_f64.to_bits()),
            coinbase_spot: AtomicU64::new(0.0_f64.to_bits()),
            kraken_spot: AtomicU64::new(0.0_f64.to_bits()),
            htx_spot: AtomicU64::new(0.0_f64.to_bits()),
            deribit_options: AtomicU64::new(0.0_f64.to_bits()),
            bitfinex_spot: AtomicU64::new(0.0_f64.to_bits()),
            binance_liquidations: AtomicU64::new(0.0_f64.to_bits()),
            agg_funding_rate: AtomicU64::new(0.0_f64.to_bits()),
            agg_open_interest: AtomicU64::new(0.0_f64.to_bits()),
            long_short_ratio: AtomicU64::new(0.0_f64.to_bits()),
            fear_greed_index: AtomicU64::new(50.0_f64.to_bits()),
            altcoin_dominance: AtomicU64::new(0.0_f64.to_bits()),
            mempool_congestion: AtomicU64::new(0.0_f64.to_bits()),
            usdt_mint_alert: AtomicU64::new(0.0_f64.to_bits()),
            exchange_inflows: AtomicU64::new(0.0_f64.to_bits()),
            exchange_outflows: AtomicU64::new(0.0_f64.to_bits()),
            whale_alert_proxy: AtomicU64::new(0.0_f64.to_bits()),
            dxy: AtomicU64::new(104.0_f64.to_bits()),
            sp500: AtomicU64::new(5100.0_f64.to_bits()),
            nasdaq: AtomicU64::new(18000.0_f64.to_bits()),
            vix: AtomicU64::new(15.0_f64.to_bits()),
            us10y: AtomicU64::new(4.2_f64.to_bits()),
            gold: AtomicU64::new(2300.0_f64.to_bits()),
            oil_wti: AtomicU64::new(80.0_f64.to_bits()),
            econ_calendar_impact: AtomicU64::new(0.0_f64.to_bits()),
            fed_interest_rate: AtomicU64::new(5.5_f64.to_bits()),
            spot_cvd: AtomicU64::new(0.0_f64.to_bits()),
            futures_cvd: AtomicU64::new(0.0_f64.to_bits()),
            taker_buy_sell_ratio: AtomicU64::new(1.0_f64.to_bits()),
            futures_basis_premium: AtomicU64::new(0.0_f64.to_bits()),
            liq_cluster_shorts: AtomicU64::new(0.0_f64.to_bits()),
            liq_cluster_longs: AtomicU64::new(0.0_f64.to_bits()),
            cme_futures_premium: AtomicU64::new(0.0_f64.to_bits()),
            cme_gap_proximity: AtomicU64::new(0.0_f64.to_bits()),
            spot_futures_arb_spread: AtomicU64::new(0.0_f64.to_bits()),
            order_flow_imbalance: AtomicU64::new(0.0_f64.to_bits()),
            dvol_index: AtomicU64::new(50.0_f64.to_bits()),
            options_25_delta_skew: AtomicU64::new(0.0_f64.to_bits()),
            put_call_ratio: AtomicU64::new(0.8_f64.to_bits()),
            max_pain_price: AtomicU64::new(0.0_f64.to_bits()),
            top_traders_pos_accounts: AtomicU64::new(1.0_f64.to_bits()),
            top_traders_pos_volume: AtomicU64::new(1.0_f64.to_bits()),
            stablecoin_supply_ratio: AtomicU64::new(0.0_f64.to_bits()),
            margin_debt_ratio: AtomicU64::new(0.0_f64.to_bits()),
            etf_net_inflows: AtomicU64::new(0.0_f64.to_bits()),
            micro_volatility: AtomicU64::new(0.0_f64.to_bits()),
            wb_us_m2_supply: AtomicU64::new(20000.0_f64.to_bits()),
            wb_us_cpi_inflation: AtomicU64::new(3.2_f64.to_bits()),
            wb_us_real_interest: AtomicU64::new(2.3_f64.to_bits()),
            wb_global_gdp_growth: AtomicU64::new(2.5_f64.to_bits()),
            macro_last_success_ms: AtomicU64::new(0),
        }
    }

    pub fn get_features(&self) -> [f64; 54] {
        let b_spot = f64::from_bits(self.binance_spot.load(Ordering::Relaxed));
        let ref_p = if b_spot.is_finite() && b_spot > 0.0 { b_spot } else { 1.0 };

        // L-0: Normalizar precios nominales cross-exchange a spreads porcentuales [-5.0, 5.0]
        // para erradicar la saturación del scaler Z-Score en DarkAlphaEngine (Causa Forense #D96).
        let norm_spread = |val_bits: u64| -> f64 {
            let p = f64::from_bits(val_bits);
            if p.is_finite() && p > 0.0 {
                (((p - ref_p) / ref_p) * 100.0).clamp(-5.0, 5.0)
            } else {
                0.0
            }
        };

        let mut feats = [
            0.0, // Referencia base Binance Spot (retorno relativo = 0.0)
            norm_spread(self.binance_futures.load(Ordering::Relaxed)),
            norm_spread(self.bybit_linear.load(Ordering::Relaxed)),
            norm_spread(self.okx_swap.load(Ordering::Relaxed)),
            norm_spread(self.bitget_futures.load(Ordering::Relaxed)),
            norm_spread(self.coinbase_spot.load(Ordering::Relaxed)),
            norm_spread(self.kraken_spot.load(Ordering::Relaxed)),
            norm_spread(self.htx_spot.load(Ordering::Relaxed)),
            norm_spread(self.deribit_options.load(Ordering::Relaxed)),
            norm_spread(self.bitfinex_spot.load(Ordering::Relaxed)),
            f64::from_bits(self.binance_liquidations.load(Ordering::Relaxed)),
            f64::from_bits(self.agg_funding_rate.load(Ordering::Relaxed)),
            f64::from_bits(self.agg_open_interest.load(Ordering::Relaxed)),
            f64::from_bits(self.long_short_ratio.load(Ordering::Relaxed)),
            f64::from_bits(self.fear_greed_index.load(Ordering::Relaxed)),
            f64::from_bits(self.altcoin_dominance.load(Ordering::Relaxed)),
            f64::from_bits(self.mempool_congestion.load(Ordering::Relaxed)),
            f64::from_bits(self.usdt_mint_alert.load(Ordering::Relaxed)),
            f64::from_bits(self.exchange_inflows.load(Ordering::Relaxed)),
            f64::from_bits(self.exchange_outflows.load(Ordering::Relaxed)),
            f64::from_bits(self.whale_alert_proxy.load(Ordering::Relaxed)),
            f64::from_bits(self.dxy.load(Ordering::Relaxed)),
            f64::from_bits(self.sp500.load(Ordering::Relaxed)),
            f64::from_bits(self.nasdaq.load(Ordering::Relaxed)),
            f64::from_bits(self.vix.load(Ordering::Relaxed)),
            f64::from_bits(self.us10y.load(Ordering::Relaxed)),
            f64::from_bits(self.gold.load(Ordering::Relaxed)),
            f64::from_bits(self.oil_wti.load(Ordering::Relaxed)),
            f64::from_bits(self.econ_calendar_impact.load(Ordering::Relaxed)),
            f64::from_bits(self.fed_interest_rate.load(Ordering::Relaxed)),
            f64::from_bits(self.spot_cvd.load(Ordering::Relaxed)),
            f64::from_bits(self.futures_cvd.load(Ordering::Relaxed)),
            f64::from_bits(self.taker_buy_sell_ratio.load(Ordering::Relaxed)),
            f64::from_bits(self.futures_basis_premium.load(Ordering::Relaxed)),
            f64::from_bits(self.liq_cluster_shorts.load(Ordering::Relaxed)),
            f64::from_bits(self.liq_cluster_longs.load(Ordering::Relaxed)),
            f64::from_bits(self.cme_futures_premium.load(Ordering::Relaxed)),
            f64::from_bits(self.cme_gap_proximity.load(Ordering::Relaxed)),
            f64::from_bits(self.spot_futures_arb_spread.load(Ordering::Relaxed)),
            f64::from_bits(self.order_flow_imbalance.load(Ordering::Relaxed)),
            f64::from_bits(self.dvol_index.load(Ordering::Relaxed)),
            f64::from_bits(self.options_25_delta_skew.load(Ordering::Relaxed)),
            f64::from_bits(self.put_call_ratio.load(Ordering::Relaxed)),
            f64::from_bits(self.max_pain_price.load(Ordering::Relaxed)),
            f64::from_bits(self.top_traders_pos_accounts.load(Ordering::Relaxed)),
            f64::from_bits(self.top_traders_pos_volume.load(Ordering::Relaxed)),
            f64::from_bits(self.stablecoin_supply_ratio.load(Ordering::Relaxed)),
            f64::from_bits(self.margin_debt_ratio.load(Ordering::Relaxed)),
            f64::from_bits(self.etf_net_inflows.load(Ordering::Relaxed)),
            f64::from_bits(self.micro_volatility.load(Ordering::Relaxed)),
            f64::from_bits(self.wb_us_m2_supply.load(Ordering::Relaxed)),
            f64::from_bits(self.wb_us_cpi_inflation.load(Ordering::Relaxed)),
            f64::from_bits(self.wb_us_real_interest.load(Ordering::Relaxed)),
            f64::from_bits(self.wb_global_gdp_growth.load(Ordering::Relaxed)),
        ];
        // FIX #1462: Sanitización de los 54 features macro
        for f in feats.iter_mut() {
            if !f.is_finite() {
                *f = 0.0;
            }
        }
        feats
    }
}

pub async fn run_bybit_ws(state: Arc<OmniState>, symbol: String) {
    let url = "wss://stream.bybit.com/v5/public/linear";
    // FIX #1462: Parseo seguro de URL sin unwrap frágil
    let url_parsed = match url::Url::parse(url) {
        Ok(u) => u,
        Err(_) => return,
    };
    let host = url_parsed.host_str().unwrap_or("stream.bybit.com");
    let port = url_parsed.port_or_known_default().unwrap_or(443);

    loop {
        if let Ok(target_addr) = tokio::net::lookup_host((host, port))
            .await
            .and_then(|mut iter| {
                iter.next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "DNS"))
            })
        {
            if let Ok(tcp_stream) = tokio::net::TcpStream::connect(target_addr).await {
                let _ = tcp_stream.set_nodelay(true);
                let std_stream = match tcp_stream.into_std() {
                    Ok(s) => s,
                    Err(e) => {
                        println!("⚠️ [BYBIT WS] Failed into_std: {}", e);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                };
                let socket = socket2::Socket::from(std_stream);
                let _ = socket.set_recv_buffer_size(1024 * 1024 * 4); // 4MB
                let keepalive =
                    socket2::TcpKeepalive::new().with_time(std::time::Duration::from_secs(30));
                let _ = socket.set_tcp_keepalive(&keepalive);
                let tcp_stream = match tokio::net::TcpStream::from_std(socket.into()) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("⚠️ [BYBIT WS] Failed from_std: {}", e);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                };

                if let Ok((mut ws_stream, _)) =
                    tokio_tungstenite::client_async_tls(url_parsed.clone(), tcp_stream).await
                {
                    let symbol_upper = symbol.to_uppercase();
                    let msg = format!(
                        r#"{{"op": "subscribe", "args": ["orderbook.1.{}"]}}"#,
                        symbol_upper
                    );
                    let _ = futures_util::SinkExt::send(
                        &mut ws_stream,
                        tokio_tungstenite::tungstenite::Message::Text(msg),
                    )
                    .await;

                    // FIX #824: Manejar Ping/Pong y extender timeout para activos de baja liquidez
                    while let Ok(Some(msg_res)) = tokio::time::timeout(Duration::from_secs(60), ws_stream.next()).await {
                        match msg_res {
                            Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                                if let Ok(json) = serde_json::from_str::<Value>(&text) {
                                    if let Some(price_str) = json
                                        .get("data")
                                        .and_then(|d| d.get("b"))
                                        .and_then(|b| b.as_array()?.first())
                                        .and_then(|f| f.as_array()?.first())
                                        .and_then(|v| v.as_str())
                                    {
                                        if let Ok(p) = price_str.parse::<f64>() {
                                            state.bybit_linear.store(p.to_bits(), Ordering::Relaxed);
                                        }
                                    }
                                }
                            }
                            Ok(tokio_tungstenite::tungstenite::Message::Ping(payload)) => {
                                let _ = futures_util::SinkExt::send(
                                    &mut ws_stream,
                                    tokio_tungstenite::tungstenite::Message::Pong(payload),
                                ).await;
                            }
                            Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => break,
                            Err(_) => break,
                            _ => {}
                        }
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

pub async fn run_okx_ws(state: Arc<OmniState>, symbol: String) {
    let url = "wss://ws.okx.com:8443/ws/v5/public";
    // OKX Perp Swap usa BTC-USDT-SWAP en lugar de BTCUSDT
    let symbol_clean = symbol.to_uppercase();
    let okx_symbol = if symbol_clean.ends_with("-SWAP") {
        symbol_clean
    } else {
        format!("{}-SWAP", symbol_clean.replace("USDT", "-USDT"))
    };
    let url_parsed = match url::Url::parse(url) {
        Ok(u) => u,
        Err(e) => {
            println!("⚠️ [OKX WS] Bad URL: {}", e);
            return;
        }
    };
    let host = url_parsed.host_str().unwrap_or("ws.okx.com");
    let port = url_parsed.port_or_known_default().unwrap_or(8443);

    loop {
        if let Ok(target_addr) = tokio::net::lookup_host((host, port))
            .await
            .and_then(|mut iter| {
                iter.next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "DNS"))
            })
        {
            if let Ok(tcp_stream) = tokio::net::TcpStream::connect(target_addr).await {
                let _ = tcp_stream.set_nodelay(true);
                let std_stream = match tcp_stream.into_std() {
                    Ok(s) => s,
                    Err(e) => {
                        println!("⚠️ [OKX WS] Failed into_std: {}", e);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                };
                let socket = socket2::Socket::from(std_stream);
                let _ = socket.set_recv_buffer_size(1024 * 1024 * 4);
                let keepalive =
                    socket2::TcpKeepalive::new().with_time(std::time::Duration::from_secs(30));
                let _ = socket.set_tcp_keepalive(&keepalive);
                let tcp_stream = match tokio::net::TcpStream::from_std(socket.into()) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("⚠️ [OKX WS] Failed from_std: {}", e);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                };

                if let Ok((mut ws_stream, _)) =
                    tokio_tungstenite::client_async_tls(url_parsed.clone(), tcp_stream).await
                {
                    let msg = format!(
                        r#"{{"op": "subscribe", "args": [{{"channel": "bbo-tbt", "instId": "{}"}}]}}"#,
                        okx_symbol
                    );
                    let _ = futures_util::SinkExt::send(
                        &mut ws_stream,
                        tokio_tungstenite::tungstenite::Message::Text(msg),
                    )
                    .await;

                    // FIX #824: Manejar Ping/Pong y extender timeout para OKX
                    while let Ok(Some(msg_res)) = tokio::time::timeout(Duration::from_secs(60), ws_stream.next()).await {
                        match msg_res {
                            Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                                if let Ok(json) = serde_json::from_str::<Value>(&text) {
                                    if let Some(price_str) = json
                                        .get("data")
                                        .and_then(|d| d.as_array()?.first())
                                        .and_then(|f| f.get("bids"))
                                        .and_then(|b| b.as_array()?.first())
                                        .and_then(|f| f.as_array()?.first())
                                        .and_then(|v| v.as_str())
                                    {
                                        if let Ok(p) = price_str.parse::<f64>() {
                                            state.okx_swap.store(p.to_bits(), Ordering::Relaxed);
                                        }
                                    }
                                }
                            }
                            Ok(tokio_tungstenite::tungstenite::Message::Ping(payload)) => {
                                let _ = futures_util::SinkExt::send(
                                    &mut ws_stream,
                                    tokio_tungstenite::tungstenite::Message::Pong(payload),
                                ).await;
                            }
                            Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => break,
                            Err(_) => break,
                            _ => {}
                        }
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

/// F2.4 — MACRO FEED VIVO (FRED + Binance PAXG).
/// El endpoint Yahoo v7 murió y fallaba EN SILENCIO: DXY/VIX/S&P quedaban
/// congelados desde hace meses mientras las features los consumían como
/// vivos. Fuentes validadas en vivo (2026-08-16):
///   - FRED (Reserva Federal, CSV sin auth, actualización diaria):
///     SP500, NASDAQCOM, VIXCLS, DGS10, DTWEXBGS (dólar trade-weighted —
///     no es el ICE DXY pero es el estándar de la Fed), DCOILWTICO.
///   - Oro: PAXG/USDT de Binance (proxy on-chain del oro, minuto a minuto,
///     más fresco que cualquier fix diario).
/// Éxito actualiza macro_last_success_ms → staleness medible (F4.1 lo
/// cablea como feature). Fallo LOGUEA (cada 10º) — jamás congelamiento mudo.
pub async fn run_macro_rest_poller(state: Arc<OmniState>) {
    let mut ticker = interval(Duration::from_secs(60));
    let client = reqwest::Client::new();
    let mut fail_count: u64 = 0;

    let fred_series: [(&str, &AtomicU64); 6] = [
        ("SP500", &state.sp500),
        ("NASDAQCOM", &state.nasdaq),
        ("VIXCLS", &state.vix),
        ("DGS10", &state.us10y),
        ("DTWEXBGS", &state.dxy),
        ("DCOILWTICO", &state.oil_wti),
    ];

    loop {
        ticker.tick().await;
        let mut updated = 0usize;

        for (series, slot) in &fred_series {
            let url = format!(
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}",
                series
            );
            if let Ok(res) = client.get(&url).send().await {
                if let Ok(csv) = res.text().await {
                    // Última fila con valor válido ("." = sin dato ese día).
                    if let Some((_, val)) = csv
                        .lines()
                        .skip(1)
                        .filter_map(|l| {
                            let mut parts = l.split(',');
                            let d = parts.next()?.trim();
                            let v = parts.next()?.trim();
                            let f = v.parse::<f64>().ok()?;
                            Some((d, f))
                        })
                        .last()
                    {
                        slot.store(val.to_bits(), Ordering::Relaxed);
                        updated += 1;
                    }
                }
            }
        }

        // Oro on-chain: PAXG/USDT (público, sin firma).
        let is_testnet = std::env::var("USE_TESTNET")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            == "true";
        let base = if is_testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        };
        if let Ok(res) = client
            .get(format!("{}/fapi/v1/ticker/price?symbol=PAXGUSDT", base))
            .send()
            .await
        {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(p) = json["price"].as_str().and_then(|s| s.parse::<f64>().ok()) {
                    state.gold.store(p.to_bits(), Ordering::Relaxed);
                    updated += 1;
                }
            }
        }

        if updated > 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            state.macro_last_success_ms.store(now, Ordering::Relaxed);
            fail_count = 0;
        } else {
            fail_count += 1;
            if fail_count % 10 == 1 {
                println!(
                    "⚠️ [MACRO] FRED/PAXG sin datos utilizables (fallo #{fail_count}) — features macro con staleness creciente"
                );
            }
        }
    }
}

pub async fn run_sentiment_onchain_poller(state: Arc<OmniState>) {
    let mut ticker = interval(Duration::from_secs(120));
    let client = reqwest::Client::new();
    let fear_url = "https://api.alternative.me/fng/";
    let is_testnet = std::env::var("USE_TESTNET")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        == "true";
    let base_url = if is_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };
    let funding_url = format!("{}/fapi/v1/premiumIndex?symbol=BTCUSDT", base_url);

    loop {
        ticker.tick().await;
        if let Ok(res) = client.get(fear_url).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(data) = json.get("data").and_then(|d| d.as_array()?.first()) {
                    if let Some(val_str) = data.get("value").and_then(|v| v.as_str()) {
                        if let Ok(val) = val_str.parse::<f64>() {
                            if val.is_finite() {
                                // FIX #1537: Clamping de índice de miedo/codicia a [0.0, 100.0]
                                let safe_fg = val.clamp(0.0, 100.0);
                                state
                                    .fear_greed_index
                                    .store(safe_fg.to_bits(), Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
        }
        if let Ok(res) = client.get(&funding_url).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(funding_str) = json.get("lastFundingRate").and_then(|v| v.as_str()) {
                    if let Ok(f) = funding_str.parse::<f64>() {
                        if f.is_finite() {
                            // FIX #1537: Clamping de tasa de fondeo a [-1.0, 1.0]
                            let safe_funding = f.clamp(-1.0, 1.0);
                            state.agg_funding_rate.store(safe_funding.to_bits(), Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }
}

pub async fn run_world_bank_poller(state: Arc<OmniState>) {
    let mut ticker = interval(Duration::from_secs(3600));
    let client = reqwest::Client::new();

    let cpi_url = "https://api.worldbank.org/v2/country/USA/indicator/FP.CPI.TOTL.ZG?format=json";
    // R2.3 — indicador CORRECTO: FM.LBL.BMNY.CD es "Broad money (current LCU)"
    // (= M2 en USD para USA, ~21e12). El anterior (FM.LBL.BMNY.GD.ZS) es
    // "Broad money, % of GDP" (~90): tras el primer poll, wb_us_m2_supply
    // saltaba 2 órdenes de magnitud rompiendo la normalización del tensor.
    let m2_url = "https://api.worldbank.org/v2/country/USA/indicator/FM.LBL.BMNY.CD?format=json&per_page=10";

    loop {
        ticker.tick().await;
        if let Ok(res) = client.get(cpi_url).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(arr) = json.as_array() {
                    if arr.len() > 1 {
                        if let Some(data) = arr[1].as_array() {
                            for item in data {
                                if let Some(val) = item.get("value").and_then(|v| v.as_f64()) {
                                    state
                                        .wb_us_cpi_inflation
                                        .store(val.to_bits(), Ordering::Relaxed);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Ok(res) = client.get(m2_url).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(arr) = json.as_array() {
                    if arr.len() > 1 {
                        if let Some(data) = arr[1].as_array() {
                            for item in data {
                                if let Some(val) = item.get("value").and_then(|v| v.as_f64()) {
                                    // R2.3: FM.LBL.BMNY.CD llega en USD corrientes
                                    // (~21.4e12); la escala canónica de la feature es
                                    // MILES DE MILLONES (~20800, igual que el tensor de
                                    // backtest y el default de arranque).
                                    let billions = (val / 1e9).max(0.0);
                                    state
                                        .wb_us_m2_supply
                                        .store(billions.to_bits(), Ordering::Relaxed);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct OmniDataHub {
    pub omni_state: Arc<OmniState>,
}

impl Default for OmniDataHub {
    fn default() -> Self {
        Self::new()
    }
}

impl OmniDataHub {
    pub fn new() -> Self {
        Self {
            omni_state: Arc::new(OmniState::new()),
        }
    }

    pub fn start_feeds(&self, target_symbol: String) {
        let state = Arc::clone(&self.omni_state);
        let symbol = target_symbol.clone();

        // 1. Supervisor Bybit WS con auto-reconexión
        {
            let state = Arc::clone(&state);
            let sym = symbol.clone();
            tokio::spawn(async move {
                let mut backoff = 1u64;
                loop {
                    run_bybit_ws(Arc::clone(&state), sym.clone()).await;
                    println!("⚠️ [OMNI MULTIPLEXER] Bybit WS reconectando en {}s...", backoff);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(30);
                }
            });
        }

        // 2. Supervisor OKX WS con auto-reconexión
        {
            let state = Arc::clone(&state);
            let sym = symbol.clone();
            tokio::spawn(async move {
                let mut backoff = 1u64;
                loop {
                    run_okx_ws(Arc::clone(&state), sym.clone()).await;
                    println!("⚠️ [OMNI MULTIPLEXER] OKX WS reconectando en {}s...", backoff);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(30);
                }
            });
        }

        // 3. Supervisor Macro REST Poller
        {
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                let mut backoff = 5u64;
                loop {
                    run_macro_rest_poller(Arc::clone(&state)).await;
                    println!("⚠️ [OMNI MULTIPLEXER] Macro REST Poller reiniciando en {}s...", backoff);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(60);
                }
            });
        }

        // 4. Supervisor Sentiment Onchain Poller
        {
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                let mut backoff = 5u64;
                loop {
                    run_sentiment_onchain_poller(Arc::clone(&state)).await;
                    println!("⚠️ [OMNI MULTIPLEXER] Sentiment Poller reiniciando en {}s...", backoff);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(60);
                }
            });
        }

        // 5. Supervisor World Bank Poller
        {
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                let mut backoff = 30u64;
                loop {
                    run_world_bank_poller(Arc::clone(&state)).await;
                    println!("⚠️ [OMNI MULTIPLEXER] World Bank Poller reiniciando en {}s...", backoff);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(300);
                }
            });
        }

        // 6. Lakehouse Telemetry MMap
        {
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                spawn_lakehouse_telemetry(state).await;
            });
        }
    }
}

/// FASE 8 & 11: Data Lakehouse (Tensor Persistence)
/// Zero-Copy MMap Implementation.
/// Evita la latencia I/O bloqueante de Windows y escribe los tensores directamente a RAM mapeada.
pub async fn spawn_lakehouse_telemetry(state: Arc<OmniState>) {
    use memmap2::MmapOptions;
    use std::fs::OpenOptions;

    let path = "data/lakehouse";
    if std::fs::metadata(path).is_err() {
        std::fs::create_dir_all(path).unwrap_or_default();
    }

    // Allocate a 1GB file for continuous tensor dumps (approx 2.3 million ticks)
    let file_path = "data/lakehouse/omni_telemetry_mmap.bin";
    let file = match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(file_path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("⚠️ [LAKEHOUSE] No se pudo abrir omni_telemetry_mmap.bin: {}", e);
            return;
        }
    };

    if file.metadata().map(|m| m.len()).unwrap_or(0) < 1024 * 1024 * 1024 {
        let _ = file.set_len(1024 * 1024 * 1024); // 1GB
    }

    let mut mmap = match unsafe { MmapOptions::new().map_mut(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("⚠️ [LAKEHOUSE] No se pudo mapear memoria para lakehouse: {}", e);
            return;
        }
    };
    let mut offset = 0usize;
    let tensor_size = 8 + 54 * 8; // Timestamp (u64) + 54 f64 features

    let mut ticker = interval(Duration::from_millis(250));
    let mut last_features = [0.0; 54];
    let mut last_flush_time = std::time::Instant::now();

    loop {
        ticker.tick().await;

        let features = state.get_features();
        let mut max_delta = 0.0f64;
        for i in 0..54 {
            let diff = (features[i] - last_features[i]).abs();
            if diff > max_delta {
                max_delta = diff;
            }
        }

        // FASE 8: Only flush if tensor changed significantly (delta > 1e-6) or 5-second heartbeat
        let should_flush = max_delta > 1e-6 || last_flush_time.elapsed().as_secs() >= 5;
        if !should_flush {
            continue;
        }

        last_features = features;
        last_flush_time = std::time::Instant::now();

        if offset + tensor_size > mmap.len() {
            // Buffer full. Rotate circular pointer in mmap region
            offset = 0;
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        mmap[offset..offset + 8].copy_from_slice(&timestamp.to_le_bytes());
        offset += 8;

        for f in features.iter() {
            mmap[offset..offset + 8].copy_from_slice(&f.to_bits().to_le_bytes());
            offset += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omni_state_initial_features_count_and_finiteness() {
        let state = OmniState::new();
        let feats = state.get_features();
        assert_eq!(feats.len(), 54);
        for (idx, &f) in feats.iter().enumerate() {
            assert!(f.is_finite(), "Feature {} debe ser finita", idx);
        }
    }

    #[test]
    fn test_omni_state_macro_staleness_flag() {
        let state = OmniState::new();
        assert_eq!(state.macro_last_success_ms.load(Ordering::Relaxed), 0);

        state.macro_last_success_ms.store(1700000000000, Ordering::Relaxed);
        assert_eq!(state.macro_last_success_ms.load(Ordering::Relaxed), 1700000000000);
    }
}

