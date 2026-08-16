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
        }
    }

    pub fn get_features(&self) -> [f64; 54] {
        [
            f64::from_bits(self.binance_spot.load(Ordering::Relaxed)),
            f64::from_bits(self.binance_futures.load(Ordering::Relaxed)),
            f64::from_bits(self.bybit_linear.load(Ordering::Relaxed)),
            f64::from_bits(self.okx_swap.load(Ordering::Relaxed)),
            f64::from_bits(self.bitget_futures.load(Ordering::Relaxed)),
            f64::from_bits(self.coinbase_spot.load(Ordering::Relaxed)),
            f64::from_bits(self.kraken_spot.load(Ordering::Relaxed)),
            f64::from_bits(self.htx_spot.load(Ordering::Relaxed)),
            f64::from_bits(self.deribit_options.load(Ordering::Relaxed)),
            f64::from_bits(self.bitfinex_spot.load(Ordering::Relaxed)),
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
        ]
    }
}

pub async fn run_bybit_ws(state: Arc<OmniState>, symbol: String) {
    let url = "wss://stream.bybit.com/v5/public/linear";
    let url_parsed = url::Url::parse(url).unwrap();
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
                let socket = socket2::Socket::from(tcp_stream.into_std().unwrap());
                let _ = socket.set_recv_buffer_size(1024 * 1024 * 4); // 4MB
                let keepalive =
                    socket2::TcpKeepalive::new().with_time(std::time::Duration::from_secs(30));
                let _ = socket.set_tcp_keepalive(&keepalive);
                let tcp_stream = tokio::net::TcpStream::from_std(socket.into()).unwrap();

                if let Ok((mut ws_stream, _)) =
                    tokio_tungstenite::client_async_tls(url_parsed.clone(), tcp_stream).await
                {
                    let msg = format!(
                        r#"{{"op": "subscribe", "args": ["orderbook.1.{}"]}}"#,
                        symbol
                    );
                    let _ = futures_util::SinkExt::send(
                        &mut ws_stream,
                        tokio_tungstenite::tungstenite::Message::Text(msg),
                    )
                    .await;

                    while let Ok(Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text)))) =
                        tokio::time::timeout(Duration::from_secs(30), ws_stream.next()).await
                    {
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
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

pub async fn run_okx_ws(state: Arc<OmniState>, symbol: String) {
    let url = "wss://ws.okx.com:8443/ws/v5/public";
    // OKX usa BTC-USDT en lugar de BTCUSDT
    let okx_symbol = symbol.replace("USDT", "-USDT");
    let url_parsed = url::Url::parse(url).unwrap();
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
                let socket = socket2::Socket::from(tcp_stream.into_std().unwrap());
                let _ = socket.set_recv_buffer_size(1024 * 1024 * 4);
                let keepalive =
                    socket2::TcpKeepalive::new().with_time(std::time::Duration::from_secs(30));
                let _ = socket.set_tcp_keepalive(&keepalive);
                let tcp_stream = tokio::net::TcpStream::from_std(socket.into()).unwrap();

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

                    while let Ok(Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text)))) =
                        tokio::time::timeout(Duration::from_secs(30), ws_stream.next()).await
                    {
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
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

pub async fn run_macro_rest_poller(state: Arc<OmniState>) {
    let mut ticker = interval(Duration::from_secs(60));
    let client = reqwest::Client::new();
    let url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=DX-Y.NYB,^GSPC,^NDX,^VIX,GC=F,CL=F,^TNX";

    loop {
        ticker.tick().await;
        if let Ok(res) = client.get(url).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(results) = json["quoteResponse"]["result"].as_array() {
                    for item in results {
                        let symbol = item["symbol"].as_str().unwrap_or("");
                        let price = item["regularMarketPrice"].as_f64().unwrap_or(0.0);
                        match symbol {
                            "DX-Y.NYB" => state.dxy.store(price.to_bits(), Ordering::Relaxed),
                            "^GSPC" => state.sp500.store(price.to_bits(), Ordering::Relaxed),
                            "^NDX" => state.nasdaq.store(price.to_bits(), Ordering::Relaxed),
                            "^VIX" => state.vix.store(price.to_bits(), Ordering::Relaxed),
                            "GC=F" => state.gold.store(price.to_bits(), Ordering::Relaxed),
                            "CL=F" => state.oil_wti.store(price.to_bits(), Ordering::Relaxed),
                            "^TNX" => state.us10y.store(price.to_bits(), Ordering::Relaxed),
                            _ => {}
                        }
                    }
                }
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
                            state
                                .fear_greed_index
                                .store(val.to_bits(), Ordering::Relaxed);
                        }
                    }
                }
            }
        }
        if let Ok(res) = client.get(&funding_url).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(funding_str) = json.get("lastFundingRate").and_then(|v| v.as_str()) {
                    if let Ok(f) = funding_str.parse::<f64>() {
                        state.agg_funding_rate.store(f.to_bits(), Ordering::Relaxed);
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
    let m2_url = "https://api.worldbank.org/v2/country/USA/indicator/FM.LBL.BMNY.GD.ZS?format=json";

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
                                    state
                                        .wb_us_m2_supply
                                        .store(val.to_bits(), Ordering::Relaxed);
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

        let mut join_set = tokio::task::JoinSet::new();

        join_set.spawn(run_bybit_ws(Arc::clone(&state), symbol.clone()));
        join_set.spawn(run_okx_ws(Arc::clone(&state), symbol.clone()));
        join_set.spawn(run_macro_rest_poller(Arc::clone(&state)));
        join_set.spawn(run_sentiment_onchain_poller(Arc::clone(&state)));

        // FASE 8: Institutional Lakehouse Persistencia Binaria Continua
        join_set.spawn(spawn_lakehouse_telemetry(Arc::clone(&state)));

        // FASE 13: Supervisor Cuántico de Tareas
        // Evita fugas de memoria y tareas fantasma al recolectar los hilos
        tokio::spawn(async move {
            while let Some(res) = join_set.join_next().await {
                if let Err(e) = res {
                    println!(
                        "⚠️ [OMNI MULTIPLEXER] Tarea de alimentación de datos colapsada: {:?}",
                        e
                    );
                }
            }
            println!("🛑 [OMNI MULTIPLEXER] Todas las conexiones externas han finalizado.");
        });
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
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
        .unwrap();

    let _ = file.set_len(1024 * 1024 * 1024); // 1GB

    let mut mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
    let mut offset = 0usize;
    let tensor_size = 8 + 54 * 8; // Timestamp (u64) + 54 f64 features

    let mut ticker = interval(Duration::from_millis(500));

    loop {
        ticker.tick().await;

        if offset + tensor_size > mmap.len() {
            // Buffer full. In a production system, we would rotate the file here and hand off to Polars Parquet compressor.
            offset = 0;
        }

        let features = state.get_features();
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
