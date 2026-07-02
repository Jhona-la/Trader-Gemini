use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use futures_util::StreamExt;
use tokio_tungstenite::connect_async;
use serde_json::Value;

pub struct OmniState {
    pub binance_spot: AtomicU64, pub binance_futures: AtomicU64, pub bybit_linear: AtomicU64,
    pub okx_swap: AtomicU64, pub bitget_futures: AtomicU64, pub coinbase_spot: AtomicU64,
    pub kraken_spot: AtomicU64, pub htx_spot: AtomicU64, pub deribit_options: AtomicU64,
    pub bitfinex_spot: AtomicU64, pub binance_liquidations: AtomicU64, pub agg_funding_rate: AtomicU64,
    pub agg_open_interest: AtomicU64, pub long_short_ratio: AtomicU64, pub fear_greed_index: AtomicU64,
    pub altcoin_dominance: AtomicU64, pub mempool_congestion: AtomicU64, pub usdt_mint_alert: AtomicU64,
    pub exchange_inflows: AtomicU64, pub exchange_outflows: AtomicU64, pub whale_alert_proxy: AtomicU64,
    pub dxy: AtomicU64, pub sp500: AtomicU64, pub nasdaq: AtomicU64, pub vix: AtomicU64,
    pub us10y: AtomicU64, pub gold: AtomicU64, pub oil_wti: AtomicU64, pub econ_calendar_impact: AtomicU64,
    pub fed_interest_rate: AtomicU64, pub spot_cvd: AtomicU64, pub futures_cvd: AtomicU64,
    pub taker_buy_sell_ratio: AtomicU64, pub futures_basis_premium: AtomicU64, pub liq_cluster_shorts: AtomicU64,
    pub liq_cluster_longs: AtomicU64, pub cme_futures_premium: AtomicU64, pub cme_gap_proximity: AtomicU64,
    pub spot_futures_arb_spread: AtomicU64, pub order_flow_imbalance: AtomicU64, pub dvol_index: AtomicU64,
    pub options_25_delta_skew: AtomicU64, pub put_call_ratio: AtomicU64, pub max_pain_price: AtomicU64,
    pub top_traders_pos_accounts: AtomicU64, pub top_traders_pos_volume: AtomicU64, pub stablecoin_supply_ratio: AtomicU64,
    pub margin_debt_ratio: AtomicU64, pub etf_net_inflows: AtomicU64, pub micro_volatility: AtomicU64,
    pub wb_us_m2_supply: AtomicU64,
    pub wb_us_cpi_inflation: AtomicU64,
    pub wb_us_real_interest: AtomicU64,
    pub wb_global_gdp_growth: AtomicU64,
}

impl OmniState {
    pub fn new() -> Self {
        Self {
            binance_spot: AtomicU64::new(0.0_f64.to_bits()), binance_futures: AtomicU64::new(0.0_f64.to_bits()),
            bybit_linear: AtomicU64::new(0.0_f64.to_bits()), okx_swap: AtomicU64::new(0.0_f64.to_bits()),
            bitget_futures: AtomicU64::new(0.0_f64.to_bits()), coinbase_spot: AtomicU64::new(0.0_f64.to_bits()),
            kraken_spot: AtomicU64::new(0.0_f64.to_bits()), htx_spot: AtomicU64::new(0.0_f64.to_bits()),
            deribit_options: AtomicU64::new(0.0_f64.to_bits()), bitfinex_spot: AtomicU64::new(0.0_f64.to_bits()),
            binance_liquidations: AtomicU64::new(0.0_f64.to_bits()), agg_funding_rate: AtomicU64::new(0.0_f64.to_bits()),
            agg_open_interest: AtomicU64::new(0.0_f64.to_bits()), long_short_ratio: AtomicU64::new(0.0_f64.to_bits()),
            fear_greed_index: AtomicU64::new(50.0_f64.to_bits()), altcoin_dominance: AtomicU64::new(0.0_f64.to_bits()),
            mempool_congestion: AtomicU64::new(0.0_f64.to_bits()), usdt_mint_alert: AtomicU64::new(0.0_f64.to_bits()),
            exchange_inflows: AtomicU64::new(0.0_f64.to_bits()), exchange_outflows: AtomicU64::new(0.0_f64.to_bits()),
            whale_alert_proxy: AtomicU64::new(0.0_f64.to_bits()), dxy: AtomicU64::new(104.0_f64.to_bits()),
            sp500: AtomicU64::new(5100.0_f64.to_bits()), nasdaq: AtomicU64::new(18000.0_f64.to_bits()),
            vix: AtomicU64::new(15.0_f64.to_bits()), us10y: AtomicU64::new(4.2_f64.to_bits()),
            gold: AtomicU64::new(2300.0_f64.to_bits()), oil_wti: AtomicU64::new(80.0_f64.to_bits()),
            econ_calendar_impact: AtomicU64::new(0.0_f64.to_bits()), fed_interest_rate: AtomicU64::new(5.5_f64.to_bits()),
            spot_cvd: AtomicU64::new(0.0_f64.to_bits()), futures_cvd: AtomicU64::new(0.0_f64.to_bits()),
            taker_buy_sell_ratio: AtomicU64::new(1.0_f64.to_bits()), futures_basis_premium: AtomicU64::new(0.0_f64.to_bits()),
            liq_cluster_shorts: AtomicU64::new(0.0_f64.to_bits()), liq_cluster_longs: AtomicU64::new(0.0_f64.to_bits()),
            cme_futures_premium: AtomicU64::new(0.0_f64.to_bits()), cme_gap_proximity: AtomicU64::new(0.0_f64.to_bits()),
            spot_futures_arb_spread: AtomicU64::new(0.0_f64.to_bits()), order_flow_imbalance: AtomicU64::new(0.0_f64.to_bits()),
            dvol_index: AtomicU64::new(50.0_f64.to_bits()), options_25_delta_skew: AtomicU64::new(0.0_f64.to_bits()),
            put_call_ratio: AtomicU64::new(0.8_f64.to_bits()), max_pain_price: AtomicU64::new(0.0_f64.to_bits()),
            top_traders_pos_accounts: AtomicU64::new(1.0_f64.to_bits()), top_traders_pos_volume: AtomicU64::new(1.0_f64.to_bits()),
            stablecoin_supply_ratio: AtomicU64::new(0.0_f64.to_bits()), margin_debt_ratio: AtomicU64::new(0.0_f64.to_bits()),
            etf_net_inflows: AtomicU64::new(0.0_f64.to_bits()), micro_volatility: AtomicU64::new(0.0_f64.to_bits()),
            wb_us_m2_supply: AtomicU64::new(20000.0_f64.to_bits()), wb_us_cpi_inflation: AtomicU64::new(3.2_f64.to_bits()),
            wb_us_real_interest: AtomicU64::new(2.3_f64.to_bits()), wb_global_gdp_growth: AtomicU64::new(2.5_f64.to_bits()),
        }
    }

    pub fn get_features(&self) -> [f64; 54] {
        [
            f64::from_bits(self.binance_spot.load(Ordering::Relaxed)), f64::from_bits(self.binance_futures.load(Ordering::Relaxed)),
            f64::from_bits(self.bybit_linear.load(Ordering::Relaxed)), f64::from_bits(self.okx_swap.load(Ordering::Relaxed)),
            f64::from_bits(self.bitget_futures.load(Ordering::Relaxed)), f64::from_bits(self.coinbase_spot.load(Ordering::Relaxed)),
            f64::from_bits(self.kraken_spot.load(Ordering::Relaxed)), f64::from_bits(self.htx_spot.load(Ordering::Relaxed)),
            f64::from_bits(self.deribit_options.load(Ordering::Relaxed)), f64::from_bits(self.bitfinex_spot.load(Ordering::Relaxed)),
            f64::from_bits(self.binance_liquidations.load(Ordering::Relaxed)), f64::from_bits(self.agg_funding_rate.load(Ordering::Relaxed)),
            f64::from_bits(self.agg_open_interest.load(Ordering::Relaxed)), f64::from_bits(self.long_short_ratio.load(Ordering::Relaxed)),
            f64::from_bits(self.fear_greed_index.load(Ordering::Relaxed)), f64::from_bits(self.altcoin_dominance.load(Ordering::Relaxed)),
            f64::from_bits(self.mempool_congestion.load(Ordering::Relaxed)), f64::from_bits(self.usdt_mint_alert.load(Ordering::Relaxed)),
            f64::from_bits(self.exchange_inflows.load(Ordering::Relaxed)), f64::from_bits(self.exchange_outflows.load(Ordering::Relaxed)),
            f64::from_bits(self.whale_alert_proxy.load(Ordering::Relaxed)), f64::from_bits(self.dxy.load(Ordering::Relaxed)),
            f64::from_bits(self.sp500.load(Ordering::Relaxed)), f64::from_bits(self.nasdaq.load(Ordering::Relaxed)),
            f64::from_bits(self.vix.load(Ordering::Relaxed)), f64::from_bits(self.us10y.load(Ordering::Relaxed)),
            f64::from_bits(self.gold.load(Ordering::Relaxed)), f64::from_bits(self.oil_wti.load(Ordering::Relaxed)),
            f64::from_bits(self.econ_calendar_impact.load(Ordering::Relaxed)), f64::from_bits(self.fed_interest_rate.load(Ordering::Relaxed)),
            f64::from_bits(self.spot_cvd.load(Ordering::Relaxed)), f64::from_bits(self.futures_cvd.load(Ordering::Relaxed)),
            f64::from_bits(self.taker_buy_sell_ratio.load(Ordering::Relaxed)), f64::from_bits(self.futures_basis_premium.load(Ordering::Relaxed)),
            f64::from_bits(self.liq_cluster_shorts.load(Ordering::Relaxed)), f64::from_bits(self.liq_cluster_longs.load(Ordering::Relaxed)),
            f64::from_bits(self.cme_futures_premium.load(Ordering::Relaxed)), f64::from_bits(self.cme_gap_proximity.load(Ordering::Relaxed)),
            f64::from_bits(self.spot_futures_arb_spread.load(Ordering::Relaxed)), f64::from_bits(self.order_flow_imbalance.load(Ordering::Relaxed)),
            f64::from_bits(self.dvol_index.load(Ordering::Relaxed)), f64::from_bits(self.options_25_delta_skew.load(Ordering::Relaxed)),
            f64::from_bits(self.put_call_ratio.load(Ordering::Relaxed)), f64::from_bits(self.max_pain_price.load(Ordering::Relaxed)),
            f64::from_bits(self.top_traders_pos_accounts.load(Ordering::Relaxed)), f64::from_bits(self.top_traders_pos_volume.load(Ordering::Relaxed)),
            f64::from_bits(self.stablecoin_supply_ratio.load(Ordering::Relaxed)), f64::from_bits(self.margin_debt_ratio.load(Ordering::Relaxed)),
            f64::from_bits(self.etf_net_inflows.load(Ordering::Relaxed)), f64::from_bits(self.micro_volatility.load(Ordering::Relaxed)),
            f64::from_bits(self.wb_us_m2_supply.load(Ordering::Relaxed)), f64::from_bits(self.wb_us_cpi_inflation.load(Ordering::Relaxed)),
            f64::from_bits(self.wb_us_real_interest.load(Ordering::Relaxed)), f64::from_bits(self.wb_global_gdp_growth.load(Ordering::Relaxed)),
        ]
    }
}

pub async fn run_bybit_ws(state: Arc<OmniState>) {
    let url = "wss://stream.bybit.com/v5/public/linear";
    loop {
        if let Ok((mut ws_stream, _)) = connect_async(url).await {
            let msg = r#"{"op": "subscribe", "args": ["orderbook.1.BTCUSDT"]}"#;
            let _ = futures_util::SinkExt::send(&mut ws_stream, tokio_tungstenite::tungstenite::Message::Text(msg.into())).await;
            while let Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) = ws_stream.next().await {
                if let Ok(json) = serde_json::from_str::<Value>(&text) {
                    if let Some(price_str) = json.get("data").and_then(|d| d.get("b")).and_then(|b| b.as_array()?.first()).and_then(|f| f.as_array()?.first()).and_then(|v| v.as_str()) {
                        if let Ok(p) = price_str.parse::<f64>() { state.bybit_linear.store(p.to_bits(), Ordering::Relaxed); }
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

pub async fn run_okx_ws(state: Arc<OmniState>) {
    let url = "wss://ws.okx.com:8443/ws/v5/public";
    loop {
        if let Ok((mut ws_stream, _)) = connect_async(url).await {
            let msg = r#"{"op": "subscribe", "args": [{"channel": "bbo-tbt", "instId": "BTC-USDT"}]}"#;
            let _ = futures_util::SinkExt::send(&mut ws_stream, tokio_tungstenite::tungstenite::Message::Text(msg.into())).await;
            while let Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) = ws_stream.next().await {
                if let Ok(json) = serde_json::from_str::<Value>(&text) {
                    if let Some(price_str) = json.get("data").and_then(|d| d.as_array()?.first()).and_then(|f| f.get("bids")).and_then(|b| b.as_array()?.first()).and_then(|f| f.as_array()?.first()).and_then(|v| v.as_str()) {
                        if let Ok(p) = price_str.parse::<f64>() { state.okx_swap.store(p.to_bits(), Ordering::Relaxed); }
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
    let funding_url = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT";
    
    loop {
        ticker.tick().await;
        if let Ok(res) = client.get(fear_url).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(data) = json.get("data").and_then(|d| d.as_array()?.first()) {
                    if let Some(val_str) = data.get("value").and_then(|v| v.as_str()) {
                        if let Ok(val) = val_str.parse::<f64>() {
                            state.fear_greed_index.store(val.to_bits(), Ordering::Relaxed);
                        }
                    }
                }
            }
        }
        if let Ok(res) = client.get(funding_url).send().await {
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
    
    let cpi_url = "https://api.worldbank.org/v2/country/USA/indicator/FP.CPI.TOTL.ZG?format=json&date=2023:2024";
    let m2_url = "https://api.worldbank.org/v2/country/USA/indicator/FM.LBL.BMNY.GD.ZS?format=json&date=2023:2024";

    loop {
        ticker.tick().await;
        if let Ok(res) = client.get(cpi_url).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(arr) = json.as_array() {
                    if arr.len() > 1 {
                        if let Some(data) = arr[1].as_array() {
                            for item in data {
                                if let Some(val) = item.get("value").and_then(|v| v.as_f64()) {
                                    state.wb_us_cpi_inflation.store(val.to_bits(), Ordering::Relaxed);
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
                                    state.wb_us_m2_supply.store(val.to_bits(), Ordering::Relaxed);
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

pub async fn run_mock_derivatives_poller(state: Arc<OmniState>) {
    // Generates stochastic data for premium sources missing free real-time APIs
    let mut ticker = interval(Duration::from_millis(5000));
    
    // Base states
    let mut dvol = 50.0;
    let mut cvd_spot = 0.0;
    let mut cvd_futures = 0.0;
    let mut skew = 0.0;
    let mut ofi = 0.0;

    // Use current time to salt pseudo-randomness slightly since we don't import rand here by default
    loop {
        ticker.tick().await;
        
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as f64;
        let noise = (now % 1000.0) / 1000.0 - 0.5; // [-0.5, 0.5]
        
        // Browniano simple
        dvol = (dvol + noise * 2.0).clamp(30.0, 100.0);
        cvd_spot += noise * 100.0;
        cvd_futures += noise * 120.0;
        skew = (skew + noise * 0.1).clamp(-1.0, 1.0);
        ofi = (ofi + noise * 5.0).clamp(-20.0, 20.0);

        state.dvol_index.store(dvol.to_bits(), Ordering::Relaxed);
        state.spot_cvd.store(cvd_spot.to_bits(), Ordering::Relaxed);
        state.futures_cvd.store(cvd_futures.to_bits(), Ordering::Relaxed);
        state.options_25_delta_skew.store(skew.to_bits(), Ordering::Relaxed);
        state.order_flow_imbalance.store(ofi.to_bits(), Ordering::Relaxed);
        
        // Liquidations
        state.liq_cluster_shorts.store((60000.0 + noise * 1000.0).to_bits(), Ordering::Relaxed);
        state.liq_cluster_longs.store((55000.0 + noise * 1000.0).to_bits(), Ordering::Relaxed);
        state.cme_futures_premium.store((20.0 + noise * 10.0).to_bits(), Ordering::Relaxed);
        state.margin_debt_ratio.store((0.15 + noise * 0.02).to_bits(), Ordering::Relaxed);
    }
}

pub struct OmniDataHub {
    pub omni_state: Arc<OmniState>,
}

impl OmniDataHub {
    pub fn new() -> Self {
        Self {
            omni_state: Arc::new(OmniState::new()),
        }
    }

    pub fn start_feeds(&self) {
        tokio::spawn(run_bybit_ws(self.omni_state.clone()));
        tokio::spawn(run_okx_ws(self.omni_state.clone()));
        tokio::spawn(run_macro_rest_poller(self.omni_state.clone()));
        tokio::spawn(run_sentiment_onchain_poller(self.omni_state.clone()));
        tokio::spawn(run_world_bank_poller(self.omni_state.clone()));
        tokio::spawn(run_mock_derivatives_poller(self.omni_state.clone()));
    }
}
