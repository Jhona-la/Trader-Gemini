use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use serde_json::Value;

pub struct OnChainState {
    pub btc_funding_rate: AtomicU64,
    pub btc_open_interest: AtomicU64,
}

impl OnChainState {
    pub fn new() -> Self {
        Self {
            btc_funding_rate: AtomicU64::new(0.0_f64.to_bits()),
            btc_open_interest: AtomicU64::new(0.0_f64.to_bits()),
        }
    }

    pub fn get_funding(&self) -> f64 { f64::from_bits(self.btc_funding_rate.load(Ordering::Relaxed)) }
    pub fn get_oi(&self) -> f64 { f64::from_bits(self.btc_open_interest.load(Ordering::Relaxed)) }
}

/// Extrae Funding Rate y Open Interest de Binance Futures
pub async fn run_onchain_feed(state: Arc<OnChainState>) {
    let mut ticker = interval(Duration::from_secs(30));
    let client = reqwest::Client::new();
    let url_funding = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT";
    let url_oi = "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT";

    loop {
        ticker.tick().await;
        
        // Fetch Funding Rate
        if let Ok(res) = client.get(url_funding).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(funding_str) = json.get("lastFundingRate").and_then(|v| v.as_str()) {
                    if let Ok(funding) = funding_str.parse::<f64>() {
                        state.btc_funding_rate.store(funding.to_bits(), Ordering::Relaxed);
                    }
                }
            }
        }

        // Fetch Open Interest
        if let Ok(res) = client.get(url_oi).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(oi_str) = json.get("openInterest").and_then(|v| v.as_str()) {
                    if let Ok(oi) = oi_str.parse::<f64>() {
                        state.btc_open_interest.store(oi.to_bits(), Ordering::Relaxed);
                    }
                }
            }
        }
    }
}
