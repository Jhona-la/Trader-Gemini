use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use serde_json::Value;

pub struct OnChainState {
    pub btc_funding_rate: AtomicU64,
    pub btc_open_interest: AtomicU64,
}

impl Default for OnChainState {
    fn default() -> Self {
        Self::new()
    }
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
    let is_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
    let base_url = if is_testnet { "https://testnet.binancefuture.com" } else { "https://fapi.binance.com" };
    let url_funding = format!("{}/fapi/v1/premiumIndex?symbol=BTCUSDT", base_url);
    let url_oi = format!("{}/fapi/v1/openInterest?symbol=BTCUSDT", base_url);

    loop {
        ticker.tick().await;
        
        // Fetch Funding Rate
        if let Ok(res) = client.get(&url_funding).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(funding_str) = json.get("lastFundingRate").and_then(|v| v.as_str()) {
                    if let Ok(funding) = funding_str.parse::<f64>() {
                        state.btc_funding_rate.store(funding.to_bits(), Ordering::Relaxed);
                    }
                }
            }
        }

        // Fetch Open Interest
        if let Ok(res) = client.get(&url_oi).send().await {
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
