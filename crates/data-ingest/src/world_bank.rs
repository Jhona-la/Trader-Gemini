use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use serde_json::Value;
use std::time::Instant;

/// World Bank Open Data Client
/// API Indicators:
/// FP.CPI.TOTL.ZG - Inflation, consumer prices (annual %)
/// FR.INR.LEND - Lending interest rate (%)
/// We will fetch these indicators periodically and update the OmniState.
pub struct WorldBankClient {
    client: reqwest::Client,
    inflation_var: Arc<AtomicU64>,
    interest_rate_var: Arc<AtomicU64>,
}

impl WorldBankClient {
    pub fn new(inflation_var: Arc<AtomicU64>, interest_rate_var: Arc<AtomicU64>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            inflation_var,
            interest_rate_var,
        }
    }

    /// Fetches a World Bank indicator and returns the latest numeric value
    async fn fetch_indicator(&self, country: &str, indicator: &str) -> Option<f64> {
        let url = format!("http://api.worldbank.org/v2/country/{}/indicator/{}?format=json&per_page=1", country, indicator);
        
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(json) = resp.json::<Value>().await {
                    if let Some(data_array) = json.as_array() {
                        if data_array.len() > 1 {
                            if let Some(records) = data_array[1].as_array() {
                                if let Some(latest) = records.first() {
                                    if let Some(value) = latest.get("value") {
                                        return value.as_f64();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ [WORLD BANK] Fetch Error: {}", e);
            }
        }
        None
    }

    /// Spawns a background task to poll the World Bank API
    pub fn start_polling(self: Arc<Self>) {
        tokio::spawn(async move {
            println!("🌍 [WORLD BANK] Daemon Started. Syncing Geopolitical Macro-Context...");
            loop {
                let start = Instant::now();
                
                // Fetch US Inflation (FP.CPI.TOTL.ZG)
                if let Some(inflation) = self.fetch_indicator("US", "FP.CPI.TOTL.ZG").await {
                    self.inflation_var.store(inflation.to_bits(), Ordering::Relaxed);
                    println!("🏦 [WORLD BANK] US Inflation Updated: {:.2}%", inflation);
                }

                // Fetch US Lending Interest Rate (FR.INR.LEND)
                if let Some(interest_rate) = self.fetch_indicator("US", "FR.INR.LEND").await {
                    self.interest_rate_var.store(interest_rate.to_bits(), Ordering::Relaxed);
                    println!("🏦 [WORLD BANK] US Interest Rate Updated: {:.2}%", interest_rate);
                }
                
                let lat = start.elapsed().as_millis();
                println!("✅ [WORLD BANK] Macro-Context Synced in {} ms.", lat);

                // Polling interval: 6 hours (Macro indicators update rarely)
                sleep(Duration::from_secs(6 * 3600)).await;
            }
        });
    }
}
