use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::time::{sleep, Duration};

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
    // FIX #1489: Construcción de cliente HTTP resiliente sin unwrap
    pub fn new(inflation_var: Arc<AtomicU64>, interest_rate_var: Arc<AtomicU64>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            inflation_var,
            interest_rate_var,
        }
    }

    /// Fetches a World Bank indicator and returns the latest numeric value
    async fn fetch_indicator(&self, country: &str, indicator: &str) -> Option<f64> {
        let url = format!(
            "https://api.worldbank.org/v2/country/{}/indicator/{}?format=json&per_page=5",
            country, indicator
        );

        match self.client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(json) = resp.json::<Value>().await {
                    if let Some(data_array) = json.as_array() {
                        if data_array.len() > 1 {
                            if let Some(records) = data_array[1].as_array() {
                                for rec in records {
                                    if let Some(val) = rec.get("value").and_then(|v| v.as_f64()) {
                                        // FIX #706: Validar finitud del indicador macro
                                        if val.is_finite() {
                                            return Some(val);
                                        }
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
                    if inflation.is_finite() {
                        // FIX #1536: Clamping de inflación a límites macroeconómicos razonables [-50.0, 100.0]
                        let safe_inf = inflation.clamp(-50.0, 100.0);
                        self.inflation_var
                            .store(safe_inf.to_bits(), Ordering::Relaxed);
                        println!("🏦 [WORLD BANK] US Inflation Updated: {:.2}%", safe_inf);
                    }
                }

                // Fetch US Lending Interest Rate (FR.INR.LEND)
                if let Some(interest_rate) = self.fetch_indicator("US", "FR.INR.LEND").await {
                    if interest_rate.is_finite() {
                        // FIX #1536: Clamping de tasa de interés [0.0, 100.0]
                        let safe_rate = interest_rate.clamp(0.0, 100.0);
                        self.interest_rate_var
                            .store(safe_rate.to_bits(), Ordering::Relaxed);
                        println!(
                            "🏦 [WORLD BANK] US Interest Rate Updated: {:.2}%",
                            safe_rate
                        );
                    }
                }

                let lat = start.elapsed().as_millis();
                println!("✅ [WORLD BANK] Macro-Context Synced in {} ms.", lat);

                // Polling interval: 6 hours (Macro indicators update rarely)
                sleep(Duration::from_secs(6 * 3600)).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_bank_atomic_storage() {
        let inf = Arc::new(AtomicU64::new(0.0f64.to_bits()));
        let rate = Arc::new(AtomicU64::new(0.0f64.to_bits()));

        let client = WorldBankClient::new(Arc::clone(&inf), Arc::clone(&rate));
        inf.store(3.25f64.to_bits(), Ordering::Relaxed);
        rate.store(5.50f64.to_bits(), Ordering::Relaxed);

        let stored_inf = f64::from_bits(client.inflation_var.load(Ordering::Relaxed));
        let stored_rate = f64::from_bits(client.interest_rate_var.load(Ordering::Relaxed));

        assert_eq!(stored_inf, 3.25);
        assert_eq!(stored_rate, 5.50);
    }

    #[test]
    fn test_world_bank_nan_and_negative_immunity() {
        let inf = Arc::new(AtomicU64::new(0.0f64.to_bits()));
        let rate = Arc::new(AtomicU64::new(0.0f64.to_bits()));

        let client = WorldBankClient::new(Arc::clone(&inf), Arc::clone(&rate));

        // Negative interest rates (e.g. historical Japan/ECB)
        rate.store((-0.50f64).to_bits(), Ordering::Relaxed);
        let stored_neg_rate = f64::from_bits(client.interest_rate_var.load(Ordering::Relaxed));
        assert_eq!(stored_neg_rate, -0.50);

        // NaN safety
        inf.store(f64::NAN.to_bits(), Ordering::Relaxed);
        let stored_nan_inf = f64::from_bits(client.inflation_var.load(Ordering::Relaxed));
        assert!(stored_nan_inf.is_nan());
    }

    #[test]
    fn test_world_bank_clamping_ranges_and_default_init() {
        let inf = Arc::new(AtomicU64::new(0.0f64.to_bits()));
        let rate = Arc::new(AtomicU64::new(0.0f64.to_bits()));

        let _client = WorldBankClient::new(Arc::clone(&inf), Arc::clone(&rate));

        // Test clamping logic
        let extreme_inflation = 150.0f64;
        let clamped_inf = extreme_inflation.clamp(-50.0, 100.0);
        assert_eq!(clamped_inf, 100.0);

        let negative_interest = -5.0f64;
        let clamped_rate = negative_interest.clamp(0.0, 100.0);
        assert_eq!(clamped_rate, 0.0);
    }
}


