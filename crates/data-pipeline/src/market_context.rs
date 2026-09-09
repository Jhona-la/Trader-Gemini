use polars::prelude::*;
use reqwest::Client;
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct BinanceFundingRate {
    symbol: String,
    #[serde(rename = "fundingTime")]
    funding_time: u64,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
}

pub struct MarketContextFetcher {
    client: Client,
}

impl Default for MarketContextFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl MarketContextFetcher {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn fetch_funding_history(
        &self,
        symbol: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
        let url = format!(
            "{}/fapi/v1/fundingRate?symbol={}&limit=1000",
            base_url, symbol
        );

        let response = self.client.get(&url).send().await?;
        if !response.status().is_success() {
            eprintln!(
                "⚠️ [MarketContextFetcher] HTTP error {} fetching funding rates for {}",
                response.status(),
                symbol
            );
            return Ok(());
        }

        let rates: Vec<BinanceFundingRate> = response.json().await?;

        if rates.is_empty() {
            println!(
                "ℹ️ [MarketContextFetcher] No funding rates returned for {}",
                symbol
            );
            return Ok(());
        }

        let timestamps: Vec<u64> = rates.iter().map(|r| r.funding_time).collect();
        let funding_rates: Vec<f64> = rates
            .iter()
            .map(|r| {
                // FIX #1538: Sanitización y clamping de tasas de fondeo históricas
                let parsed = r.funding_rate.parse::<f64>().unwrap_or(0.0);
                if parsed.is_finite() {
                    parsed.clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect();

        let time_series = Series::new("timestamp".into(), timestamps);
        let rate_series = Series::new("funding_rate".into(), funding_rates);

        let mut df = DataFrame::new(vec![time_series, rate_series])?;

        let data_dir = Path::new("data/historical");
        if !data_dir.exists() {
            std::fs::create_dir_all(data_dir)?;
        }

        let file_path = data_dir.join(format!("{}_FUNDING.parquet", symbol));
        let tmp_path = data_dir.join(format!("{}_FUNDING.parquet.tmp", symbol));
        let mut file = File::create(&tmp_path)?;

        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut df)?;

        file.sync_all()?;
        if file_path.exists() {
            let _ = std::fs::remove_file(&file_path);
        }
        std::fs::rename(&tmp_path, &file_path)?;

        println!("✅ Funding Rates guardadas atómicamente para {}", symbol);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_context_fetcher_instantiation() {
        let fetcher = MarketContextFetcher::new();
        let default_fetcher = MarketContextFetcher::default();
        let _ = fetcher;
        let _ = default_fetcher;
    }

    #[test]
    fn test_binance_funding_rate_deserialization() {
        let json_str = r#"{
            "symbol": "BTCUSDT",
            "fundingTime": 1672531200000,
            "fundingRate": "0.00010000"
        }"#;

        let rate: BinanceFundingRate = serde_json::from_str(json_str).unwrap();
        assert_eq!(rate.symbol, "BTCUSDT");
        assert_eq!(rate.funding_time, 1672531200000);
        assert_eq!(rate.funding_rate, "0.00010000");
    }
}
