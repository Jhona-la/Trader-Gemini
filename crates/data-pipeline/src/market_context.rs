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
            return Ok(());
        }

        let rates: Vec<BinanceFundingRate> = response.json().await?;

        if rates.is_empty() {
            return Ok(());
        }

        let timestamps: Vec<u64> = rates.iter().map(|r| r.funding_time).collect();
        let funding_rates: Vec<f64> = rates
            .iter()
            .map(|r| r.funding_rate.parse::<f64>().unwrap_or(0.0))
            .collect();

        let time_series = Series::new("timestamp".into(), timestamps);
        let rate_series = Series::new("funding_rate".into(), funding_rates);

        let mut df = DataFrame::new(vec![time_series, rate_series])?;

        let data_dir = Path::new("data/historical");
        if !data_dir.exists() {
            std::fs::create_dir_all(data_dir)?;
        }

        let file_path = data_dir.join(format!("{}_FUNDING.parquet", symbol));
        let mut file = File::create(&file_path)?;

        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut df)?;

        println!("✅ Funding Rates guardadas para {}", symbol);
        Ok(())
    }
}
