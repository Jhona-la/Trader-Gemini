use reqwest::Client;
use serde::Deserialize;
use std::fs::File;
use std::path::Path;
use polars::prelude::*;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct BinanceFundingRate {
    symbol: String,
    fundingTime: u64,
    fundingRate: String,
}

pub struct MarketContextFetcher {
    client: Client,
}

impl MarketContextFetcher {
    pub fn new() -> Self {
        Self { client: Client::new() }
    }

    pub async fn fetch_funding_history(&self, symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("https://fapi.binance.com/fapi/v1/fundingRate?symbol={}&limit=1000", symbol);
        
        let response = self.client.get(&url).send().await?;
        if !response.status().is_success() {
            return Ok(());
        }

        let rates: Vec<BinanceFundingRate> = response.json().await?;
        
        if rates.is_empty() {
            return Ok(());
        }

        let timestamps: Vec<u64> = rates.iter().map(|r| r.fundingTime).collect();
        let funding_rates: Vec<f64> = rates.iter().map(|r| r.fundingRate.parse::<f64>().unwrap_or(0.0)).collect();

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
