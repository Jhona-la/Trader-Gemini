use yahoo_finance_api as yahoo;
use polars::prelude::*;
use std::fs::File;
use std::path::Path;

pub struct MacroFetcher;

impl MacroFetcher {
    pub async fn fetch_and_save_macro_data() -> Result<(), Box<dyn std::error::Error>> {
        let provider = yahoo::YahooConnector::new()?;
        
        let symbols = vec![
            ("^GSPC", "SP500"),
            ("DX-Y.NYB", "DXY"),
            ("^VIX", "VIX"),
        ];

        let data_dir = Path::new("data/historical");
        if !data_dir.exists() {
            std::fs::create_dir_all(data_dir)?;
        }

        for (ticker, name) in symbols {
            // Bajamos los ultimos 6 meses
            let response = provider.get_quote_range(ticker, "1d", "6mo").await?;
            let quotes = response.quotes()?;
            
            if quotes.is_empty() {
                continue;
            }

            let timestamps: Vec<u64> = quotes.iter().map(|q| (q.timestamp * 1000) as u64).collect();
            let closes: Vec<f64> = quotes.iter().map(|q| q.close).collect();

            let time_series = Series::new("timestamp".into(), timestamps);
            let close_series = Series::new(format!("{}_close", name).into(), closes);

            let mut df = DataFrame::new(vec![time_series, close_series])?;

            let file_path = data_dir.join(format!("MACRO_{}.parquet", name));
            let mut file = File::create(&file_path)?;
            
            ParquetWriter::new(&mut file)
                .with_compression(ParquetCompression::Zstd(None))
                .finish(&mut df)?;

            println!("✅ Macro Data guardada: {}", name);
        }

        Ok(())
    }
}
