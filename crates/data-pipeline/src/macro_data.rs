use polars::prelude::*;
use std::fs::File;
use std::path::Path;
use yahoo_finance_api as yahoo;

pub struct MacroFetcher;

impl MacroFetcher {
    pub async fn fetch_and_save_macro_data() -> Result<(), Box<dyn std::error::Error>> {
        let provider = yahoo::YahooConnector::new()?;

        let symbols = vec![("^GSPC", "SP500"), ("DX-Y.NYB", "DXY"), ("^VIX", "VIX")];

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

            // FIX #692: Filtrar registros con timestamps negativos y precios inválidos o NaN
            let mut timestamps = Vec::with_capacity(quotes.len());
            let mut closes = Vec::with_capacity(quotes.len());

            for q in quotes {
                if q.timestamp > 0 && q.close > 0.0 && q.close.is_finite() {
                    timestamps.push((q.timestamp * 1000) as u64);
                    closes.push(q.close);
                }
            }

            if timestamps.is_empty() {
                continue;
            }

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

#[cfg(test)]
mod tests {
    #[test]
    fn test_macro_fetcher_filtering_logic() {
        struct MockQuote {
            timestamp: i64,
            close: f64,
        }

        let quotes = vec![
            MockQuote { timestamp: 1672531200, close: 4000.0 },
            MockQuote { timestamp: -100, close: 4000.0 }, // Invalid timestamp
            MockQuote { timestamp: 1672531300, close: f64::NAN }, // Invalid NaN close
            MockQuote { timestamp: 1672531400, close: -50.0 }, // Invalid negative close
            MockQuote { timestamp: 1672531500, close: 4050.0 },
        ];

        let mut valid_ts = Vec::new();
        let mut valid_closes = Vec::new();

        for q in quotes {
            if q.timestamp > 0 && q.close > 0.0 && q.close.is_finite() {
                valid_ts.push((q.timestamp * 1000) as u64);
                valid_closes.push(q.close);
            }
        }

        assert_eq!(valid_ts.len(), 2);
        assert_eq!(valid_closes.len(), 2);
        assert_eq!(valid_closes[0], 4000.0);
        assert_eq!(valid_closes[1], 4050.0);
    }
}

