use data_pipeline::historical::Kline;
use data_pipeline::macro_data::MacroFetcher;
use data_pipeline::market_context::MarketContextFetcher;
use std::fs::File;
use std::path::Path;
use polars::prelude::*;
use reqwest::Client;
use std::io::{Cursor, Read};

const SYMBOLS: [&str; 40] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "TRXUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "ATOMUSDT", "UNIUSDT", "XMRUSDT", "ETCUSDT", "FILUSDT", "ICPUSDT",
    "VETUSDT", "NEARUSDT", "AAVEUSDT", "ALGOUSDT", "EGLDUSDT", "SANDUSDT", "THETAUSDT", "AXSUSDT", "MANAUSDT", "FTMUSDT",
    "APEUSDT", "GALAUSDT", "RUNEUSDT", "CHZUSDT", "CRVUSDT", "MKRUSDT", "GRTUSDT", "LDOUSDT", "OPUSDT", "ARBUSDT"
];

// Descargar los 6 meses más recientes
const MONTHS: [&str; 6] = [
    "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🌊 TRADER GEMINI V5 - DATA LAKE DOWNLOADER (BINANCE VISION + MACRO)");
    println!("============================================================");

    // 1. Fetch Macro Data (SP500, DXY, VIX)
    println!("📊 Descargando Contexto Macroeconómico...");
    if let Err(e) = MacroFetcher::fetch_and_save_macro_data().await {
        println!("⚠️ Error descargando Macro Data: {}", e);
    }

    let client = Client::new();
    let market_fetcher = MarketContextFetcher::new();
    let data_dir = Path::new("data/historical");
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir)?;
    }

    for symbol in SYMBOLS.iter() {
        // 2. Fetch Funding Rates para este símbolo
        if let Err(e) = market_fetcher.fetch_funding_history(symbol).await {
            println!("⚠️ Error descargando Funding para {}: {}", symbol, e);
        }

        let file_path = data_dir.join(format!("{}_6M.parquet", symbol));
        let mut all_klines: Vec<Kline> = Vec::with_capacity(300_000);

        for month in MONTHS.iter() {
            let url = format!(
                "https://data.binance.vision/data/futures/um/monthly/klines/{}/1m/{}-1m-{}.zip",
                symbol, symbol, month
            );
            
            println!("Descargando {}...", url);
            let response = match client.get(&url).send().await {
                Ok(resp) => resp,
                Err(e) => {
                    println!("⚠️ Error de red para {}: {}", url, e);
                    continue;
                }
            };

            if response.status().is_success() {
                let bytes = response.bytes().await?;
                let cursor = Cursor::new(bytes);
                let mut archive = match ::zip::ZipArchive::new(cursor) {
                    Ok(a) => a,
                    Err(e) => {
                        println!("⚠️ Error abriendo zip para {} ({}): {}", symbol, month, e);
                        continue;
                    }
                };

                if archive.len() == 0 {
                    continue;
                }
                
                let mut csv_file = archive.by_index(0)?;
                let mut csv_content = String::new();
                csv_file.read_to_string(&mut csv_content)?;

                let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_reader(csv_content.as_bytes());
                for result in rdr.records() {
                    let record = match result {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    
                    if record.len() < 8 { continue; }
                    
                    let open_time = record[0].parse::<u64>().unwrap_or(0);
                    let open = record[1].parse::<f64>().unwrap_or(0.0);
                    let high = record[2].parse::<f64>().unwrap_or(0.0);
                    let low = record[3].parse::<f64>().unwrap_or(0.0);
                    let close = record[4].parse::<f64>().unwrap_or(0.0);
                    let volume = record[5].parse::<f64>().unwrap_or(0.0);
                    let close_time = record[6].parse::<u64>().unwrap_or(0);

                    all_klines.push(Kline {
                        open_time, open, high, low, close, volume, close_time
                    });
                }
            } else {
                println!("⚠️ Archivo no encontrado (404) para {}", url);
            }
        }

        if all_klines.is_empty() {
            println!("❌ No se encontraron datos para {}", symbol);
            continue;
        }

        // Ordenar
        all_klines.sort_by_key(|k| k.open_time);

        // Guardar a Parquet
        let open_time_series = Series::new("open_time".into(), all_klines.iter().map(|k| k.open_time).collect::<Vec<_>>());
        let open_series = Series::new("open".into(), all_klines.iter().map(|k| k.open).collect::<Vec<_>>());
        let high_series = Series::new("high".into(), all_klines.iter().map(|k| k.high).collect::<Vec<_>>());
        let low_series = Series::new("low".into(), all_klines.iter().map(|k| k.low).collect::<Vec<_>>());
        let close_series = Series::new("close".into(), all_klines.iter().map(|k| k.close).collect::<Vec<_>>());
        let volume_series = Series::new("volume".into(), all_klines.iter().map(|k| k.volume).collect::<Vec<_>>());
        let close_time_series = Series::new("close_time".into(), all_klines.iter().map(|k| k.close_time).collect::<Vec<_>>());

        let mut df = DataFrame::new(vec![
            open_time_series, open_series, high_series, low_series,
            close_series, volume_series, close_time_series
        ])?;

        let mut file = File::create(&file_path)?;
        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut df)?;

        println!("✅ Guardado Parquet para {} ({} velas)", symbol, all_klines.len());
    }

    Ok(())
}
