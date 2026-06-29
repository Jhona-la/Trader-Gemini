/// CSV to Binary converter for historical data.
/// Converts large CSV files to memory-mapped binary format [closes][highs][lows][volumes]
/// Usage: cargo run --release --bin csv_to_bin

use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn convert_csv_to_bin(csv_path: &str, bin_path: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::open(csv_path)?;
    let reader = BufReader::new(file);
    
    let mut closes = Vec::new();
    let mut highs = Vec::new();
    let mut lows = Vec::new();
    let mut volumes = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 { continue; } // Skip header
        
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6 { continue; }
        
        // CSV format: open_time,open,high,low,close,volume,...
        let high: f64 = fields[2].parse().unwrap_or(0.0);
        let low: f64 = fields[3].parse().unwrap_or(0.0);
        let close: f64 = fields[4].parse().unwrap_or(0.0);
        let volume: f64 = fields[5].parse().unwrap_or(0.0);
        
        if close > 0.0 {
            closes.push(close);
            highs.push(high);
            lows.push(low);
            volumes.push(volume);
        }
    }
    
    let count = closes.len();
    if count == 0 {
        println!("⚠️ No valid data found in {}", csv_path);
        return Ok(0);
    }
    
    let mut bin_file = File::create(bin_path)?;
    
    // Write contiguous blocks: [closes][highs][lows][volumes]
    let write_slice = |file: &mut File, data: &[f64]| -> std::io::Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
        file.write_all(bytes)
    };
    
    write_slice(&mut bin_file, &closes)?;
    write_slice(&mut bin_file, &highs)?;
    write_slice(&mut bin_file, &lows)?;
    write_slice(&mut bin_file, &volumes)?;
    
    println!("✅ Converted {} → {} ({} candles, {} bytes)", csv_path, bin_path, count, count * 4 * 8);
    Ok(count)
}

fn main() {
    println!("========================================================");
    println!("📊 CSV → BINARY CONVERTER (Memory-Map Ready)");
    println!("========================================================");
    
    let conversions = vec![
        // Large CSVs with underscore format (100K+ candles each)
        ("data/historical/BTC_USDT_1m.csv", "data/historical/BTCUSDT_1m.bin"),
        ("data/historical/ETH_USDT_1m.csv", "data/historical/ETHUSDT_1m.bin"),
        ("data/historical/SOL_USDT_1m.csv", "data/historical/SOLUSDT_1m.bin"),
        ("data/historical/XRP_USDT_1m.csv", "data/historical/XRPUSDT_1m.bin"),
        ("data/historical/BNB_USDT_1m.csv", "data/historical/BNBUSDT_1m.bin"),
        ("data/historical/DOGE_USDT_1m.csv", "data/historical/DOGEUSDT_1m.bin"),
        ("data/historical/ADA_USDT_1m.csv", "data/historical/ADAUSDT_1m.bin"),
        ("data/historical/AVAX_USDT_1m.csv", "data/historical/AVAXUSDT_1m.bin"),
        ("data/historical/DOT_USDT_1m.csv", "data/historical/DOTUSDT_1m.bin"),
        ("data/historical/LINK_USDT_1m.csv", "data/historical/LINKUSDT_1m.bin"),
        ("data/historical/LTC_USDT_1m.csv", "data/historical/LTCUSDT_1m.bin"),
        ("data/historical/UNI_USDT_1m.csv", "data/historical/UNIUSDT_1m.bin"),
    ];
    
    let mut total_candles = 0usize;
    
    for (csv, bin) in &conversions {
        match convert_csv_to_bin(csv, bin) {
            Ok(count) => total_candles += count,
            Err(e) => println!("⚠️ Skipped {}: {}", csv, e),
        }
    }
    
    println!("========================================================");
    println!("🎯 Total: {} candles converted across {} symbols", total_candles, conversions.len());
    println!("   At 1m resolution: ~{:.1} days of data per symbol", total_candles as f64 / conversions.len() as f64 / 1440.0);
}
