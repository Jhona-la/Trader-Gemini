use reqwest::Client;
use serde_json::Value;
use std::fs::{self, File};
use std::io::Write;
use chrono::prelude::*;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================");
    println!("📡 GOD ENGINE - NATIVE DATA LOADER (BINANCE VISION)");
    println!("========================================================");

    let config_str = fs::read_to_string("data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
    let config_json: Value = serde_json::from_str(&config_str).unwrap_or(serde_json::json!({}));
    
    let default_symbols = vec!["btcusdt".to_string(), "ethusdt".to_string()];
    let symbols: Vec<String> = config_json["symbols"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or(default_symbols);

    fs::create_dir_all("data/historical")?;

    let client = Client::new();
    let intervals = vec!["1m", "1h"];

    for symbol in symbols {
        let sym_upper = symbol.to_uppercase();
        for interval in &intervals {
            println!("📥 Downloading {} for {}...", interval, sym_upper);
            let url = format!(
                "https://fapi.binance.com/fapi/v1/klines?symbol={}&interval={}&limit=1500",
                sym_upper, interval
            );

            let mut retries = 3;
            loop {
                match client.get(&url).send().await {
                    Ok(res) => {
                        if res.status().is_success() {
                            let data: Value = res.json().await?;
                            let file_path = format!("data/historical/{}_{}.csv", sym_upper, interval);
                            let mut file = File::create(&file_path)?;
                            
                            writeln!(file, "open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore")?;
                            
                            if let Some(arr) = data.as_array() {
                                for row in arr {
                                    let row_arr = row.as_array().unwrap();
                                    let line: Vec<String> = row_arr.iter().map(|v| v.to_string().replace("\"", "")).collect();
                                    writeln!(file, "{}", line.join(","))?;
                                }
                                println!("✅ Saved {} records to {}", arr.len(), file_path);
                            }
                            break;
                        } else {
                            println!("⚠️ Error downloading {}: HTTP {}", url, res.status());
                        }
                    },
                    Err(e) => {
                        println!("⚠️ Network Error: {}", e);
                    }
                }
                
                retries -= 1;
                if retries == 0 {
                    println!("❌ Failed to download {} after 3 retries.", sym_upper);
                    break;
                }
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
    
    println!("🎯 Data Loading Complete.");
    Ok(())
}
