use reqwest::Client;
use std::fs::File;
use std::io::{Write, copy};
use std::path::Path;
use chrono::{Utc, Duration};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let symbol = if args.len() > 1 { &args[1] } else { "BTCUSDT" };
    
    // Binance daily aggTrades usually have a 1-2 day delay for the previous day. We use 2 days ago.
    let target_date = Utc::now() - Duration::days(2);
    let date_str = target_date.format("%Y-%m-%d").to_string();
    
    let url = format!(
        "https://data.binance.vision/data/futures/um/daily/aggTrades/{}/{}-aggTrades-{}.zip",
        symbol, symbol, date_str
    );

    let raw_dir = "data/raw";
    std::fs::create_dir_all(raw_dir)?;
    
    let zip_path = format!("{}/{}-aggTrades-{}.zip", raw_dir, symbol, date_str);
    
    println!("🌍 [DOWNLOADER] Fetching {}...", url);
    let client = Client::new();
    let mut response = client.get(&url).send().await?;
    
    if response.status().is_success() {
        let mut file = File::create(&zip_path)?;
        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk)?;
        }
        println!("✅ Download complete. Unzipping...");
        
        let file = File::open(&zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = match file.enclosed_name() {
                Some(path) => path.to_owned(),
                None => continue,
            };
            
            let final_out_path = Path::new(raw_dir).join(outpath);
            if let Some(p) = final_out_path.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(&p)?;
                }
            }
            let mut outfile = File::create(&final_out_path)?;
            copy(&mut file, &mut outfile)?;
            
            // Rename to standard target
            let target = format!("data/{}_ticks.csv", symbol);
            let _ = std::fs::remove_file(&target);
            std::fs::rename(final_out_path, target.clone())?;
            println!("🚀 Data ready at {}", target);
        }
    } else {
        println!("❌ [DOWNLOADER] Error fetching: HTTP {}", response.status());
    }
    
    Ok(())
}
