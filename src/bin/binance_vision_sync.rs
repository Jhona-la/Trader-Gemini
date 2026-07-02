use reqwest::blocking::Client;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use zip::ZipArchive;

const SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
    "DOTUSDT", "DOGEUSDT", "LINKUSDT", "TRXUSDT", "LTCUSDT", "BCHUSDT",
    "ATOMUSDT", "UNIUSDT", "XMRUSDT", "ETCUSDT", "FILUSDT", "ICPUSDT", "VETUSDT",
    "NEARUSDT", "AAVEUSDT", "ALGOUSDT", "EGLDUSDT", "SANDUSDT", "THETAUSDT",
    "AXSUSDT", "MANAUSDT", "XLMUSDT", "GALAUSDT", "FTMUSDT", "RUNEUSDT",
    "WAVESUSDT", "ZECUSDT", "DASHUSDT", "ENJUSDT", "BATUSDT", "ZILUSDT",
    "COMPUSDT", "SNXUSDT"
];

const MONTHS: &[&str] = &["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"];

fn main() {
    println!("============================================================");
    println!("🌐 BINANCE VISION INSTITUTIONAL DOWNLOADER (Top 40 Assets)");
    println!("============================================================");

    let client = Client::new();
    let data_dir = Path::new("data/vision");
    std::fs::create_dir_all(data_dir).unwrap();

    for symbol in SYMBOLS {
        println!("🚀 Iniciando descarga para {}", symbol);
        
        let mut out_file_path = data_dir.join(format!("{}_6M.csv", symbol));
        let mut out_file = File::create(&out_file_path).unwrap();
        
        // CSV Header (Binance Vision Format)
        writeln!(out_file, "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore").unwrap();

        for month in MONTHS {
            let url = format!(
                "https://data.binance.vision/data/futures/um/monthly/klines/{}/1m/{}-1m-{}.zip",
                symbol, symbol, month
            );
            
            let zip_path = data_dir.join(format!("{}-{}.zip", symbol, month));
            
            println!("   📥 Descargando: {}", url);
            let mut resp = match client.get(&url).send() {
                Ok(r) => {
                    if !r.status().is_success() {
                        println!("   ❌ HTTP Error {} for {}", r.status(), url);
                        continue;
                    }
                    r
                },
                Err(e) => {
                    println!("   ❌ Reqwest Error: {}", e);
                    continue;
                }
            };
            
            let mut dest = File::create(&zip_path).unwrap();
            resp.copy_to(&mut dest).unwrap();
            
            // Unzip the file and append to CSV
            let zip_file = File::open(&zip_path).unwrap();
            if let Ok(mut archive) = ZipArchive::new(zip_file) {
                if archive.len() > 0 {
                    let mut file = archive.by_index(0).unwrap();
                    let mut content = String::new();
                    file.read_to_string(&mut content).unwrap();
                    out_file.write_all(content.as_bytes()).unwrap();
                }
            } else {
                println!("   ❌ Error reading ZIP: {:?}", zip_path);
            }
            
            // Eliminar zip temporal
            let _ = std::fs::remove_file(&zip_path);
        }
        
        println!("✅ {} completado y ensamblado.", symbol);
    }
}
