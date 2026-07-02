use reqwest::blocking::Client;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    println!("============================================================");
    println!("🌍 MACRO ECONOMIC HISTORY DOWNLOADER (SP500, VIX, DXY)");
    println!("============================================================");

    let client = Client::new();
    let data_dir = Path::new("data/macro");
    std::fs::create_dir_all(data_dir).unwrap();

    // Yahoo Finance tickers
    let symbols = vec![
        ("^VIX", "VIX_Volatility_Index"),
        ("^GSPC", "SP500_Index"),
        ("DX-Y.NYB", "DXY_Dollar_Index"),
    ];

    // Timestamp for last 10 years roughly
    let period1 = 1420070400; // Jan 1 2015
    let period2 = 1719792000; // July 2026 approx

    for (ticker, name) in symbols {
        println!("🚀 Descargando historial macro para: {}", name);
        
        let url = format!(
            "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history&includeAdjustedClose=true",
            ticker, period1, period2
        );
        
        match client.get(&url).send() {
            Ok(r) => {
                if r.status().is_success() {
                    if let Ok(content) = r.text() {
                        let out_path = data_dir.join(format!("{}.csv", name));
                        let mut file = File::create(&out_path).unwrap();
                        file.write_all(content.as_bytes()).unwrap();
                        println!("   ✅ Guardado en {:?}", out_path);
                    }
                } else {
                    println!("   ❌ Error HTTP {}: {}", r.status(), url);
                }
            },
            Err(e) => {
                println!("   ❌ Error de conexión: {}", e);
            }
        }
    }
    
    println!("✅ Descarga de datos macroeconómicos completada.");
}
