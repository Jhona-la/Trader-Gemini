use reqwest::blocking::Client;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

fn main() {
    println!("============================================================");
    println!("🌍 MACRO ECONOMIC HISTORY DOWNLOADER (SP500, VIX, DXY)");
    println!("============================================================");

    let client = Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| Client::new());

    let data_dir = Path::new("data/macro");
    std::fs::create_dir_all(data_dir).unwrap();

    // Macro tickers
    let symbols = vec![
        ("^VIX", "VIX_Volatility_Index", 20.0),
        ("^GSPC", "SP500_Index", 5000.0),
        ("DX-Y.NYB", "DXY_Dollar_Index", 104.0),
    ];

    let period1 = 1420070400; // Jan 1 2015
    // R2.3: fin de ventana = HOY (antes 1751328000, congelado en jul-2025).
    let period2 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(1751328000);

    for (ticker, name, base_val) in symbols {
        println!("🚀 Descargando historial macro para: {}", name);

        let url = format!(
            "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history&includeAdjustedClose=true",
            ticker, period1, period2
        );

        let out_path = data_dir.join(format!("{}.csv", name));
        let mut downloaded = false;

        if let Ok(r) = client.get(&url).send() {
            if r.status().is_success() {
                if let Ok(content) = r.text() {
                    if content.contains("Date,") && content.lines().count() > 10 {
                        if let Ok(mut file) = File::create(&out_path) {
                            let _ = file.write_all(content.as_bytes());
                            println!("   ✅ Descargado y guardado en {:?}", out_path);
                            downloaded = true;
                        }
                    }
                }
            }
        }

        if !downloaded {
            // R2.3 — DATO SINTÉTICO MARCADO: el fallback se escribe con sufijo
            // .SYNTHETIC.csv y cabecera marcada, NUNCA como el archivo real.
            // Un random-walk etiquetado como historia de VIX/SP500/DXY
            // contaminaría cualquier feature macro que lo ingiriera.
            let synthetic_path = data_dir.join(format!("{}.SYNTHETIC.csv", name));
            println!(
                "   🚨 [R2.3] Endpoint remoto NO disponible para {}. Fallback SINTÉTICO en {:?} — NO es historia real y no debe ingerirse como tal.",
                ticker,
                synthetic_path
            );
            if let Ok(mut file) = File::create(&synthetic_path) {
                let _ = writeln!(file, "# SYNTHETIC RANDOM-WALK — NOT REAL MARKET DATA — DO NOT INGEST");
                let _ = writeln!(file, "Date,Open,High,Low,Close,Adj Close,Volume");
                let mut price = base_val;
                for i in 0..1000 {
                    let seed = (i as u64) ^ 0x5DEECE66D;
                    let pct = (((seed % 200) as f64) - 100.0) / 5000.0;
                    price = (price * (1.0 + pct)).max(1.0);
                    let _ = writeln!(
                        file,
                        "2022-01-{:02},{:.2},{:.2},{:.2},{:.2},{:.2},1000000",
                        (i % 28) + 1,
                        price * 0.998,
                        price * 1.005,
                        price * 0.995,
                        price,
                        price
                    );
                }
                println!("   ⚠️ Fallback sintético generado (marcado). El archivo real {}.csv NO fue creado.", name);
            }
        }
    }

    println!("✅ Descarga y preparación de datos macroeconómicos completada.");
}
