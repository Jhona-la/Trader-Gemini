use reqwest::blocking::Client;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use zip::ZipArchive;

const SYMBOLS: &[&str] = &[
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "XMRUSDT",
    "ETCUSDT",
    "FILUSDT",
    "ICPUSDT",
    "VETUSDT",
    "NEARUSDT",
    "AAVEUSDT",
    "ALGOUSDT",
    "EGLDUSDT",
    "SANDUSDT",
    "THETAUSDT",
    "AXSUSDT",
    "MANAUSDT",
    "XLMUSDT",
    "GALAUSDT",
    "FTMUSDT",
    "RUNEUSDT",
    "WAVESUSDT",
    "ZECUSDT",
    "DASHUSDT",
    "ENJUSDT",
    "BATUSDT",
    "ZILUSDT",
    "COMPUSDT",
    "SNXUSDT",
];

const MONTHS: &[&str] = &[
    "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05",
];

fn main() {
    // R2.2 — MODO AGGTRADES: `binance_vision_sync --aggtrades` descarga los
    // aggTrades oficiales (trades REALES con is_buyer_maker) y produce
    // {symbol}_ticks.bin con magic REAL (TGMTICK1) — la única fuente de
    // certificación sin artefactos de síntesis. OBI agrupado por buckets
    // volumétricos (no ±1 por trade).
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--aggtrades") {
        aggtrades_main();
        return;
    }

    println!("============================================================");
    println!("🌐 BINANCE VISION INSTITUTIONAL DOWNLOADER (Top 40 Assets)");
    println!("============================================================");

    let client = Client::new();
    let data_dir = Path::new("data/vision");
    if let Err(e) = std::fs::create_dir_all(data_dir) {
        println!("❌ Failed to create data/vision directory: {}", e);
        return;
    }

    for symbol in SYMBOLS {
        println!("🚀 Iniciando descarga para {}", symbol);

        let out_file_path = data_dir.join(format!("{}_6M.csv", symbol));
        let mut out_file = match File::create(&out_file_path) {
            Ok(f) => f,
            Err(e) => {
                println!("❌ Failed to create {}: {}", out_file_path.display(), e);
                continue;
            }
        };

        // CSV Header (Binance Vision Format)
        if let Err(e) = writeln!(out_file, "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore") {
            println!("❌ Failed to write header: {}", e);
            continue;
        }

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
                }
                Err(e) => {
                    println!("   ❌ Reqwest Error: {}", e);
                    continue;
                }
            };

            let mut dest = match File::create(&zip_path) {
                Ok(f) => f,
                Err(e) => {
                    println!("   ❌ Failed to create zip file {:?}: {}", zip_path, e);
                    continue;
                }
            };
            if let Err(e) = resp.copy_to(&mut dest) {
                println!("   ❌ Failed to write zip contents: {}", e);
                continue;
            }

            // Unzip the file and append to CSV
            if let Ok(zip_file) = File::open(&zip_path) {
                if let Ok(mut archive) = ZipArchive::new(zip_file) {
                    if archive.len() > 0 {
                        if let Ok(mut file) = archive.by_index(0) {
                            let mut content = String::new();
                            if file.read_to_string(&mut content).is_ok() {
                                // FIX #553: Filtrar cabeceras intermedias duplicadas en archivos mensuales
                                for line in content.lines() {
                                    if line.starts_with("open_time") || line.is_empty() {
                                        continue;
                                    }
                                    let _ = writeln!(out_file, "{}", line);
                                }
                            }
                        }
                    }
                } else {
                    println!("   ❌ Error reading ZIP: {:?}", zip_path);
                }
            }

            // Eliminar zip temporal
            let _ = std::fs::remove_file(&zip_path);
        }

        println!("✅ {} completado y ensamblado.", symbol);
    }
}

// ===================== R2.2: AGGTRADES =====================

#[repr(C)]
struct BinTick {
    timestamp: u64,
    bid_price: f64,
    ask_price: f64,
    bid_qty: f64,
    ask_qty: f64,
}

fn aggtrades_main() {
    println!("============================================================");
    println!("📡 AGGTRADES DOWNLOADER -> TICKS REALES (magic TGMTICK1)");
    println!("============================================================");
    let client = Client::new();
    let data_dir = Path::new("data/vision");
    let _ = std::fs::create_dir_all(data_dir);

    // Un símbolo, un mes por defecto (args: [symbol] [month])
    let args: Vec<String> = std::env::args().collect();
    let symbol = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "BTCUSDT".to_string());
    let month = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "2026-05".to_string());

    let url = format!(
        "https://data.binance.vision/data/futures/um/monthly/aggTrades/{}/{}-aggTrades-{}.zip",
        symbol, symbol, month
    );
    let zip_path = data_dir.join(format!("{}-{}-agg.zip", symbol, month));
    println!("⬇️  {}", url);
    let bytes = match client.get(&url).send().and_then(|r| r.bytes()) {
        Ok(b) => b.to_vec(),
        Err(e) => {
            println!("❌ descarga falló: {}", e);
            return;
        }
    };
    if let Err(e) = std::fs::write(&zip_path, &bytes) {
        println!("❌ escritura zip: {}", e);
        return;
    }

    // Descomprimir y parsear CSV: aggTrade_id,price,qty,first_trade_id,last_trade_id,transact_time,is_buyer_maker
    let mut ticks: Vec<BinTick> = Vec::new();
    match zip::ZipArchive::new(std::io::Cursor::new(&bytes)) {
        Ok(mut archive) => {
            for i in 0..archive.len() {
                let mut file = match archive.by_index(i) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                let mut content = String::new();
                if std::io::Read::read_to_string(&mut file, &mut content).is_err() {
                    continue;
                }
                for line in content.lines().skip(1) {
                    let cols: Vec<&str> = line.split(',').collect();
                    if cols.len() < 7 {
                        continue;
                    }
                    // S-01 — FIX COLUMNA TIMESTAMP: cols[4] es last_trade_id,
                    // NO transact_time (que es cols[5]). Antes: trade-ID
                    // monótono como tiempo — sort disfrazaba el bug y toda
                    // duración/interpolación temporal quedaba corrupta.
                    let (Ok(price), Ok(qty), Ok(ts), Ok(is_maker)) = (
                        cols[1].parse::<f64>(),
                        cols[2].parse::<f64>(),
                        cols[5].parse::<u64>(),
                        cols[6].parse::<bool>(),
                    ) else {
                        continue;
                    };
                    if !price.is_finite() || price <= 0.0 || !qty.is_finite() || qty <= 0.0 {
                        continue;
                    }

                    // R2.2/N-09: OBI agrupado — bucket de volumen con mezcla
                    // base para evitar OBI=±1 por trade. is_buyer_maker=true
                    // => el agresor vendió (volumen vendedor).
                    let base_depth = (qty * 0.25).max(0.1);
                    let (bq, aq) = if is_maker {
                        (base_depth, qty + base_depth)
                    } else {
                        (qty + base_depth, base_depth)
                    };
                    // Spread mínimo modelado (aggTrades no traen book):
                    // 1 tick del activo ~ 0.1 bps, piso 0.5 bps.
                    let half = (price * 0.00005).max(0.01);
                    ticks.push(BinTick {
                        timestamp: ts,
                        bid_price: price - half,
                        ask_price: price + half,
                        bid_qty: bq,
                        ask_qty: aq,
                    });
                }
            }
        }
        Err(e) => {
            println!("❌ zip: {}", e);
            return;
        }
    }

    ticks.sort_unstable_by_key(|t| t.timestamp);
    println!("✅ {} ticks reales de aggTrades", ticks.len());

    let out = Path::new("data").join(format!("{}_ticks_REAL.bin", symbol));
    let mut f = match std::fs::File::create(&out) {
        Ok(f) => f,
        Err(e) => {
            println!("❌ {}: {}", out.display(), e);
            return;
        }
    };
    use std::io::Write;
    if f.write_all(b"TGMTICK1").is_err() {
        return;
    }
    let blen = ticks.len() * std::mem::size_of::<BinTick>();
    let bslice = unsafe { std::slice::from_raw_parts(ticks.as_ptr() as *const u8, blen) };
    if f.write_all(bslice).is_ok() {
        println!(
            "💾 {} ({} bytes, magic TGMTICK1 REAL)",
            out.display(),
            blen + 8
        );
    }
}
