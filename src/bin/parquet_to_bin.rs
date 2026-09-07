use polars::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct BinTick {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================");
    println!("📊 PARQUET → BINARY CONVERTER (For Evolution Engine)");
    println!("========================================================");

    let symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"];
    let data_dir = Path::new("data/historical");
    let out_dir = Path::new("data");

    for symbol in symbols.iter() {
        let in_path = data_dir.join(format!("{}_6M.parquet", symbol));
        let out_path = out_dir.join(format!("{}_ticks.bin", symbol));

        if !in_path.exists() {
            println!("⚠️ Skipping {} (Parquet not found)", symbol);
            continue;
        }

        let mut file = File::open(&in_path)?;
        let df = ParquetReader::new(&mut file).finish()?;

        let opens = df.column("open").ok().and_then(|c| c.f64().ok());
        let highs = df.column("high")?.f64()?;
        let lows = df.column("low")?.f64()?;
        let closes = df.column("close")?.f64()?;
        let volumes = df.column("volume")?.f64()?;
        let open_times = df.column("open_time")?.u64()?;

        let count = df.height();
        let mut ticks: Vec<BinTick> = Vec::with_capacity(count * 4);

        for i in 0..count {
            let raw_close = closes.get(i).unwrap_or(0.0);
            if raw_close <= 0.0 || !raw_close.is_finite() {
                continue;
            }
            let close = raw_close;
            let open = opens.as_ref().and_then(|o| o.get(i)).unwrap_or(close);
            let raw_high = highs.get(i).unwrap_or(close);
            let high = if raw_high.is_finite() { raw_high.max(close).max(open) } else { close };
            let raw_low = lows.get(i).unwrap_or(close);
            let low = if raw_low.is_finite() { raw_low.min(close).min(open).max(1e-6) } else { close * 0.999 };
            let raw_vol = volumes.get(i).unwrap_or(1.0);
            let volume = if raw_vol.is_finite() && raw_vol >= 0.0 { raw_vol } else { 1.0 };
            let ts = open_times.get(i).unwrap_or(0);

            // Causalidad estricta: determinar la tendencia inicial a partir del paso previo, no del cierre futuro
            let prev_open = if i > 0 { opens.as_ref().and_then(|o| o.get(i - 1)).unwrap_or(open) } else { open };
            let initial_trend_up = open >= prev_open;
            let spread = (high - low).max(close * 0.0001);
            let quarter_vol = (volume * 0.25).max(0.001);

            // Sub-tick 1: Apertura (t + 0s)
            let bid1 = (open - spread * 0.5).max(1e-6);
            let ask1 = (open + spread * 0.5).max(bid1 + 1e-6);
            ticks.push(BinTick {
                timestamp: ts,
                bid_price: bid1,
                ask_price: ask1,
                bid_qty: quarter_vol,
                ask_qty: quarter_vol,
            });

            // Sub-tick 2: Extremo 1 (t + 15s) - Explora extremo inicial guiado causalmente
            let p2 = if initial_trend_up { high } else { low };
            let bid2 = (p2 - spread * 0.5).max(1e-6);
            let ask2 = (p2 + spread * 0.5).max(bid2 + 1e-6);
            let p2_up = p2 >= open;
            let (b_qty2, a_qty2) = if p2_up {
                (quarter_vol * 1.15, quarter_vol * 0.85) // Imbalance causal local
            } else {
                (quarter_vol * 0.85, quarter_vol * 1.15)
            };
            ticks.push(BinTick {
                timestamp: ts + 15_000,
                bid_price: bid2,
                ask_price: ask2,
                bid_qty: b_qty2,
                ask_qty: a_qty2,
            });

            // Sub-tick 3: Extremo 2 (t + 35s) - Explora el extremo opuesto
            let p3 = if initial_trend_up { low } else { high };
            let bid3 = (p3 - spread * 0.5).max(1e-6);
            let ask3 = (p3 + spread * 0.5).max(bid3 + 1e-6);
            let p3_up = p3 >= p2;
            let (b_qty3, a_qty3) = if p3_up {
                (quarter_vol * 1.15, quarter_vol * 0.85)
            } else {
                (quarter_vol * 0.85, quarter_vol * 1.15)
            };
            ticks.push(BinTick {
                timestamp: ts + 35_000,
                bid_price: bid3,
                ask_price: ask3,
                bid_qty: b_qty3,
                ask_qty: a_qty3,
            });

            // Sub-tick 4: Cierre (t + 55s) - Convergencia al cierre de la vela
            let bid4 = (close - spread * 0.5).max(1e-6);
            let ask4 = (close + spread * 0.5).max(bid4 + 1e-6);
            let p4_up = close >= p3;
            let (b_qty4, a_qty4) = if p4_up {
                (quarter_vol * 1.15, quarter_vol * 0.85)
            } else {
                (quarter_vol * 0.85, quarter_vol * 1.15)
            };
            ticks.push(BinTick {
                timestamp: ts + 55_000,
                bid_price: bid4,
                ask_price: ask4,
                bid_qty: b_qty4,
                ask_qty: a_qty4,
            });
        }

        let mut bin_file = File::create(&out_path)?;
        // R2.4 — header magic+version: permite al lector validar formato y
        // rechazar ruidosamente archivos de otra versión (antes: basura
        // deserializada en silencio).
        bin_file.write_all(backtest_engine::tick_replayer::TICK_MAGIC)?;
        let byte_len = ticks.len() * std::mem::size_of::<BinTick>();
        let bytes = unsafe { std::slice::from_raw_parts(ticks.as_ptr() as *const u8, byte_len) };
        bin_file.write_all(bytes)?;

        println!(
            "✅ Converted {} → {} ({} BinTicks, {} bytes)",
            in_path.display(),
            out_path.display(),
            ticks.len(),
            byte_len
        );
    }

    Ok(())
}
