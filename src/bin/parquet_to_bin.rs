use polars::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

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

        let opens = df.column("open")?.f64()?;
        let highs = df.column("high")?.f64()?;
        let lows = df.column("low")?.f64()?;
        let closes = df.column("close")?.f64()?;
        let volumes = df.column("volume")?.f64()?;
        let open_times = df.column("open_time")?.u64()?;

        let count = df.height();
        
        let mut c_arr: Vec<f64> = Vec::with_capacity(count);
        let mut h_arr: Vec<f64> = Vec::with_capacity(count);
        let mut l_arr: Vec<f64> = Vec::with_capacity(count);
        let mut v_arr: Vec<f64> = Vec::with_capacity(count);

        for i in 0..count {
            c_arr.push(closes.get(i).unwrap_or(0.0));
            h_arr.push(highs.get(i).unwrap_or(0.0));
            l_arr.push(lows.get(i).unwrap_or(0.0));
            v_arr.push(volumes.get(i).unwrap_or(0.0));
        }

        let mut bin_file = File::create(&out_path)?;

        let write_slice = |file: &mut File, data: &[f64]| -> std::io::Result<()> {
            let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
            file.write_all(bytes)
        };

        // Evolution.rs format is: [closes][highs][lows][volumes]
        write_slice(&mut bin_file, &c_arr)?;
        write_slice(&mut bin_file, &h_arr)?;
        write_slice(&mut bin_file, &l_arr)?;
        write_slice(&mut bin_file, &v_arr)?;

        println!("✅ Converted {} → {} ({} candles)", in_path.display(), out_path.display(), count);
    }

    Ok(())
}
