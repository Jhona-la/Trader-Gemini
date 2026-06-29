use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("data/BTCUSDT_ticks.csv")?;
    let reader = BufReader::new(file);

    let mut timestamps = Vec::new();
    let mut prices = Vec::new();
    let mut quantities = Vec::new();
    let mut is_buyer_maker = Vec::new();

    let mut count = 0;
    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; } // skip header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 7 { continue; }

        if let (Ok(p), Ok(q), Ok(t), is_bm) = (
            parts[1].parse::<f64>(),
            parts[2].parse::<f64>(),
            parts[5].parse::<f64>(),
            parts[6] == "true" || parts[6] == "True"
        ) {
            prices.push(p);
            quantities.push(q);
            timestamps.push(t);
            is_buyer_maker.push(if is_bm { 1.0 } else { 0.0 });
            count += 1;
        }
    }

    let bin_path = "data/BTCUSDT_ticks.bin";
    let mut bin_file = File::create(bin_path)?;

    // Write contiguous blocks
    let slice_t = unsafe { std::slice::from_raw_parts(timestamps.as_ptr() as *const u8, timestamps.len() * 8) };
    bin_file.write_all(slice_t)?;

    let slice_p = unsafe { std::slice::from_raw_parts(prices.as_ptr() as *const u8, prices.len() * 8) };
    bin_file.write_all(slice_p)?;

    let slice_q = unsafe { std::slice::from_raw_parts(quantities.as_ptr() as *const u8, quantities.len() * 8) };
    bin_file.write_all(slice_q)?;

    let slice_bm = unsafe { std::slice::from_raw_parts(is_buyer_maker.as_ptr() as *const u8, is_buyer_maker.len() * 8) };
    bin_file.write_all(slice_bm)?;

    println!("✅ Parsed {} ticks and saved to {}", count, bin_path);

    Ok(())
}
