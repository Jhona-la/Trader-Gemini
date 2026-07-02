use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use god_engine_core::stateful_engine::StatefulEngine;

fn main() {
    println!("============================================================");
    println!("🌌 RUST QUANTUM FEATURE EXPORTER (1:1 ALIGNMENT)");
    println!("============================================================");

    let symbol = "BTCUSDT";
    let input_path = format!("data/{}_ticks.bin", symbol);
    
    let file = match File::open(&input_path) {
        Ok(f) => f,
        Err(_) => {
            println!("⚠️ Failed to open {}", input_path);
            return;
        }
    };
    
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).expect("Failed to mmap file") };
    let bytes_len = mmap.len();
    let len = bytes_len / (5 * 8); // 5 arrays of f64
    
    if len == 0 {
        println!("❌ No data loaded or file is empty.");
        return;
    }
    
    let ptr = mmap.as_ptr() as *const f64;
    let _timestamps = unsafe { std::slice::from_raw_parts(ptr, len) };
    let closes = unsafe { std::slice::from_raw_parts(ptr.add(len), len) };
    let highs = unsafe { std::slice::from_raw_parts(ptr.add(len * 2), len) };
    let lows = unsafe { std::slice::from_raw_parts(ptr.add(len * 3), len) };
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(len * 4), len) };
    
    println!("✅ Loaded {} candles for {}", len, symbol);
    
    let out_path = format!("data/{}_FEATURES.csv", symbol);
    let mut out_file = File::create(&out_path).unwrap();
    
    // Escribir cabeceras
    let mut header = String::from("target_5m");
    for i in 0..25 {
        header.push_str(&format!(",feature_{}", i));
    }
    writeln!(out_file, "{}", header).unwrap();
    
    let mut feature_engine = StatefulEngine::new();
    let mut written = 0;
    
    for i in 0..len {
        let current_close = closes[i];
        let current_vol = volumes[i];
        
        let prev_close = if i > 0 { closes[i - 1] } else { current_close };
        let delta = current_close - prev_close;
        
        let mut bid_ratio = 0.5;
        if delta > 0.0 { bid_ratio = 1.0; } 
        else if delta < 0.0 { bid_ratio = 0.0; }
        
        let bid_qty = current_vol * bid_ratio;
        let ask_qty = current_vol * (1.0 - bid_ratio);
        let bid = lows[i];
        let ask = highs[i];
        let total_vol = bid_qty + ask_qty;
        let depth_obi = if total_vol > 0.0 { (bid_qty - ask_qty) / total_vol } else { 0.0 };
        
        let mid_price = (bid + ask) / 2.0;
        let pseudo_maker = bid_qty > ask_qty;
        
        feature_engine.process_tick(mid_price, total_vol);
        feature_engine.update_trade_flow(total_vol, pseudo_maker);
        let _ = feature_engine.update_ofi(bid, ask, bid_qty, ask_qty);
        
        // Calcular Target a futuro (Forward Return de 5 periodos)
        if i >= 100 && i + 5 < len {
            let future_return = (closes[i + 5] - closes[i]) / closes[i];
            // Generar features idénticas a GodEngineCore
            let stateful_feats = feature_engine.get_features();
            
            let mut features = [0.0; 25];
            features[0] = stateful_feats[0]; // EMA trend
            features[1] = stateful_feats[1]; // Hurst
            features[2] = stateful_feats[2]; // OFI
            features[3] = feature_engine.get_atr_pct() as f32;
            
            let mut row = format!("{:.6}", future_return);
            for f in &features {
                row.push_str(&format!(",{:.6}", f));
            }
            writeln!(out_file, "{}", row).unwrap();
            written += 1;
        }
    }
    
    println!("🚀 Exporter finished! Wrote {} rows to {}", written, out_path);
}
