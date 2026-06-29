use quantum_engine::unified_engine::{ffi_run_unified_backtest, UnifiedConfig};
use quantum_engine::ml_inference::NanoForest;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn random_f64(min: f64, max: f64) -> f64 {
    let rand_val = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as f64 / 1_000_000_000.0;
    min + rand_val * (max - min)
}

fn main() {
    println!("============================================================");
    println!("🌌 RUST QUANTUM EVOLUTION ENGINE (GENETIC ALGORITHM)");
    println!("============================================================");

    let forest_path = "models/nano_forest.json";
    if let Ok(_) = NanoForest::load_from_json(forest_path) {
        println!("✅ NanoForest Loaded for Evolution: {}", forest_path);
    } else {
        println!("⚠️ NanoForest NOT FOUND! Evolution will proceed with dummy/random probabilities.");
    }

    let mmap_path = "data/backtest_mmap.bin";
    let mut file = match File::open(mmap_path) {
        Ok(f) => f,
        Err(_) => {
            println!("❌ Failed to open backtest_mmap.bin. Please run run_nanosecond_backtest.py first.");
            return;
        }
    };
    
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    
    let float_count = buffer.len() / 8;
    let len = float_count / 4;
    
    let ptr = buffer.as_ptr() as *const f64;
    let closes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let highs = unsafe { std::slice::from_raw_parts(ptr.add(len), len) };
    let lows = unsafe { std::slice::from_raw_parts(ptr.add(len * 2), len) };
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(len * 3), len) };
    
    println!("✅ Loaded {} ticks for Backtest Evolution.", len);
    
    let generations = 2000;
    let pop_size = 200;
    
    let mut best_config = UnifiedConfig {
        sl_pct: 0.015,
        tp_pct: 0.030,
        ml_threshold_l: 0.8,
        ml_threshold_s: 0.8,
        tech_threshold_l: 0.002,
        tech_threshold_s: 0.002,
    };
    
    let mut best_score = -9999999.0;
    
    println!("🚀 Starting Evolution: {} Gens x {} Pop = {} Nano-Backtests", generations, pop_size, generations * pop_size);
    let start_time = Instant::now();
    
    for gen in 0..generations {
        let mut gen_best_score = -9999999.0;
        
        for _ in 0..pop_size {
            let mut test_cfg = best_config.clone();
            test_cfg.sl_pct += random_f64(-0.005, 0.005);
            test_cfg.tp_pct += random_f64(-0.01, 0.01);
            test_cfg.ml_threshold_l += random_f64(-0.1, 0.1);
            test_cfg.ml_threshold_s += random_f64(-0.1, 0.1);
            
            if test_cfg.sl_pct < 0.001 { test_cfg.sl_pct = 0.001; }
            if test_cfg.tp_pct < 0.002 { test_cfg.tp_pct = 0.002; }
            if test_cfg.ml_threshold_l > 0.99 { test_cfg.ml_threshold_l = 0.99; }
            if test_cfg.ml_threshold_l < 0.4 { test_cfg.ml_threshold_l = 0.4; }
            if test_cfg.ml_threshold_s > 0.99 { test_cfg.ml_threshold_s = 0.99; }
            if test_cfg.ml_threshold_s < 0.4 { test_cfg.ml_threshold_s = 0.4; }
            
            let mut out_pnl = vec![0.0; len];
            let mut out_stats = [0.0; 4];
            
            ffi_run_unified_backtest(
                closes.as_ptr(),
                highs.as_ptr(),
                lows.as_ptr(),
                volumes.as_ptr(),
                len,
                &test_cfg,
                out_pnl.as_mut_ptr(),
                out_stats.as_mut_ptr()
            );
            
            let win_rate = out_stats[0];
            let trades = out_stats[1];
            let capital = out_stats[2];
            let dd = out_stats[3];
            
            let mut score = capital;
            if trades < 1.0 { score -= 500.0; } // We need at least some operations
            
            // Phase 28: Supreme 100% Win Rate priority (User explicitly requested 100% WR for Scalping thresholds)
            if win_rate < 0.999 {
                score -= 50000.0 * (1.0 - win_rate);
            } else {
                score += win_rate * 10000.0;
            }
            
            if dd > 0.05 { score -= dd * 5000.0; } // Extreme drawdown penalty
            if score > best_score {
                best_score = score;
                best_config = test_cfg.clone();
            }
            if score > gen_best_score {
                gen_best_score = score;
            }
        }
        
        if gen % 50 == 0 || gen == generations - 1 {
            println!("🧬 Gen {}: Best Score = {:.2} | SL: {:.4} | TP: {:.4} | WR: Target Max", gen, gen_best_score, best_config.sl_pct, best_config.tp_pct);
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("============================================================");
    println!("✅ Evolution Complete in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("🏆 Best Config Found:");
    println!("   SL %      : {:.4}", best_config.sl_pct);
    println!("   TP %      : {:.4}", best_config.tp_pct);
    println!("   ML Thresh L: {:.4}", best_config.ml_threshold_l);
    println!("   ML Thresh S: {:.4}", best_config.ml_threshold_s);
    
    let out_json = format!(
        "{{\n  \"sl_pct\": {:.4},\n  \"tp_pct\": {:.4},\n  \"ml_threshold_l\": {:.4},\n  \"ml_threshold_s\": {:.4},\n  \"tech_threshold_l\": {:.4},\n  \"tech_threshold_s\": {:.4}\n}}",
        best_config.sl_pct, best_config.tp_pct, best_config.ml_threshold_l, best_config.ml_threshold_s, best_config.tech_threshold_l, best_config.tech_threshold_s
    );
    
    std::fs::write("data/dynamic_config.json", out_json).expect("Unable to write dynamic_config.json");
    println!("💾 Exported to data/dynamic_config.json successfully.");
}
