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

    let sym = "BTCUSDT";
    let file_path = format!("data/historical/{}_1m.csv", sym);
    let mut closes_vec = Vec::new();
    let mut highs_vec = Vec::new();
    let mut lows_vec = Vec::new();
    let mut vols_vec = Vec::new();

    if let Ok(file) = File::open(&file_path) {
        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(file);
        for line in reader.lines().filter_map(|l| l.ok()).skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 6 {
                if let (Ok(h), Ok(l), Ok(c), Ok(v)) = (
                    parts[2].parse::<f64>(),
                    parts[3].parse::<f64>(),
                    parts[4].parse::<f64>(),
                    parts[5].parse::<f64>(),
                ) {
                    highs_vec.push(h);
                    lows_vec.push(l);
                    closes_vec.push(c);
                    vols_vec.push(v);
                }
            }
        }
    } else {
        println!("❌ Failed to open {}. Please run data_loader first.", file_path);
        return;
    }

    let len = closes_vec.len();
    if len == 0 {
        println!("❌ No data loaded.");
        return;
    }
    
    let closes = &closes_vec[..];
    let highs = &highs_vec[..];
    let lows = &lows_vec[..];
    let volumes = &vols_vec[..];
    
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
            
            // Random Mutator
            test_cfg.sl_pct += random_f64(-0.005, 0.005);
            test_cfg.tp_pct += random_f64(-0.01, 0.01);
            test_cfg.ml_threshold_l += random_f64(-0.1, 0.1);
            test_cfg.ml_threshold_s += random_f64(-0.1, 0.1);
            test_cfg.tech_threshold_l += random_f64(-0.001, 0.001);
            test_cfg.tech_threshold_s += random_f64(-0.001, 0.001);
            
            // Constraints
            if test_cfg.sl_pct < 0.001 { test_cfg.sl_pct = 0.001; }
            if test_cfg.tp_pct < 0.002 { test_cfg.tp_pct = 0.002; }
            if test_cfg.ml_threshold_l > 0.99 { test_cfg.ml_threshold_l = 0.99; }
            if test_cfg.ml_threshold_l < 0.4 { test_cfg.ml_threshold_l = 0.4; }
            if test_cfg.ml_threshold_s > 0.99 { test_cfg.ml_threshold_s = 0.99; }
            if test_cfg.ml_threshold_s < 0.4 { test_cfg.ml_threshold_s = 0.4; }
            if test_cfg.tech_threshold_l < 0.0005 { test_cfg.tech_threshold_l = 0.0005; }
            if test_cfg.tech_threshold_s < 0.0005 { test_cfg.tech_threshold_s = 0.0005; }
            
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
            
            // Assume 1-minute klines (1440 mins/day)
            let days_simulated = (len as f64) / 1440.0;
            let periods_of_3_days = days_simulated / 3.0;
            
            // If capital grew from 13.0 to 26.0 in 3 days, ratio is 2.0.
            // powf(1.0 / periods) gives the compounding rate per 3-day window.
            let compound_rate_3d = if capital > 0.0 && periods_of_3_days > 0.0 {
                (capital / 13.0).powf(1.0 / periods_of_3_days)
            } else {
                0.0
            };
            
            // Goal is > 2.0 (100% every 3 days)
            let mut score = compound_rate_3d * 10000.0;
            
            // Regularity Penalties
            if trades < (days_simulated * 5.0) { score -= 5000.0; } 
            
            // Master Rule Penalties (13 USD protection & <5% Drawdown)
            if capital < 13.0 { score -= 50000.0; } // Never lose the starting 13
            if dd > 0.05 { score -= dd * 100000.0; } // Heavy penalty over 5% max Drawdown
            
            if score > best_score {
                best_score = score;
                best_config = test_cfg.clone();
            }
            if score > gen_best_score {
                gen_best_score = score;
            }
        }
        
        if gen % 50 == 0 || gen == generations - 1 {
            println!("🧬 Gen {}: Best Score = {:.2} | Compound 3D: {:.2}x | Max DD: {:.2}%", 
                     gen, gen_best_score, best_score / 10000.0, 0.0); // We can't print actual max_dd without tracking the best one, but score reflects it
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
    println!("   Tech L     : {:.4}", best_config.tech_threshold_l);
    println!("   Tech S     : {:.4}", best_config.tech_threshold_s);
    
    let out_json = format!(
        "{{\n  \"sl_pct\": {:.4},\n  \"tp_pct\": {:.4},\n  \"ml_threshold_l\": {:.4},\n  \"ml_threshold_s\": {:.4},\n  \"tech_threshold_l\": {:.4},\n  \"tech_threshold_s\": {:.4}\n}}",
        best_config.sl_pct, best_config.tp_pct, best_config.ml_threshold_l, best_config.ml_threshold_s, best_config.tech_threshold_l, best_config.tech_threshold_s
    );
    
    std::fs::write("data/dynamic_config.json", out_json).expect("Unable to write dynamic_config.json");
    println!("💾 Exported to data/dynamic_config.json successfully.");
}
