use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use serde_json::Value;

fn main() {
    println!("========================================================");
    println!("🧪 GOD ENGINE - QUANTUM BACKTEST SIMULATOR (NATIVE)");
    println!("========================================================");

    let config_str = std::fs::read_to_string("data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
    let config_json: Value = serde_json::from_str(&config_str).unwrap_or(serde_json::json!({}));
    
    let default_symbols = vec!["btcusdt".to_string(), "ethusdt".to_string()];
    let symbols: Vec<String> = config_json["symbols"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or(default_symbols);

    let initial_capital = 13.0; // The 13 USD constraint
    let target_capital = initial_capital * 2.0; // 100% in 3 days
    
    let mut unified_cfg = quantum_engine::unified_engine::UnifiedConfig {
        sl_pct: config_json["sl_pct"].as_f64().unwrap_or(0.01),
        tp_pct: config_json["tp_pct"].as_f64().unwrap_or(0.02),
        ml_threshold_l: config_json["ml_threshold_l"].as_f64().unwrap_or(0.95),
        ml_threshold_s: config_json["ml_threshold_s"].as_f64().unwrap_or(0.95),
        tech_threshold_l: config_json["tech_threshold_l"].as_f64().unwrap_or(0.005),
        tech_threshold_s: config_json["tech_threshold_s"].as_f64().unwrap_or(0.005),
    };

    println!("💵 Initial Capital: ${:.2}", initial_capital);
    println!("🎯 Target Capital: ${:.2} (100% Growth)", target_capital);
    println!("⚙️  Unified Config: SL: {:.2}% | TP: {:.2}% | ML: {:.2}/{:.2}", 
        unified_cfg.sl_pct * 100.0, unified_cfg.tp_pct * 100.0, unified_cfg.ml_threshold_l, unified_cfg.ml_threshold_s);

    for symbol in &symbols {
        let sym_upper = symbol.to_uppercase();
        let file_path = format!("data/historical/{}_1m.csv", sym_upper);
        
        let start_time = Instant::now();
        if let Ok(file) = File::open(&file_path) {
            println!("--------------------------------------------------------");
            println!("📊 Loading Data for {}...", sym_upper);
            let reader = BufReader::new(file);
            let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
            
            if lines.len() <= 1 { continue; }
            
            let mut closes = Vec::with_capacity(lines.len());
            let mut highs = Vec::with_capacity(lines.len());
            let mut lows = Vec::with_capacity(lines.len());
            let mut volumes = Vec::with_capacity(lines.len());

            for i in 1..lines.len() {
                let parts: Vec<&str> = lines[i].split(',').collect();
                if parts.len() < 6 { continue; }
                
                // CSV Format: open_time,open,high,low,close,volume,...
                highs.push(parts[2].parse::<f64>().unwrap_or(0.0));
                lows.push(parts[3].parse::<f64>().unwrap_or(0.0));
                closes.push(parts[4].parse::<f64>().unwrap_or(0.0));
                volumes.push(parts[5].parse::<f64>().unwrap_or(0.0));
            }
            
            let len = closes.len();
            println!("✅ Loaded {} rows in {:?}", len, start_time.elapsed());
            
            let mut out_pnl = vec![0.0; len];
            let mut out_stats = vec![0.0; 4]; // win_rate, total_trades, final_capital, max_dd

            let sim_start = Instant::now();
            let total_trades = quantum_engine::unified_engine::run_backtest_native(
                &closes, &highs, &lows, &volumes,
                &unified_cfg,
                &mut out_pnl,
                &mut out_stats
            );
            
            let sim_elapsed = sim_start.elapsed();
            
            let win_rate = out_stats[0] * 100.0;
            let final_capital = out_stats[2];
            let max_dd = out_stats[3] * 100.0;
            let growth = ((final_capital - initial_capital) / initial_capital) * 100.0;
            
            println!("🏁 BACKTEST COMPLETE for {} in {:?}", sym_upper, sim_elapsed);
            println!("Trades Executed: {}", total_trades);
            println!("Win Rate: {:.2}%", win_rate);
            println!("Final Capital: ${:.2}", final_capital);
            println!("Max Drawdown: {:.2}%", max_dd);
            println!("Total Growth: {:.2}%", growth);
            
            if final_capital >= target_capital {
                println!("🚀 SUCCESS: Achieved 100% Growth Target!");
            } else {
                println!("❌ FAILED: Did not reach target growth.");
            }
        } else {
            println!("⚠️ Missing data for {}. Run data_loader first.", sym_upper);
        }
    }
}

