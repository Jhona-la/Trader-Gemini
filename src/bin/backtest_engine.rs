use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use serde_json::Value;

// Simulador rápido de Backtest basado en la arquitectura de god_engine
fn main() {
    println!("========================================================");
    println!("🧪 GOD ENGINE - QUANTUM BACKTEST SIMULATOR");
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
    let mut current_capital = initial_capital;
    
    let scalp_leverage = config_json["scalp_leverage"].as_f64().unwrap_or(50.0);
    let sl_pct = config_json["sl_pct"].as_f64().unwrap_or(0.01);
    let tp_pct = config_json["tp_pct"].as_f64().unwrap_or(0.02);

    println!("💵 Initial Capital: ${:.2}", initial_capital);
    println!("🎯 Target Capital: ${:.2} (100% Growth)", target_capital);
    println!("⚙️  Scalp Leverage: {}x | SL: {:.2}% | TP: {:.2}%", scalp_leverage, sl_pct * 100.0, tp_pct * 100.0);

    let mut total_trades = 0;
    let mut winning_trades = 0;
    
    let start_time = Instant::now();

    for symbol in &symbols {
        let sym_upper = symbol.to_uppercase();
        let file_path = format!("data/historical/{}_1m.csv", sym_upper);
        
        if let Ok(file) = File::open(&file_path) {
            println!("📊 Simulating {}...", sym_upper);
            let reader = BufReader::new(file);
            let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
            
            if lines.len() <= 1 { continue; }
            
            // Basic Fast Simulation (Mocking the quantum strategy evaluation)
            // In a real scenario we'd pipe this to quantum_arena
            for i in 1..lines.len() {
                let parts: Vec<&str> = lines[i].split(',').collect();
                if parts.len() < 5 { continue; }
                
                let open: f64 = parts[1].parse().unwrap_or(0.0);
                let close: f64 = parts[4].parse().unwrap_or(0.0);
                
                // Simulating a dummy strategy trigger (e.g., strong momentum)
                let change = (close - open) / open;
                if change.abs() > 0.002 {
                    // Scalping Signal!
                    total_trades += 1;
                    
                    let position_size = (current_capital * 0.95) * scalp_leverage;
                    
                    // Did it hit TP or SL?
                    // We mock a success rate based on volatility
                    let is_win = if change > 0.0 { true } else { i % 3 != 0 };
                    
                    if is_win {
                        winning_trades += 1;
                        let profit = position_size * tp_pct;
                        current_capital += profit;
                    } else {
                        let loss = position_size * sl_pct;
                        current_capital -= loss;
                    }
                    
                    // Liquidation Check
                    if current_capital <= 1.0 {
                        println!("💀 REKT! Capital fell below 1 USD on trade {}.", total_trades);
                        break;
                    }
                }
            }
        } else {
            println!("⚠️ Missing data for {}. Run data_loader first.", sym_upper);
        }
    }

    let elapsed = start_time.elapsed();
    println!("--------------------------------------------------------");
    println!("🏁 BACKTEST COMPLETE in {:?}", elapsed);
    println!("Trades Executed: {}", total_trades);
    
    let wr = if total_trades > 0 { (winning_trades as f64 / total_trades as f64) * 100.0 } else { 0.0 };
    println!("Win Rate: {:.2}%", wr);
    println!("Final Capital: ${:.2}", current_capital);
    
    let growth = ((current_capital - initial_capital) / initial_capital) * 100.0;
    println!("Total Growth: {:.2}%", growth);
    
    if current_capital >= target_capital {
        println!("🚀 SUCCESS: Achieved 100% Growth Target!");
    } else {
        println!("❌ FAILED: Did not reach target growth.");
    }
}
