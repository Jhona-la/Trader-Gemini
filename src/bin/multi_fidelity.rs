use quantum_engine::unified_engine::{run_backtest_native, UnifiedConfig};
use quantum_engine::ml_inference::NanoForest;
use std::fs::File;
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

static RNG_STATE: AtomicU64 = AtomicU64::new(0);

fn random_f64(min: f64, max: f64) -> f64 {
    let mut s = RNG_STATE.fetch_add(1, Ordering::Relaxed)
        .wrapping_add(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    RNG_STATE.store(s, Ordering::Relaxed);
    let norm = (s as f64) / (u64::MAX as f64);
    min + norm * (max - min)
}

fn main() {
    println!("============================================================");
    println!("⚔️ MULTI-FIDELITY TOURNAMENT ORCHESTRATOR (QUICK-DEATH SIM)");
    println!("============================================================");

    let symbol = "BTCUSDT";
    let path = format!("models/{}_SCALP.json", symbol);
    if NanoForest::load_global(&format!("{}_SCALP", symbol), &path).is_err() {
        println!("⚠️ Warning: Failed to load NanoForest from {}. Baseline metrics only.", path);
    }

    let file_path = format!("data/{}_ticks.bin", symbol);
    let file = match File::open(&file_path) {
        Ok(f) => f,
        Err(_) => {
            println!("❌ Failed to open data file: {}", file_path);
            return;
        }
    };
    
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).expect("Failed to mmap file") };
    let bytes_len = mmap.len();
    let len = bytes_len / (4 * 8); 
    
    if len < 50_000 {
        println!("❌ Insufficient data for tournament (Need >50k ticks).");
        return;
    }
    
    // Quick-Death Window: First 10,000 ticks
    let low_fidelity_len = 10_000;
    
    let ptr = mmap.as_ptr() as *const f64;
    let _timestamps = unsafe { std::slice::from_raw_parts(ptr, len) };
    let closes = unsafe { std::slice::from_raw_parts(ptr.add(len), len) };
    let highs = closes;
    let lows = closes;
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(len * 2), len) };
    
    let generation_size = 50;
    
    println!("🧬 Generating {} candidates...", generation_size);
    let mut candidates = Vec::new();
    for _ in 0..generation_size {
        candidates.push(UnifiedConfig {
            sl_pct: random_f64(0.001, 0.02),
            tp_pct: random_f64(0.002, 0.05),
            ml_threshold_l: random_f64(0.5, 0.9),
            ml_threshold_s: random_f64(0.5, 0.9),
            tech_threshold_l: random_f64(0.001, 0.005),
            tech_threshold_s: random_f64(0.001, 0.005),
        });
    }

    // Phase 1: Low-Fidelity (Quick Death)
    println!("🏎️ Phase 1: Low-Fidelity Quick Death ({} ticks)", low_fidelity_len);
    let start_p1 = Instant::now();
    let mut survivors = Vec::new();
    
    for (_i, cfg) in candidates.iter().enumerate() {
        let mut out_pnl = vec![0.0; low_fidelity_len];
        let mut out_stats = [0.0; 4]; // [win_rate, trades, capital, max_dd]
        
        let lf_closes = &closes[..low_fidelity_len];
        let lf_highs = &highs[..low_fidelity_len];
        let lf_lows = &lows[..low_fidelity_len];
        let lf_volumes = &volumes[..low_fidelity_len];
        
        run_backtest_native(
            lf_closes, lf_highs, lf_lows, lf_volumes,
            cfg, &mut out_pnl, &mut out_stats, symbol
        );
        
        let capital = out_stats[2];
        let dd = out_stats[3];
        
        // Survival Rules: Must not lose initial 13.0, and drawdown < 5%
        if capital >= 13.0 && dd < 0.05 {
            survivors.push(cfg.clone());
        }
    }
    
    println!("🏁 Phase 1 Complete in {:?}. Survivors: {}/{}", start_p1.elapsed(), survivors.len(), generation_size);
    
    if survivors.is_empty() {
        println!("💀 All candidates failed Quick-Death. Consider loosening thresholds.");
        return;
    }
    
    // Phase 2: High-Fidelity
    println!("🔬 Phase 2: High-Fidelity Deep Simulation ({} ticks)", len);
    let start_p2 = Instant::now();
    
    let mut best_cfg = survivors[0].clone();
    let mut best_capital = 0.0;
    
    for (i, cfg) in survivors.iter().enumerate() {
        let mut out_pnl = vec![0.0; len];
        let mut out_stats = [0.0; 4]; 
        
        run_backtest_native(
            closes, highs, lows, volumes,
            cfg, &mut out_pnl, &mut out_stats, symbol
        );
        
        let capital = out_stats[2];
        if capital > best_capital {
            best_capital = capital;
            best_cfg = cfg.clone();
        }
        
        println!("   Survivor {} => Final Capital: ${:.2}", i, capital);
    }
    
    println!("🏆 Tournament Winner Found in {:?}", start_p2.elapsed());
    println!("   Best Capital: ${:.2}", best_capital);
    println!("   SL: {:.4}, TP: {:.4}, ML_L: {:.4}, ML_S: {:.4}", 
             best_cfg.sl_pct, best_cfg.tp_pct, best_cfg.ml_threshold_l, best_cfg.ml_threshold_s);
}
