use std::fs::File;
use std::time::Instant;
use quantum_engine::unified_engine::{run_backtest_native, UnifiedConfig};
use quantum_engine::ml_inference::NanoForest;

fn main() {
    println!("========================================================");
    println!("🧪 STRESS SIMULATOR (QUANTUM ENGINE - ZERO ALLOC)");
    println!("========================================================");

    let symbol = "BTCUSDT";

    // Intentar cargar modelo de IA
    let path = format!("models/{}_SCALP.json", symbol);
    if let Err(_) = NanoForest::load_global(&format!("{}_SCALP", symbol), &path) {
        println!("⚠️ No NanoForest loaded, using fallback EMA logic.");
    }

    let file_path = format!("data/{}_ticks.bin", symbol);
    let file = match File::open(&file_path) {
        Ok(f) => f,
        Err(_) => {
            println!("❌ Data file not found: {}", file_path);
            return;
        }
    };

    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).expect("Failed to mmap file") };
    let bytes_len = mmap.len();
    let len = bytes_len / (4 * 8); 

    if len == 0 {
        println!("❌ No data loaded.");
        return;
    }

    let ptr = mmap.as_ptr() as *const f64;
    let _timestamps = unsafe { std::slice::from_raw_parts(ptr, len) };
    let closes = unsafe { std::slice::from_raw_parts(ptr.add(len), len) };
    let highs = closes;
    let lows = closes;
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(len * 2), len) };

    println!("✅ Memory-Mapped {} ticks for Stress Simulation", len);

    let cfg = UnifiedConfig {
        sl_pct: 0.0015,
        tp_pct: 0.0035,
        ml_threshold_l: 0.55,
        ml_threshold_s: 0.55,
        tech_threshold_l: 0.001,
        tech_threshold_s: 0.001,
        starting_capital: 13.0,
        scalp_leverage: 100.0, // High leverage for stress
        swing_leverage: 20.0,
        scalp_sl_ratio: 1.0,
        scalp_tp_ratio: 1.0,
    };

    let mut out_pnl = vec![0.0; len];
    let mut out_stats = [0.0; 4];

    println!("🚀 Firing QuantumEngine Core...");
    let start_time = Instant::now();
    
    let trades = run_backtest_native(
        closes, highs, lows, volumes,
        &cfg,
        &mut out_pnl,
        &mut out_stats,
        symbol
    );
    
    let elapsed = start_time.elapsed();
    let total_nanos = elapsed.as_nanos();
    let nanos_per_tick = total_nanos as f64 / len as f64;

    let total_capital = out_stats[2];
    let max_dd = out_stats[1];

    println!("✅ SIMULACIÓN COMPLETADA");
    println!("⏱️ Tiempo Total: {:?}", elapsed);
    println!("⚡ Rendimiento: {:.2} nanosegundos / tick", nanos_per_tick);
    
    if nanos_per_tick < 1000.0 {
        println!("🟢 APROBADO: Latencia sub-microsegundo (Regla de 1 microsegundo cumplida).");
    } else {
        println!("🔴 REPROBADO: Latencia excede 1 microsegundo por evento.");
    }

    println!("💰 Capital Final: ${:.4} USD (Inicio: $13.0000 USD)", total_capital);
    println!("📊 Total Trades: {}", trades);
    println!("📉 Max Drawdown: {:.2}%", max_dd * 100.0);
    
    let multiplier = total_capital / 13.0;
    if multiplier >= 2.0 {
        println!("🚀 SINGULARIDAD LOGRADA: {:.2}x COMPOUNDING CONFIRMADO", multiplier);
    } else {
        println!("⚠️ Falla de Singularidad: {:.2}x", multiplier);
    }
}
