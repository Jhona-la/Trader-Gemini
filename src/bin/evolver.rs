use quantum_arena::{GlobalArena, TickEvent};
use god_engine_core::GodEngineCore;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::fs::File;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use rand::RngExt;
use std::time::Duration;
use memmap2::MmapOptions;
use std::mem::size_of;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Genotype {
    pub global_leverage: f64,
    pub trend_threshold: f64,
    pub maker_spread_pct: f64,
    pub maker_obi_threshold: f64,
    pub scalp_tp: f64,
    pub scalp_sl: f64,
    pub swing_tp: f64,
    pub swing_sl: f64,
    pub scalp_z_target: f64,
    pub capital_split_scalp: f64,
    pub min_confidence: f64,
    pub explosive_leverage_multiplier: f64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct BinTick {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    println!("============================================================");
    println!("🧬 TRADER GEMINI V5 - TICK-LEVEL REALITY EVOLVER");
    println!("============================================================");

    let arena_for_cap = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| Arc::new(GlobalArena::default()))
        .unwrap()
        .join()
        .unwrap();
        
    let initial_capital = arena_for_cap.config.base_capital.load(Ordering::Relaxed);
    println!("💰 Initial Capital: ${:.2}", initial_capital);

    let bin_path = "data/BTCUSDT_ticks.bin";
    println!("📥 Cargando datos REALES de alta frecuencia: {}", bin_path);
    
    let file = File::open(bin_path).expect("❌ Archivo BTCUSDT_ticks.bin no encontrado. Ejecuta los simuladores anteriores primero.");
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    
    let tick_size = size_of::<BinTick>();
    let num_ticks = mmap.len() / tick_size;
    println!("📊 Ticks reales cargados (Cero Fricción simulada): {}", num_ticks);
    
    let mut master_stream = Vec::with_capacity(num_ticks);
    
    for i in 0..num_ticks {
        let start = i * tick_size;
        let end = start + tick_size;
        let bytes = &mmap[start..end];
        let tick: BinTick = unsafe { std::ptr::read(bytes.as_ptr() as *const _) };
        
        // Filter out bad ticks just in case
        if tick.bid_price > 0.0 && tick.ask_price > tick.bid_price {
            master_stream.push(TickEvent {
                coin_id: 0, // BTCUSDT is index 0
                timestamp: tick.timestamp,
                bid_price: tick.bid_price,
                ask_price: tick.ask_price,
                bid_qty: tick.bid_qty,
                ask_qty: tick.ask_qty,
            });
        }
    }
    
    println!("✅ Ticks válidos en memoria: {}", master_stream.len());

    let pop_size = 50;
    let generations = 20;
    let mutation_rate = 0.3;
    let mut population: Vec<Genotype> = (0..pop_size).map(|_| {
        Genotype {
            global_leverage: rand::rng().random_range(10.0..50.0),
            trend_threshold: rand::rng().random_range(0.4..0.8),
            maker_spread_pct: rand::rng().random_range(0.0001..0.0010),
            maker_obi_threshold: rand::rng().random_range(0.4..0.8),
            scalp_tp: rand::rng().random_range(0.001..0.005),
            scalp_sl: rand::rng().random_range(0.0005..0.002),
            swing_tp: rand::rng().random_range(0.005..0.020),
            swing_sl: rand::rng().random_range(0.002..0.008),
            scalp_z_target: rand::rng().random_range(1.5..3.5),
            capital_split_scalp: rand::rng().random_range(0.5..0.9), // Prioritize scalp for 100% WR
            min_confidence: rand::rng().random_range(0.6..0.9),
            explosive_leverage_multiplier: rand::rng().random_range(1.5..4.0),
        }
    }).collect();

    println!("🚀 Iniciando Evolución de Supervivencia: {} Individuos x {} Generaciones...", pop_size, generations);

    let mut best_all_time = (population[0].clone(), 0.0_f64, 0);

    for gen in 1..=generations {
        println!("== GENERACIÓN {} ==", gen);
        
        let mut results: Vec<_> = population
            .par_iter()
            .map(|genome| {
                let arena = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(|| Arc::new(GlobalArena::default()))
                    .unwrap()
                    .join()
                    .unwrap();
                let initial_capital = arena.config.base_capital.load(Ordering::Relaxed);
                
                arena.config.global_leverage.store(genome.global_leverage, Ordering::Relaxed);
                arena.config.trend_threshold.store(genome.trend_threshold, Ordering::Relaxed);
                arena.config.scalp_obi_threshold.store(genome.scalp_z_target, Ordering::Relaxed);
                arena.config.maker_spread_pct.store(genome.maker_spread_pct, Ordering::Relaxed);
                arena.config.maker_obi_threshold.store(genome.maker_obi_threshold, Ordering::Relaxed);
                arena.config.scalp_tp_base.store(genome.scalp_tp, Ordering::Relaxed);
                arena.config.scalp_sl_base.store(genome.scalp_sl, Ordering::Relaxed);
                arena.config.swing_tp_base.store(genome.swing_tp, Ordering::Relaxed);
                arena.config.swing_sl_base.store(genome.swing_sl, Ordering::Relaxed);
                arena.config.capital_split_scalp.store(genome.capital_split_scalp, Ordering::Relaxed);
                arena.config.min_confidence_btc.store(genome.min_confidence, Ordering::Relaxed);
                arena.config.explosive_leverage_multiplier.store(genome.explosive_leverage_multiplier, Ordering::Relaxed);
                
                // Allow extreme compounding - accept 90% drawdown max
                arena.config.global_max_drawdown.store(0.90, Ordering::Relaxed);
                
                let mut engine = GodEngineCore::new(arena.clone());
                let mut total_trades = 0;
                let mut max_drawdown = 0.0;
                let mut peak_capital = initial_capital;

                for tick in &master_stream {
                    arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty);
                    let (_new_sc, _new_sw, closed_sc, closed_sw, _) = engine.process_tick(
                        tick.coin_id,
                        tick.bid_price,
                        tick.ask_price,
                        tick.bid_qty,
                        tick.ask_qty,
                        tick.timestamp, &[0.0; 54]);
                    
                    if closed_sc.is_some() || closed_sw.is_some() {
                        total_trades += 1;
                        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
                        if current_cap > peak_capital {
                            peak_capital = current_cap;
                        }
                        let dd = (peak_capital - current_cap) / peak_capital.max(0.001);
                        if dd > max_drawdown {
                            max_drawdown = dd;
                        }
                    }
                }

                let final_capital = arena.unified_capital.load(Ordering::Relaxed);
                
                // Fitness heavily penalizes negative expectancy and low trades
                let fitness = if max_drawdown > 0.90 || final_capital < initial_capital {
                    0.0
                } else {
                    // Reward high final capital, but penalize drawdown. Also reward high trade count for statistical significance.
                    (final_capital - initial_capital) * (1.0 - max_drawdown) * (total_trades as f64).ln().max(1.0)
                };
                (genome.clone(), final_capital, total_trades, fitness)
            })
            .collect();
            
        results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        
        let best_gen = &results[0];
        if best_gen.1 > best_all_time.1 {
            best_all_time = (best_gen.0.clone(), best_gen.1, best_gen.2);
        }
        
        println!("   Mejor de G{}: Cap ${:.2} (Fitness: {:.2}, Trades: {})", gen, best_gen.1, best_gen.3, best_gen.2);
        
        if gen == generations {
            break;
        }
        
        let mut next_gen = Vec::with_capacity(pop_size);
        let elites_count = pop_size / 5; // 20% elites
        for i in 0..elites_count {
            next_gen.push(results[i].0.clone());
        }
        
        while next_gen.len() < pop_size {
            let p1 = &results[rand::rng().random_range(0..(pop_size/2))].0;
            let p2 = &results[rand::rng().random_range(0..(pop_size/2))].0;
            
            let mut child = Genotype {
                global_leverage: if rand::rng().random_bool(0.5) { p1.global_leverage } else { p2.global_leverage },
                trend_threshold: if rand::rng().random_bool(0.5) { p1.trend_threshold } else { p2.trend_threshold },
                maker_spread_pct: if rand::rng().random_bool(0.5) { p1.maker_spread_pct } else { p2.maker_spread_pct },
                maker_obi_threshold: if rand::rng().random_bool(0.5) { p1.maker_obi_threshold } else { p2.maker_obi_threshold },
                scalp_tp: if rand::rng().random_bool(0.5) { p1.scalp_tp } else { p2.scalp_tp },
                scalp_sl: if rand::rng().random_bool(0.5) { p1.scalp_sl } else { p2.scalp_sl },
                swing_tp: if rand::rng().random_bool(0.5) { p1.swing_tp } else { p2.swing_tp },
                swing_sl: if rand::rng().random_bool(0.5) { p1.swing_sl } else { p2.swing_sl },
                scalp_z_target: if rand::rng().random_bool(0.5) { p1.scalp_z_target } else { p2.scalp_z_target },
                capital_split_scalp: if rand::rng().random_bool(0.5) { p1.capital_split_scalp } else { p2.capital_split_scalp },
                min_confidence: if rand::rng().random_bool(0.5) { p1.min_confidence } else { p2.min_confidence },
                explosive_leverage_multiplier: if rand::rng().random_bool(0.5) { p1.explosive_leverage_multiplier } else { p2.explosive_leverage_multiplier },
            };
            
            if rand::rng().random_bool(mutation_rate) { child.global_leverage *= rand::rng().random_range(0.8..1.2); }
            if rand::rng().random_bool(mutation_rate) { child.trend_threshold *= rand::rng().random_range(0.9..1.1); }
            if rand::rng().random_bool(mutation_rate) { child.maker_spread_pct *= rand::rng().random_range(0.5..2.0); }
            if rand::rng().random_bool(mutation_rate) { child.maker_obi_threshold *= rand::rng().random_range(0.8..1.2); }
            if rand::rng().random_bool(mutation_rate) { child.scalp_tp *= rand::rng().random_range(0.7..1.5); }
            if rand::rng().random_bool(mutation_rate) { child.scalp_sl *= rand::rng().random_range(0.7..1.5); }
            if rand::rng().random_bool(mutation_rate) { child.swing_tp *= rand::rng().random_range(0.7..1.5); }
            if rand::rng().random_bool(mutation_rate) { child.swing_sl *= rand::rng().random_range(0.7..1.5); }
            if rand::rng().random_bool(mutation_rate) { child.scalp_z_target *= rand::rng().random_range(0.8..1.2); }
            if rand::rng().random_bool(mutation_rate) { child.capital_split_scalp *= rand::rng().random_range(0.8..1.2); }
            if rand::rng().random_bool(mutation_rate) { child.min_confidence *= rand::rng().random_range(0.9..1.1); }
            if rand::rng().random_bool(mutation_rate) { child.explosive_leverage_multiplier *= rand::rng().random_range(0.5..2.0); }
            
            child.global_leverage = child.global_leverage.clamp(1.0, 100.0);
            child.trend_threshold = child.trend_threshold.clamp(0.1, 0.9);
            child.maker_spread_pct = child.maker_spread_pct.clamp(0.0001, 0.05);
            child.maker_obi_threshold = child.maker_obi_threshold.clamp(0.1, 0.95);
            child.scalp_tp = child.scalp_tp.clamp(0.0005, 0.02);
            child.scalp_sl = child.scalp_sl.clamp(0.0005, 0.01);
            child.swing_tp = child.swing_tp.clamp(0.001, 0.05);
            child.swing_sl = child.swing_sl.clamp(0.0005, 0.02);
            child.scalp_z_target = child.scalp_z_target.clamp(0.5, 5.0);
            child.capital_split_scalp = child.capital_split_scalp.clamp(0.1, 1.0);
            child.min_confidence = child.min_confidence.clamp(0.5, 0.99);
            child.explosive_leverage_multiplier = child.explosive_leverage_multiplier.clamp(1.0, 10.0);
            
            next_gen.push(child);
        }
        population = next_gen;
    }

    println!("============================================================");
    println!("🏆 MEJOR CONFIGURACIÓN DE TODA LA EVOLUCIÓN:");
    if best_all_time.1 > 0.0 {
        let (ref params, best_pnl, best_trades) = best_all_time;
        let pnl_pct = ((best_pnl - initial_capital) / initial_capital) * 100.0;
        println!("Leverage: {:.2}x (Explosive Mult: {:.2}x)", params.global_leverage, params.explosive_leverage_multiplier);
        println!("Capital Split Scalp: {:.1}%", params.capital_split_scalp * 100.0);
        println!("Min Confidence: {:.2}", params.min_confidence);
        println!("Trend Threshold: {:.2}", params.trend_threshold);
        println!("Scalp TP: {:.3}% | SL: {:.3}% | RR: {:.1}:1", params.scalp_tp*100.0, params.scalp_sl*100.0, params.scalp_tp/params.scalp_sl);
        println!("Swing TP: {:.3}% | SL: {:.3}% | RR: {:.1}:1", params.swing_tp*100.0, params.swing_sl*100.0, params.swing_tp/params.swing_sl);
        println!("Scalp Z-Target: {:.2}", params.scalp_z_target);
        println!("Trades: {}", best_trades);
        println!("Capital Final: ${:.2} ({:.2}% Crecimiento)", best_pnl, pnl_pct);
    } else {
        println!("Ninguna configuración sobrevivió.");
    }
    println!("============================================================");

    // Escribir active_genome.json si tuvimos éxito
    if best_all_time.1 > initial_capital {
        let (ref best_genome, _, _) = best_all_time;
        match serde_json::to_string_pretty(best_genome) {
            Ok(json_str) => {
                let dir_path = "config_dir/genotypes";
                let _ = std::fs::create_dir_all(dir_path);
                let file_path = format!("{}/active_genome.json", dir_path);
                
                if let Err(e) = std::fs::write(&file_path, json_str) {
                    println!("❌ Error al escribir active_genome.json en {}: {}", file_path, e);
                } else {
                    println!("🧬 ✅ active_genome.json actualizado en {}. Live Trader lo cargará en <60s.", file_path);
                }
            }
            Err(e) => println!("❌ Error serializando genome: {}", e),
        }
    }
    
    Ok(())
}


