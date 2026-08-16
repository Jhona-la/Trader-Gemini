use god_engine_core::GodEngineCore;
use quantum_arena::genome::SuperGenotype;
use quantum_arena::{GlobalArena, TickEvent};
use rand::RngExt;
use rayon::prelude::*;
use std::fs::File;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use memmap2::MmapOptions;
use std::env;
use std::mem::size_of;

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
    println!("🧬 TRADER GEMINI V5 - TICK-LEVEL REALITY EVOLVER (SuperGenotype)");
    println!("============================================================");

    let initial_capital_str = env::var("INITIAL_CAPITAL").unwrap_or_else(|_| "13.0".to_string());
    let mut initial_capital: f64 = initial_capital_str.parse().unwrap_or(13.0);
    if initial_capital <= 0.0 {
        println!("⚠️ INITIAL_CAPITAL <= 0.0 detected. Defaulting to $13.00 base capital.");
        initial_capital = 13.0;
    }

    let _arena_for_cap = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || Arc::new(GlobalArena::new(initial_capital)))
        .unwrap()
        .join()
        .unwrap();

    println!("💰 Initial Capital: ${:.2}", initial_capital);

    // INICIALIZAR EL SYMBOL REGISTRY PARA BTCUSDT (coin_id = 0)
    quantum_arena::symbol_registry::update_registry(vec![
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "BTCUSDT".to_string(),
            step_size: 0.001,
            tick_size: 0.10,
            min_qty: 0.001,
            min_notional: 5.0,
            max_leverage: 125,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        },
    ]);

    let bin_path = "data/BTCUSDT_ticks.bin";
    println!("📥 Cargando datos REALES de alta frecuencia: {}", bin_path);

    let file = File::open(bin_path).expect(
        "❌ Archivo BTCUSDT_ticks.bin no encontrado. Ejecuta los simuladores anteriores primero.",
    );
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };

    let tick_size = size_of::<BinTick>();
    let num_ticks = mmap.len() / tick_size;
    println!(
        "📊 Ticks reales cargados (Cero Fricción simulada): {}",
        num_ticks
    );

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

    // --- CONFIGURACIÓN EVOLUTIVA ---
    let pop_size = 16;
    let generations = 8;
    let mutation_rate = 0.20; // CMA-ES mutation strength

    // Inicializar población con SuperGenotype::new_random()
    let mut population: Vec<SuperGenotype> =
        (0..pop_size).map(|_| SuperGenotype::new_random()).collect();

    println!("🚀 Iniciando Evolución SuperGenotype: {} Individuos x {} Generaciones (100+ genes cada uno)...", pop_size, generations);

    let mut best_all_time: Option<(SuperGenotype, f64, usize, f64)> = None; // (genome, capital, trades, fitness)

    for gen in 1..=generations {
        println!("== GENERACIÓN {} ==", gen);

        let mut results: Vec<_> = population
            .par_iter()
            .map(|genome| {
                let arena = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || Arc::new(GlobalArena::new(initial_capital)))
                    .unwrap()
                    .join()
                    .unwrap();

                // CRITICAL FIX: apply_to_arena() sets ALL 100+ parameters including
                // dynamic_atr_min, dynamic_obi_threshold, dynamic_ema_trend, dynamic_ofi_threshold
                // which control ENTRY conditions in process_tick
                genome.apply_to_arena(&arena);

                // Allow extreme compounding - accept 90% drawdown max
                arena
                    .config
                    .global_max_drawdown
                    .store(0.90, Ordering::Relaxed);

                let mut engine = GodEngineCore::new(arena.clone());
                let mut total_trades: usize = 0;
                let mut wins: usize = 0;
                let mut max_drawdown: f64 = 0.0;
                let mut peak_capital = initial_capital;

                for tick in &master_stream {
                    arena.update_market_data(
                        tick.coin_id,
                        tick.bid_price,
                        tick.ask_price,
                        tick.bid_qty,
                        tick.ask_qty,
                        tick.timestamp,
                    );
                    let (_new_sc, _new_sw, closed_sc, closed_sw, _) = engine.process_tick(
                        tick.coin_id,
                        tick.bid_price,
                        tick.ask_price,
                        tick.bid_qty,
                        tick.ask_qty,
                        tick.timestamp,
                        &[0.0; 54],
                    );

                    if let Some((_, pnl, qty)) = closed_sc {
                        total_trades += 1;
                        if pnl > 0.0 {
                            wins += 1;
                        }
                        if total_trades <= 10 {
                            println!(
                                "🔍 TRACE [Scalp {}]: PnL=${:.10}, qty={:.8}",
                                total_trades, pnl, qty
                            );
                        }
                    }
                    if let Some((_, pnl, qty)) = closed_sw {
                        total_trades += 1;
                        if pnl > 0.0 {
                            wins += 1;
                        }
                        if total_trades <= 10 {
                            println!(
                                "🔍 TRACE [Swing {}]: PnL=${:.10}, qty={:.8}",
                                total_trades, pnl, qty
                            );
                        }
                    }

                    if closed_sc.is_some() || closed_sw.is_some() {
                        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
                        if total_trades <= 10 {
                            println!("🔍 TRACE [Capital]: current=${:.10}", current_cap);
                        }
                        if current_cap > peak_capital {
                            peak_capital = current_cap;
                        } else {
                            let dd = (peak_capital - current_cap) / peak_capital.max(0.001);
                            if dd > max_drawdown {
                                max_drawdown = dd;
                            }
                        }
                    }
                }

                let final_capital = arena.unified_capital.load(Ordering::Relaxed);
                let win_rate = if total_trades > 0 {
                    wins as f64 / total_trades as f64
                } else {
                    0.0
                };

                // Fitness: reward compound growth + positive expectancy, penalize catastrophic ruin
                let fitness = if max_drawdown > 0.90 || total_trades < 5 {
                    0.0
                } else {
                    let pnl = final_capital - initial_capital;
                    let growth = pnl / initial_capital;
                    let dd_penalty = (1.0 - max_drawdown).powf(1.5).max(0.01);
                    let trade_factor = (total_trades as f64).sqrt().clamp(1.0, 20.0);
                    let wr_factor = (win_rate * 2.0).clamp(0.5, 2.0);

                    if pnl > 0.0 {
                        10.0 + (growth * dd_penalty * trade_factor * wr_factor * 50.0)
                    } else {
                        let loss_pct = pnl.abs() / initial_capital;
                        (10.0 - (loss_pct * 10.0)).max(0.01)
                    }
                };
                (
                    genome.clone(),
                    final_capital,
                    total_trades,
                    fitness,
                    win_rate,
                    max_drawdown,
                )
            })
            .collect();

        results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        let best_gen = &results[0];
        let worst_gen = &results[results.len() - 1];

        // Track best of all time
        if best_all_time.is_none() || best_gen.3 > best_all_time.as_ref().unwrap().3 {
            best_all_time = Some((best_gen.0.clone(), best_gen.1, best_gen.2, best_gen.3));
        }

        let growth_pct = ((best_gen.1 - initial_capital) / initial_capital) * 100.0;
        println!("   Mejor G{}: Cap ${:.2} ({:+.1}%) | WR: {:.0}% | Trades: {} | DD: {:.1}% | Fitness: {:.2}", 
            gen, best_gen.1, growth_pct, best_gen.4 * 100.0, best_gen.2, best_gen.5 * 100.0, best_gen.3);
        println!(
            "   Peor  G{}: Cap ${:.2} | Trades: {} | Fitness: {:.2}",
            gen, worst_gen.1, worst_gen.2, worst_gen.3
        );

        // Key parameters of best genome
        println!(
            "   🧬 ATR_min={:.5} OBI_thr={:.3} EMA_thr={:.5} OFI_thr={:.3}",
            best_gen.0.dynamic_atr_min,
            best_gen.0.dynamic_obi_threshold,
            best_gen.0.dynamic_ema_trend,
            best_gen.0.dynamic_ofi_threshold
        );
        println!(
            "   🧬 Scalp TP={:.3}% SL={:.3}% | Swing TP={:.2}% SL={:.2}% | Lev={:.1}x",
            best_gen.0.scalp_tp_base * 100.0,
            best_gen.0.scalp_sl_base * 100.0,
            best_gen.0.swing_tp_base * 100.0,
            best_gen.0.swing_sl_base * 100.0,
            best_gen.0.global_leverage
        );

        if gen == generations {
            break;
        }

        // --- NEXT GENERATION ---
        let mut next_gen = Vec::with_capacity(pop_size);

        // 20% elites (unchanged)
        let elites_count = (pop_size / 5).max(2);
        for i in 0..elites_count {
            next_gen.push(results[i].0.clone());
        }

        // 80% children via CMA-ES mutation of top 50%
        while next_gen.len() < pop_size {
            let parent_idx = rand::rng().random_range(0..(pop_size / 2));
            let parent = &results[parent_idx].0;
            let child = parent.mutate_cmaes(mutation_rate);
            next_gen.push(child);
        }

        population = next_gen;
    }

    println!("============================================================");
    println!("🏆 MEJOR CONFIGURACIÓN DE TODA LA EVOLUCIÓN:");
    if let Some((ref genome, best_cap, best_trades, best_fitness)) = best_all_time {
        let pnl_pct = ((best_cap - initial_capital) / initial_capital) * 100.0;
        println!("Leverage: {:.2}x", genome.global_leverage);
        println!(
            "Capital Split Scalp: {:.1}%",
            genome.capital_split_scalp * 100.0
        );
        println!("Min Confidence: {:.3}", genome.min_confidence_btc);
        println!("Dynamic ATR Min: {:.6}", genome.dynamic_atr_min);
        println!("Dynamic OBI Threshold: {:.4}", genome.dynamic_obi_threshold);
        println!("Dynamic EMA Trend: {:.6}", genome.dynamic_ema_trend);
        println!("Dynamic OFI Threshold: {:.4}", genome.dynamic_ofi_threshold);
        println!(
            "Scalp TP: {:.3}% | SL: {:.3}% | RR: {:.1}:1",
            genome.scalp_tp_base * 100.0,
            genome.scalp_sl_base * 100.0,
            genome.scalp_tp_base / genome.scalp_sl_base.max(0.0001)
        );
        println!(
            "Swing TP: {:.3}% | SL: {:.3}% | RR: {:.1}:1",
            genome.swing_tp_base * 100.0,
            genome.swing_sl_base * 100.0,
            genome.swing_tp_base / genome.swing_sl_base.max(0.0001)
        );
        println!("Trend Threshold: {:.3}", genome.trend_threshold);
        println!(
            "CVD Veto: {:.3} | Wall Veto: {:.2}",
            genome.cvd_veto_threshold, genome.wall_veto_threshold
        );
        println!("Trades: {}", best_trades);
        println!("Fitness: {:.4}", best_fitness);
        println!(
            "Capital Final: ${:.2} ({:+.2}% Crecimiento)",
            best_cap, pnl_pct
        );

        // Save as SuperGenotype JSON
        if best_cap > initial_capital {
            let dir_path = "config_dir/genotypes";
            let _ = std::fs::create_dir_all(dir_path);
            let file_path = format!("{}/active_genome.json", dir_path);

            match serde_json::to_string_pretty(genome) {
                Ok(json_str) => {
                    if let Err(e) = std::fs::write(&file_path, &json_str) {
                        println!(
                            "❌ Error al escribir active_genome.json en {}: {}",
                            file_path, e
                        );
                    } else {
                        println!("🧬 ✅ active_genome.json actualizado en {}. Live Trader lo cargará en <60s.", file_path);
                    }
                }
                Err(e) => println!("❌ Error serializando genome: {}", e),
            }
        } else {
            println!("⚠️ Ninguna configuración generó ganancia. No se guarda active_genome.json.");
        }
    } else {
        println!("Ninguna configuración sobrevivió.");
    }
    println!("============================================================");

    Ok(())
}
