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
    // FIX #1519: Sanitización estricta de capital inicial finito
    let mut initial_capital: f64 = initial_capital_str.parse().unwrap_or(13.0);
    if !initial_capital.is_finite() || initial_capital <= 0.0 {
        println!("⚠️ INITIAL_CAPITAL inválido o <= 0.0 detectado. Usando $13.00 como capital base.");
        initial_capital = 13.0;
    }

    let _arena_for_cap = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || Arc::new(GlobalArena::new(initial_capital)))
        .unwrap()
        .join()
        .unwrap();

    println!("💰 Initial Capital: ${:.2}", initial_capital);

    // INICIALIZAR EL SYMBOL REGISTRY PARA LOS 5 ACTIVOS PRINCIPALES
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
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "ETHUSDT".to_string(),
            step_size: 0.01,
            tick_size: 0.01,
            min_qty: 0.01,
            min_notional: 5.0,
            max_leverage: 100,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        },
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "SOLUSDT".to_string(),
            step_size: 0.1,
            tick_size: 0.01,
            min_qty: 0.1,
            min_notional: 5.0,
            max_leverage: 50,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        },
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "BNBUSDT".to_string(),
            step_size: 0.01,
            tick_size: 0.01,
            min_qty: 0.01,
            min_notional: 5.0,
            max_leverage: 50,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        },
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "XRPUSDT".to_string(),
            step_size: 1.0,
            tick_size: 0.0001,
            min_qty: 1.0,
            min_notional: 5.0,
            max_leverage: 75,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        },
    ]);

    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"];
    let mut master_stream = Vec::new();

    for (coin_id, symbol) in symbols.iter().enumerate() {
        let bin_path = format!("data/{}_ticks.bin", symbol);
        println!("📥 Cargando ticks multiactivo [{}]: {}", symbol, bin_path);

        if let Ok(file) = File::open(&bin_path) {
            if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
                let tick_size = size_of::<BinTick>();
                let num_ticks = mmap.len() / tick_size;
                for i in 0..num_ticks {
                    let start = i * tick_size;
                    let end = start + tick_size;
                    let bytes = &mmap[start..end];
                    let tick: BinTick = unsafe { std::ptr::read(bytes.as_ptr() as *const _) };
                    if tick.bid_price > 0.0 && tick.ask_price > tick.bid_price {
                        master_stream.push(TickEvent {
                            coin_id,
                            timestamp: tick.timestamp,
                            bid_price: tick.bid_price,
                            ask_price: tick.ask_price,
                            bid_qty: tick.bid_qty,
                            ask_qty: tick.ask_qty,
                        });
                    }
                }
            }
        }
    }

    // Ordenar cronológicamente el stream multiactivo para simulación temporal estricta
    master_stream.sort_by_key(|t| t.timestamp);
    println!("✅ Stream multiactivo ordenado en memoria: {} ticks totales", master_stream.len());

    // --- CONFIGURACIÓN EVOLUTIVA AVANZADA: MODELO DE 4 ISLAS + TEMPERATURA RECOCIDA ---
    let pop_size = 24; // 6 individuos por isla x 4 islas
    let generations = 16;
    let num_islands = 4;
    let island_size = pop_size / num_islands;

    // Inicializar población:
    // Isla 0: Especialistas Scalping (altos OBI/OFI, TP/SL micro)
    // Isla 1: Especialistas Swing (alta persistencia macro, Hurst > 0.65)
    // Isla 2: Especialistas Mean Reversion (Z-Score y bandas de Bollinger)
    // Isla 3: Especialistas Híbridos Cuánticos (Equilibrio de Confluencia y Mínimo DD)
    let current_champion = SuperGenotype::load_or_default();
    let mut population: Vec<SuperGenotype> = Vec::with_capacity(pop_size);

    // Isla 0 (0..6): Especialistas Scalping L2 de Alta Velocidad (OBI/OFI dinámicos [0.35, 0.65])
    for _ in 0..island_size {
        let mut ind = current_champion.mutate_cmaes(0.15);
        ind.dynamic_obi_threshold = rand::random_range(0.35..0.65);
        ind.dynamic_ofi_threshold = rand::random_range(0.40..0.70);
        ind.scalp_sl_base = rand::random_range(0.0008..0.0015);
        ind.scalp_tp_base = ind.scalp_sl_base * rand::random_range(2.2..3.5);
        ind.capital_split_scalp = rand::random_range(0.60..0.85);
        ind.global_leverage = rand::random_range(25.0..35.0);
        population.push(ind);
    }

    // Isla 1 (6..12): Especialistas Swing Macro Trend (Hurst > 0.35, TP/SL > 3.5:1)
    for _ in 0..island_size {
        let mut ind = current_champion.mutate_cmaes(0.15);
        ind.trend_threshold = rand::random_range(0.25..0.45);
        ind.swing_sl_base = rand::random_range(0.008..0.015);
        ind.swing_tp_base = ind.swing_sl_base * rand::random_range(3.5..5.5);
        ind.capital_split_scalp = rand::random_range(0.10..0.30);
        ind.global_leverage = rand::random_range(20.0..30.0);
        population.push(ind);
    }

    // Isla 2 (12..18): Especialistas Mean Reversion & SMR
    for _ in 0..island_size {
        let mut ind = current_champion.mutate_cmaes(0.15);
        ind.dynamic_ema_trend = rand::random_range(0.00010..0.00030);
        ind.scalp_sl_base = rand::random_range(0.0010..0.0018);
        ind.scalp_tp_base = ind.scalp_sl_base * rand::random_range(2.0..3.0);
        ind.capital_split_scalp = 0.50;
        population.push(ind);
    }

    // Isla 3 (18..24): Especialistas Híbridos Cuánticos & Kelly Adaptive (Semilla con Campeón Actual)
    population.push(current_champion.clone());
    for _ in 1..island_size {
        population.push(current_champion.mutate_cmaes(0.10));
    }

    println!("🚀 Iniciando Co-Evolución por Islas: {} Individuos ({} islas x {}) x {} Generaciones...", 
        pop_size, num_islands, island_size, generations);

    let mut best_all_time: Option<(SuperGenotype, f64, usize, f64)> = None; // (genome, capital, trades, fitness)

    for gen in 1..=generations {
        // Tasa de mutación adaptativa con enfriamiento por recocido simulado (Simulated Annealing)
        // Alta exploración inicial (0.35) -> Refinamiento de precisión milimétrica final (0.04)
        let progress = (gen as f64 - 1.0) / (generations as f64 - 1.0).max(1.0);
        let mutation_rate = 0.04 + (0.35 - 0.04) * (1.0 - progress).powf(1.5);

        println!("== GENERACIÓN {}/{} (Temp/Mut: {:.4}) ==", gen, generations, mutation_rate);

        let mut results: Vec<_> = population
            .par_iter()
            .map(|genome| {
                let arena = std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || Arc::new(GlobalArena::new(initial_capital)))
                    .unwrap()
                    .join()
                    .unwrap();

                genome.apply_to_arena(&arena);

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
                    let mut omni = [0.0f64; 54];
                    let swing_feats = engine.feature_engines[tick.coin_id].get_swing_features();
                    for (idx, &f) in swing_feats.iter().enumerate() {
                        if idx < 54 {
                            omni[idx] = f as f64;
                        }
                    }
                    let total_qty = tick.bid_qty + tick.ask_qty;
                    let mid = (tick.bid_price + tick.ask_price) / 2.0;
                    if total_qty > 0.0 && mid > 0.0 {
                        omni[0] = tick.bid_price;
                        omni[1] = tick.ask_price;
                        omni[30] = tick.bid_qty - tick.ask_qty;
                        omni[31] = (tick.bid_qty - tick.ask_qty) * 1.2;
                        omni[39] = (tick.bid_qty - tick.ask_qty) / total_qty;
                    }
                    let (_new_sc, _new_sw, closed_sc, closed_sw, _) = engine.process_tick(
                        tick.coin_id,
                        tick.bid_price,
                        tick.ask_price,
                        tick.bid_qty,
                        tick.ask_qty,
                        tick.timestamp,
                        &omni,
                    );

                    if let Some((_, pnl, _qty)) = closed_sc {
                        total_trades += 1;
                        if pnl > 0.0 {
                            wins += 1;
                        }
                    }
                    if let Some((_, pnl, _qty)) = closed_sw {
                        total_trades += 1;
                        if pnl > 0.0 {
                            wins += 1;
                        }
                    }

                    if closed_sc.is_some() || closed_sw.is_some() {
                        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
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

                // FITNESS CUANTITATIVO MULTIOBJETIVO MULTIACTIVO INSTITUCIONAL (López de Prado)
                let fitness = if max_drawdown > 0.40 {
                    0.0
                } else {
                    let pnl = final_capital - initial_capital;
                    let growth = pnl / initial_capital;
                    let dd_penalty = (1.0 - max_drawdown).powf(3.0).max(0.001);
                    let trade_factor = if total_trades >= 20 && total_trades <= 250 {
                        1.8
                    } else if total_trades >= 10 {
                        1.2
                    } else {
                        (total_trades as f64 / 10.0).max(0.01)
                    };
                    let wr_factor = (1.0 + win_rate).powf(2.0);

                    if total_trades < 5 {
                        0.001 // Penalización de inactividad
                    } else if pnl > 0.0 {
                        10.0 + (growth * dd_penalty * trade_factor * wr_factor * 80.0)
                    } else {
                        let loss_pct = pnl.abs() / initial_capital;
                        (10.0 - (loss_pct * 15.0) - (2.0 / (total_trades as f64).max(1.0))).max(0.001)
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

        // Rastrear el mejor histórico
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

        // --- MIGRACIÓN Y CO-EVOLUCIÓN ENTRE ISLAS ---
        let mut next_gen = Vec::with_capacity(pop_size);

        // Elitismo del 25% superior (los 6 mejores individuos globales)
        let elites_count = (pop_size / 4).max(2);
        for i in 0..elites_count {
            next_gen.push(results[i].0.clone());
        }

        // Crossover y mutación CMA-ES recocida para las 4 islas
        while next_gen.len() < pop_size {
            let parent1_idx = rand::rng().random_range(0..(pop_size / 2));
            let parent = &results[parent1_idx].0;
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
            // F4.3: embudo único — envelope versionado con linaje (generación,
            // fuente, métricas) + historia inmutable + espejo legacy atómico.
            // El write directo a active_genome.json queda abolido: sin versión
            // ni auditoría era imposible saber quién promovió qué ni revertir.
            match quantum_arena::genome_store::GenomeEnvelope::promote(
                genome.clone(),
                "ga_evolver",
                &format!(
                    "capital {:.2} → {:.2} ({:+.2}%)",
                    initial_capital, best_cap, pnl_pct
                ),
            ) {
                Ok(env) => println!(
                    "🧬 ✅ Genoma generación {} promovido (padre {}). Envelope + historia + espejo legacy escritos.",
                    env.generation, env.parent_generation
                ),
                Err(e) => println!("❌ Error promoviendo genoma al almacén: {}", e),
            }
        } else {
            println!("⚠️ Ninguna configuración generó ganancia. No se promueve genoma.");
        }
    } else {
        println!("Ninguna configuración sobrevivió.");
    }
    println!("============================================================");

    Ok(())
}
