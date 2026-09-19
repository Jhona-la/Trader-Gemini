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

#[derive(Clone)]
struct IslandResult {
    pub island_idx: usize,
    pub genome: SuperGenotype,
    pub final_capital: f64,
    pub total_trades: usize,
    pub fitness: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    println!("============================================================");
    println!("🧬 TRADER GEMINI V5 - TICK-LEVEL REALITY EVOLVER (SuperGenotype)");
    println!("🏛️ TOPOLOGÍA: 4 ISLAS SEGREGADAS CON RING MIGRATION (CERO CONTAMINACIÓN)");
    println!("============================================================");

    let initial_capital_str = env::var("INITIAL_CAPITAL").unwrap_or_else(|_| "13.0".to_string());
    // FIX #1519: Sanitización estricta de capital inicial finito
    let mut initial_capital: f64 = initial_capital_str.parse().unwrap_or(13.0);
    if !initial_capital.is_finite() || initial_capital <= 0.0 {
        println!(
            "⚠️ INITIAL_CAPITAL inválido o <= 0.0 detectado. Usando $13.00 como capital base."
        );
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
    println!("✅ Stream multiactivo ordenado en memoria: {} ticks totales (100% Ticks Reales - Sin Submuestreo)", master_stream.len());

    // Inicializar Ghost Flusher Lock-Free para evitar bloqueos de consola
    telemetry_server::macros::init_telemetry_logger();

    // --- CONFIGURACIÓN EVOLUTIVA AVANZADA: TOPOLOGÍA DE 4 ISLAS SEGREGADAS ---
    let num_islands = 4;
    let island_size = 4; // 4 individuos por isla = 16 individuos totales
    let pop_size = num_islands * island_size;
    let generations = 6;

    // U-5 (MOTOR UNIVERSAL CONTINUO): islas por BANDA del continuo temporal,
    // no por estrategia. La especialización fija el EXTREMO de su banda en
    // las CURVAS TP/SL (autoritativas desde C-10: fijar `scalp_tp_base`/
    // `swing_tp_base` se AUTODESTRUÍA en el roundtrip to_vector/from_vector
    // que re-deriva las anclas — la evolución de islas llevaba ese bug).
    let island_names = [
        "Isla 0 (Banda Rápida τ<19m — microestructura)",
        "Isla 1 (Banda Lenta τ>19m — tendencia macro)",
        "Isla 2 (Mean Reversion & SMR)",
        "Isla 3 (Híbrido Cuántico)",
    ];

    /// Fija el extremo `fast` o `slow` de las curvas TP/SL con el RR dado,
    /// preservando el otro extremo del campeón, y re-deriva las anclas.
    fn specialize_band_tp_sl(ind: &mut SuperGenotype, fast: bool, sl_base: f64, rr: f64) {
        use quantum_arena::temporal_spectrum::{
            HorizonCurve, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS,
        };
        let (keep_sl, keep_tp) = if fast {
            (
                ind.sl_horizon_curve.eval(TAU_ANCHOR_SLOW_MS),
                ind.tp_horizon_curve.eval(TAU_ANCHOR_SLOW_MS),
            )
        } else {
            (
                ind.sl_horizon_curve.eval(TAU_ANCHOR_FAST_MS),
                ind.tp_horizon_curve.eval(TAU_ANCHOR_FAST_MS),
            )
        };
        if fast {
            ind.sl_horizon_curve = HorizonCurve::through_two_points(
                TAU_ANCHOR_FAST_MS,
                sl_base,
                TAU_ANCHOR_SLOW_MS,
                keep_sl,
            );
            ind.tp_horizon_curve = HorizonCurve::through_two_points(
                TAU_ANCHOR_FAST_MS,
                sl_base * rr,
                TAU_ANCHOR_SLOW_MS,
                keep_tp,
            );
        } else {
            ind.sl_horizon_curve = HorizonCurve::through_two_points(
                TAU_ANCHOR_FAST_MS,
                keep_sl,
                TAU_ANCHOR_SLOW_MS,
                sl_base,
            );
            ind.tp_horizon_curve = HorizonCurve::through_two_points(
                TAU_ANCHOR_FAST_MS,
                keep_tp,
                TAU_ANCHOR_SLOW_MS,
                sl_base * rr,
            );
        }
        ind.derive_anchors_from_curves();
    }

    // Inicializar 4 islas segregadas:
    // Isla 0: Banda Rápida (microestructura: OBI/OFI altos, TP/SL micro en τ corto)
    // Isla 1: Banda Lenta (persistencia macro, Hurst > 0.55, RR >= 3:1 en τ largo)
    // Isla 2: Mean Reversion & SMR (Z-Score y bandas de Bollinger, balance 50/50)
    // Isla 3: Híbridos Cuánticos (Equilibrio de Confluencia, Mínimo DD, Semilla Campeón)
    let current_champion = SuperGenotype::load_or_default();
    let mut islands: Vec<Vec<SuperGenotype>> = Vec::with_capacity(num_islands);

    // Isla 0: Banda Rápida — microestructura de alta velocidad (OBI/OFI [0.18, 0.38])
    let mut island_0 = Vec::with_capacity(island_size);
    for _ in 0..island_size {
        let mut ind = current_champion.mutate_cmaes(0.15);
        ind.dynamic_obi_threshold = rand::random_range(0.18..0.38);
        ind.dynamic_ofi_threshold = rand::random_range(0.20..0.45);
        ind.dynamic_ema_trend = rand::random_range(0.00003..0.00012);
        ind.ml_threshold_long = rand::random_range(0.52..0.65);
        ind.ml_threshold_short = rand::random_range(0.52..0.65);
        let sl_fast = rand::random_range(0.0012..0.0022);
        specialize_band_tp_sl(&mut ind, true, sl_fast, rand::random_range(2.0..3.0));
        ind.capital_split_scalp = rand::random_range(0.55..0.75);
        ind.global_leverage = rand::random_range(25.0..35.0);
        island_0.push(ind);
    }
    islands.push(island_0);

    // Isla 1: Banda Lenta — tendencia macro (Hurst > 0.55, RR ≥ 3:1 en τ largo)
    let mut island_1 = Vec::with_capacity(island_size);
    for _ in 0..island_size {
        let mut ind = current_champion.mutate_cmaes(0.15);
        ind.trend_threshold = rand::random_range(0.20..0.40);
        ind.hurst_trend_threshold = rand::random_range(0.55..0.70);
        let sl_slow = rand::random_range(0.006..0.015);
        specialize_band_tp_sl(&mut ind, false, sl_slow, rand::random_range(3.0..5.0));
        ind.capital_split_scalp = rand::random_range(0.20..0.40);
        ind.global_leverage = rand::random_range(15.0..25.0);
        island_1.push(ind);
    }
    islands.push(island_1);

    // Isla 2: Mean Reversion & SMR (banda rápida con RR moderado — la reversión
    // vive en τ corto)
    let mut island_2 = Vec::with_capacity(island_size);
    for _ in 0..island_size {
        let mut ind = current_champion.mutate_cmaes(0.15);
        ind.dynamic_ema_trend = rand::random_range(0.00002..0.00008);
        let sl_fast = rand::random_range(0.0012..0.0020);
        specialize_band_tp_sl(&mut ind, true, sl_fast, rand::random_range(2.0..2.8));
        ind.dynamic_obi_threshold = rand::random_range(0.20..0.35);
        ind.range_threshold = rand::random_range(0.40..0.70);
        ind.capital_split_scalp = 0.50;
        island_2.push(ind);
    }
    islands.push(island_2);

    // Isla 3: Especialistas Híbridos Cuánticos & Kelly Adaptive (Semilla con Campeón Actual)
    let mut island_3 = Vec::with_capacity(island_size);
    island_3.push(current_champion.clone());
    for _ in 1..island_size {
        island_3.push(current_champion.mutate_cmaes(0.10));
    }
    islands.push(island_3);

    println!(
        "🚀 Iniciando Co-Evolución por Islas: {} Individuos ({} islas x {}) x {} Generaciones...",
        pop_size, num_islands, island_size, generations
    );

    let mut best_all_time: Option<(SuperGenotype, f64, usize, f64)> = None; // (genome, capital, trades, fitness)

    // Cargar modelo DarkAlphaEngine 54D para co-evolución simbiótica
    let trained_nn = match dark_alpha_engine::DarkAlphaEngine::load_json(
        "models/DarkAlpha_BTCUSDT.json",
    ) {
        Ok(mut m) => {
            let is_corrupt = m
                .layer1
                .weights
                .iter()
                .any(|&w| w.is_nan() || w.is_infinite());
            if is_corrupt {
                println!("⚠️ [EVOLVER] Modelo en disco corrupto. Usando default_model 54D.");
                let mut def = dark_alpha_engine::DarkAlphaEngine::default_model();
                def.init_buffers();
                def
            } else {
                m.init_buffers();
                println!(
                    "🧠 [EVOLVER] DarkAlphaEngine 54D cargado exitosamente para co-evolución."
                );
                m
            }
        }
        Err(_) => {
            println!("ℹ️ [EVOLVER] models/DarkAlpha_BTCUSDT.json no encontrado. Usando default_model 54D.");
            let mut def = dark_alpha_engine::DarkAlphaEngine::default_model();
            def.init_buffers();
            def
        }
    };

    for gen in 1..=generations {
        let progress = (gen as f64 - 1.0) / (generations as f64 - 1.0).max(1.0);
        let mutation_rate = 0.04 + (0.35 - 0.04) * (1.0 - progress).powf(1.5);

        println!(
            "== GENERACIÓN {}/{} (Temp/Mut: {:.4}) ==",
            gen, generations, mutation_rate
        );

        // Aplanar tareas de evaluación conservando identidad de isla
        let eval_tasks: Vec<(usize, SuperGenotype)> = islands
            .iter()
            .enumerate()
            .flat_map(|(isl_idx, isl)| isl.iter().map(move |ind| (isl_idx, ind.clone())))
            .collect();

        let raw_results: Vec<IslandResult> = eval_tasks
            .par_iter()
            .map(|(isl_idx, genome)| {
                let arena = Arc::new(GlobalArena::new(initial_capital));
                genome.apply_to_arena(&arena);
                arena
                    .config
                    .global_max_drawdown
                    .store(0.90, Ordering::Relaxed);

                let mut engine = GodEngineCore::new(arena.clone());
                let mut local_nn = trained_nn.clone();
                local_nn.init_buffers();
                engine.swing_nn = Some(local_nn);

                let mut total_trades: usize = 0;
                let mut wins: usize = 0;
                let mut max_drawdown: f64 = 0.0;
                let mut peak_capital = initial_capital;

                // Microestructura 100% exacta: Procesamiento de todos los ticks sin step_by
                for tick in master_stream.iter() {
                    arena.update_market_data(
                        tick.coin_id,
                        tick.bid_price,
                        tick.ask_price,
                        tick.bid_qty,
                        tick.ask_qty,
                        tick.timestamp,
                    );
                    let mut omni = [0.0f64; 54];
                    let swing_feats = engine.feature_engines[tick.coin_id].get_universal_features();
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
                    let (_new_pos, closed_pos, _) = engine.process_tick(
                        tick.coin_id,
                        tick.bid_price,
                        tick.ask_price,
                        tick.bid_qty,
                        tick.ask_qty,
                        tick.timestamp,
                        &omni,
                    );

                    if let Some((_, pnl, _qty)) = closed_pos {
                        total_trades += 1;
                        if pnl > 0.0 {
                            wins += 1;
                        }

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

                let mut final_capital = arena.unified_capital.load(Ordering::Relaxed);
                
                // Aplicar Slippage Estricto de Realidad al Capital Final
                let raw_pnl = final_capital - initial_capital;
                // Asumiendo fee de 0.0004 y notional promedio de 35.0
                let penalized_pnl = evolution_engine::entropy_fitness::EntropyFitness::reality_slippage_penalty_with_notional(raw_pnl, total_trades, 0.0004, 35.0);
                final_capital = initial_capital + penalized_pnl;

                let win_rate = if total_trades > 0 {
                    wins as f64 / total_trades as f64
                } else {
                    0.0
                };

                // CERT-M5-H01 — FITNESS UNIFICADO (D-652/D-653/D-654):
                // La fórmula anterior `10 + growth^1.5 × (1−dd)² × trade_factor ×
                // (1+wr)³ × 100` era exactamente la patología que D-653
                // documentó: crecimiento CONVEXO en leverage y win-rate al cubo
                // premiaban tamaño de apuesta sobre calidad. Además daba +10
                // gratis a cualquier genoma rentable y scoreaba 0.001 la
                // inacción (D-654: inacción es INVIABLE, no intermedia).
                // Ahora TODO promotor usa la MISMA función: fitness::compute
                // (utilidad Kelly log − λ·dd², inacción = INVIABLE).
                let fitness = evolution_engine::fitness::compute(
                    &evolution_engine::fitness::FitnessInputs {
                        initial_capital,
                        final_capital,
                        max_drawdown_pct: max_drawdown,
                        total_trades: total_trades as u32,
                        min_trades_required: 5,
                        oos_start_capital: final_capital, // mismo período (sin split IS/OOS aquí)
                        oos_end_capital: final_capital,
                    },
                );

                IslandResult {
                    island_idx: *isl_idx,
                    genome: genome.clone(),
                    final_capital,
                    total_trades,
                    fitness,
                    win_rate,
                    max_drawdown,
                }
            })
            .collect();

        // Agrupar resultados por isla segregada
        let mut island_results: Vec<Vec<IslandResult>> =
            vec![Vec::with_capacity(island_size); num_islands];
        for res in raw_results {
            island_results[res.island_idx].push(res);
        }

        // Ordenar internamente cada isla por fitness decreciente
        for isl in 0..num_islands {
            island_results[isl].sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Telemetría por Isla y Detección de Mejor Histórico
        let mut global_gen_best: Option<&IslandResult> = None;

        for isl in 0..num_islands {
            let best_i = &island_results[isl][0];
            let growth_pct = ((best_i.final_capital - initial_capital) / initial_capital) * 100.0;
            println!("   🏝️  {} | Mejor: Cap ${:.2} ({:+.1}%) | WR: {:.0}% | Trades: {} | DD: {:.1}% | Fit: {:.2}",
                island_names[isl], best_i.final_capital, growth_pct, best_i.win_rate * 100.0, best_i.total_trades, best_i.max_drawdown * 100.0, best_i.fitness);

            if global_gen_best.is_none() || best_i.fitness > global_gen_best.unwrap().fitness {
                global_gen_best = Some(best_i);
            }
        }

        if let Some(best_gen) = global_gen_best {
            if best_all_time.is_none() || best_gen.fitness > best_all_time.as_ref().unwrap().3 {
                best_all_time = Some((
                    best_gen.genome.clone(),
                    best_gen.final_capital,
                    best_gen.total_trades,
                    best_gen.fitness,
                ));
            }
            let growth_pct = ((best_gen.final_capital - initial_capital) / initial_capital) * 100.0;
            println!("   ⭐ Campeón Global G{}: Cap ${:.2} ({:+.1}%) | WR: {:.0}% | Trades: {} | DD: {:.1}% | Fitness: {:.2}", 
                gen, best_gen.final_capital, growth_pct, best_gen.win_rate * 100.0, best_gen.total_trades, best_gen.max_drawdown * 100.0, best_gen.fitness);
            println!(
                "      🧬 Scalp TP={:.3}% SL={:.3}% | Swing TP={:.2}% SL={:.2}% | Lev={:.1}x | OBI_thr={:.3} OFI_thr={:.3}",
                best_gen.genome.scalp_tp_base * 100.0,
                best_gen.genome.scalp_sl_base * 100.0,
                best_gen.genome.swing_tp_base * 100.0,
                best_gen.genome.swing_sl_base * 100.0,
                best_gen.genome.global_leverage,
                best_gen.genome.dynamic_obi_threshold,
                best_gen.genome.dynamic_ofi_threshold
            );
        }

        if gen == generations {
            break;
        }

        // --- REPRODUCCIÓN INTRA-ISLA (PRESERVA ESPECIALIZACIÓN) ---
        let mut next_islands: Vec<Vec<SuperGenotype>> = Vec::with_capacity(num_islands);

        for k in 0..num_islands {
            let mut next_island = Vec::with_capacity(island_size);
            // Elitismo intra-isla: Preservar el mejor individuo de la isla
            next_island.push(island_results[k][0].genome.clone());

            // Reproducción dentro de la propia isla con mutación CMA-ES
            while next_island.len() < island_size {
                let parent_idx = rand::rng().random_range(0..(island_size / 2).max(1));
                let parent = &island_results[k][parent_idx].genome;
                let mut child = parent.mutate_cmaes(mutation_rate);

                // Reforzar la especialización fenotípica por nicho de isla
                match k {
                    0 => {
                        // Isla 0: Scalp L2
                        child.capital_split_scalp = child.capital_split_scalp.clamp(0.50, 0.85);
                        child.dynamic_obi_threshold = child.dynamic_obi_threshold.clamp(0.15, 0.45);
                        child.dynamic_ofi_threshold = child.dynamic_ofi_threshold.clamp(0.18, 0.50);
                        child.global_leverage = child.global_leverage.clamp(25.0, 35.0);
                    }
                    1 => {
                        // Isla 1: Swing Macro
                        child.capital_split_scalp = child.capital_split_scalp.clamp(0.15, 0.45);
                        child.trend_threshold = child.trend_threshold.clamp(0.15, 0.45);
                        if child.swing_tp_base < child.swing_sl_base * 2.5 {
                            child.swing_tp_base = child.swing_sl_base * 3.5;
                        }
                    }
                    2 => {
                        // Isla 2: Mean Reversion
                        child.capital_split_scalp = 0.50;
                        child.dynamic_ema_trend = child.dynamic_ema_trend.clamp(0.00001, 0.00010);
                        child.range_threshold = child.range_threshold.clamp(0.35, 0.75);
                    }
                    _ => {} // Isla 3: Híbrido Cuántico (Espacio libre sin restricciones rígidas)
                }

                next_island.push(child);
            }
            next_islands.push(next_island);
        }

        // --- TOPOLOGÍA RING MIGRATION (MIGRACIÓN EN ANILLO) ---
        // El campeón de la Isla K migra a la Isla (K + 1) % 4, reemplazando al peor individuo
        let champions_to_migrate: Vec<SuperGenotype> = (0..num_islands)
            .map(|k| island_results[k][0].genome.clone())
            .collect();

        for k in 0..num_islands {
            let dest_island = (k + 1) % num_islands;
            let worst_idx = next_islands[dest_island].len() - 1;
            // El migrante de la isla k ocupa el puesto inferior de dest_island sin desplazar a su propio elite
            next_islands[dest_island][worst_idx] = champions_to_migrate[k].clone();
        }

        println!("   🔄 [RING MIGRATION] Campeones transferidos en anillo: Isla 0 → Isla 1 → Isla 2 → Isla 3 → Isla 0");

        islands = next_islands;
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
        // CERT-M5-H01: umbral en la NUEVA escala de fitness::compute (log
        // utility). La escala antigua daba +10 gratis; la nueva: 0 = sin
        // crecimiento, >0 = crecimiento log positivo. Umbral 0.01 = cualquier
        // crecimiento neto positivo tras la penalización de drawdown.
        if best_cap > initial_capital || best_fitness > 0.01 {
            // F4.3: embudo único — envelope versionado con linaje (generación,
            // fuente, métricas) + historia inmutable + espejo legacy atómico.
            // El write directo a active_genome.json queda abolido: sin versión
            // ni auditoría era imposible saber quién promovió qué ni revertir.
            match quantum_arena::genome_store::GenomeEnvelope::promote(
                genome.clone(),
                "ring_island_evolver",
                &format!(
                    "capital {:.2} → {:.2} ({:+.2}%) | fit: {:.2}",
                    initial_capital, best_cap, pnl_pct, best_fitness
                ),
            ) {
                Ok(env) => println!(
                    "🧬 ✅ Genoma generación {} promovido (padre {}). Envelope + historia + espejo legacy escritos.",
                    env.generation, env.parent_generation
                ),
                Err(e) => println!("❌ Error promoviendo genoma al almacén: {}", e),
            }
        } else {
            println!(
                "⚠️ Ninguna configuración superó el umbral de viabilidad. No se promueve genoma."
            );
        }
    } else {
        println!("Ninguna configuración sobrevivió.");
    }
    println!("============================================================");
    Ok(())
}
