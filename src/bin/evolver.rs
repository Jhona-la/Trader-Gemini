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

/// Resultado de recorrer un flujo de ticks con un genoma.
struct Evaluacion {
    final_capital: f64,
    total_trades: usize,
    wins: usize,
    max_drawdown: f64,
}

/// LG-17 — recorre `stream` con `genome` en un núcleo nuevo.
///
/// - El arena se construye en su propia pila (`build_in_own_stack`): dentro de
///   un worker de rayon `GlobalArena::new` desbordaba la pila (panic = abort).
/// - El omni es el neutro del replay sin historia macro, con su frescura
///   (`macro_staleness_ms` = desconocido). Antes se copiaban 34 features de
///   microestructura sobre los slots 0..33 del omni (macro, funding, OI) y se
///   inyectaban constantes de DXY/SP500 en otra escala.
/// - Sin DarkAlpha validado el NN no opina (`None`), como en vivo (F5-B-H1).
/// - El capital final es el del motor, que ya cobra comisiones y
///   deslizamiento: no se vuelve a cobrar la fricción.
fn evaluar(
    genome: &SuperGenotype,
    stream: &[TickEvent],
    initial_capital: f64,
    nn: Option<&dark_alpha_engine::DarkAlphaEngine>,
    omni: &[f64; 54],
) -> Evaluacion {
    let arena = GlobalArena::build_in_own_stack(initial_capital);
    genome.apply_to_arena(&arena);
    arena
        .config
        .global_max_drawdown
        .store(0.90, Ordering::Relaxed);
    arena.registry.set("macro_staleness_ms", u64::MAX as f64);

    let mut engine = GodEngineCore::new(arena.clone());
    engine.swing_nn = nn.map(|m| {
        let mut local = m.clone();
        local.init_buffers();
        local
    });

    let mut total_trades: usize = 0;
    let mut wins: usize = 0;
    let mut max_drawdown: f64 = 0.0;
    let mut peak_capital = initial_capital;
    for tick in stream {
        arena.update_market_data(
            tick.coin_id,
            tick.bid_price,
            tick.ask_price,
            tick.bid_qty,
            tick.ask_qty,
            tick.timestamp,
        );
        let (_new_pos, closed_pos, _) = engine.process_tick(
            tick.coin_id,
            tick.bid_price,
            tick.ask_price,
            tick.bid_qty,
            tick.ask_qty,
            tick.timestamp,
            omni,
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
    Evaluacion {
        final_capital: arena.unified_capital.load(Ordering::Relaxed),
        total_trades,
        wins,
        max_drawdown,
    }
}

/// Aptitud de una evaluación con la función única (`fitness::compute`).
/// `juicio` indica que el flujo es la partición fuera de muestra: entonces
/// sus extremos de capital son los de la partición OOS y se exige
/// `MIN_OPERACIONES_JUICIO`.
fn aptitud(e: &Evaluacion, initial_capital: f64, juicio: bool) -> f64 {
    let (min_trades, oos_ini) = if juicio {
        (MIN_OPERACIONES_JUICIO, initial_capital)
    } else {
        // Selección: convención legacy sin split (no acredita un OOS).
        (5, e.final_capital)
    };
    evolution_engine::fitness::compute(&evolution_engine::fitness::FitnessInputs {
        initial_capital,
        final_capital: e.final_capital,
        max_drawdown_pct: e.max_drawdown,
        total_trades: e.total_trades as u32,
        min_trades_required: min_trades,
        oos_start_capital: oos_ini,
        oos_end_capital: e.final_capital,
    })
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

/// LG-17: fracción del flujo (por orden cronológico) con la que se selecciona;
/// la cola restante sólo juzga. Es el mismo corte 70/30 de los demás
/// promotores (`oos_split_index` del demonio, `train_forest`).
const FRACCION_SELECCION: f64 = 0.70;

/// LG-17: operaciones mínimas en la partición de juicio. Es el mínimo
/// muestral del examen del demonio (`WF_MIN_TRADES`, regla X-014): por debajo
/// la aptitud es INVIABLE (D-654), no mediocre.
const MIN_OPERACIONES_JUICIO: u32 = 30;

/// Índice del primer tick de la partición de juicio.
fn indice_de_corte(n: usize) -> usize {
    ((n as f64) * FRACCION_SELECCION).floor() as usize
}

/// LG-17 — ¿se promueve el campeón de la selección?
///
/// Sólo se juzga sobre aggTrades reales (D-691: con microestructura
/// fabricada el veredicto no vale). Con un incumbente medido en la misma
/// partición de juicio se exige el margen común de los promotores
/// (`darwin::meets_promotion_margin`). Si el incumbente no tiene evidencia
/// (aptitud no finita: menos de `MIN_OPERACIONES_JUICIO` operaciones), el
/// candidato debe al menos crecer en log neto de la penalización de drawdown.
fn decidir_promocion(candidato_oos: f64, incumbente_oos: f64, tapes_reales: bool) -> bool {
    if !tapes_reales || !candidato_oos.is_finite() {
        return false;
    }
    if incumbente_oos.is_finite() {
        god_engine_core::darwin::meets_promotion_margin(candidato_oos, incumbente_oos)
    } else {
        candidato_oos > 0.0
    }
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

    // LG-17: sin entorno explícito el almacén escribe en el silo
    // `__UNSET__`, que nadie lee: la evolución quemaba CPU y anunciaba éxito.
    if env::var("TG_GENOME_ENV").map(|v| v.trim().is_empty()).unwrap_or(true) {
        eprintln!("❌ [EVOLVER] TG_GENOME_ENV no está definida (backtest | demo | prod). Sin entorno no hay a dónde promover: no se evoluciona.");
        return Err("TG_GENOME_ENV no definida".into());
    }
    // LG-17: EVOLVER_LOW_RESOURCES (lo fija LAUNCHER.bat) acota los hilos de
    // rayon a la mitad de los núcleos para no competir con el motor vivo.
    if env::var_os("EVOLVER_LOW_RESOURCES").is_some() {
        let nucleos = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads((nucleos / 2).max(1))
            .build_global();
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
    // LG-17: el veredicto sólo vale sobre aggTrades reales (D-691).
    let mut tapes_reales = true;
    let mut tapes_cargados = 0usize;

    for (coin_id, symbol) in symbols.iter().enumerate() {
        let bin_path = format!("data/{}_ticks.bin", symbol);
        println!("📥 Cargando ticks multiactivo [{}]: {}", symbol, bin_path);

        if let Ok(file) = File::open(&bin_path) {
            if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
                // LG-15: saltar la cabecera de 8 B del tape (TGMSYNT1 / TGMTRAW…).
                let (origen, cuerpo) = backtest_engine::tick_replayer::cuerpo_de_ticks(&mmap);
                println!("   origen: {}", origen.descripcion());
                tapes_reales &= origen.es_real();
                tapes_cargados += 1;
                let tick_size = size_of::<BinTick>();
                let num_ticks = cuerpo.len() / tick_size;
                for i in 0..num_ticks {
                    let start = i * tick_size;
                    let end = start + tick_size;
                    let bytes = &cuerpo[start..end];
                    let tick: BinTick = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const _) };
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
    tapes_reales &= tapes_cargados > 0;
    // LG-17: selección sobre el 70 % inicial; el 30 % final sólo juzga.
    let corte = indice_de_corte(master_stream.len());
    let (seleccion, juicio) = master_stream.split_at(corte);
    println!(
        "✅ Flujo multiactivo ordenado: {} ticks ({} de selección, {} de juicio). Tapes reales: {}",
        master_stream.len(),
        seleccion.len(),
        juicio.len(),
        tapes_reales
    );
    // Omni neutro del replay sin historia macro.
    let omni_neutro = data_pipeline::omni_multiplexer::OmniState::new().get_features();

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
    // LG-17: sin modelo validado el NN no opina, como en vivo (F5-B-H1). Antes
    // se fabricaba una red aleatoria que votaba ruido en el ensamble.
    let trained_nn: Option<dark_alpha_engine::DarkAlphaEngine> = match dark_alpha_engine::DarkAlphaEngine::load_json(
        "models/DarkAlpha_BTCUSDT.json",
    ) {
        Ok(mut m) => {
            let is_corrupt = m
                .layer1
                .weights
                .iter()
                .any(|&w| w.is_nan() || w.is_infinite());
            if is_corrupt {
                println!("⚠️ [EVOLVER] Modelo DarkAlpha en disco corrupto: el NN no opina.");
                None
            } else {
                m.init_buffers();
                println!(
                    "🧠 [EVOLVER] DarkAlphaEngine 54D cargado exitosamente para co-evolución."
                );
                Some(m)
            }
        }
        Err(_) => {
            println!("ℹ️ [EVOLVER] models/DarkAlpha_BTCUSDT.json no encontrado: el NN no opina.");
            None
        }
    };

    // FASE AI HONESTA: Cargar todos los modelos NanoForest en el registro global para co-evolución de IA (Punto #11)
    println!("🌲 [EVOLVER] Cargando modelos NanoForest para co-evolución honesta...");
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let ext = path.extension().and_then(|s| s.to_str());
            if ext == Some("json") || ext == Some("bin") {
                if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if file_stem.contains("MOTOR") {
                        if let Ok(_) = god_engine_core::ml_inference::NanoForest::load_global(
                            file_stem,
                            path.to_str().unwrap(),
                        ) {
                            println!("🌲 [EVOLVER] NanoForest Brain cargado: {}", file_stem);
                        }
                    }
                }
            }
        }
    }
    // LG-17: sin `{SYM}_MOTOR` propio la moneda no opera (B3.25), igual que en
    // vivo. Antes se le prestaba el bosque FDUSD o el de BTC y la evolución
    // seleccionaba genomas en un mundo donde todas operaban con el de BTC.
    for sym in &symbols {
        let key = format!("{}_MOTOR", sym);
        if god_engine_core::ml_inference::NanoForest::get_global(&key).is_none() {
            println!("🌲 [EVOLVER] {} sin modelo propio: no opera en la evolución.", key);
        }
    }

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
                // LG-17: selección sobre el 70 % inicial del flujo.
                let e = evaluar(genome, seleccion, initial_capital, trained_nn.as_ref(), &omni_neutro);
                let final_capital = e.final_capital;
                let total_trades = e.total_trades;
                let max_drawdown = e.max_drawdown;
                let win_rate = if total_trades > 0 {
                    e.wins as f64 / total_trades as f64
                } else {
                    0.0
                };
                // CERT-M5-H01: función única de aptitud (utilidad log − λ·dd²).
                let fitness = aptitud(&e, initial_capital, false);

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

        // LG-17: el campeón de la selección y el incumbente se juzgan en la
        // MISMA partición final, que la selección no vio. Antes se promovía
        // con la aptitud de selección y con un OR que aceptaba cualquier
        // capital final > inicial aunque la aptitud fuera INVIABLE.
        // En el pool de rayon: el núcleo no cabe en la pila del hilo principal.
        let (e_cand, e_inc) = rayon::join(
            || evaluar(genome, juicio, initial_capital, trained_nn.as_ref(), &omni_neutro),
            || evaluar(&current_champion, juicio, initial_capital, trained_nn.as_ref(), &omni_neutro),
        );
        let fit_cand = aptitud(&e_cand, initial_capital, true);
        let fit_inc = aptitud(&e_inc, initial_capital, true);
        println!(
            "⚖️  Juicio fuera de muestra: candidato ${:.2} ({} ops, DD {:.1}%, apt {:.4}) vs incumbente ${:.2} ({} ops, DD {:.1}%, apt {:.4})",
            e_cand.final_capital, e_cand.total_trades, e_cand.max_drawdown * 100.0, fit_cand,
            e_inc.final_capital, e_inc.total_trades, e_inc.max_drawdown * 100.0, fit_inc
        );
        if decidir_promocion(fit_cand, fit_inc, tapes_reales) {
            // F4.3: embudo único — envelope versionado con linaje.
            match quantum_arena::genome_store::GenomeEnvelope::promote(
                genome.clone(),
                "ring_island_evolver",
                &format!(
                    "selección: capital {:.2} → {:.2} ({:+.2}%) apt {:.4} | juicio OOS: {:.2} → {:.2} apt {:.4} vs incumbente {:.4}",
                    initial_capital, best_cap, pnl_pct, best_fitness,
                    initial_capital, e_cand.final_capital, fit_cand, fit_inc
                ),
            ) {
                Ok(env) => println!(
                    "🧬 ✅ Genoma generación {} promovido (padre {}). Envelope + historia + espejo legacy escritos.",
                    env.generation, env.parent_generation
                ),
                Err(e) => println!("❌ Error promoviendo genoma al almacén: {}", e),
            }
        } else if !tapes_reales {
            println!(
                "⚠️ Tapes sin origen real (sintéticos o sin cabecera, D-691): no se promueve."
            );
        } else {
            println!(
                "⚠️ El campeón no supera al incumbente fuera de muestra. No se promueve genoma."
            );
        }
    } else {
        println!("Ninguna configuración sobrevivió.");
    }
    println!("============================================================");
    Ok(())
}

#[cfg(test)]
mod tests_lg17 {
    use super::*;

    #[test]
    fn lg17_sin_tapes_reales_no_se_promueve() {
        assert!(!decidir_promocion(0.50, 0.10, false));
        assert!(!decidir_promocion(0.50, f64::NEG_INFINITY, false));
    }

    #[test]
    fn lg17_contra_un_incumbente_medido_exige_el_margen_comun() {
        // meets_promotion_margin: con base ≥ 0, > max(1,05·base, base + 1e-4).
        assert!(!decidir_promocion(0.104, 0.10, true));
        assert!(decidir_promocion(0.20, 0.10, true));
        assert!(!decidir_promocion(f64::NEG_INFINITY, 0.10, true));
    }

    #[test]
    fn lg17_contra_un_incumbente_sin_evidencia_exige_aptitud_positiva() {
        assert!(!decidir_promocion(-0.05, f64::NEG_INFINITY, true));
        assert!(!decidir_promocion(0.0, f64::NEG_INFINITY, true));
        assert!(decidir_promocion(0.05, f64::NEG_INFINITY, true));
    }

    #[test]
    fn lg17_el_corte_deja_la_cola_cronologica_para_juzgar() {
        assert_eq!(indice_de_corte(0), 0);
        assert_eq!(indice_de_corte(10), 7);
        assert_eq!(indice_de_corte(1_000_000), 700_000);
    }

    /// Cableado: el evolver ya no presta bosques, no cobra dos veces la
    /// fricción, no pisa el omni con features de microestructura ni
    /// promueve con aptitud de selección.
    #[test]
    fn lg17_cableado_del_evolver() {
        let src = include_str!("evolver.rs");
        let fin = src.find("#[cfg(test)]").unwrap_or(src.len());
        let codigo = &src[..fin];
        let prohibido = [
            ["NanoForest::store_global(&key", ""].concat(),
            ["reality_slippage_penalty", "_with_notional"].concat(),
            ["omni[21] = ", "104.0"].concat(),
            ["best_cap > initial_capital ", "|| best_fitness"].concat(),
            ["DarkAlphaEngine::", "default_model()"].concat(),
        ];
        for p in &prohibido {
            assert!(!codigo.contains(p.as_str()), "el evolver aún contiene «{p}»");
        }
        assert!(codigo.contains("decidir_promocion("));
        assert!(codigo.contains("TG_GENOME_ENV"));
    }
}
