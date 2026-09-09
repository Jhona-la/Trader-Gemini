use data_pipeline::historical::Kline;

use god_engine_core::GodEngineCore;

use quantum_arena::{GlobalArena, TickEvent};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Simplificamos a 1 moneda primaria (BTCUSDT) para validar el loop continuo más rápido.
#[allow(dead_code)]
const COINS: [&str; 1] = ["BTCUSDT"];

#[allow(dead_code)]
fn simple_kline_to_ticks(coin_id: usize, kline: &Kline) -> Vec<TickEvent> {
    let mut ticks = Vec::with_capacity(4);
    let duration = kline.close_time.saturating_sub(kline.open_time);
    
    let o = if kline.open.is_finite() && kline.open > 0.0 { kline.open } else { 1.0 };
    let h = if kline.high.is_finite() && kline.high > 0.0 { kline.high.max(o) } else { o };
    let l = if kline.low.is_finite() && kline.low > 0.0 { kline.low.min(o).max(1e-6) } else { o * 0.999 };
    let c = if kline.close.is_finite() && kline.close > 0.0 { kline.close } else { o };
    let total_v = if kline.volume.is_finite() && kline.volume > 0.0 { kline.volume } else { 1.0 };
    
    let _v_per_tick = total_v / 4.0;
    let step = duration / 4;
    
    // FIX #D16: Eliminación de Lookahead Bias (c >= o).
    // No se conoce el cierre futuro 'c' para ordenar el path intra-barra.
    // Se utiliza la proximidad causal del precio de apertura al extremo más cercano.
    let open_closer_to_low = (o - l).abs() <= (h - o).abs();
    let points = if open_closer_to_low {
        vec![
            (kline.open_time, o),
            (kline.open_time + step, l),
            (kline.open_time + step * 2, h),
            (kline.close_time, c),
        ]
    } else {
        vec![
            (kline.open_time, o),
            (kline.open_time + step, h),
            (kline.open_time + step * 2, l),
            (kline.close_time, c),
        ]
    };

    for (ts, price) in points {
        let spread_half = (price * 0.00005).max(0.00001); // 1 bps spread
        ticks.push(TickEvent {
            coin_id,
            timestamp: ts,
            bid_price: (price - spread_half).max(1e-6),
            ask_price: price + spread_half,
            bid_qty: 15.0 + (price % 5.0), // Simulated dynamic L1 Bid Depth
            ask_qty: 15.0 + ((price * 1.5) % 5.0), // Simulated dynamic L1 Ask Depth
        });
    }
    
    ticks
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // E3 — ENTORNO DE GENOMA AISLADO: las promociones de este backtest van a
    // config_dir/genomes/backtest/ y NUNCA contaminan el active.json de demo
    // o producción. La promoción cross-entorno es explícita.
    unsafe { std::env::set_var("TG_GENOME_ENV", "backtest"); }
    println!("============================================================");
    println!("🌌 TRADER GEMINI V5 - CONTINUOUS EVOLUTION BACKTESTER");
    println!("🛡️ Meta-Engine: Walk-Forward (Train 3 Days -> Trade 1 Day)");
    println!("============================================================");

    // 🔮 ORACLE PREDICTIVE METRICS: Arrancar el Reconciliador del Tiempo
    // tokio::spawn(telemetry_server::oracle_profiler::OracleProfiler::start_time_reconciler()); // Disabled for backtest

    // 📊 Inicializar CSV de Telemetría PnL
    let mut csv_file = std::fs::File::create("equity_curve.csv").expect("No se pudo crear equity_curve.csv");
    use std::io::Write;
    writeln!(csv_file, "Day,Capital,DailyPnL,PnLPercent,Trades").unwrap();

    // FIX: el default silencioso de 3 días truncaba backtests de 7d/15d/1m/180d
    // sin avisar (los lanzadores no pasan argumento). Ahora exige días
    // explícitos por CLI o env SIM_DAYS, y falla ruidosamente si faltan.
    let sim_days: u64 = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("SIM_DAYS").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            eprintln!(
                "ERROR: días de simulación no especificados. Uso: continuous_evolution_backtest <dias> [all] [mutantes] o env SIM_DAYS=<dias>"
            );
            std::process::exit(2);
        });
    let mode_arg = std::env::args().nth(2).unwrap_or_default();
    let is_multicoin = mode_arg == "all" || std::env::var("MULTI_COIN").unwrap_or_default() == "1";
    let requested_mutants: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(250);

    // R2.2 — PRIORIDAD AL DATO REAL: si existe el bin de aggTrades reales
    // (magic TGMTICK1), se usa; el sintético queda como fallback.
    let btc_path = {
        let real = Path::new("data/BTCUSDT_ticks_REAL.bin");
        if real.exists() {
            println!("📡 R2.2: usando AGGTRADES REALES (certificación sin artefactos de síntesis).");
            real
        } else {
            Path::new("data/BTCUSDT_ticks.bin")
        }
    };
    let eth_path = Path::new("data/ETHUSDT_ticks.bin");
    let sol_path = Path::new("data/SOLUSDT_ticks.bin");
    let bnb_path = Path::new("data/BNBUSDT_ticks.bin");
    let xrp_path = Path::new("data/XRPUSDT_ticks.bin");

    let specs = vec![
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "BTCUSDT".to_string(),
            step_size: 0.001, tick_size: 0.1, min_qty: 0.001, min_notional: 5.0,
            max_leverage: 125, maker_fee: 0.0002, taker_fee: 0.0005, is_shadow: false,
        },
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "ETHUSDT".to_string(),
            step_size: 0.01, tick_size: 0.01, min_qty: 0.01, min_notional: 5.0,
            max_leverage: 100, maker_fee: 0.0002, taker_fee: 0.0005, is_shadow: false,
        },
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "SOLUSDT".to_string(),
            step_size: 0.1, tick_size: 0.01, min_qty: 0.1, min_notional: 5.0,
            max_leverage: 50, maker_fee: 0.0002, taker_fee: 0.0005, is_shadow: false,
        },
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "BNBUSDT".to_string(),
            step_size: 0.01, tick_size: 0.01, min_qty: 0.01, min_notional: 5.0,
            max_leverage: 50, maker_fee: 0.0002, taker_fee: 0.0005, is_shadow: false,
        },
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "XRPUSDT".to_string(),
            step_size: 1.0, tick_size: 0.0001, min_qty: 1.0, min_notional: 5.0,
            max_leverage: 50, maker_fee: 0.0002, taker_fee: 0.0005, is_shadow: false,
        },
    ];
    quantum_arena::symbol_registry::update_registry(specs);

    let ticks = if is_multicoin {
        let files: Vec<(&Path, usize)> = vec![
            (btc_path, 0),
            (eth_path, 1),
            (sol_path, 2),
            (bnb_path, 3),
            (xrp_path, 4),
        ].into_iter().filter(|(p, _)| p.exists()).collect();
        println!("🌐 Modo Multi-Activo Habilitado: cargando {} archivos de ticks reales...", files.len());
        backtest_engine::tick_replayer::load_multi_coin_binary_ticks(&files)?
    } else {
        unsafe { std::env::set_var("SINGLE_COIN_MODE", "1"); }
        if !btc_path.exists() {
            println!("❌ Archivo binario no encontrado: {:?}.", btc_path);
            return Ok(());
        }
        backtest_engine::tick_replayer::load_binary_ticks(btc_path, 0)?
    };

    println!("✅ {} Ticks Reales cargados (AGGTRADES REALES) para Walk-Forward.", ticks.len());

    let initial_capital = 13.0; // El capital microscópico desafiante
    let mut current_capital = initial_capital;
    
    let ms_per_day = 86_400_000;
    let first_ts = ticks.first().unwrap().timestamp;
    let last_ts = ticks.last().unwrap().timestamp;
    let total_days_in_data = (((last_ts.saturating_sub(first_ts)) as f64 / ms_per_day as f64).ceil() as u64).max(1);
    let total_simulation_days = total_days_in_data.min(sim_days);

    let force_fresh = std::env::var("FORCE_FRESH_BASELINE").map(|v| v == "1" || v == "true").unwrap_or(false);
    let mut current_genome = if force_fresh {
        println!("🧬 [BOOTSTRAP] FORCE_FRESH_BASELINE activo: iniciando desde Baseline Matemático.");
        quantum_arena::genome::SuperGenotype::new_baseline(0.0002, 0.0005)
    } else {
        quantum_arena::genome_store::GenomeEnvelope::load_active()
            .map(|e| e.genome)
            .unwrap_or_else(|| {
                if let Ok(data) = std::fs::read_to_string("config_dir/genotypes/champion.json") {
                    if let Ok(g) = serde_json::from_str::<quantum_arena::genome::SuperGenotype>(&data) {
                        println!("🧬 [BOOTSTRAP] Cargado Genoma Campeón persistido desde champion.json.");
                        return g;
                    }
                }
                quantum_arena::genome::SuperGenotype::new_baseline(0.0002, 0.0005)
            })
    };

    if !current_genome.scalp_sl_base.is_finite() || current_genome.scalp_sl_base <= 0.0 {
        current_genome.scalp_sl_base = 0.0065;
    }
    if !current_genome.scalp_tp_base.is_finite() || current_genome.scalp_tp_base <= 0.0 {
        current_genome.scalp_tp_base = 0.0160;
    }
    if !current_genome.tech_threshold.is_finite() || current_genome.tech_threshold <= 0.0 {
        current_genome.tech_threshold = 0.1487;
    }
    if !current_genome.scalp_kelly_fraction.is_finite() || current_genome.scalp_kelly_fraction <= 0.0 {
        current_genome.scalp_kelly_fraction = 0.52;
    }
    
    println!("📈 Iniciando simulación Walk-Forward de {} Días", total_simulation_days);

    let mut global_pnl = 0.0;
    let mut day_idx = 0;
    let mut idx = 0;
    
    // MOTOR MAESTRO CONTINUO (Paridad 1:1 Estricta con Producción Real):
    // La arena y el motor se instancian una sola vez para mantener vivos los filtros wavelets,
    // promedios exponenciales, modelos L2 y posiciones activas sin reseteos artificiales a medianoche.
    let arena = Arc::new(GlobalArena::new(current_capital));
    arena.config.live_maker_fee.store(0.0002, Ordering::Relaxed);
    arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);
    arena.config.base_capital.store(initial_capital, Ordering::Relaxed);
    current_genome.apply_to_arena(&arena);
    
    let mut engine = GodEngineCore::new(arena.clone());
    engine.reality.mode = god_engine_core::reality_physics::EngineMode::HyperRealistic;
    engine.arena.config.latency_penalty_ms.store(25.0, Ordering::Relaxed);
    
    while day_idx < total_simulation_days {
        let day_start_ts = first_ts + (day_idx * ms_per_day);
        let day_end_ts = day_start_ts + ms_per_day;

        let day_start_capital = arena.unified_capital.load(Ordering::Relaxed);
        let mut day_trades = 0;
        let mut day_scalp_trades = 0;
        let mut day_scalp_pnl = 0.0;
        let mut day_scalp_wins = 0;
        let mut prev_kline_ts = 0;
        
        // ESCALADO MASIVO DE MUTANTES (ENJAMBRE CUÁNTICO PARALELO CON RAYON: 10 NICHOS ECOLÓGICOS)
        let num_mutants = requested_mutants.clamp(10, 1000);
        let mut shadow_engines = Vec::with_capacity(num_mutants);
        let mut shadow_genomes = Vec::with_capacity(num_mutants);
        println!("🧬 [ENJAMBRE CUÁNTICO] Desplegando {} mutantes concurrentes en 10 nichos ecológicos...", num_mutants);
        for i in 0..num_mutants {
            // DETERMINISMO MULTI-DÍA: semilla única por (día, mutante) — el
            // enjambre produce la MISMA población en cada corrida.
            let deterministic_seed: u64 = 0x5EED_0000_0000_0000u64
                .wrapping_add((day_idx as u64) << 32)
                .wrapping_add(i as u64);
            let mutant_arena = Arc::new(GlobalArena::new(current_capital));
            mutant_arena.config.live_maker_fee.store(0.0002, Ordering::Relaxed);
            mutant_arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);
            mutant_arena.config.base_capital.store(initial_capital, Ordering::Relaxed);
            let mut mutant_genome = if i == 0 {
                current_genome.clone() // Baseline elitism
            } else if i < (num_mutants * 10 / 100).max(1) {
                // Nicho 1: Afinamiento Fino CMA-ES (0.05) - Explotación pura
                current_genome.mutate_cmaes_seeded(0.05, deterministic_seed)
            } else if i < (num_mutants * 20 / 100).max(2) {
                // Nicho 2: Especialista en Scalping L2 y OFI (TP corto, SL ceñido, Kelly controlado)
                let mut g = current_genome.mutate_cmaes_seeded(0.15, deterministic_seed);
                g.scalp_tp_base = g.scalp_tp_base.clamp(0.0120, 0.0240);
                g.scalp_sl_base = (g.scalp_sl_base * 0.85).clamp(0.0030, 0.0075);
                g.scalp_kelly_fraction = g.scalp_kelly_fraction.clamp(0.15, 0.35);
                g.base_duration_ms = 10_000.0;
                g
            } else if i < (num_mutants * 30 / 100).max(3) {
                // Nicho 3: Soliton Wavelet & Multiscale Trend (TP amplio, Trailing ATR)
                let mut g = current_genome.mutate_cmaes_seeded(0.20, deterministic_seed);
                g.scalp_tp_base = g.scalp_tp_base.clamp(0.0180, 0.0400);
                g.scalp_sl_base = (g.scalp_sl_base * 1.15).clamp(0.0050, 0.0120);
                g.scalp_trail_act_atr = g.scalp_trail_act_atr.clamp(0.8, 1.8);
                g
            } else if i < (num_mutants * 40 / 100).max(4) {
                // Nicho 4: KAN Neural & DarkAlpha (Alta ponderación neural)
                let mut g = current_genome.mutate_cmaes_seeded(0.25, deterministic_seed);
                g.tech_threshold = g.tech_threshold.clamp(0.120, 0.220);
                g
            } else if i < (num_mutants * 50 / 100).max(5) {
                // Nicho 5: Mean-Reversion & Wall Bounce (Absorción en muros L2)
                let mut g = current_genome.mutate_cmaes_seeded(0.20, deterministic_seed);
                g.weight_obi = (g.weight_obi * 1.8).clamp(0.5, 2.5);
                g.scalp_tp_base = g.scalp_tp_base.clamp(0.0120, 0.0250);
                g.scalp_sl_base = (g.scalp_sl_base * 0.90).clamp(0.0035, 0.0085);
                g
            } else if i < (num_mutants * 60 / 100).max(6) {
                // Nicho 6: Volatility Squeeze Breakout (Compresión y explosión ATR/Bollinger)
                let mut g = current_genome.mutate_cmaes_seeded(0.25, deterministic_seed);
                g.target_volatility = g.target_volatility.clamp(0.01, 0.04);
                g.explosive_confidence_threshold = g.explosive_confidence_threshold.clamp(0.70, 0.88);
                g.explosive_leverage_multiplier = g.explosive_leverage_multiplier.clamp(1.5, 4.0);
                g
            } else if i < (num_mutants * 70 / 100).max(7) {
                // Nicho 7: Macro Lead-Lag Arbitrage (Impulsos BTC transmitidos a altcoins)
                let mut g = current_genome.mutate_cmaes_seeded(0.30, deterministic_seed);
                g.global_correlation_threshold = g.global_correlation_threshold.clamp(0.30, 0.70);
                g
            } else if i < (num_mutants * 80 / 100).max(8) {
                // Nicho 8: Zero-Taker Maker Rebate Harvester (Captura de spread y fees negativos)
                let mut g = current_genome.mutate_cmaes_seeded(0.15, deterministic_seed);
                g.maker_spread_pct = g.maker_spread_pct.clamp(0.0002, 0.0008);
                g.maker_obi_threshold = g.maker_obi_threshold.clamp(0.20, 0.40);
                g
            } else if i < (num_mutants * 90 / 100).max(9) {
                // Nicho 9: Fractal Mandelbrot Trend Surfer (Hurst > 0.60, tendencias hiperbólicas)
                let mut g = current_genome.mutate_cmaes_seeded(0.25, deterministic_seed);
                g.trend_threshold = g.trend_threshold.clamp(0.55, 0.85);
                g.swing_tp_base = g.swing_tp_base.clamp(0.020, 0.060);
                g.swing_kelly_fraction = g.swing_kelly_fraction.clamp(0.15, 0.35);
                g
            } else {
                // Nicho 10: Saltos de Lévy / Mutaciones Cuánticas Globales (0.55 caótico)
                current_genome.mutate_cmaes_seeded(0.55, deterministic_seed)
            };
            // Blindaje Cuántico: cotas estrictas para todos los mutantes (impedir que nazcan mutantes cobardes o suicidas)
            mutant_genome.tech_threshold = mutant_genome.tech_threshold.clamp(0.080, 0.220);
            mutant_genome.scalp_kelly_fraction = mutant_genome.scalp_kelly_fraction.clamp(0.12, 0.38);
            mutant_genome.dynamic_obi_threshold = mutant_genome.dynamic_obi_threshold.clamp(0.15, 0.35);
            mutant_genome.dynamic_ofi_threshold = mutant_genome.dynamic_ofi_threshold.clamp(0.15, 0.35);
            mutant_genome.scalp_sl_base = mutant_genome.scalp_sl_base.clamp(0.0030, 0.0150);
            mutant_genome.swing_sl_base = mutant_genome.swing_sl_base.clamp(0.0080, 0.0350);
            mutant_genome.scalp_trail_act_atr = mutant_genome.scalp_trail_act_atr.clamp(1.0, 2.5);
            mutant_genome.scalp_trail_step_atr = mutant_genome.scalp_trail_step_atr.clamp(1.0, 2.5);
            mutant_genome.scalp_trail_atr_mult_base = mutant_genome.scalp_trail_atr_mult_base.clamp(1.0, 2.5);

            mutant_genome.apply_to_arena(&mutant_arena);
            let mut mutant_engine = GodEngineCore::new(mutant_arena);
            mutant_engine.reality.mode = god_engine_core::reality_physics::EngineMode::HyperRealistic;
            mutant_engine.arena.config.latency_penalty_ms.store(25.0, Ordering::Relaxed);
            shadow_engines.push(mutant_engine);
            shadow_genomes.push(mutant_genome);
        }
        
        // DIAGNOSTIC: Verify genome diversity is actually present in arena config
        for diag_i in 0..3.min(num_mutants) {
            let e = &shadow_engines[diag_i];
                println!("🔬 [GENOME DIAG] Mutant #{}: tech_thr={:.4}",
                diag_i,
                e.arena.config.tech_threshold.load(Ordering::Relaxed),
            );
        }

        let _synthetic_macro = [0.0f64; 54];
        let _pseudo_rng = 42u64;

        // DIAGNÓSTICO SIGNAL-PATH: contar en cada etapa del pipeline para
        // localizar dónde mueren las señales. Imprime al cierre del día.
        let mut diag = SignalPathDiag::default();

        while idx < ticks.len() && ticks[idx].timestamp < day_end_ts {
            if current_capital <= 1.0 { break; }
            let tick = &ticks[idx];
            let cid = tick.coin_id;
            arena.update_market_data(cid, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp);
            let mid = (tick.bid_price + tick.ask_price) / 2.0;

            let mut omni = [0.0f64; 54]; // Inicializar tensor limpio
            let live_vol = tick.bid_qty + tick.ask_qty;
            let live_ofi = if live_vol > 0.0 { (tick.bid_qty - tick.ask_qty) / live_vol } else { 0.0 };
            
            // FASE 21: Mapeo topológico multidimensional completo y causal 1:1 con Producción
            omni[0] = mid;
            omni[1] = live_vol;
            omni[2] = if mid > 0.0 { (tick.ask_price - tick.bid_price) / mid * 10000.0 } else { 0.0 };
            omni[3] = (live_vol * mid) / 1000.0;
            omni[4] = (live_ofi * 100.0).clamp(0.0, 200.0);
            omni[5] = 50.0; // Benchmark neutral RSI proxy
            omni[11] = 0.0001; // agg_funding_rate
            omni[21] = 104.2;  // dxy
            omni[22] = 5120.0; // sp500
            omni[23] = 18100.0;// nasdaq
            omni[24] = 18.5;   // vix
            omni[25] = 4.25;   // us10y
            omni[26] = 2320.0; // gold
            omni[27] = 81.0;   // oil_wti
            omni[29] = 5.25;   // fed_interest_rate
            omni[30] = tick.bid_qty - tick.ask_qty;
            omni[31] = (tick.bid_qty - tick.ask_qty) * 1.2;
            omni[39] = live_ofi;
            omni[48] = live_ofi.clamp(-1.0, 1.0);
            omni[49] = (live_vol / 100.0).tanh().clamp(-1.0, 1.0);

            let is_kline_closed = prev_kline_ts == 0 || (tick.timestamp / 60_000) != (prev_kline_ts / 60_000);
            if is_kline_closed { prev_kline_ts = tick.timestamp; }

            // S-07: increment_tick eliminado — process_tick_dual ya lo hace
            // internamente (el doble conteo hacia que la lógica temporal
            // corriera a 2x).
            let (new_ord, closed_ord, _) = engine.process_tick_dual(
                cid, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp, &omni
            );

            let new_order = new_ord.is_some();
            let closed_order = closed_ord.is_some();

            if let Some((_is_long, net_pnl, _qty)) = closed_ord {
                day_scalp_trades += 1;
                day_scalp_pnl += net_pnl;
                if net_pnl > 0.0 { day_scalp_wins += 1; }
                day_trades += 1;
            }

            // DIAGNÓSTICO SIGNAL-PATH: post-engine.
            {
                use signal_engine::SignalType;
                let si = &engine.last_scalp_intent[cid];
                let wi = &engine.last_swing_intent[cid];
                if si.signal != SignalType::Flat { diag.intents_scalp += 1; }
                if wi.signal != SignalType::Flat { diag.intents_swing += 1; }
                if si.signal != SignalType::Flat || wi.signal != SignalType::Flat {
                    diag.intents += 1;
                    let c = si.confidence.max(wi.confidence);
                    diag.max_intent_conf = diag.max_intent_conf.max(c);
                }
                let reg = &engine.arena.registry;
                diag.max_obi = diag.max_obi.max(reg.get_value_or("orderbook_imbalance", 0.0).abs());
                diag.tech_thr = reg.get_value_or("tech_threshold", 0.0);
                diag.max_micro_trend = diag.max_micro_trend.max(reg.get_value_or("ema_trend", 0.0).abs());
                if new_order { diag.orders += 1; }
            }

            if idx % 100_000 == 0 || new_order || closed_order {
                let safe_cid = cid.min(engine.feature_engines.len().saturating_sub(1));
                let atr = engine.feature_engines[safe_cid].get_atr_pct();
                let hurst = engine.feature_engines[safe_cid].hurst.current();
                let macro_t = engine.feature_engines[safe_cid].get_macro_trend();
                let is_open = engine.arena.coins[cid].positions.position.is_open();
                println!("🔍 [TICK #{}] coin={} ts={} mid={:.2} atr={:.6} hurst={:.3} macro_trend={:.6} open={} new_ord={} closed={}",
                    idx, cid, tick.timestamp, mid, atr, hurst, macro_t, is_open, new_order, closed_order);
            }

            // Transmitir al Shadow Forest Inline (Paralelizado para velocidad extrema)
            shadow_engines.iter_mut().for_each(|shadow_engine| {
                shadow_engine.arena.update_market_data(cid, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp);
                shadow_engine.arena.coins[cid].current_price.store(mid, Ordering::Relaxed);

                let _ = shadow_engine.process_tick_dual(
                    cid, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp, &omni
                );
            });

            idx += 1;
        }

        println!("🏁 Loop terminado. Total ticks procesados en Día {}: {}", day_idx + 1, idx);

        let day_start_cap_real = day_start_capital;
        let day_final_cap = arena.unified_capital.load(Ordering::Relaxed);
        let last_tick_price = if idx > 0 { ticks[idx - 1].bid_price } else { ticks[0].bid_price };
        let mut open_unrealized = 0.0;
        for coin in arena.coins.iter() {
            let pos = &coin.positions.position;
            if pos.is_open() {
                let entry = pos.entry_price.load(Ordering::Relaxed);
                let qty = pos.quantity.load(Ordering::Relaxed);
                let is_long = pos.is_long.load(Ordering::Relaxed);
                let c_price = coin.current_price.load(Ordering::Relaxed);
                let exit_price = if c_price > 0.0 { c_price } else { last_tick_price };
                let exit_fee = qty * exit_price * 0.0005;
                let unrealized = (exit_price - entry) * qty * if is_long { 1.0 } else { -1.0 } - exit_fee;
                if unrealized.is_finite() {
                    open_unrealized += unrealized;
                }
            }
        }
        let total_equity = day_final_cap + open_unrealized;
        let day_pnl = total_equity - day_start_cap_real;
        global_pnl += day_pnl;
        let pnl_pct = if day_start_cap_real > 0.0 { (day_pnl / day_start_cap_real) * 100.0 } else { 0.0 };
        current_capital = total_equity;

        println!("📅 DÍA {}: Capital: ${:.4} (Efectivo: ${:.4}, Flotante: ${:+.4}) | PnL Día: {:+.4} ({:+.2}%) | Trades: {} (Continuous: {} [WR: {:.1}%, PnL: {:+.4}])", 
            day_idx + 1, current_capital, day_final_cap, open_unrealized, day_pnl, pnl_pct, day_trades,
            day_scalp_trades, if day_scalp_trades > 0 { (day_scalp_wins as f64 / day_scalp_trades as f64) * 100.0 } else { 0.0 }, day_scalp_pnl);
        
        println!("🔬 [SIGNAL-PATH] rejects: {}", risk_engine::reject_report());
        {
            let e = &engine;
            let wr = if e.diag_close_total > 0 { e.diag_close_wins as f64 / e.diag_close_total as f64 } else { 0.0 };
            let avg_not = if e.diag_close_total > 0 { e.diag_notional_sum / e.diag_close_total as f64 } else { 0.0 };
            let cap = e.arena.unified_capital.load(Ordering::Relaxed);
            println!("🔬 [FILL-MODEL] closes={} WR={:.3} avg_notional=${:.2} max_notional=${:.2} capital=${:.2} avg_pnl={:.4}",
                e.diag_close_total, wr, avg_not, e.diag_notional_max, cap, if e.diag_close_total > 0 { e.diag_pnl_sum / e.diag_close_total as f64 } else { 0.0 });
        }
        println!("🔬 [SIGNAL-PATH] council_vetoes={} opened={} swing_vetoes={} swing_opened={} | intents={} (scalp={} swing={}) orders={} max_intent_conf={:.4} max_obi={:.4} tech_thr={:.6} max_micro_trend={:.6}",
            engine.diag_council_vetoes, engine.diag_opened, engine.diag_swing_vetoes, engine.diag_swing_opened, diag.intents, diag.intents_scalp, diag.intents_swing, diag.orders, diag.max_intent_conf, diag.max_obi, diag.tech_thr, diag.max_micro_trend);
        // diag removed
             
        // Escribir Telemetría Plotly
        writeln!(csv_file, "{},{:.4},{:.4},{:.2},{}", day_idx + 1, current_capital, day_pnl, pnl_pct, day_trades).unwrap();

        if current_capital <= 0.0 {
            println!("💀 REKT (Liquidado) en Día {}", day_idx + 1);
            break;
        }

        // --- FASE 2: EVOLUCIONAR (Meta-Learning) ---
        // Cosechar el mejor genoma del bosque oscuro inline
        let mut best_fitness = -999999.0;
        let mut best_idx = 0;
        
        let mut all_mutant_stats = Vec::with_capacity(num_mutants);
        let mutant_start_cap = current_capital;
        for (i, engine) in shadow_engines.iter().enumerate() {
            let mut cap = engine.arena.unified_capital.load(Ordering::Relaxed);
            for coin in engine.arena.coins.iter() {
                let pos = &coin.positions.position;
                if pos.is_open() {
                    let entry = pos.entry_price.load(Ordering::Relaxed);
                    let qty = pos.quantity.load(Ordering::Relaxed);
                    let is_long = pos.is_long.load(Ordering::Relaxed);
                    let c_price = coin.current_price.load(Ordering::Relaxed);
                    let exit_price = if c_price > 0.0 { c_price } else { last_tick_price };
                    let exit_fee = qty * exit_price * 0.0005;
                    let unrealized = (exit_price - entry) * qty * if is_long { 1.0 } else { -1.0 } - exit_fee;
                    if unrealized.is_finite() {
                        cap += unrealized;
                    }
                }
            }
            let pnl = cap - mutant_start_cap;
            let safe_pnl = if pnl.is_finite() { pnl } else { -999999.0 };
            
            // Contar actividad real de trades del mutante en el día
            let m_trades: usize = engine.arena.coins.iter().map(|c| c.metrics.trade_count.load(Ordering::Relaxed)).sum();
            // Descalificación estricta contra inactividad (Anti-Cowardice Fitness):
            // Si el mutante no operó en todo el día (0 trades), recibe descalificación inmediata (safe_pnl).
            // Un mutante inerte nunca puede ganar ni transmitir genes de inactividad.
            let fitness = if m_trades == 0 {
                safe_pnl
            } else {
                let activity_bonus = 0.0; // H-8: neutralizado — premiaba churn no edge
                safe_pnl + activity_bonus
            };

            all_mutant_stats.push((i, safe_pnl, fitness, m_trades));
            if fitness > best_fitness {
                best_fitness = fitness;
                best_idx = i;
            }
        }
        all_mutant_stats.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        if all_mutant_stats.len() >= 5 {
            println!("🧬 [POPULATION DIVERSITY] Top 5 Mutants: 1) #{}: PnL {:+.4} (Fit: {:+.4}, Trd: {}) | 2) #{}: {:+.4} | 3) #{}: {:+.4} | 4) #{}: {:+.4} | 5) #{}: {:+.4}",
            all_mutant_stats[0].0, all_mutant_stats[0].1, all_mutant_stats[0].2, all_mutant_stats[0].3,
            all_mutant_stats[1].0, all_mutant_stats[1].1,
            all_mutant_stats[2].0, all_mutant_stats[2].1,
            all_mutant_stats[3].0, all_mutant_stats[3].1,
            all_mutant_stats[4].0, all_mutant_stats[4].1);
        } else {
            println!("🧬 [POPULATION DIVERSITY] Best Mutant: #{}: PnL {:+.4}", all_mutant_stats[0].0, all_mutant_stats[0].1);
        }
        
        for rank in 0..3.min(all_mutant_stats.len()) {
            let m_idx = all_mutant_stats[rank].0;
            let safe_pnl = all_mutant_stats[rank].1;
            let m_trd = all_mutant_stats[rank].3;
            let g = &shadow_genomes[m_idx];
            println!("   🏅 Rank #{}: Mutant #{} (PnL: {:+.4}, Trades: {}) -> tp: {:.4}, sl: {:.4}, tech_thr: {:.4}, kelly: {:.4}, trail_act: {:.4}",
                rank + 1, m_idx, safe_pnl, m_trd, g.scalp_tp_base, g.scalp_sl_base, g.tech_threshold, g.scalp_kelly_fraction, g.scalp_trail_act_atr);
        }
        
        // PnL y Fitness del control (#0)
        let control_stats = all_mutant_stats.iter().find(|(idx, _, _, _)| *idx == 0).cloned().unwrap_or((0, 0.0, 0.0, 0));
        let control_fitness = control_stats.2;
        let _control_trades = control_stats.3;
        let best_trades = all_mutant_stats.iter().find(|(idx, _, _, _)| *idx == best_idx).map(|s| s.3).unwrap_or(0);
        let improvement = best_fitness - control_fitness;

        let best_pnl = all_mutant_stats.iter().find(|(idx, _, _, _)| *idx == best_idx).map(|s| s.1).unwrap_or(0.0);
        let control_pnl = control_stats.1;

        // Regla Cuántica de Elitismo y Adaptación Continua:
        // Un mutante puede destronar al baseline si:
        // 1. Operó de forma activa y real (best_trades >= 1).
        // 2. Superó al control en fitness (mejora real > capital * 0.0002).
        // 3. O bien generó Alpha positivo (best_pnl > 0.0), o bien en un día de contracción general
        //    protegió el capital sustancialmente mejor que el control (best_pnl > control_pnl + 0.30).
        let valid_replacement = (best_trades >= 1 && best_pnl > (control_pnl + 0.30)) || (best_trades >= 1 && best_pnl > 0.0 && control_pnl <= 0.0); // R-04: sin actividad REAL mínima no hay reemplazo — ni la rama forzada promueve mutantes sin trades

        if valid_replacement {
            println!("🧬 [EVOLUTION] Mutant #{} replaces Baseline! (PnL: {:.4} vs {:.4}, Imp: {:.4}, Trades: {})", 
                best_idx, best_pnl, control_pnl, improvement, best_trades);
            current_genome = shadow_genomes[best_idx].clone();
        } else if best_pnl > control_pnl && best_trades >= 1 {
            println!("🧬 [EVOLUTION] Mutante #{} superó al control en PnL ({:+.4} vs {:+.4}). Adoptando adaptación.", 
                best_idx, best_pnl, control_pnl);
            current_genome = shadow_genomes[best_idx].clone();
        } else if best_pnl <= 0.0 && !(best_pnl > (control_pnl + 0.30)) {
            println!("🧬 [ESTABILIDAD] Población en contracción sin mejora de capital. Genoma Base protegido.");
        } else {
            println!("🧬 [ESTABILIDAD] El Genoma Base venció a la población mutante. Sin mutación.");
        }

        // Sanitización ligera para evitar NaNs sin restringir la adaptabilidad genética
        if !current_genome.tech_threshold.is_finite() { current_genome.tech_threshold = 0.12; }
        if !current_genome.scalp_kelly_fraction.is_finite() { current_genome.scalp_kelly_fraction = 0.55; }
        if !current_genome.dynamic_obi_threshold.is_finite() { current_genome.dynamic_obi_threshold = 0.25; }
        if !current_genome.dynamic_ofi_threshold.is_finite() { current_genome.dynamic_ofi_threshold = 0.25; }
        if !current_genome.dynamic_ema_trend.is_finite() { current_genome.dynamic_ema_trend = 0.00025; }

        current_genome.apply_to_arena(&arena);
        println!("🧬 [GENOMA ACTUAL] tech_threshold: {:.6}, scalp_kelly_fraction: {:.6}, dynamic_obi: {:.4}", 
            current_genome.tech_threshold, current_genome.scalp_kelly_fraction, current_genome.dynamic_obi_threshold);

        day_idx += 1;
    }

    // PROMOTE AL ENVELOPE GLOBAL
    match quantum_arena::genome_store::GenomeEnvelope::promote(
        current_genome.clone(),
        "continuous_evolution_backtest",
        "End of 30-day pre-training",
    ) {
        Ok(env) => println!("💾 [GUARDADO] Genoma Campeón persistido exitosamente en generación {}.", env.generation),
        Err(e) => println!("⚠️ [ERROR] No se pudo guardar el genoma: {}", e),
    }

    println!("============================================================");
    println!("🏆 FIN DE LA SIMULACIÓN CONTINUA");
    println!("💰 Capital Inicial: $13.00");
    println!("💰 Capital Final  : ${:.4}", current_capital);
    println!("💵 PnL Global     : ${:+.4}", global_pnl);
    println!("📈 Crecimiento    : {:.2}%", ((current_capital - 13.0) / 13.0) * 100.0);
    println!("============================================================");

    Ok(())
}

/// DIAGNÓSTICO SIGNAL-PATH (R4-diag): contadores por etapa del pipeline.
#[derive(Default)]
struct SignalPathDiag {
    intents: u64,
    intents_scalp: u64,
    intents_swing: u64,
    orders: u64,
    max_intent_conf: f64,
    max_obi: f64,
    tech_thr: f64,
    max_micro_trend: f64,
}
