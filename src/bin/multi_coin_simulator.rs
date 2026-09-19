use chrono::{DateTime, Utc};
use data_pipeline::historical::Kline;
use data_pipeline::multiplexer::multiplex_ticks;
use data_pipeline::omni_multiplexer::OmniState;
use god_engine_core::GodEngineCore;
use phase_runner::{Phase, PhaseExecutor};
use polars::prelude::{ParquetReader, SerReader};
use quantum_arena::{GlobalArena, TickEvent};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const COINS: [&str; 30] = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "XLMUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "XMRUSDT",
    "ETCUSDT",
    "FILUSDT",
    "ICPUSDT",
    "VETUSDT",
    "NEARUSDT",
    "AAVEUSDT",
    "ALGOUSDT",
    "EGLDUSDT",
    "SANDUSDT",
    "THETAUSDT",
    "AXSUSDT",
    "MANAUSDT",
    "FTMUSDT",
];

// Convertir 1 Kline en 4 ticks determinísticos con spread institucional realista (1.5 bps).
// R2.1c — SIN LOOKAHEAD: el desequilibrio de volumen (OBI) de los 4 ticks se
// deriva de la dirección de la vela PREVIA (`prev_bullish`), no de `c >= o` de
// la vela en curso. Antes, observar el OBI del primer tick equivalía a conocer
// el cierre 45 segundos antes — el "edge" del backtest forense era el sesgo.
fn simple_kline_to_ticks(coin_id: usize, kline: &Kline, prev_bullish: bool) -> [TickEvent; 4] {
    let step = kline.close_time.saturating_sub(kline.open_time) / 4;
    // FIX #1480: Sanitización de precios y volúmenes en generación de ticks
    let o = if kline.open.is_finite() && kline.open > 0.0 {
        kline.open
    } else {
        1.0
    };
    let h = if kline.high.is_finite() && kline.high > 0.0 {
        kline.high.max(o)
    } else {
        o
    };
    let l = if kline.low.is_finite() && kline.low > 0.0 {
        kline.low.min(o).max(1e-6)
    } else {
        o * 0.999
    };
    let c = if kline.close.is_finite() && kline.close > 0.0 {
        kline.close
    } else {
        o
    };
    let v = if kline.volume.is_finite() && kline.volume > 0.0 {
        kline.volume / 4.0
    } else {
        0.25
    };

    let spread_half = (o * 0.000075).max(0.00001); // 0.75 bps = 1.5 bps total spread

    // R2.1: Trayectoria intra-vela sin coreografía rígida (50% high-first / 50% low-first)
    let high_first =
        ((kline.open_time.wrapping_mul(0x9E3779B97F4A7C15) ^ (coin_id as u64)) & 1) == 0;
    let (p2, p3) = if high_first { (h, l) } else { (l, h) };

    // D-415: Modelo causal Lee-Ready sin sesgo artificial fijo 54/46.
    // El volumen comprador/vendedor se deriva del delta de precio real entre sub-ticks contiguos.
    let compute_volumes = |p_curr: f64, p_prev: f64, vol: f64| -> (f64, f64) {
        let delta = p_curr - p_prev;
        let norm_delta = if spread_half > 0.0 {
            (delta / (spread_half * 4.0)).clamp(-0.25, 0.25)
        } else {
            0.0
        };
        let b = vol * (0.50 + norm_delta);
        let a = vol * (0.50 - norm_delta);
        (b.max(0.001), a.max(0.001))
    };

    let p_prev0 = if prev_bullish {
        o - spread_half
    } else {
        o + spread_half
    };
    let (b0, a0) = compute_volumes(o, p_prev0, v);
    let (b2, a2) = compute_volumes(p2, o, v);
    let (b3, a3) = compute_volumes(p3, p2, v);
    let (b4, a4) = compute_volumes(c, p3, v);

    [
        TickEvent {
            coin_id,
            timestamp: kline.open_time,
            bid_price: (o - spread_half).max(1e-6),
            ask_price: o + spread_half,
            bid_qty: b0,
            ask_qty: a0,
        },
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step,
            bid_price: (p2 - spread_half).max(1e-6),
            ask_price: p2 + spread_half,
            bid_qty: b2,
            ask_qty: a2,
        },
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step * 2,
            bid_price: (p3 - spread_half).max(1e-6),
            ask_price: p3 + spread_half,
            bid_qty: b3,
            ask_qty: a3,
        },
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step * 3,
            bid_price: (c - spread_half).max(1e-6),
            ask_price: c + spread_half,
            bid_qty: b4,
            ask_qty: a4,
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🌌 TRADER GEMINI V5 - MULTI-COIN QUANTUM SIMULATOR (30 COINS)");
    println!("🛡️ AUDIT FORENSIC ENGINE — Full process_event Pipeline & VIP0 Fees");
    println!("============================================================");

    // 1. Inicialización de especificaciones oficiales de símbolos Binance VIP0 (D-391 paridad estricta 1:1)
    let mut specs = Vec::with_capacity(COINS.len());
    for symbol in COINS.iter() {
        specs.push(quantum_arena::symbol_registry::get_official_binance_spec(
            symbol,
        ));
    }
    quantum_arena::symbol_registry::update_registry(specs);

    // 2. Default de capital $13.0 USD sin pánico y prevención de contaminación de balance
    let initial_capital_str =
        std::env::var("INITIAL_CAPITAL").unwrap_or_else(|_| "13.0".to_string());
    let mut initial_capital: f64 = initial_capital_str.parse().unwrap_or(13.0);
    if initial_capital <= 0.0 || !initial_capital.is_finite() {
        println!("⚠️ INITIAL_CAPITAL no válido detectado. Asignando $13.00 USD nominales.");
        initial_capital = 13.0;
    }
    println!("💰 Initial Capital: ${:.2} USD", initial_capital);

    // 3. Spawn de hilo con 64MB stack resiliente para GlobalArena
    let arena = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || Arc::new(GlobalArena::new(initial_capital)))
        .map_err(|e| format!("Failed to spawn arena thread: {}", e))?
        .join()
        .map_err(|_| "Arena thread panicked during initialization")?;

    // 4. Configuración atómica de comisiones Binance VIP0 en GlobalArena
    arena.config.live_maker_fee.store(0.0002, Ordering::Relaxed);
    arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);
    println!("💳 [FEES] Binance VIP0 Comisiones Atómicas Activadas: Maker 0.02% | Taker 0.05%");

    // 5. Carga y aplicación del Genoma Campeón Activo (SuperGenotype)
    let genome = quantum_arena::genome::SuperGenotype::load_or_default();
    println!(
        "🧬 [GENOME] Active Champion Genome Cargado (Lev {:.1}x, Scalp TP: {:.4}, Scalp SL: {:.4}, Swing TP: {:.4}, Swing SL: {:.4})",
        genome.global_leverage, genome.scalp_tp_base, genome.scalp_sl_base, genome.swing_tp_base, genome.swing_sl_base
    );
    genome.apply_to_arena(&arena);
    arena
        .config
        .global_max_drawdown
        .store(0.90, Ordering::Relaxed);

    let sim_days: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var("SIM_DAYS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(30);

    let requested_rows = (sim_days * 24 * 60) as usize;

    println!(
        "📥 Loading Klines (1m resolution) from Parquet for the last {} days ({} bars)...",
        sim_days, requested_rows
    );

    let mut coin_ticks: Vec<Vec<TickEvent>> = Vec::with_capacity(30);

    for (id, &symbol) in COINS.iter().enumerate() {
        print!("  -> Loading {}... ", symbol);
        let file_path = format!("data/historical/{}_6M.parquet", symbol);

        if !Path::new(&file_path).exists() {
            println!(
                "❌ File not found: {}. Please run download_history first.",
                file_path
            );
            continue;
        }

        let mut file = std::fs::File::open(&file_path)?;
        let df = ParquetReader::new(&mut file).finish()?;

        let mut klines = Vec::with_capacity(df.height());

        let open_times = df.column("open_time")?.u64()?;
        let opens = df.column("open")?.f64()?;
        let highs = df.column("high")?.f64()?;
        let lows = df.column("low")?.f64()?;
        let closes = df.column("close")?.f64()?;
        let volumes = df.column("volume")?.f64()?;
        let close_times = df.column("close_time")?.u64()?;
        let total_rows = df.height();
        let start_idx = if total_rows > requested_rows {
            total_rows - requested_rows
        } else {
            0
        };

        for i in start_idx..total_rows {
            let o = opens.get(i).unwrap_or(0.0);
            let c = closes.get(i).unwrap_or(o);
            if o <= 0.0 || c <= 0.0 || !o.is_finite() || !c.is_finite() {
                continue;
            }
            let h = highs.get(i).unwrap_or(o.max(c));
            let l = lows.get(i).unwrap_or(o.min(c));
            let v = volumes.get(i).unwrap_or(1.0);
            klines.push(Kline {
                open_time: open_times.get(i).unwrap_or(0),
                open: o,
                high: if h.is_finite() { h } else { o.max(c) },
                low: if l.is_finite() { l } else { o.min(c) },
                close: c,
                volume: if v.is_finite() && v >= 0.0 { v } else { 1.0 },
                close_time: close_times.get(i).unwrap_or(0),
            });
        }

        let mut ticks = Vec::with_capacity(klines.len() * 4);
        // R2.1c: dirección causal por pares consecutivos (vela previa -> actual).
        for (i, k) in klines.iter().enumerate() {
            let prev_bullish = if i > 0 {
                klines[i - 1].close >= klines[i - 1].open
            } else {
                k.close >= k.open // primera vela: sin historia previa disponible
            };
            ticks.extend(simple_kline_to_ticks(id, k, prev_bullish));
        }
        println!("{} ticks generated.", ticks.len());
        coin_ticks.push(ticks);
    }

    println!("🔄 Multiplexing and sorting chronologically (Merge Sort O(N log N))...");
    let start_sort = Instant::now();
    let master_stream = multiplex_ticks(coin_ticks);
    println!(
        "✅ Multiplexing done in {:?}. Total Ticks: {}",
        start_sort.elapsed(),
        master_stream.len()
    );

    let backtest_thread = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            println!("🚀 LAUNCHING MULTI-ASSET FORENSIC ENGINE (process_event pipeline)...");
            let start_backtest = Instant::now();

            // Carga de modelos NanoForest para todos los símbolos disponibles (.bin y .json)
            for symbol in COINS.iter() {
                let bin_path = format!("models/{}_MOTOR.bin", symbol);
                let json_path = format!("models/{}_MOTOR.json", symbol);
                let key = format!("{}_MOTOR", symbol);
                if std::path::Path::new(&bin_path).exists() {
                    let _ = god_engine_core::ml_inference::NanoForest::load_global(&key, &bin_path);
                } else if std::path::Path::new(&json_path).exists() {
                    let _ = god_engine_core::ml_inference::NanoForest::load_global(&key, &json_path);
                }
            }

            // Instanciación de GodEngineCore con paridad completa de producción
            let mut engine = GodEngineCore::new(arena.clone());

            // Carga y validación estricta de la Red Neuronal 54D DarkAlphaEngine
            let model_res = dark_alpha_engine::DarkAlphaEngine::load_json("models/DarkAlpha_BTCUSDT.json");
            let nn = match model_res {
                Ok(mut m) => {
                    let is_corrupt = m.layer1.weights.iter().any(|&w| w.is_nan() || w.is_infinite());
                    if is_corrupt {
                        println!("⚠️ [DARK ALPHA] Pesos no finitos detectados en models/DarkAlpha_BTCUSDT.json. Regenerando modelo Xavier 54D.");
                        let mut clean = dark_alpha_engine::DarkAlphaEngine::default_model();
                        clean.freeze(); // T-04: sin estadísticos mutantes
                        let _ = clean.save_json("models/DarkAlpha_BTCUSDT.json");
                        clean
                    } else {
                        m.init_buffers();
                        // T-04: inferencia con normalizadores congelados
                        m.freeze();
                        println!("🧠 [DARK ALPHA] 54D Neural Model cargado exitosamente: models/DarkAlpha_BTCUSDT.json (in_features: {})", m.layer1.in_features);
                        m
                    }
                }
                Err(e) => {
                    println!("⚠️ [DARK ALPHA] models/DarkAlpha_BTCUSDT.json no disponible ({}). Creando modelo 54D inicializado.", e);
                    let clean = dark_alpha_engine::DarkAlphaEngine::default_model();
                    let _ = clean.save_json("models/DarkAlpha_BTCUSDT.json");
                    clean
                }
            };
            engine.swing_nn = Some(nn);

            // Estado macro y omnidireccional 54D base
            let omni_state = Arc::new(OmniState::new());
            let mut prev_kline_ts = [0u64; 30];

            let mut total_trades = 0;
            let mut wins_count = 0;
            let mut losses_count = 0;
            let mut gross_profit = 0.0;
            let mut gross_loss = 0.0;
            let mut total_opens = 0;
            let mut total_closes = 0;
            let mut long_trades = 0;
            let mut short_trades = 0;
            let mut peak_capital = initial_capital;
            let mut max_drawdown_dollars = 0.0;
            let mut max_drawdown_pct = 0.0;

            let total_ticks_len = master_stream.len() as u32;
            let warmup_ticks = 50_000.min(master_stream.len() / 10);
            println!("🔥 [WARM-UP] Alimentando {} ticks multiactivo para calibración de tensores...", warmup_ticks);

            for tick in master_stream.iter().take(warmup_ticks) {
                arena.update_market_data(
                    tick.coin_id,
                    tick.bid_price,
                    tick.ask_price,
                    tick.bid_qty,
                    tick.ask_qty,
                    tick.timestamp,
                );
                let mid_price = (tick.bid_price + tick.ask_price) / 2.0;
                let total_qty = tick.bid_qty + tick.ask_qty;
                let is_buyer_maker = tick.ask_qty > tick.bid_qty;
                if tick.coin_id < engine.feature_engines.len() {
                    engine.feature_engines[tick.coin_id].process_tick(mid_price, total_qty, tick.timestamp);
                    engine.feature_engines[tick.coin_id].update_trade_flow(total_qty, is_buyer_maker);
                    engine.feature_engines[tick.coin_id].update_ofi(tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty);
                }
            }
            arena.unified_capital.store(initial_capital, Ordering::Relaxed);
            println!("✅ [WARM-UP] Calibración completa. Iniciando simulación multiactivo 1:1...");

            let first_ts = master_stream.get(warmup_ticks).map(|t| t.timestamp).unwrap_or(0);
            let last_ts = master_stream.last().map(|t| t.timestamp).unwrap_or(0);

            for tick in master_stream.into_iter().skip(warmup_ticks) {
                // Inyectar tick en Arena
                arena.update_market_data(
                    tick.coin_id,
                    tick.bid_price,
                    tick.ask_price,
                    tick.bid_qty,
                    tick.ask_qty,
                    tick.timestamp,
                );

                let mid_price = (tick.bid_price + tick.ask_price) / 2.0;
                let total_qty = tick.bid_qty + tick.ask_qty;
                let real_obi = if total_qty > 0.0 {
                    (tick.bid_qty - tick.ask_qty) / total_qty
                } else {
                    0.0
                };

                let mut omni = omni_state.get_features();
                // FIX #1520: Verificación de límites de coin_id para acceso seguro a feature_engines
                if tick.coin_id < engine.feature_engines.len() {
                    let swing_feats = engine.feature_engines[tick.coin_id].get_universal_features();
                    for (idx, &f) in swing_feats.iter().enumerate() {
                        if idx < 34 && f.is_finite() {
                            omni[idx] = f as f64;
                        }
                    }
                }
                // FIX #1521: Sanitización y cálculo protegido de features de microestructura
                if total_qty > 0.0 && mid_price > 0.0 && total_qty.is_finite() && mid_price.is_finite() {
                    omni[0] = tick.bid_price;
                    omni[1] = tick.ask_price;
                    omni[30] = (tick.bid_qty - tick.ask_qty).clamp(-1e9, 1e9);
                    omni[31] = ((tick.bid_qty - tick.ask_qty) * 1.2).clamp(-1e9, 1e9);
                    omni[39] = real_obi.clamp(-1.0, 1.0);
                }

                // Detección precisa de cierre de vela de 1 minuto por activo
                let is_kline_closed = prev_kline_ts[tick.coin_id] == 0
                    || (tick.timestamp / 60_000) != (prev_kline_ts[tick.coin_id] / 60_000);
                if is_kline_closed {
                    prev_kline_ts[tick.coin_id] = tick.timestamp;
                }

                let is_buyer_maker = real_obi < 0.0;
                // D-420: Desacoplamiento estricto de eventos @depth y @trade con paridad WebSocket en vivo
                // 1) Actualización de L2 Depth & OFI en el order book
                let (_, closed_1) = engine.process_event(
                    tick.coin_id,
                    false,           // is_trade: false en frame de profundidad
                    is_kline_closed, // is_kline_closed
                    true,            // is_depth: true para actualizar L2, OFI y ATR
                    mid_price,
                    0.0,
                    tick.bid_price,
                    tick.ask_price,
                    tick.bid_qty,
                    tick.ask_qty,
                    real_obi,
                    0.0,
                    tick.timestamp,
                    false,
                    &omni,
                    is_buyer_maker,
                );

                // 2) Procesamiento de transacción agresiva y evaluación de estrategias
                let (new_order, closed_2) = engine.process_event(
                    tick.coin_id,
                    true,  // is_trade: true para evaluar estrategias y ejecución
                    false, // is_kline_closed: ya consumido
                    false, // is_depth: false
                    mid_price,
                    total_qty,
                    tick.bid_price,
                    tick.ask_price,
                    tick.bid_qty,
                    tick.ask_qty,
                    real_obi,
                    0.0,
                    tick.timestamp,
                    false, // latency_panic
                    &omni,
                    is_buyer_maker,
                );

                let closed_order = closed_1.or(closed_2);

                if new_order.is_some() {
                    total_opens += 1;
                }

                // Registro de cierres continuos (net_pnl ya deduce atómicamente entry fee y exit fee VIP0)
                if let Some((is_long, net_pnl, _qty)) = closed_order {
                    total_trades += 1;
                    total_closes += 1;
                    if is_long { long_trades += 1; } else { short_trades += 1; }
                    if net_pnl > 0.0 {
                        wins_count += 1;
                        gross_profit += net_pnl;
                    } else {
                        losses_count += 1;
                        gross_loss += net_pnl.abs();
                    }
                }

                let current_cap = arena.unified_capital.load(Ordering::Relaxed);
                if current_cap > peak_capital {
                    peak_capital = current_cap;
                }
                let dd_dollars = peak_capital - current_cap;
                let dd_pct = if peak_capital > 0.0 { (dd_dollars / peak_capital) * 100.0 } else { 0.0 };
                if dd_dollars > max_drawdown_dollars {
                    max_drawdown_dollars = dd_dollars;
                }
                if dd_pct > max_drawdown_pct {
                    max_drawdown_pct = dd_pct;
                }

                // Disparo de PhaseRunner cada 1,000,000 de ticks para simulación de auditoría
                if total_ticks_len > 0 && arena.tick_counter.load(Ordering::Relaxed) % 1_000_000 == 0 {
                    let _result = PhaseExecutor::run(Phase::Zeta, std::time::Duration::from_millis(10));
                }
            }

            let backtest_duration = start_backtest.elapsed();

            let mut total_pnl_realized = 0.0;
            for c in arena.coins.iter() {
                // D-739: fuente única del PnL realizado.
                let realized = c.metrics.pnl_realized.load(Ordering::Relaxed);
                total_pnl_realized += realized;
            }

            let final_capital = arena.unified_capital.load(Ordering::Relaxed);
            let net_growth_pct = ((final_capital - initial_capital) / initial_capital) * 100.0;
            let win_rate = if total_trades > 0 { (wins_count as f64 / total_trades as f64) * 100.0 } else { 0.0 };
            let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else if gross_profit > 0.0 { 999.0 } else { 0.0 };

            let start_dt = DateTime::<Utc>::from_timestamp((first_ts / 1000) as i64, 0).unwrap_or_default();
            let end_dt = DateTime::<Utc>::from_timestamp((last_ts / 1000) as i64, 0).unwrap_or_default();
            let days_sim = (last_ts.saturating_sub(first_ts)) as f64 / (1000.0 * 60.0 * 60.0 * 24.0);

            println!("============================================================");
            println!("🏁 FORENSIC BACKTEST COMPLETE (MULTI-ASSET 30 COINS)");
            println!("⏱️ Execution Time       : {:?}", backtest_duration);
            println!("⚡ Latency per Tick      : {:?}", backtest_duration / total_ticks_len.max(1));
            println!("🗓️ Period               : {} to {} ({:.2} days)", start_dt.format("%Y-%m-%d %H:%M:%S"), end_dt.format("%Y-%m-%d %H:%M:%S"), days_sim);
            println!("📊 Total Closed Trades  : {}", total_trades);
            println!("🎯 Global Win Rate      : {:.2}% ({} Wins / {} Losses)", win_rate, wins_count, losses_count);
            println!("⚖️ Profit Factor        : {:.3} (Gross Profit: ${:.2} / Gross Loss: ${:.2})", profit_factor, gross_profit, gross_loss);
            println!("🌊 Max Drawdown         : {:.2}% (${:.2})", max_drawdown_pct, max_drawdown_dollars);
            println!("⚡ Continuous Engine    : {} Opens | {} Closes", total_opens, total_closes);
            println!("📈 Directional Breakdown: {} Longs | {} Shorts", long_trades, short_trades);
            println!("💰 Initial Capital      : ${:.4} USD", initial_capital);
            println!("💵 Final Capital        : ${:.4} USD", final_capital);
            println!("📉 Net Realized PnL     : ${:.4} ({:.2}% ROI con comisiones Binance VIP0)", total_pnl_realized, net_growth_pct);
            println!("============================================================");
        })
        .unwrap();

    backtest_thread.join().unwrap();

    Ok(())
}
