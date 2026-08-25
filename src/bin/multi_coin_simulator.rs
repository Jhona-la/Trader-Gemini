use chrono::{DateTime, Utc};
use data_pipeline::historical::Kline;
use data_pipeline::multiplexer::multiplex_ticks;
use god_engine_core::GodEngineCore;
use phase_runner::{Phase, PhaseExecutor};
use polars::prelude::{ParquetReader, SerReader};
use quantum_arena::{GlobalArena, TickEvent};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

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

// Convertir 1 Kline en 4 ticks determinísticos con spread institucional realista (1.5 bps) y desequilibrio de volumen direccional
fn simple_kline_to_ticks(coin_id: usize, kline: &Kline) -> [TickEvent; 4] {
    let step = kline.close_time.saturating_sub(kline.open_time) / 4;
    // FIX #1480: Sanitización de precios y volúmenes en generación de ticks
    let o = if kline.open.is_finite() && kline.open > 0.0 { kline.open } else { 1.0 };
    let h = if kline.high.is_finite() && kline.high > 0.0 { kline.high.max(o) } else { o };
    let l = if kline.low.is_finite() && kline.low > 0.0 { kline.low.min(o).max(1e-6) } else { o * 0.999 };
    let c = if kline.close.is_finite() && kline.close > 0.0 { kline.close } else { o };
    let v = if kline.volume.is_finite() && kline.volume > 0.0 { kline.volume / 4.0 } else { 0.25 };

    let spread_half = (o * 0.000075).max(0.00001); // 0.75 bps = 1.5 bps total spread
    let is_bullish = c >= o;
    let (bid_v, ask_v) = if is_bullish {
        (v * 0.58, v * 0.42)
    } else {
        (v * 0.42, v * 0.58)
    };

    [
        TickEvent {
            coin_id,
            timestamp: kline.open_time,
            bid_price: (o - spread_half).max(1e-6),
            ask_price: o + spread_half,
            bid_qty: bid_v.max(0.001),
            ask_qty: ask_v.max(0.001),
        },
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step,
            bid_price: (h - spread_half).max(1e-6),
            ask_price: h + spread_half,
            bid_qty: (bid_v * 1.1).max(0.001),
            ask_qty: (ask_v * 0.9).max(0.001),
        },
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step * 2,
            bid_price: (l - spread_half).max(1e-6),
            ask_price: l + spread_half,
            bid_qty: (bid_v * 0.9).max(0.001),
            ask_qty: (ask_v * 1.1).max(0.001),
        },
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step * 3,
            bid_price: (c - spread_half).max(1e-6),
            ask_price: c + spread_half,
            bid_qty: bid_v.max(0.001),
            ask_qty: ask_v.max(0.001),
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🌌 TRADER GEMINI V5 - MULTI-COIN QUANTUM SIMULATOR (30 COINS)");
    println!("============================================================");

    let mut specs = Vec::with_capacity(COINS.len());
    for symbol in COINS.iter() {
        specs.push(quantum_arena::symbol_registry::SymbolSpec {
            symbol: symbol.to_string(),
            step_size: 0.00000001, // ultra fine to prevent issues
            tick_size: 0.00001,
            min_qty: 0.0001,
            min_notional: 1.0,
            max_leverage: 20,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        });
    }
    quantum_arena::symbol_registry::update_registry(specs);

    // FIX #732: Default de capital $13.0 USD sin pánico y prevención de contaminación de balance
    let initial_capital_str = std::env::var("INITIAL_CAPITAL").unwrap_or_else(|_| "13.0".to_string());
    let mut initial_capital: f64 = initial_capital_str
        .parse()
        .unwrap_or(13.0);
    if initial_capital <= 0.0 || !initial_capital.is_finite() {
        println!("⚠️ INITIAL_CAPITAL no válido detectado. Asignando $13.00 USD nominales.");
        initial_capital = 13.0;
    }
    println!("💰 Initial Capital: ${:.2}", initial_capital);

    // FIX #1480: Spawn de hilo con 64MB stack resiliente
    let arena = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || Arc::new(GlobalArena::new(initial_capital)))
        .map_err(|e| format!("Failed to spawn arena thread: {}", e))?
        .join()
        .map_err(|_| "Arena thread panicked during initialization")?;

    let three_days_rows = 3 * 24 * 60; // 4320 minutos (3 días)

    println!("📥 Loading Klines (1m resolution) from Parquet for the last 3 days...");

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

        // Polars filter is possible but iterating is simple enough since it's only 6M
        let open_times = df.column("open_time")?.u64()?;
        let opens = df.column("open")?.f64()?;
        let highs = df.column("high")?.f64()?;
        let lows = df.column("low")?.f64()?;
        let closes = df.column("close")?.f64()?;
        let volumes = df.column("volume")?.f64()?;
        let close_times = df.column("close_time")?.u64()?;
        let total_rows = df.height();
        let start_idx = if total_rows > three_days_rows {
            total_rows - three_days_rows
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
        for k in klines.iter() {
            ticks.extend(simple_kline_to_ticks(id, k));
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
            println!("🚀 LAUNCHING HFT BACKTEST ENGINE...");
            let start_backtest = Instant::now();

            // Carga de modelos para todos los símbolos disponibles (.bin y .json)
            for symbol in COINS.iter() {
                let bin_path = format!("models/{}_SCALP.bin", symbol);
                let json_path = format!("models/{}_SCALP.json", symbol);
                let key = format!("{}_SCALP", symbol);
                if std::path::Path::new(&bin_path).exists() {
                    let _ = god_engine_core::ml_inference::NanoForest::load_global(&key, &bin_path);
                } else if std::path::Path::new(&json_path).exists() {
                    let _ = god_engine_core::ml_inference::NanoForest::load_global(&key, &json_path);
                }
            }

            let mut engine = GodEngineCore::new(arena.clone());
            let mut total_trades = 0;
            let total_ticks_len = master_stream.len() as u32;
            let first_ts = master_stream.first().map(|t| t.timestamp).unwrap_or(0);
            let last_ts = master_stream.last().map(|t| t.timestamp).unwrap_or(0);

            for tick in master_stream {
                // Inject tick to arena directly
                arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp);

                let mut omni = [0.0f64; 54];
                // FIX #1520: Verificación de límites de coin_id para acceso seguro a feature_engines
                if tick.coin_id < engine.feature_engines.len() {
                    let swing_feats = engine.feature_engines[tick.coin_id].get_swing_features();
                    for (idx, &f) in swing_feats.iter().enumerate() {
                        if idx < 54 && f.is_finite() {
                            omni[idx] = f as f64;
                        }
                    }
                }
                let total_qty = tick.bid_qty + tick.ask_qty;
                let mid = (tick.bid_price + tick.ask_price) / 2.0;
                // FIX #1521: Sanitización y cálculo protegido de features de microestructura
                if total_qty > 0.0 && mid > 0.0 && total_qty.is_finite() && mid.is_finite() {
                    // FIX #1413: No sobreescribir ranuras [0..34] de features ML (price_change, hurst, etc.)
                    omni[35] = ((tick.bid_price - mid) / mid.max(1e-5)).clamp(-1.0, 1.0);
                    omni[36] = ((tick.ask_price - mid) / mid.max(1e-5)).clamp(-1.0, 1.0);
                    omni[37] = (tick.bid_qty - tick.ask_qty).clamp(-1e9, 1e9);
                    omni[38] = ((tick.bid_qty - tick.ask_qty) * 1.2).clamp(-1e9, 1e9);
                    omni[39] = ((tick.bid_qty - tick.ask_qty) / total_qty).clamp(-1.0, 1.0);
                }

                let (_new_sc, _new_sw, closed_sc, closed_sw, _maker) = engine.process_tick(
                    tick.coin_id,
                    tick.bid_price, tick.ask_price,
                    tick.bid_qty, tick.ask_qty,
                    tick.timestamp,
                    &omni
                );

                if closed_sc.is_some() || closed_sw.is_some() {
                    total_trades += 1;
                }

                // Disparo de PhaseRunner cada 1,000,000 de ticks para simulacion de auditoria
                if total_ticks_len > 0 && arena.tick_counter.load(Ordering::Relaxed) % 1_000_000 == 0 {
                    let _result = PhaseExecutor::run(Phase::Zeta, std::time::Duration::from_millis(10));
                }
            }

            let backtest_duration = start_backtest.elapsed();

            // Sumary
            let mut total_pnl_realized = 0.0;
            for c in arena.coins.iter() {
                let realized = c.scalp.pnl_realized.load(Ordering::Relaxed) + c.swing.pnl_realized.load(Ordering::Relaxed);
                total_pnl_realized += realized;
                // Para calcular el Gross, necesitaríamos agregar los fees cobrados, pero simplificaremos asumiendo:
                // Gross = Net + Fees, aunque PnL Realized internamente ya descontó los fees.
                // Estimación rápida de Fees: 0.04% por trade
            }

            let final_capital = arena.unified_capital.load(Ordering::Relaxed);
            let net_growth_pct = ((final_capital - initial_capital) / initial_capital) * 100.0;

            // Asumiendo fees promedio del 0.05% (taker) por cada trade abierto y cerrado (0.10% total)
            let estimated_total_fees = total_trades as f64 * (initial_capital / 15.0) * 0.001;
            let total_gross_pnl = total_pnl_realized + estimated_total_fees;
            let gross_growth_pct = (total_gross_pnl / initial_capital) * 100.0;

            let start_dt = DateTime::<Utc>::from_timestamp((first_ts / 1000) as i64, 0).unwrap_or_default();
            let end_dt = DateTime::<Utc>::from_timestamp((last_ts / 1000) as i64, 0).unwrap_or_default();
            let days_sim = (last_ts.saturating_sub(first_ts)) as f64 / (1000.0 * 60.0 * 60.0 * 24.0);

            println!("============================================================");
            println!("🏁 QUANTUM BACKTEST COMPLETE");
            println!("⏱️ Execution Time  : {:?}", backtest_duration);
            println!("⚡ Latency per Tick: {:?}", backtest_duration / total_ticks_len.max(1));
            println!("🗓️ Period          : {} to {} ({:.2} days)", start_dt.format("%Y-%m-%d %H:%M:%S"), end_dt.format("%Y-%m-%d %H:%M:%S"), days_sim);
            println!("📊 Total Trades    : {}", total_trades);
            println!("💰 Initial Capital : ${:.2}", initial_capital);
            println!("💵 Final Capital   : ${:.2}", final_capital);
            println!("📈 Gross PnL       : ${:.2} ({:.2}% ROI sin fees)", total_gross_pnl, gross_growth_pct);

            // SISTEMA SUPREMO: Proyección Exponencial Matemática
            let base_growth = (1.0f64 + net_growth_pct / 100.0f64).max(0.0f64);
            let expected_3day_multiplier = if base_growth > 0.0 {
                base_growth.powf(3.0 / days_sim.max(0.1))
            } else {
                0.0
            };
            println!("🚀 3-Day Compounding Velocity: {:.2}x (Meta: 2.00x)", expected_3day_multiplier);
            if expected_3day_multiplier >= 2.0 {
                println!("✅ [SUPREME STATUS] Exponential Velocity target achieved (100% every 3 days)!");
            } else {
                println!("⚠️ [SUPREME STATUS] Compounding velocity is below the 2.0x 3-day target. Optimization required.");
            }
            println!("📉 Net PnL         : ${:.2} ({:.2}% ROI con fees reales)", total_pnl_realized, net_growth_pct);
            println!("============================================================");

            if final_capital >= initial_capital * 2.0 {
                println!("🏆 100% GROWTH IN 3 DAYS ACHIEVED! EXPONENTIAL TARGET MET!");
            } else {
                println!("⚠️ Target not met. Need optimization to achieve 100% growth.");
            }
        })
        .unwrap();

    backtest_thread.join().unwrap();

    Ok(())
}
