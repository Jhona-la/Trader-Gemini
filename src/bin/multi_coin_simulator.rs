use quantum_arena::{GlobalArena, TickEvent};
use god_engine_core::GodEngineCore;
use phase_runner::{Phase, PhaseExecutor};
use data_pipeline::historical::Kline;
use data_pipeline::multiplexer::{kline_to_ticks, multiplex_ticks};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use polars::prelude::*;
use std::path::Path;

const COINS: [&str; 30] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "TRXUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "ATOMUSDT", "UNIUSDT", "XMRUSDT", "ETCUSDT", "FILUSDT", "ICPUSDT",
    "VETUSDT", "NEARUSDT", "AAVEUSDT", "ALGOUSDT", "EGLDUSDT", "SANDUSDT", "THETAUSDT", "AXSUSDT", "MANAUSDT", "FTMUSDT"
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🌌 TRADER GEMINI V5 - MULTI-COIN QUANTUM SIMULATOR (30 COINS)");
    println!("============================================================");

    let initial_capital = 13.0; // The holy $13
    println!("💰 Initial Capital: ${:.2}", initial_capital);

    let arena = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024) // 64 MB stack to prevent 30MB GlobalArena stack overflow
        .spawn(move || {
            Arc::new(GlobalArena::new(initial_capital))
        })
        .unwrap()
        .join()
        .unwrap();

    let three_days_rows = 3 * 24 * 60; // 4320 minutos (3 días)

    println!("📥 Loading Klines (1m resolution) from Parquet for the last 3 days...");

    let mut coin_ticks: Vec<Vec<TickEvent>> = Vec::with_capacity(30);

    for (id, &symbol) in COINS.iter().enumerate() {
        print!("  -> Loading {}... ", symbol);
        let file_path = format!("data/historical/{}_6M.parquet", symbol);
        
        if !Path::new(&file_path).exists() {
            println!("❌ File not found: {}. Please run download_history first.", file_path);
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
        let start_idx = if total_rows > three_days_rows { total_rows - three_days_rows } else { 0 };

        for i in start_idx..total_rows {
            klines.push(Kline {
                open_time: open_times.get(i).unwrap(),
                open: opens.get(i).unwrap(),
                high: highs.get(i).unwrap(),
                low: lows.get(i).unwrap(),
                close: closes.get(i).unwrap(),
                volume: volumes.get(i).unwrap(),
                close_time: close_times.get(i).unwrap(),
            });
        }
        
        let mut ticks = Vec::with_capacity(klines.len() * 4);
        for k in klines.iter() {
            ticks.extend(kline_to_ticks(id, k));
        }
        println!("{} ticks generated.", ticks.len());
        coin_ticks.push(ticks);
    }

    println!("🔄 Multiplexing and sorting chronologically (Merge Sort O(N log N))...");
    let start_sort = Instant::now();
    let master_stream = multiplex_ticks(coin_ticks);
    println!("✅ Multiplexing done in {:?}. Total Ticks: {}", start_sort.elapsed(), master_stream.len());

    let backtest_thread = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            println!("🚀 LAUNCHING HFT BACKTEST ENGINE...");
            let start_backtest = Instant::now();
            
            let mut engine = GodEngineCore::new(arena.clone());
            let mut total_trades = 0;
            let total_ticks_len = master_stream.len() as u32;
            
            for tick in master_stream {
                // Inject tick to arena directly
                arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty);
                
                let (_new_sc, _new_sw, closed_sc, closed_sw, _maker) = engine.process_tick(
                    tick.coin_id,
                    tick.bid_price, tick.ask_price,
                    tick.bid_qty, tick.ask_qty,
                    tick.timestamp,
                    &[0.0; 54]
                );

                if closed_sc.is_some() || closed_sw.is_some() {
                    total_trades += 1;
                }
                
                // Disparo de PhaseRunner cada 1,000,000 de ticks para simulacion de auditoria
                if total_ticks_len > 0 && arena.tick_counter.load(Ordering::Relaxed) % 1_000_000 == 0 {
                    let result = PhaseExecutor::run(Phase::Zeta, std::time::Duration::from_millis(10));
                    // println!("🔄 [PHASE RUNNER] Executed phase {:?}", result.phase);
                }
            }
            
            let backtest_duration = start_backtest.elapsed();

            // Sumary
            let mut total_pnl_realized = 0.0;
            for c in arena.coins.iter() {
                total_pnl_realized += c.scalp.pnl_realized.load(Ordering::Relaxed);
                total_pnl_realized += c.swing.pnl_realized.load(Ordering::Relaxed);
            }

            let final_capital = arena.unified_capital.load(Ordering::Relaxed);
            let growth_pct = ((final_capital - initial_capital) / initial_capital) * 100.0;
            
            println!("============================================================");
            println!("🏁 BACKTEST COMPLETE");
            println!("⏱️ Execution Time: {:?}", backtest_duration);
            println!("⚡ Latency per Tick: {:?}", backtest_duration / total_ticks_len.max(1));
            println!("📊 Total Trades: {}", total_trades);
            println!("💰 Initial Capital: ${:.2}", initial_capital);
            println!("💵 Final Capital:   ${:.2}", final_capital);
            println!("📈 Net PnL:         ${:.2} ({:.2}%)", total_pnl_realized, growth_pct);
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
