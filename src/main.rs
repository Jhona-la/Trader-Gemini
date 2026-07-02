use data_pipeline::ws_client::BinanceStreamer;
use execution_engine::executor::{ExecutionProvider, OrderExecutor};
use evolution_engine::EvolutionEngine;
use telemetry_server::start_telemetry_server;
use quantum_arena::GlobalArena;
use god_engine_core::GodEngineCore;
use std::sync::Arc;
use std::env;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();
    println!("🚀 TRADER GEMINI V5 - INITIALIZING GENESIS SEQUENCE (PHASE 20)...");

    let initial_capital: f64 = env::var("INITIAL_CAPITAL").unwrap_or_else(|_| "13.0".to_string()).parse().expect("INITIAL_CAPITAL debe ser un número");
    let arena = Arc::new(GlobalArena::new(initial_capital));
    
    arena.config.global_leverage.store(10.0, std::sync::atomic::Ordering::Relaxed);
    arena.config.global_max_drawdown.store(0.15, std::sync::atomic::Ordering::Relaxed);
    
    let evolution_engine = EvolutionEngine::new(arena.clone());
    tokio::spawn(async move { evolution_engine.start_evolution_loop().await; });

    let telemetry_arena = arena.clone();
    tokio::spawn(async move { start_telemetry_server(telemetry_arena).await; });

    let api_key = env::var("BINANCE_API_KEY").unwrap_or_else(|_| "DUMMY_KEY".to_string());
    let api_secret = env::var("BINANCE_API_SECRET").unwrap_or_else(|_| "DUMMY_SECRET".to_string());
    let executor = Arc::new(OrderExecutor::new(api_key, api_secret));
    
    let mut god_engine = GodEngineCore::new(arena.clone());
    let symbol = env::var("TRADING_SYMBOL").unwrap_or_else(|_| "btcusdt".to_string());
    
    let (tx, mut rx) = mpsc::unbounded_channel();
    let arena_stream = arena.clone();
    let coin_id = 0; // Hardcoded for single coin for now
    let streamer = BinanceStreamer::new(coin_id, &symbol, arena_stream);
    
    tokio::spawn(async move {
        streamer.start(move |event| {
            let _ = tx.send(event);
        }).await;
    });

    println!("⚡ SISTEMAS EN LÍNEA. CONECTANDO AL NERVIO ÓPTICO (HFT)...");

    let step_size = 0.001;
    
    while let Some(event) = rx.recv().await {
        // Enlazar Kill Switch
        if arena.kill_switch_active.load(std::sync::atomic::Ordering::Relaxed) {
            println!("🚨 [KILL SWITCH] Triggered in Main loop! Executing emergency shutdown...");
            executor.trigger_kill_switch();
            break;
        }

        let (new_scalp, new_swing, closed_scalp, closed_swing, _maker_quote) = god_engine.process_tick(
            coin_id,
            event.bid_price, 
            event.ask_price, 
            event.bid_qty, 
            event.ask_qty
        );
        
        let s = symbol.to_uppercase();
        
        if let Some((is_long, _price, qty)) = new_scalp {
            println!("⚡ [SCALP] Señal generada -> LONG: {}, QTY: {}", is_long, qty);
            let exec = Arc::clone(&executor);
            let sym = s.clone();
            tokio::spawn(async move {
                let _ = exec.execute_raw_qty(&sym, is_long, qty, step_size).await;
            });
        }
        if let Some((is_long, _price, qty)) = new_swing {
            println!("🌊 [SWING] Señal generada -> LONG: {}, QTY: {}", is_long, qty);
            let exec = Arc::clone(&executor);
            let sym = s.clone();
            tokio::spawn(async move {
                let _ = exec.execute_raw_qty(&sym, is_long, qty, step_size).await;
            });
        }
        if let Some((is_long, _price, qty)) = closed_scalp {
            println!("❌ [SCALP CLOSE] Cerrando posición");
            let exec = Arc::clone(&executor);
            let sym = s.clone();
            tokio::spawn(async move {
                let _ = exec.execute_raw_qty(&sym, !is_long, qty, step_size).await;
            });
        }
        if let Some((is_long, _price, qty)) = closed_swing {
            println!("❌ [SWING CLOSE] Cerrando posición");
            let exec = Arc::clone(&executor);
            let sym = s.clone();
            tokio::spawn(async move {
                let _ = exec.execute_raw_qty(&sym, !is_long, qty, step_size).await;
            });
        }
    }
}
