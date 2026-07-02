use std::sync::Arc;
use dotenvy::dotenv;

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use data_pipeline::ws_client::BinanceStreamer;
use data_pipeline::omni_multiplexer::OmniDataHub;
// use os_guardian::{OsGuardian, ResourceLimits};
use dark_alpha_engine::DarkAlphaEngine;
use execution_engine::executor::OrderExecutor;
use execution_engine::executor::ExecutionProvider;
use tokio::sync::mpsc;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Inicializando Live Trader (Trader Gemini v5) 🚀");
    dotenv().ok();

    /* let limits = ResourceLimits {
        max_memory_mb: 2048,
        max_cpu_percent: 85,
        max_disk_io_mb_s: 500,
        max_threads: 16,
    };
    let _guardian = OsGuardian::new(limits);
    println!("🛡️ OS Guardian inicializado (Límites Estrictos de Memoria y CPU)"); */

    let arena = Arc::new(GlobalArena::new(13.0));
    println!("⚛️ Arena Cuántica inicializada para 30 activos");

    let weights_path = "models/DarkAlpha_BTCUSDT.json";
    let ml_engine = if std::path::Path::new(weights_path).exists() {
        println!("🧠 Cargando pesos neuronales nativos desde {}...", weights_path);
        let engine = DarkAlphaEngine::load_json(weights_path).ok();
        engine
    } else {
        println!("⚠️ No se encontró {}. El motor funcionará con 50% de probabilidad (Random Walk).", weights_path);
        None
    };

    let omni_hub = OmniDataHub::new();
omni_hub.start_feeds();
println!("🌐 Omni-Data Pipeline inicializado con 10 fuentes (TradFi, On-Chain, Cross-Exchange)");

    let mut god_engine = GodEngineCore::new(arena.clone());
    if let Some(engine) = ml_engine {
        god_engine.swing_nn = Some(engine);
    }

    let is_testnet = std::env::var("USE_TESTNET").unwrap_or("true".to_string()) == "true";
    if is_testnet {
        println!("🌐 Ejecutor de Binance inicializado en MODO TESTNET");
    } else {
        println!("⚠️ 🌐 EJECUTOR DE BINANCE INICIALIZADO EN MAINNET (PRODUCCIÓN REAL) ⚠️");
    }

    const COINS: [&str; 30] = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
        "TRXUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "ATOMUSDT", "UNIUSDT", "XMRUSDT", "ETCUSDT", "FILUSDT", "ICPUSDT",
        "VETUSDT", "NEARUSDT", "AAVEUSDT", "ALGOUSDT", "EGLDUSDT", "SANDUSDT", "THETAUSDT", "AXSUSDT", "MANAUSDT", "FTMUSDT"
    ];

    println!("🔌 Conectando a Binance WebSockets para 30 activos y esperando Ticks de Mercado en Tiempo Real...");

    let api_key = if is_testnet { std::env::var("BINANCE_TESTNET_API_KEY").unwrap_or_default() } else { std::env::var("BINANCE_API_KEY").unwrap_or_default() };
    let api_secret = if is_testnet { std::env::var("BINANCE_TESTNET_SECRET_KEY").unwrap_or_default() } else { std::env::var("BINANCE_SECRET_KEY").unwrap_or_default() };
    let executor = OrderExecutor::new(api_key, api_secret);
    println!("⚔️ OrderExecutor inicializado");

    println!("⚡ Motor Cuántico Listo para Producción.");

    let (tx_tick, mut rx_tick) = mpsc::unbounded_channel();

    // Lanzar un streamer por cada moneda
    for (id, &symbol) in COINS.iter().enumerate() {
        let ws_client = BinanceStreamer::new(id, symbol, arena.clone());
        let tx_tick_clone = tx_tick.clone();
        tokio::spawn(async move {
            ws_client.start(move |event| {
                let _ = tx_tick_clone.send(event);
            }).await;
        });
    }

    let step_size = 0.00001; // Ajustar según activo, para BTC aprox 0.00001

    let mut last_heartbeat = tokio::time::Instant::now();
    let heartbeat_interval = tokio::time::Duration::from_secs(10);

    loop {
        tokio::select! {
            Some(tick) = rx_tick.recv() => {
                // Obtenemos vector macro
                let omni_features = omni_hub.omni_state.get_features();

                // Procesamos el tick en el motor
                let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
                let (new_sc, new_sw, _closed_sc, _closed_sw, _maker) = god_engine.process_tick(
                    tick.coin_id,
                    tick.bid_price,
                    tick.ask_price,
                    tick.bid_qty,
                    tick.ask_qty,
                    ts,
                    &omni_features
                );

                // Ejecutamos posiciones SCALP (si las hay)
                if let Some((is_long, _price, qty)) = new_sc {
                    let current_price = (tick.bid_price + tick.ask_price) / 2.0;
                    let symbol = COINS[tick.coin_id];
                    println!("🚀 [SCALP SIGNAL] {} is_long: {}, qty: {:.4} a precio {}", symbol, is_long, qty, current_price);
                    let ex = &executor;
                    let _ = ex.execute_raw_qty(symbol, is_long, qty, step_size).await;
                }

                // Ejecutamos posiciones SWING (si las hay)
                if let Some((is_long, _price, qty)) = new_sw {
                    let current_price = (tick.bid_price + tick.ask_price) / 2.0;
                    let symbol = COINS[tick.coin_id];
                    println!("🦅 [SWING SIGNAL] {} is_long: {}, qty: {:.4} a precio {}", symbol, is_long, qty, current_price);
                    let ex = &executor;
                    let _ = ex.execute_raw_qty(symbol, is_long, qty, step_size).await;
                }

            }
            
            _ = tokio::time::sleep_until(last_heartbeat + heartbeat_interval) => {
                last_heartbeat = tokio::time::Instant::now();
                let ndx = f64::from_bits(omni_hub.omni_state.nasdaq.load(std::sync::atomic::Ordering::Relaxed));
                let dxy = f64::from_bits(omni_hub.omni_state.dxy.load(std::sync::atomic::Ordering::Relaxed));
                
                if ndx < 17000.0 || dxy > 108.0 {
                    if !arena.kill_switch_active.load(std::sync::atomic::Ordering::Relaxed) {
                        println!("🚨 [KILL SWITCH ACTIVADO] Colapso TradFi detectado (NASDAQ={:.2}, DXY={:.2}). Desactivando compras.", ndx, dxy);
                        arena.kill_switch_active.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                } else if arena.kill_switch_active.load(std::sync::atomic::Ordering::Relaxed) {
                    println!("✅ [KILL SWITCH DESACTIVADO] TradFi recuperado. Reanudando compras.");
                    arena.kill_switch_active.store(false, std::sync::atomic::Ordering::Relaxed);
                }

                let ticks = arena.tick_counter.load(std::sync::atomic::Ordering::Relaxed);
                println!("⏱️ Latido... Capital: ${:.2} | Bybit: {:.2} | DXY: {:.2} | S&P500: {:.2} | Fund: {:.4}% | Ticks: {}", 
                    arena.unified_capital.load(std::sync::atomic::Ordering::Relaxed), 
                    f64::from_bits(omni_hub.omni_state.bybit_linear.load(std::sync::atomic::Ordering::Relaxed)), 
                    f64::from_bits(omni_hub.omni_state.dxy.load(std::sync::atomic::Ordering::Relaxed)), 
                    f64::from_bits(omni_hub.omni_state.sp500.load(std::sync::atomic::Ordering::Relaxed)), 
                    f64::from_bits(omni_hub.omni_state.agg_funding_rate.load(std::sync::atomic::Ordering::Relaxed)) * 100.0,
                    ticks
                );
            }
        }
    }
}





