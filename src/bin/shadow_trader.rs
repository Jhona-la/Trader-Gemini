use std::sync::Arc;
use dotenvy::dotenv;

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use data_pipeline::ws_client::BinanceStreamer;
use data_pipeline::omni_multiplexer::OmniDataHub;
// use os_guardian::{OsGuardian, ResourceLimits};
use dark_alpha_engine::DarkAlphaEngine;

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

    let _ws_client = BinanceStreamer::new(0, "btcusdt", arena.clone());
    println!("🔌 Conectando a Binance WebSockets y esperando Ticks de Mercado en Tiempo Real...");

    println!("⚡ Motor Cuántico Listo para Producción.");

    // Mantenemos vivo el hilo principal
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
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

        println!("⏱️ Latido... Capital: ${:.2} | Bybit: {:.2} | DXY: {:.2} | S&P500: {:.2} | Fund: {:.4}%", arena.unified_capital.load(std::sync::atomic::Ordering::Relaxed), f64::from_bits(omni_hub.omni_state.bybit_linear.load(std::sync::atomic::Ordering::Relaxed)), f64::from_bits(omni_hub.omni_state.dxy.load(std::sync::atomic::Ordering::Relaxed)), f64::from_bits(omni_hub.omni_state.sp500.load(std::sync::atomic::Ordering::Relaxed)), f64::from_bits(omni_hub.omni_state.agg_funding_rate.load(std::sync::atomic::Ordering::Relaxed)) * 100.0);
    }
}





