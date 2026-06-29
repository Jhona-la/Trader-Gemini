use data_pipeline::BinanceStreamer;
use execution_engine::executor::OrderExecutor;
use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{ScalpEngine, SwingEngine, SignalType};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::env;

#[tokio::main]
async fn main() {
    // 0. Cargar variables de entorno
    dotenvy::dotenv().ok();

    println!("🚀 TRADER GEMINI V5 - INITIALIZING GENESIS SEQUENCE...");

    let initial_capital: f64 = env::var("INITIAL_CAPITAL")
        .unwrap_or_else(|_| "13.0".to_string())
        .parse()
        .expect("INITIAL_CAPITAL debe ser un número");

    // 1. Instanciar la Arena Cuántica (Memoria Compartida O(1))
    let mut arena = GlobalArena::new(initial_capital);
    
    // Configurar límites agresivos pero seguros
    arena.config.global_leverage.store(10.0, Ordering::Relaxed);
    arena.config.global_max_drawdown.store(0.15, Ordering::Relaxed); // 15% Max Drawdown
    
    // Convertir la Arena en un puntero atómico para poder leerlo desde los motores
    let arena_ptr = Arc::new(arena);

    // 2. Encender los Motores de Señal
    let mut scalp_engine = ScalpEngine::new();
    let mut swing_engine = SwingEngine::new();

    // 3. Encender el Escudo
    let mut risk_engine = RiskEngine::new(initial_capital);

    // 4. Encender el Gatillo (Las Manos)
    let api_secret = env::var("BINANCE_API_SECRET")
        .unwrap_or_else(|_| "DUMMY_SECRET_KEY_FOR_NOW_REPLACE_IN_PROD".to_string());
    let executor = OrderExecutor::new(api_secret);

    // 5. Configurar el par de trading
    let symbol = env::var("TRADING_SYMBOL")
        .unwrap_or_else(|_| "bnbusdt".to_string());

    println!("⚡ SISTEMAS EN LÍNEA. CONECTANDO AL NERVIO ÓPTICO (HFT)...");

    // 6. Iniciar el Data Pipeline (WebSockets)
    let streamer = BinanceStreamer::new(symbol);
    
    // Clones para el closure
    let arena_ref = Arc::clone(&arena_ptr);
    let symbol_upper = symbol.to_uppercase();

    // El closure se ejecutará sincrónicamente en cada tick que reciba Tokio.
    // Esto es el Hot Path. No hay Mutexes. No hay esperas.
    streamer.start(move |tick| {
        // --- 1. INYECTAR DATOS EN LA ARENA ---
        // Aquí actualizaríamos características del precio en la Arena. 
        // Por simplicidad, asumimos que el precio de ejecución es el ASK para compras y BID para ventas.
        let mid_price = (tick.bid_price + tick.ask_price) / 2.0;

        // --- 2. EVALUAR CEREBROS ---
        let scalp_intent = scalp_engine.evaluate(&arena_ref);
        let _swing_intent = swing_engine.evaluate(&arena_ref);

        // --- 3. EVALUAR RIESGO (Prioridad Scalp por ahora) ---
        if scalp_intent.signal != SignalType::Flat {
            let validated_order = risk_engine.evaluate(scalp_intent, &arena_ref, true);
            
            // --- 4. EJECUTAR ---
            if validated_order.signal != SignalType::Flat {
                let step_size = 0.001; // BNB step size típico
                if let Some(payload) = executor.build_payload(&validated_order, &symbol_upper, mid_price, step_size) {
                    println!("[HFT FIRE] 💥 ORDEN GENERADA: {:?}", payload);
                    // Aquí se enviaría el HTTP/TCP payload real
                }
            }
        }
        
    }).await;
}
