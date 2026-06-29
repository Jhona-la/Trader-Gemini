use data_pipeline::BinanceStreamer;
use execution_engine::executor::OrderExecutor;
use evolution_engine::EvolutionEngine;
use telemetry_server::start_telemetry_server;
use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{ScalpEngine, SwingEngine, SignalType};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::env;

#[tokio::main]
async fn main() {
    // 0. Cargar variables de entorno (Ignoramos si falla, en prod podemos pasar env vars directo)
    let _ = dotenvy::dotenv();

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

    // 1.5. Instanciar el Cerebro Analítico (Evolución)
    let evolution_engine = EvolutionEngine::new(arena_ptr.clone());
    
    // Desplegar la evolución en un hilo asíncrono background
    tokio::spawn(async move {
        evolution_engine.start_evolution_loop().await;
    });

    // 1.8. Desplegar el Servidor Táctico (Telemetría)
    let telemetry_arena = arena_ptr.clone();
    tokio::spawn(async move {
        start_telemetry_server(telemetry_arena).await;
    });

    // 2. Encender los Motores de Señal
    let mut scalp_engine = ScalpEngine::new();
    let mut swing_engine = SwingEngine::default(); // Usa 12/26 MACD default

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

    let http_client = reqwest::Client::new();
    let binance_url = "https://fapi.binance.com/fapi/v1/order".to_string();

    // 6. Iniciar el Data Pipeline (WebSockets)
    let streamer = BinanceStreamer::new(&symbol);
    
    // Clones para el closure
    let arena_ref = Arc::clone(&arena_ptr);
    let symbol_upper = symbol.to_uppercase();
    let client_for_closure = http_client.clone();
    let url_for_closure = binance_url.clone();
    let secret_for_closure = api_secret.clone();

    // El closure se ejecutará sincrónicamente en cada tick que reciba Tokio.
    // Esto es el Hot Path. No hay Mutexes. No hay esperas.
    streamer.start(move |tick| {
        // --- 0. HEARTBEAT ---
        arena_ref.increment_tick();

        // --- 1. INYECTAR DATOS EN LA ARENA ---
        let mid_price = (tick.bid_price + tick.ask_price) / 2.0;

        // --- 2. GESTIÓN DE POSICIONES ACTIVAS (TP/SL) ---
        let scalp_tp = arena_ref.config.scalp_tp_base.load(Ordering::Relaxed);
        let scalp_sl = arena_ref.config.scalp_sl_base.load(Ordering::Relaxed);

        if arena_ref.positions.scalp_position.is_open() {
            let is_long = arena_ref.positions.scalp_position.is_long.load(Ordering::Relaxed);
            let entry = arena_ref.positions.scalp_position.entry_price.load(Ordering::Relaxed);
            let qty = arena_ref.positions.scalp_position.quantity.load(Ordering::Relaxed);
            
            let pnl_pct = if is_long {
                (mid_price - entry) / entry
            } else {
                (entry - mid_price) / entry
            };

            // Update unrealized PnL
            let unrealized = pnl_pct * qty * entry;
            arena_ref.scalp.pnl_unrealized.store(unrealized, Ordering::Relaxed);

            // Check TP / SL
            if pnl_pct >= scalp_tp || pnl_pct <= -scalp_sl {
                // Close position
                arena_ref.positions.scalp_position.close();
                
                // Update realized PnL
                let old_pnl = arena_ref.scalp.pnl_realized.load(Ordering::Relaxed);
                arena_ref.scalp.pnl_realized.store(old_pnl + unrealized, Ordering::Relaxed);
                
                // Update capital
                let old_cap = arena_ref.unified_capital.load(Ordering::Relaxed);
                arena_ref.unified_capital.store(old_cap + unrealized, Ordering::Relaxed);
                
                println!("[HFT FIRE] 🛑 SCALP POSICIÓN CERRADA. PnL: {:.4} USD", unrealized);
                
                // TODO: Enviar HTTP POST a Binance para cerrar la posición
            }
        } else {
            // --- 3. EVALUAR CEREBROS SI NO HAY POSICIÓN ---
            let obi_threshold = arena_ref.config.scalp_obi_threshold.load(Ordering::Relaxed);
            let scalp_intent = scalp_engine.evaluate_microstructure(tick.bid_qty, tick.ask_qty, obi_threshold);
            let swing_intent = swing_engine.evaluate_trend(mid_price);

            // --- 4. EVALUAR RIESGO & EJECUTAR ---
            let step_size = 0.001; // BNB step size típico
            
            // Evaluar Scalp
            if scalp_intent.signal != SignalType::Flat {
                let validated_scalp = risk_engine.evaluate(scalp_intent, &arena_ref, true);
                if validated_scalp.signal != SignalType::Flat {
                    if let Some(payload) = executor.build_payload(&validated_scalp, &symbol_upper, mid_price, step_size) {
                        println!("[HFT FIRE] 💥 SCALP ORDEN GENERADA: {:?}", payload);
                        
                        let is_long = validated_scalp.signal == SignalType::Long;
                        arena_ref.positions.scalp_position.open(is_long, mid_price, payload.quantity);
                        
                        let client = client_for_closure.clone();
                        let url = url_for_closure.clone();
                        let secret = secret_for_closure.clone();
                        
                        tokio::spawn(async move {
                            let body = format!("symbol={}&side={}&type={}&quantity={}&timestamp={}&signature={}", 
                                payload.symbol, payload.side, payload.order_type, payload.quantity, payload.timestamp, payload.signature);
                            
                            let res = client.post(&url)
                                .header("X-MBX-APIKEY", secret)
                                .header("Content-Type", "application/x-www-form-urlencoded")
                                .body(body)
                                .send()
                                .await;
                                
                            match res {
                                Ok(r) => println!("[EXECUTOR] Binance API Response: {}", r.status()),
                                Err(e) => println!("[EXECUTOR] API Error: {}", e),
                            }
                        });
                    }
                }
            }
            
            // Evaluar Swing
            if swing_intent.signal != SignalType::Flat {
                let validated_swing = risk_engine.evaluate(swing_intent, &arena_ref, false);
                if validated_swing.signal != SignalType::Flat && !arena_ref.positions.swing_position.is_open() {
                    if let Some(payload) = executor.build_payload(&validated_swing, &symbol_upper, mid_price, step_size) {
                        println!("[HFT FIRE] 🌊 SWING ORDEN GENERADA: {:?}", payload);
                        
                        let is_long = validated_swing.signal == SignalType::Long;
                        arena_ref.positions.swing_position.open(is_long, mid_price, payload.quantity);
                        
                        let client = client_for_closure.clone();
                        let url = url_for_closure.clone();
                        let secret = secret_for_closure.clone();
                        
                        tokio::spawn(async move {
                            let body = format!("symbol={}&side={}&type={}&quantity={}&timestamp={}&signature={}", 
                                payload.symbol, payload.side, payload.order_type, payload.quantity, payload.timestamp, payload.signature);
                            
                            let res = client.post(&url)
                                .header("X-MBX-APIKEY", secret)
                                .header("Content-Type", "application/x-www-form-urlencoded")
                                .body(body)
                                .send()
                                .await;
                                
                            if let Err(e) = res {
                                println!("[EXECUTOR] API Error: {}", e);
                            }
                        });
                    }
                }
            }
        }
        
    }).await;
}
