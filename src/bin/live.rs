use data_pipeline::ws_client::BinanceStreamer;
use execution_engine::executor::{ExecutionProvider, OrderExecutor};
use quantum_arena::GlobalArena;
use std::sync::Arc;
use tokio::sync::mpsc;
use quantum_engine::multi_asset_orchestrator::MultiAssetOrchestrator;
use signal_engine::SignalType;
use quantum_engine::risk::RiskManager;
use quantum_engine::portfolio::Portfolio;
#[tokio::main]
async fn main() {
    // 0. Inicializar Protección del Sistema Operativo Windows (Axioma XX)
    // Afinidad: Núcleos 8-15 (0xFF00)
    // Límite de Memoria: 4096 MB (4 GB)
    os_guardian::init_guardian(0xFF00, 4096);

    // 1. Cargar configuración de entorno para credenciales
    dotenvy::dotenv().ok();
    let api_key = std::env::var("BINANCE_API_KEY").unwrap_or_else(|_| "".to_string());
    let api_secret = std::env::var("BINANCE_API_SECRET").unwrap_or_else(|_| "".to_string());

    if api_key.is_empty() || api_secret.is_empty() {
        eprintln!("ADVERTENCIA: Iniciando en modo Dry Run (Faltan API Keys)");
    }

    // 2. Inicializar el Arena (Memoria compartida estricta O(1))
    // We instantiate the arena using the base capital defined in its own Default implementation.
    let arena = Arc::new(GlobalArena::default()); 
    let initial_capital = arena.config.base_capital.load(std::sync::atomic::Ordering::Relaxed);
    let leverage = arena.config.global_leverage.load(std::sync::atomic::Ordering::Relaxed);
    
    // Iniciar servidor de telemetría sin bloqueo
    let telemetry_arena = arena.clone();
    tokio::spawn(async move {
        telemetry_server::start_telemetry_server(telemetry_arena).await;
    });

    // 2.5 Iniciar el Darwin Daemon (Auto-Evolución Continua Online)
    let darwin_arena = arena.clone();
    tokio::spawn(async move {
        println!("🧬 Darwin Daemon Initialized. Online Evolution is ACTIVE.");
        let daemon = god_engine_core::darwin::DarwinDaemon::new(darwin_arena);
        loop {
            // Evaluar el mercado cada 5 minutos
            tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;
            
            // Darwin Daemon opera lock-free sobre el snapshot en memoria
            tokio::task::spawn_blocking({
                let daemon_clone = god_engine_core::darwin::DarwinDaemon::new(daemon.live_arena.clone());
                move || {
                    daemon_clone.evolve_online();
                }
            }).await.unwrap();
        }
    });

    // 3. Crear el Orquestador Multi-Activo
    let mut orchestrator = MultiAssetOrchestrator::new(arena.clone());
    orchestrator.btc_inventory_usd = 0.0; // Todo en USD, sin posición de BTC al arrancar.
    let executor = Arc::new(OrderExecutor::new(api_key, api_secret));

    println!("🌌 Trader Gemini V5 - Live Engine Ignition (Phase 49 - MARKET MAKER)");
    println!("Capital inicial: ${} USD", initial_capital);
    
    // Usaremos un canal asíncrono puro para recibir los ticks desde el WebSocket
    let (tx, mut rx) = mpsc::unbounded_channel();
    
    let step_size_btc = 0.001; // Para BTC
    let tick_size_btc = 0.1; // Asumiendo tick size de 0.1
    let step_size_eth = 0.01;
    let tick_size_eth = 0.01;

    // 4. Lanzar el Data Pipeline (WebSocket Stream)
    // BTCUSDT
    let tx_btc = tx.clone();
    let arena_btc = arena.clone();
    let streamer_btc = BinanceStreamer::new("btcusdt", arena_btc);
    tokio::spawn(async move {
        streamer_btc.start(move |mut event| {
            event.symbol = "btcusdt".to_string();
            let _ = tx_btc.send(event);
        }).await;
    });

    // ETHUSDT
    let tx_eth = tx.clone();
    let arena_eth = arena.clone();
    let streamer_eth = BinanceStreamer::new("ethusdt", arena_eth);
    tokio::spawn(async move {
        streamer_eth.start(move |mut event| {
            event.symbol = "ethusdt".to_string();
            let _ = tx_eth.send(event);
        }).await;
    });

    println!("📡 Conectado a Binance WebSocket (BTCUSDT, ETHUSDT). Esperando flujos...");

    let mut tick_counter = 0;

    // 5. El God Loop Asíncrono Principal
    while let Some(event) = rx.recv().await {
        tick_counter += 1;

        // Ejecutar el orquestador
        let (quote_opt, arb_opt) = orchestrator.on_tick(
            &event.symbol, 
            event.bid_price, 
            event.ask_price, 
            event.bid_qty, 
            event.ask_qty
        );
        
        let exec = Arc::clone(&executor);

        // Procesar Quotes (Market Making en BTC)
        if let Some(quote) = quote_opt {
            if tick_counter % 1000 == 0 {
                println!("🛡️ [MAKER] Cotizando -> BID: {:.2} | ASK: {:.2} | SPREAD: {:.2}", 
                    quote.bid_price, quote.ask_price, quote.ask_price - quote.bid_price);
            }
            // En Producción (API Keys Válidas):
            // let q_qty = 13.0 / quote.bid_price; 
            // tokio::spawn(async move { exec.execute_limit_order("BTCUSDT", true, q_qty, quote.bid_price, step_size_btc, tick_size_btc, "maker_bid_1").await; });
        }
        
        // Procesar StatArb
        if let Some(arb) = arb_opt {
            let p_eth_bid = orchestrator.eth_bid;
            let p_eth_ask = orchestrator.eth_ask;
            
            if arb.signal == SignalType::Long {
                println!("⚡ [STAT_ARB] Long BTC / Short ETH (Z-Score: {:.2})", arb.confidence);
                let exec_arb = Arc::clone(&executor);
                
                let leg_capital = initial_capital / 2.0;

                // Risk Check
                if RiskManager::can_open_position("BTCUSDT", 1, leg_capital / event.ask_price, event.ask_price, leverage) &&
                   RiskManager::can_open_position("ETHUSDT", 1, leg_capital / p_eth_bid, p_eth_bid, leverage) {
                    Portfolio::set_position("BTCUSDT", 1, 1, event.ask_price, leg_capital / event.ask_price);
                    Portfolio::set_position("ETHUSDT", 1, -1, p_eth_bid, leg_capital / p_eth_bid);
                    
                    tokio::spawn(async move {
                        let _ = exec_arb.execute_raw_qty("BTCUSDT", true, leg_capital, step_size_btc).await;
                        let _ = exec_arb.execute_raw_qty("ETHUSDT", false, leg_capital, step_size_eth).await;
                    });
                } else {
                    println!("⚠️ [RISK] Margen Insuficiente o Posición Existente para StatArb LONG.");
                }
            } else if arb.signal == SignalType::Short {
                println!("⚡ [STAT_ARB] Short BTC / Long ETH (Z-Score: {:.2})", arb.confidence);
                let exec_arb = Arc::clone(&executor);
                
                let leg_capital = initial_capital / 2.0;

                if RiskManager::can_open_position("BTCUSDT", 1, leg_capital / event.bid_price, event.bid_price, leverage) &&
                   RiskManager::can_open_position("ETHUSDT", 1, leg_capital / p_eth_ask, p_eth_ask, leverage) {
                    Portfolio::set_position("BTCUSDT", 1, -1, event.bid_price, leg_capital / event.bid_price);
                    Portfolio::set_position("ETHUSDT", 1, 1, p_eth_ask, leg_capital / p_eth_ask);
                    
                    tokio::spawn(async move {
                        let _ = exec_arb.execute_raw_qty("BTCUSDT", false, leg_capital, step_size_btc).await;
                        let _ = exec_arb.execute_raw_qty("ETHUSDT", true, leg_capital, step_size_eth).await;
                    });
                } else {
                    println!("⚠️ [RISK] Margen Insuficiente o Posición Existente para StatArb SHORT.");
                }
            } else if arb.signal == SignalType::Flat {
                if Portfolio::get_position("BTCUSDT").is_some() {
                    println!("❌ [STAT_ARB] Cerrando posiciones de arbitraje (Reversión a la media)");
                    Portfolio::clear_position("BTCUSDT");
                    Portfolio::clear_position("ETHUSDT");
                    // Aquí se mandarían las órdenes inversas
                }
            }
        }
    }
}
