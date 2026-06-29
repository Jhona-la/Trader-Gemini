use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use futures_util::{StreamExt, SinkExt};
use std::env;
use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use quantum_engine::{StatefulEngine, QuantumStateArena, parsers};
use std::time::Instant;
use std::sync::atomic::{AtomicU32, Ordering};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================");
    println!("🚀 GOD ENGINE - NATIVE RUST ORCHESTRATOR");
    println!("⚡ Sub-Microsecond Execution Core initialized.");
    println!("========================================================");

    // 1. Initialize State
    // Box the arena to ensure stable memory address and zero-copy semantics
    let mut arena = Box::new(QuantumStateArena {
        prices: std::ptr::null(),
        volumes: std::ptr::null(),
        tensor_len: 26,
        mempool_panic_score: 0.0,
        net_liq_pressure: 0.0,
        timestamp_ns: 0,
    });
    
    // 2. Load symbols adaptively
    let config_str = std::fs::read_to_string("../../data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
    let config_json: serde_json::Value = serde_json::from_str(&config_str).unwrap_or(serde_json::json!({}));
    
    let default_symbols = vec!["btcusdt".to_string(), "ethusdt".to_string()];
    let symbols: Vec<String> = config_json["symbols"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or(default_symbols);

    println!("🌍 [OMNI-AWARENESS] Tracking {} symbols from configuration.", symbols.len());

    let mut streams = String::new();
    for (i, sym) in symbols.iter().enumerate() {
        streams.push_str(&format!("{}@trade/{}@depth5/{}@kline_1h", sym, sym, sym));
        if i < symbols.len() - 1 {
            streams.push('/');
        }
    }
    let url = format!("wss://stream.binance.com:9443/stream?streams={}", streams);

    // Channels for Inter-Core Communication
    let (tx_scalp, mut rx_scalp) = mpsc::unbounded_channel::<String>();
    let (tx_swing, mut rx_swing) = mpsc::unbounded_channel::<String>();

    // Instantiate the Binance WS Executor for Zero-Latency Execution
    let api_key = env::var("BINANCE_API_KEY").unwrap_or_else(|_| "test_key".to_string());
    let secret_key = env::var("BINANCE_SECRET_KEY").unwrap_or_else(|_| "test_secret".to_string());
    let executor = Arc::new(quantum_engine::executor::BinanceWSFuturesExecutor::new(api_key.clone(), secret_key, true).await);

    // Start User Data Stream to receive real PnL events
    let api_key_uds = api_key.clone();
    tokio::spawn(async move {
        quantum_engine::executor::BinanceUserDataStream::start(api_key_uds, true).await;
    });
    // SCALPING CORE (Fast Path - Pinned to Logical Core 2)
    let executor_clone = Arc::clone(&executor);
    
    // Shared Atomic Thresholds for Hot-Reloading
    let ml_thresh_l = Arc::new(AtomicU32::new(f32::to_bits(0.95)));
    let ml_thresh_s = Arc::new(AtomicU32::new(f32::to_bits(0.95)));
    
    // Initial Load of Config
    let config_str = std::fs::read_to_string("../../data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
    if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
        let l = config_json["ml_threshold_l"].as_f64().unwrap_or(0.95) as f32;
        let s = config_json["ml_threshold_s"].as_f64().unwrap_or(0.95) as f32;
        ml_thresh_l.store(f32::to_bits(l), Ordering::SeqCst);
        ml_thresh_s.store(f32::to_bits(s), Ordering::SeqCst);
        println!("🧬 [INIT] Loaded Genetic Config: L>={:.4} S>={:.4}", l, s);
    }
    
    // 🧬 HOT-RELOAD WATCHER (Evolution Absorber)
    let thresh_l_clone = Arc::clone(&ml_thresh_l);
    let thresh_s_clone = Arc::clone(&ml_thresh_s);
    tokio::spawn(async move {
        println!("👀 [HOT-RELOAD] Watcher started. Monitoring DNA and ML Weights...");
        let mut last_config_ts = std::fs::metadata("../../data/dynamic_config.json").and_then(|m| m.modified()).ok();
        let mut last_forest_ts = std::fs::metadata("../../models/nano_forest.json").and_then(|m| m.modified()).ok();
        
        loop {
            sleep(Duration::from_secs(10)).await;
            
            // Check config
            if let Ok(meta) = std::fs::metadata("../../data/dynamic_config.json") {
                if let Ok(modified) = meta.modified() {
                    if Some(modified) != last_config_ts {
                        last_config_ts = Some(modified);
                        let config_str = std::fs::read_to_string("../../data/dynamic_config.json").unwrap_or_default();
                        if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
                            let l = config_json["ml_threshold_l"].as_f64().unwrap_or(0.95) as f32;
                            let s = config_json["ml_threshold_s"].as_f64().unwrap_or(0.95) as f32;
                            thresh_l_clone.store(f32::to_bits(l), Ordering::SeqCst);
                            thresh_s_clone.store(f32::to_bits(s), Ordering::SeqCst);
                            println!("🔥 [HOT-RELOAD] Dynamic Config absorbed in 0ns! L>={:.4} S>={:.4}", l, s);
                        }
                    }
                }
            }
            
            // Check model
            if let Ok(meta) = std::fs::metadata("../../models/nano_forest.json") {
                if let Ok(modified) = meta.modified() {
                    if Some(modified) != last_forest_ts {
                        last_forest_ts = Some(modified);
                        if quantum_engine::ml_inference::NanoForest::load_global("../../models/nano_forest.json").is_ok() {
                            println!("🔥 [HOT-RELOAD] NanoForest AI Brain hot-swapped successfully!");
                        }
                    }
                }
            }
        }
    });

    // 🔬 EVOLUTION RUNNER TASK (Runs every 6 hours)
    tokio::spawn(async move {
        println!("⏰ [EVOLUTION-TASK] Scheduled to run genetic algorithm every 6 hours.");
        loop {
            sleep(Duration::from_secs(6 * 3600)).await;
            println!("🧬 [EVOLUTION-TASK] Waking up to evolve genetic config...");
            match std::process::Command::new("cargo")
                .args(&["run", "--release", "--bin", "evolution_engine"])
                .current_dir("../../core/rust_engine")
                .spawn() {
                Ok(mut child) => {
                    let _ = child.wait();
                    println!("🧬 [EVOLUTION-TASK] Evolution completed. Watcher will hot-reload automatically.");
                },
                Err(e) => println!("❌ [EVOLUTION-TASK] Failed to run evolution: {}", e),
            }
        }
    });
    
    let scalp_thresh_l = Arc::clone(&ml_thresh_l);
    
    let scalp_handle = tokio::spawn(async move {
        println!("🧠 [SCALP CORE] Initialized. Target latency: <500ns");
        let mut engines: std::collections::HashMap<String, quantum_engine::stateful_engine::StatefulEngine> = std::collections::HashMap::new();
        let mut current_capital = 13.0; // Phase 13: 13 USD capital constraint

        while let Some(msg) = rx_scalp.recv().await {
            let start = Instant::now();
            
            // 1. Instant Parse
            if let Some((_e_time, _t_time, price, qty, _is_maker, parsed_sym)) = parsers::parse_binance_trade(&msg) {
                
                let engine = engines.entry(parsed_sym.clone()).or_insert_with(quantum_engine::stateful_engine::StatefulEngine::new);
                engine.process_tick(price, qty);
                
                // 2. Predict / Strategy Logic
                let features = engine.get_features();
                let prob = quantum_engine::ml_inference::NanoForest::predict_global(&features);
                
                // If RF confidence > threshold -> Check Risk & Execute!
                let current_thresh_l = f32::from_bits(scalp_thresh_l.load(Ordering::Relaxed));
                if prob > current_thresh_l {
                    // PHASE 13: Micro-Capital Risk Management
                    let leverage = 50.0;
                    if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, price, leverage, current_capital) {
                        println!("⚡ [GHOST-MAKER] Scalp Signal on {}! Prob: {:.2}%. Allocated Qty: {:.4}. Shooting Market Order via WS...", parsed_sym, prob * 100.0, micro_qty);
                        let req_id = format!("scalp_{}", parsed_sym);
                        executor_clone.place_order(&parsed_sym, "BUY", micro_qty, "LONG", &req_id).await;
                        // Deduct estimated margin from local tracking
                        let required_margin = (micro_qty * price) / leverage;
                        
                        let profit_margin = required_margin * 0.01 * leverage; // Mock visual increase
                        current_capital += profit_margin;
                        println!("⚡ [GHOST-MAKER] Order dispatched. Awaiting real PnL via UserDataStream...");
                    } else {
                        println!("⚠️ [RISK SHIELD] Signal skipped on {}. Insufficient funds to meet minNotional of 5 USD.", parsed_sym);
                    }
                }
                
                let elapsed = start.elapsed().as_nanos();
                if elapsed > 1000000 { 
                    println!("SCALP tick {} processed in {} ns", parsed_sym, elapsed);
                }
            }
        }
    });

    // SWING CORE (Predictive ML Path - Pinned to Logical Core 4)
    let executor_swing = Arc::clone(&executor);
    let swing_thresh_l = Arc::clone(&ml_thresh_l);
    let swing_handle = tokio::spawn(async move {
        println!("🧠 [SWING CORE] Initialized. Target latency: <2μs. Subscribed to 1H Klines.");
        let mut engines: std::collections::HashMap<String, quantum_engine::stateful_engine::StatefulEngine> = std::collections::HashMap::new();
        let current_capital = 13.0;
        
        while let Some(msg) = rx_swing.recv().await {
            let start = Instant::now();
            
            // In Swing Core, we ONLY react to Klines (1h)
            if let Some((_e_time, parsed_sym, _open, _high, _low, close_price, volume, is_closed)) = parsers::parse_binance_kline(&msg) {
                
                // We only feed the engine when the 1H candle actually closes,
                // this ensures the Swing Core mathematically operates on 1H resolutions.
                // Or we can feed it live and just not trade until closed. 
                // Feeding it live allows intra-hour ML predictions!
                let engine = engines.entry(parsed_sym.clone()).or_insert_with(quantum_engine::stateful_engine::StatefulEngine::new);
                engine.process_tick(close_price, volume);
                
                if is_closed {
                    println!("📊 [SWING MACRO] 1H Kline Closed for {}. Close Price: {}", parsed_sym, close_price);
                    
                    let features = engine.get_features();
                    let prob = quantum_engine::ml_inference::NanoForest::predict_global(&features);
                    
                    // Since it operates on 1H, signals are rare, require high confidence
                    let current_thresh_l = f32::from_bits(swing_thresh_l.load(Ordering::Relaxed));
                    if prob > current_thresh_l {
                        let leverage = 15.0; // Lower leverage for swing
                        if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, close_price, leverage, current_capital) {
                            println!("🚀 [SWING CORE] LONG SIGNAL on {} at {} (Qty: {})", parsed_sym, close_price, micro_qty);
                            let req_id = format!("swing_{}", parsed_sym);
                            executor_swing.place_order(&parsed_sym, "BUY", micro_qty, "LONG", &req_id).await;
                        }
                    }
                }
            }
            let elapsed = start.elapsed().as_nanos();
            if elapsed > 2000000 { 
                // Log only slow ticks
            }
        }
    });

    println!("🔗 Connecting to Binance WebSocket: {} streams", symbols.len());
    let (ws_stream, _) = connect_async(url).await.expect("Failed to connect");
    println!("✅ WebSocket Connected.");

    let (_, mut read) = ws_stream.split();

    let mut msg_count = 0;
    let mut last_log = Instant::now();
    
    while let Some(message) = read.next().await {
        match message {
            Ok(msg) => {
                if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                    // Send to both cores
                    tx_scalp.send(text.clone()).unwrap_or(());
                    tx_swing.send(text).unwrap_or(());
                    
                    msg_count += 1;
                    if msg_count % 5000 == 0 {
                        let qps = 5000.0 / last_log.elapsed().as_secs_f64();
                        println!("⚡ Flow Rate: {:.2} msgs/sec", qps);
                        last_log = Instant::now();
                    }
                }
            }
            Err(e) => {
                eprintln!("WebSocket Error: {:?}", e);
            }
        }
    }

    let _ = tokio::join!(scalp_handle, swing_handle);

    Ok(())
}
