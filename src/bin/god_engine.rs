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

    let mut arena = Box::new(QuantumStateArena {
        prices: std::ptr::null(),
        volumes: std::ptr::null(),
        tensor_len: 26,
        mempool_panic_score: 0.0,
        net_liq_pressure: 0.0,
        timestamp_ns: 0,
    });
    
    let config_str = std::fs::read_to_string("data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
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
    let url = format!("wss://fstream.binance.com/stream?streams={}", streams);

    let (tx_scalp, mut rx_scalp) = mpsc::unbounded_channel::<String>();
    let (tx_swing, mut rx_swing) = mpsc::unbounded_channel::<String>();

    let api_key = env::var("BINANCE_API_KEY").unwrap_or_else(|_| "test_key".to_string());
    let secret_key = env::var("BINANCE_SECRET_KEY").unwrap_or_else(|_| "test_secret".to_string());
    let executor = Arc::new(quantum_engine::executor::BinanceWSFuturesExecutor::new(api_key.clone(), secret_key, true).await);

    let api_key_uds = api_key.clone();
    tokio::spawn(async move {
        quantum_engine::executor::BinanceUserDataStream::start(api_key_uds, true).await;
    });

    let executor_clone = Arc::clone(&executor);
    
    // Shared Atomic Thresholds for Hot-Reloading
    let ml_thresh_l = Arc::new(AtomicU32::new(f32::to_bits(0.95)));
    let ml_thresh_s = Arc::new(AtomicU32::new(f32::to_bits(0.95)));
    let sl_pct = Arc::new(AtomicU32::new(f32::to_bits(0.01)));
    let tp_pct = Arc::new(AtomicU32::new(f32::to_bits(0.02)));
    let tech_thresh_l = Arc::new(AtomicU32::new(f32::to_bits(0.005)));
    let tech_thresh_s = Arc::new(AtomicU32::new(f32::to_bits(0.005)));
    
    let scalp_lev = Arc::new(AtomicU32::new(f32::to_bits(50.0)));
    let swing_lev = Arc::new(AtomicU32::new(f32::to_bits(15.0)));
    
    let config_str = std::fs::read_to_string("data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
    if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
        let l = config_json["ml_threshold_l"].as_f64().unwrap_or(0.95) as f32;
        let s = config_json["ml_threshold_s"].as_f64().unwrap_or(0.95) as f32;
        let sl = config_json["sl_pct"].as_f64().unwrap_or(0.01) as f32;
        let tp = config_json["tp_pct"].as_f64().unwrap_or(0.02) as f32;
        let tl = config_json["tech_threshold_l"].as_f64().unwrap_or(0.005) as f32;
        let ts = config_json["tech_threshold_s"].as_f64().unwrap_or(0.005) as f32;
        let scl = config_json["scalp_leverage"].as_f64().unwrap_or(50.0) as f32;
        let swl = config_json["swing_leverage"].as_f64().unwrap_or(15.0) as f32;
        
        ml_thresh_l.store(f32::to_bits(l), Ordering::SeqCst);
        ml_thresh_s.store(f32::to_bits(s), Ordering::SeqCst);
        sl_pct.store(f32::to_bits(sl), Ordering::SeqCst);
        tp_pct.store(f32::to_bits(tp), Ordering::SeqCst);
        tech_thresh_l.store(f32::to_bits(tl), Ordering::SeqCst);
        tech_thresh_s.store(f32::to_bits(ts), Ordering::SeqCst);
        scalp_lev.store(f32::to_bits(scl), Ordering::SeqCst);
        swing_lev.store(f32::to_bits(swl), Ordering::SeqCst);
        println!("🧬 [INIT] Loaded Config: L>={:.4} S>={:.4} SL={:.4} TP={:.4} SCL={:.1} SWL={:.1}", l, s, sl, tp, scl, swl);
    }
    
    let thresh_l_clone = Arc::clone(&ml_thresh_l);
    let thresh_s_clone = Arc::clone(&ml_thresh_s);
    let sl_pct_clone = Arc::clone(&sl_pct);
    let tp_pct_clone = Arc::clone(&tp_pct);
    let tech_thresh_l_clone = Arc::clone(&tech_thresh_l);
    let tech_thresh_s_clone = Arc::clone(&tech_thresh_s);
    let scalp_lev_clone = Arc::clone(&scalp_lev);
    let swing_lev_clone = Arc::clone(&swing_lev);
    
    tokio::spawn(async move {
        println!("👀 [HOT-RELOAD] Watcher started. Monitoring DNA and ML Weights...");
        let mut last_config_ts = std::fs::metadata("data/dynamic_config.json").and_then(|m| m.modified()).ok();
        let mut last_forest_ts = std::fs::metadata("models/nano_forest.json").and_then(|m| m.modified()).ok();
        
        loop {
            sleep(Duration::from_secs(10)).await;
            
            if let Ok(meta) = std::fs::metadata("data/dynamic_config.json") {
                if let Ok(modified) = meta.modified() {
                    if Some(modified) != last_config_ts {
                        last_config_ts = Some(modified);
                        let config_str = std::fs::read_to_string("data/dynamic_config.json").unwrap_or_default();
                        if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
                            let l = config_json["ml_threshold_l"].as_f64().unwrap_or(0.95) as f32;
                            let s = config_json["ml_threshold_s"].as_f64().unwrap_or(0.95) as f32;
                            let sl = config_json["sl_pct"].as_f64().unwrap_or(0.01) as f32;
                            let tp = config_json["tp_pct"].as_f64().unwrap_or(0.02) as f32;
                            let tl = config_json["tech_threshold_l"].as_f64().unwrap_or(0.005) as f32;
                            let ts = config_json["tech_threshold_s"].as_f64().unwrap_or(0.005) as f32;
                            let scl = config_json["scalp_leverage"].as_f64().unwrap_or(50.0) as f32;
                            let swl = config_json["swing_leverage"].as_f64().unwrap_or(15.0) as f32;
                            
                            thresh_l_clone.store(f32::to_bits(l), Ordering::SeqCst);
                            thresh_s_clone.store(f32::to_bits(s), Ordering::SeqCst);
                            sl_pct_clone.store(f32::to_bits(sl), Ordering::SeqCst);
                            tp_pct_clone.store(f32::to_bits(tp), Ordering::SeqCst);
                            tech_thresh_l_clone.store(f32::to_bits(tl), Ordering::SeqCst);
                            tech_thresh_s_clone.store(f32::to_bits(ts), Ordering::SeqCst);
                            scalp_lev_clone.store(f32::to_bits(scl), Ordering::SeqCst);
                            swing_lev_clone.store(f32::to_bits(swl), Ordering::SeqCst);
                            
                            println!("🔥 [HOT-RELOAD] Dynamic Config absorbed! L>={:.4} S>={:.4} SL={:.4} TP={:.4} SCL={:.1} SWL={:.1}", l, s, sl, tp, scl, swl);
                        }
                    }
                }
            }
            
            if let Ok(meta) = std::fs::metadata("models/nano_forest.json") {
                if let Ok(modified) = meta.modified() {
                    if Some(modified) != last_forest_ts {
                        last_forest_ts = Some(modified);
                        if quantum_engine::ml_inference::NanoForest::load_global("models/nano_forest.json").is_ok() {
                            println!("🔥 [HOT-RELOAD] NanoForest AI Brain hot-swapped successfully!");
                        }
                    }
                }
            }
        }
    });

    tokio::spawn(async move {
        println!("⏰ [EVOLUTION-TASK] Scheduled to run genetic algorithm every 6 hours.");
        loop {
            sleep(Duration::from_secs(6 * 3600)).await;
            println!("🧬 [EVOLUTION-TASK] Waking up to evolve genetic config...");
            match std::process::Command::new("cargo")
                .args(&["run", "--release", "--bin", "evolution_engine"])
                .current_dir(".")
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
    let scalp_thresh_s = Arc::clone(&ml_thresh_s);
    let scalp_sl = Arc::clone(&sl_pct);
    let scalp_tp = Arc::clone(&tp_pct);
    
    let scalp_lev = Arc::clone(&scalp_lev);
    
    let scalp_handle = tokio::spawn(async move {
        println!("🧠 [SCALP CORE] Initialized. Target latency: <500ns");
        let mut engines: std::collections::HashMap<String, quantum_engine::stateful_engine::StatefulEngine> = std::collections::HashMap::new();
        let mut current_capital = 13.0; // Phase 13: 13 USD capital constraint
        
        let mut in_pos = false;
        let mut side = 0_i32; 
        let mut entry_price = 0.0;
        let mut current_qty = 0.0;

        while let Some(msg) = rx_scalp.recv().await {
            let start = Instant::now();
            
            if let Some((_e_time, _t_time, price, qty, _is_maker, parsed_sym)) = parsers::parse_binance_trade(&msg) {
                let engine = engines.entry(parsed_sym.clone()).or_insert_with(quantum_engine::stateful_engine::StatefulEngine::new);
                engine.process_tick(price, qty);
                
                let current_sl = f32::from_bits(scalp_sl.load(Ordering::Relaxed)) as f64 * 0.33;
                let current_tp = f32::from_bits(scalp_tp.load(Ordering::Relaxed)) as f64 * 0.33;

                // EXIT LOGIC
                if in_pos {
                    let pnl_pct = if side == 1 { (price - entry_price) / entry_price } else { (entry_price - price) / entry_price };
                    
                    let mut close_pos = false;
                    if pnl_pct >= current_tp {
                        println!("💰 [SCALP CORE] TAKE PROFIT HIT! PnL: {:.2}%", pnl_pct * 100.0);
                        close_pos = true;
                    } else if pnl_pct <= -current_sl {
                        println!("🛑 [SCALP CORE] STOP LOSS HIT! PnL: {:.2}%", pnl_pct * 100.0);
                        close_pos = true;
                    }

                    if close_pos {
                        let order_side = if side == 1 { "SELL" } else { "BUY" };
                        let req_id = format!("scalp_close_{}", parsed_sym);
                        executor_clone.place_order(&parsed_sym, order_side, current_qty, "MARKET", &req_id).await;
                        
                        let pnl_amount = current_qty * (price - entry_price) * (side as f64);
                        current_capital += pnl_amount;
                        
                        println!("⚡ [SCALP CORE] Closed Position on {}. New estimated capital: {:.2}", parsed_sym, current_capital);
                        in_pos = false;
                    }
                }

                // ENTRY LOGIC
                if !in_pos {
                    let features = engine.get_features();
                    let prob = quantum_engine::ml_inference::NanoForest::predict_global(&features);
                    
                    let tl = f32::from_bits(scalp_thresh_l.load(Ordering::Relaxed));
                    let ts = f32::from_bits(scalp_thresh_s.load(Ordering::Relaxed));
                    
                    if prob > tl {
                        let leverage = f32::from_bits(scalp_lev.load(Ordering::Relaxed)) as f64;
                        if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, price, leverage, current_capital) {
                            println!("⚡ [GHOST-MAKER] Scalp LONG Signal! Prob: {:.2}%. Shooting Market Order via WS...", prob * 100.0);
                            let req_id = format!("scalp_{}", parsed_sym);
                            executor_clone.place_order(&parsed_sym, "BUY", micro_qty, "LONG", &req_id).await;
                            in_pos = true;
                            side = 1;
                            entry_price = price;
                            current_qty = micro_qty;
                        }
                    } else if prob < (1.0 - ts) {
                        let leverage = f32::from_bits(scalp_lev.load(Ordering::Relaxed)) as f64;
                        if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, price, leverage, current_capital) {
                            println!("🩸 [GHOST-MAKER] Scalp SHORT Signal! Prob: {:.2}%. Shooting Market Order via WS...", prob * 100.0);
                            let req_id = format!("scalp_{}", parsed_sym);
                            executor_clone.place_order(&parsed_sym, "SELL", micro_qty, "SHORT", &req_id).await;
                            in_pos = true;
                            side = -1;
                            entry_price = price;
                            current_qty = micro_qty;
                        }
                    }
                }
                
                let elapsed = start.elapsed().as_nanos();
                if elapsed > 1000000 { 
                    println!("SCALP tick {} processed in {} ns", parsed_sym, elapsed);
                }
            }
        }
    });

    let executor_swing = Arc::clone(&executor);
    let swing_thresh_l = Arc::clone(&tech_thresh_l);
    let swing_thresh_s = Arc::clone(&tech_thresh_s);
    let swing_sl = Arc::clone(&sl_pct);
    let swing_tp = Arc::clone(&tp_pct);
    let swing_lev = Arc::clone(&swing_lev);

    let swing_handle = tokio::spawn(async move {
        println!("🧠 [SWING CORE] Initialized. Target latency: <2μs. Subscribed to 1H Klines.");
        let mut engines: std::collections::HashMap<String, quantum_engine::stateful_engine::StatefulEngine> = std::collections::HashMap::new();
        let mut current_capital = 13.0;
        
        let mut in_pos = false;
        let mut side = 0_i32; 
        let mut entry_price = 0.0;
        let mut current_qty = 0.0;

        while let Some(msg) = rx_swing.recv().await {
            let start = Instant::now();
            
            if let Some((_e_time, parsed_sym, _open, _high, _low, close_price, volume, is_closed)) = parsers::parse_binance_kline(&msg) {
                
                let engine = engines.entry(parsed_sym.clone()).or_insert_with(quantum_engine::stateful_engine::StatefulEngine::new);
                engine.process_tick(close_price, volume);
                
                let current_sl_pct = f32::from_bits(swing_sl.load(Ordering::Relaxed)) as f64;
                let current_tp_pct = f32::from_bits(swing_tp.load(Ordering::Relaxed)) as f64;

                // EXIT LOGIC (Continuous monitoring even if not closed)
                if in_pos {
                    let pnl_pct = if side == 1 { (close_price - entry_price) / entry_price } else { (entry_price - close_price) / entry_price };
                    
                    let mut close_pos = false;
                    if pnl_pct >= current_tp_pct {
                        println!("💰 [SWING CORE] TAKE PROFIT HIT! PnL: {:.2}%", pnl_pct * 100.0);
                        close_pos = true;
                    } else if pnl_pct <= -current_sl_pct {
                        println!("🛑 [SWING CORE] STOP LOSS HIT! PnL: {:.2}%", pnl_pct * 100.0);
                        close_pos = true;
                    }

                    if close_pos {
                        let order_side = if side == 1 { "SELL" } else { "BUY" };
                        let req_id = format!("swing_close_{}", parsed_sym);
                        executor_swing.place_order(&parsed_sym, order_side, current_qty, "MARKET", &req_id).await;
                        
                        let pnl_amount = current_qty * (close_price - entry_price) * (side as f64);
                        current_capital += pnl_amount;
                        
                        println!("🚀 [SWING CORE] Closed Position on {}. New estimated capital: {:.2}", parsed_sym, current_capital);
                        in_pos = false;
                    }
                }

                // ENTRY LOGIC (Only on 1H close)
                if !in_pos && is_closed {
                    println!("📊 [SWING MACRO] 1H Kline Closed for {}. Close Price: {}", parsed_sym, close_price);
                    
                    let fast = engine.ema_fast;
                    let slow = engine.ema_slow;
                    
                    let tl = f32::from_bits(swing_thresh_l.load(Ordering::Relaxed)) as f64;
                    let ts = f32::from_bits(swing_thresh_s.load(Ordering::Relaxed)) as f64;
                    
                    if fast > slow * (1.0 + tl) {
                        let leverage = f32::from_bits(swing_lev.load(Ordering::Relaxed)) as f64;
                        if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, close_price, leverage, current_capital) {
                            println!("🚀 [SWING CORE] LONG SIGNAL on {} at {} (Qty: {})", parsed_sym, close_price, micro_qty);
                            let req_id = format!("swing_{}", parsed_sym);
                            executor_swing.place_order(&parsed_sym, "BUY", micro_qty, "LONG", &req_id).await;
                            in_pos = true;
                            side = 1;
                            entry_price = close_price;
                            current_qty = micro_qty;
                        }
                    } else if fast < slow * (1.0 - ts) {
                        let leverage = f32::from_bits(swing_lev.load(Ordering::Relaxed)) as f64;
                        if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, close_price, leverage, current_capital) {
                            println!("🩸 [SWING CORE] SHORT SIGNAL on {} at {} (Qty: {})", parsed_sym, close_price, micro_qty);
                            let req_id = format!("swing_{}", parsed_sym);
                            executor_swing.place_order(&parsed_sym, "SELL", micro_qty, "SHORT", &req_id).await;
                            in_pos = true;
                            side = -1;
                            entry_price = close_price;
                            current_qty = micro_qty;
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
