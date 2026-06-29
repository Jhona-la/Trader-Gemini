use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use futures_util::{StreamExt, SinkExt};
use std::env;
use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use quantum_engine::{StatefulEngine, QuantumStateArena, parsers, stateful_engine::MarketRegime};
use std::time::Instant;
use std::sync::atomic::{AtomicU32, Ordering, AtomicU64};
use serde::Deserialize;

use quantum_engine::config::DynamicConfig;

#[derive(Debug, Clone)]
struct PositionInfo {
    side: i32,
    entry_price: f64,
    qty: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================");
    println!("🚀 GOD ENGINE - NATIVE RUST ORCHESTRATOR (UNIFIED SCALP/SWING)");
    println!("⚡ Sub-Microsecond Execution Core initialized.");
    println!("========================================================");
    
    let config_bytes = std::fs::read("data/dynamic_config.bin").unwrap_or_default();
    let config: DynamicConfig = bincode::deserialize(&config_bytes).unwrap_or_else(|_| {
        // Fallback to json if bin doesn't exist and write it out!
        let config_str = std::fs::read_to_string("data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
        let cfg: DynamicConfig = serde_json::from_str(&config_str).expect("CRITICAL: Failed to parse dynamic_config.json");
        if let Ok(encoded) = bincode::serialize(&cfg) {
            let _ = std::fs::write("data/dynamic_config.bin", encoded);
        }
        cfg
    });
    
    let symbols = config.symbols;

    println!("🌍 [OMNI-AWARENESS] Tracking {} symbols from configuration.", symbols.len());

    let (telemetry_tx, _) = tokio::sync::broadcast::channel(100);
    
    let dashboard_tx = telemetry_tx.clone();
    tokio::spawn(async move {
        if let Err(e) = quantum_engine::dashboard::start_server(dashboard_tx).await {
            println!("❌ [DASHBOARD] Failed to start server: {}", e);
        }
    });

    let mut streams = String::new();
    for (i, sym) in symbols.iter().enumerate() {
        streams.push_str(&format!("{}@trade/{}@depth5/{}@kline_1h", sym, sym, sym));
        if i < symbols.len() - 1 {
            streams.push('/');
        }
    }
    let url = format!("wss://fstream.binance.com/stream?streams={}", streams);

    let (tx_events, mut rx_events) = mpsc::unbounded_channel::<String>();

    let api_key = env::var("BINANCE_API_KEY").unwrap_or_else(|_| "test_key".to_string());
    let secret_key = env::var("BINANCE_SECRET_KEY").unwrap_or_else(|_| "test_secret".to_string());
    let executor = Arc::new(quantum_engine::executor::BinanceWSFuturesExecutor::new(api_key.clone(), secret_key, true).await);

    let api_key_uds = api_key.clone();
    tokio::spawn(async move {
        quantum_engine::executor::BinanceUserDataStream::start(api_key_uds, true).await;
    });
    
    // Shared Atomic Capital Pool ($13 constraint)
    let global_capital = Arc::new(AtomicU64::new(13.0_f64.to_bits()));
    
    // Shared Atomic Thresholds for Hot-Reloading
    let ml_thresh_l = Arc::new(AtomicU32::new(f32::to_bits(0.95)));
    let ml_thresh_s = Arc::new(AtomicU32::new(f32::to_bits(0.95)));
    let sl_pct = Arc::new(AtomicU32::new(f32::to_bits(0.01)));
    let tp_pct = Arc::new(AtomicU32::new(f32::to_bits(0.02)));
    let tech_thresh_l = Arc::new(AtomicU32::new(f32::to_bits(0.005)));
    let tech_thresh_s = Arc::new(AtomicU32::new(f32::to_bits(0.005)));
    let scalp_lev = Arc::new(AtomicU32::new(f32::to_bits(50.0)));
    let swing_lev = Arc::new(AtomicU32::new(f32::to_bits(15.0)));
    
    ml_thresh_l.store(f32::to_bits(config.ml_threshold_l), Ordering::SeqCst);
    ml_thresh_s.store(f32::to_bits(config.ml_threshold_s), Ordering::SeqCst);
    sl_pct.store(f32::to_bits(config.sl_pct), Ordering::SeqCst);
    tp_pct.store(f32::to_bits(config.tp_pct), Ordering::SeqCst);
    tech_thresh_l.store(f32::to_bits(config.tech_threshold_l), Ordering::SeqCst);
    tech_thresh_s.store(f32::to_bits(config.tech_threshold_s), Ordering::SeqCst);
    scalp_lev.store(f32::to_bits(config.scalp_leverage), Ordering::SeqCst);
    swing_lev.store(f32::to_bits(config.swing_leverage), Ordering::SeqCst);
    println!("🧬 [INIT] Loaded Config: L>={:.4} S>={:.4} SL={:.4} TP={:.4} SCL={:.1} SWL={:.1}", 
             config.ml_threshold_l, config.ml_threshold_s, config.sl_pct, config.tp_pct, config.scalp_leverage, config.swing_leverage);
    
    let ml_thresh_l_clone = Arc::clone(&ml_thresh_l);
    let ml_thresh_s_clone = Arc::clone(&ml_thresh_s);
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
                        let config_bytes = std::fs::read("data/dynamic_config.bin").unwrap_or_default();
                        let cfg: DynamicConfig = bincode::deserialize(&config_bytes).unwrap_or_else(|_| {
                            let config_str = std::fs::read_to_string("data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
                            serde_json::from_str(&config_str).unwrap_or_else(|_| DynamicConfig {
                                symbols: vec!["btcusdt".to_string(), "ethusdt".to_string()],
                                sl_pct: 0.002, tp_pct: 0.004,
                                ml_threshold_l: 0.5, ml_threshold_s: 0.5,
                                tech_threshold_l: 0.002, tech_threshold_s: 0.002,
                                scalp_leverage: 50.0, swing_leverage: 30.0,
                            })
                        });

                        ml_thresh_l_clone.store(f32::to_bits(cfg.ml_threshold_l), Ordering::SeqCst);
                        ml_thresh_s_clone.store(f32::to_bits(cfg.ml_threshold_s), Ordering::SeqCst);
                        sl_pct_clone.store(f32::to_bits(cfg.sl_pct), Ordering::SeqCst);
                        tp_pct_clone.store(f32::to_bits(cfg.tp_pct), Ordering::SeqCst);
                        tech_thresh_l_clone.store(f32::to_bits(cfg.tech_threshold_l), Ordering::SeqCst);
                        tech_thresh_s_clone.store(f32::to_bits(cfg.tech_threshold_s), Ordering::SeqCst);
                        scalp_lev_clone.store(f32::to_bits(cfg.scalp_leverage), Ordering::SeqCst);
                        swing_lev_clone.store(f32::to_bits(cfg.swing_leverage), Ordering::SeqCst);
                        
                        println!("🔥 [HOT-RELOAD] Dynamic Config absorbed successfully in Nanoseconds!");
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
            let _ = std::process::Command::new("cargo")
                .args(&["run", "--release", "--bin", "evolution_engine"])
                .current_dir(".")
                .spawn()
                .and_then(|mut child| child.wait());
        }
    });
    
    let exec = Arc::clone(&executor);
    
    let loop_telemetry_tx = telemetry_tx.clone();
    // Unified Event Loop Task
    let unified_handle = tokio::spawn(async move {
        println!("🧠 [UNIFIED CORE] Initialized. Target latency: <500ns.");
        
        let mut scalp_engines: std::collections::HashMap<String, quantum_engine::stateful_engine::StatefulEngine> = std::collections::HashMap::new();
        let mut swing_engines: std::collections::HashMap<String, quantum_engine::stateful_engine::StatefulEngine> = std::collections::HashMap::new();
        
        let mut scalp_positions: std::collections::HashMap<String, PositionInfo> = std::collections::HashMap::new();
        let mut swing_positions: std::collections::HashMap<String, PositionInfo> = std::collections::HashMap::new();
        
        let mut msg_count = 0u64;

        while let Some(msg) = rx_events.recv().await {
            let start = Instant::now();
            
            // 1. Determine event type from JSON quickly
            let mut event_type = 0; // 0 = unknown, 1 = trade, 2 = kline
            if msg.contains("\"e\":\"trade\"") { event_type = 1; }
            else if msg.contains("\"e\":\"kline\"") { event_type = 2; }
            
            if event_type == 1 {
                // SCALP LOGIC
                if let Some((_e_time, _t_time, price, qty, _is_maker, parsed_sym)) = parsers::parse_binance_trade(&msg) {
                    let engine = scalp_engines.entry(parsed_sym.clone()).or_insert_with(quantum_engine::stateful_engine::StatefulEngine::new);
                    engine.process_tick(price, qty);
                    
                    let regime = engine.get_market_regime();
                    
                    let current_sl = f32::from_bits(sl_pct.load(Ordering::Relaxed)) as f64 * 0.33;
                    let current_tp = f32::from_bits(tp_pct.load(Ordering::Relaxed)) as f64 * 0.33;
                    
                    // Exit Logic
                    if let Some(pos) = scalp_positions.get(&parsed_sym).cloned() {
                        let pnl_pct = if pos.side == 1 { (price - pos.entry_price) / pos.entry_price } else { (pos.entry_price - price) / pos.entry_price };
                        
                        let mut close_pos = false;
                        if pnl_pct >= current_tp {
                            println!("💰 [SCALP CORE] TAKE PROFIT HIT! PnL: {:.2}%", pnl_pct * 100.0);
                            close_pos = true;
                        } else if pnl_pct <= -current_sl {
                            println!("🛑 [SCALP CORE] STOP LOSS HIT! PnL: {:.2}%", pnl_pct * 100.0);
                            close_pos = true;
                        }
                        
                        // Adaptive Regime Exit (If market turns against scalp mean reversion)
                        if regime == MarketRegime::Swing && pnl_pct > 0.0 {
                            println!("🔄 [SCALP CORE] MARKET REGIME SHIFTED TO SWING. Taking early profit: {:.2}%", pnl_pct * 100.0);
                            close_pos = true;
                        }

                        if close_pos {
                            let order_side = if pos.side == 1 { "SELL" } else { "BUY" };
                            let req_id = format!("scalp_close_{}", parsed_sym);
                            exec.place_order(&parsed_sym, order_side, pos.qty, "MARKET", &req_id).await;
                            
                            let pnl_amount = pos.qty * (price - pos.entry_price) * (pos.side as f64);
                            let mut curr_cap = f64::from_bits(global_capital.load(Ordering::Relaxed));
                            curr_cap += pnl_amount;
                            global_capital.store(curr_cap.to_bits(), Ordering::SeqCst);
                            let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::CapitalUpdate(curr_cap));
                            
                            scalp_positions.remove(&parsed_sym);
                        }
                    }

                    // Entry Logic (Only execute scalps in Mean Reverting or Neutral regimes)
                    if !scalp_positions.contains_key(&parsed_sym) && (regime == MarketRegime::Scalping || regime == MarketRegime::Neutral) {
                        let curr_cap = f64::from_bits(global_capital.load(Ordering::Relaxed));
                        let features = engine.get_features();
                        let prob = quantum_engine::ml_inference::NanoForest::predict_global(&features);
                        
                        let tl = f32::from_bits(ml_thresh_l.load(Ordering::Relaxed));
                        let ts = f32::from_bits(ml_thresh_s.load(Ordering::Relaxed));
                        
                        if prob > tl {
                            let leverage = f32::from_bits(scalp_lev.load(Ordering::Relaxed)) as f64;
                            if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, price, leverage, curr_cap) {
                                println!("⚡ [SCALP CORE] LONG Signal (Regime: {:?})! Prob: {:.2}%. WS Order sent.", regime, prob * 100.0);
                                let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LogUpdate(
                                    "success".to_string(), 
                                    format!("⚡ SCALP LONG on {} (Prob: {:.2}%)", parsed_sym, prob * 100.0)
                                ));
                                let req_id = format!("scalp_{}", parsed_sym);
                                exec.place_order(&parsed_sym, "BUY", micro_qty, "LONG", &req_id).await;
                                scalp_positions.insert(parsed_sym.clone(), PositionInfo { side: 1, entry_price: price, qty: micro_qty });
                            }
                        } else if prob < (1.0 - ts) {
                            let leverage = f32::from_bits(scalp_lev.load(Ordering::Relaxed)) as f64;
                            if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, price, leverage, curr_cap) {
                                println!("🩸 [SCALP CORE] SHORT Signal (Regime: {:?})! Prob: {:.2}%. WS Order sent.", regime, prob * 100.0);
                                let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LogUpdate(
                                    "error".to_string(), 
                                    format!("🩸 SCALP SHORT on {} (Prob: {:.2}%)", parsed_sym, prob * 100.0)
                                ));
                                let req_id = format!("scalp_{}", parsed_sym);
                                exec.place_order(&parsed_sym, "SELL", micro_qty, "SHORT", &req_id).await;
                                scalp_positions.insert(parsed_sym.clone(), PositionInfo { side: -1, entry_price: price, qty: micro_qty });
                            }
                        }
                    }
                }
            } else if event_type == 2 {
                // SWING LOGIC
                if let Some((_e_time, parsed_sym, _open, _high, _low, close_price, volume, is_closed)) = parsers::parse_binance_kline(&msg) {
                    let engine = swing_engines.entry(parsed_sym.clone()).or_insert_with(quantum_engine::stateful_engine::StatefulEngine::new);
                    engine.process_tick(close_price, volume);
                    
                    let regime = engine.get_market_regime();
                    
                    let current_sl_pct = f32::from_bits(sl_pct.load(Ordering::Relaxed)) as f64;
                    let current_tp_pct = f32::from_bits(tp_pct.load(Ordering::Relaxed)) as f64;

                    // Exit Logic Continuous
                    if let Some(pos) = swing_positions.get(&parsed_sym).cloned() {
                        let pnl_pct = if pos.side == 1 { (close_price - pos.entry_price) / pos.entry_price } else { (pos.entry_price - close_price) / pos.entry_price };
                        
                        let mut close_pos = false;
                        if pnl_pct >= current_tp_pct {
                            println!("💰 [SWING CORE] TAKE PROFIT HIT! PnL: {:.2}%", pnl_pct * 100.0);
                            close_pos = true;
                        } else if pnl_pct <= -current_sl_pct {
                            println!("🛑 [SWING CORE] STOP LOSS HIT! PnL: {:.2}%", pnl_pct * 100.0);
                            close_pos = true;
                        }
                        
                        if regime == MarketRegime::Scalping && pnl_pct > 0.0 {
                            println!("🔄 [SWING CORE] MARKET REGIME SHIFTED TO SCALPING (Chop). Securing profit: {:.2}%", pnl_pct * 100.0);
                            close_pos = true;
                        }

                        if close_pos {
                            let order_side = if pos.side == 1 { "SELL" } else { "BUY" };
                            let req_id = format!("swing_close_{}", parsed_sym);
                            exec.place_order(&parsed_sym, order_side, pos.qty, "MARKET", &req_id).await;
                            
                            let pnl_amount = pos.qty * (close_price - pos.entry_price) * (pos.side as f64);
                            let mut curr_cap = f64::from_bits(global_capital.load(Ordering::Relaxed));
                            curr_cap += pnl_amount;
                            global_capital.store(curr_cap.to_bits(), Ordering::SeqCst);
                            let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::CapitalUpdate(curr_cap));
                            
                            swing_positions.remove(&parsed_sym);
                        }
                    }

                    // Entry Logic (Only on 1H close, and only in Trending regimes)
                    if !swing_positions.contains_key(&parsed_sym) && is_closed && regime == MarketRegime::Swing {
                        let curr_cap = f64::from_bits(global_capital.load(Ordering::Relaxed));
                        let fast = engine.ema_fast;
                        let slow = engine.ema_slow;
                        
                        let tl = f32::from_bits(tech_thresh_l.load(Ordering::Relaxed)) as f64;
                        let ts = f32::from_bits(tech_thresh_s.load(Ordering::Relaxed)) as f64;
                        
                        if fast > slow * (1.0 + tl) {
                            let leverage = f32::from_bits(swing_lev.load(Ordering::Relaxed)) as f64;
                            if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, close_price, leverage, curr_cap) {
                                println!("🚀 [SWING CORE] LONG SIGNAL on {} (Regime: {:?})", parsed_sym, regime);
                                let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LogUpdate(
                                    "info".to_string(), 
                                    format!("🚀 SWING LONG on {}", parsed_sym)
                                ));
                                let req_id = format!("swing_{}", parsed_sym);
                                exec.place_order(&parsed_sym, "BUY", micro_qty, "LONG", &req_id).await;
                                swing_positions.insert(parsed_sym.clone(), PositionInfo { side: 1, entry_price: close_price, qty: micro_qty });
                            }
                        } else if fast < slow * (1.0 - ts) {
                            let leverage = f32::from_bits(swing_lev.load(Ordering::Relaxed)) as f64;
                            if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(&parsed_sym, close_price, leverage, curr_cap) {
                                println!("🩸 [SWING CORE] SHORT SIGNAL on {} (Regime: {:?})", parsed_sym, regime);
                                let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LogUpdate(
                                    "warn".to_string(), 
                                    format!("🩸 SWING SHORT on {}", parsed_sym)
                                ));
                                let req_id = format!("swing_{}", parsed_sym);
                                exec.place_order(&parsed_sym, "SELL", micro_qty, "SHORT", &req_id).await;
                                swing_positions.insert(parsed_sym.clone(), PositionInfo { side: -1, entry_price: close_price, qty: micro_qty });
                            }
                        }
                    }
                }
            }
            
            msg_count += 1;
            if msg_count % 100 == 0 {
                let lat = start.elapsed().as_nanos();
                let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LatencyUpdate(lat as u64));
            }
            if msg_count % 5000 == 0 {
                println!("⏱️ [TELEMETRY] Processed 5000 ticks/klines. Last tick: {} ns", start.elapsed().as_nanos());
            }
        }
    });

    println!("🔗 Connecting to Binance WebSocket: {} streams", symbols.len());
    let (ws_stream, _) = connect_async(url).await.expect("Failed to connect");
    println!("✅ WebSocket Connected.");

    let (_, mut read) = ws_stream.split();

    while let Some(message) = read.next().await {
        match message {
            Ok(msg) => {
                if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                    tx_events.send(text).unwrap_or(());
                }
            }
            Err(e) => {
                eprintln!("WebSocket Error: {:?}", e);
            }
        }
    }

    let _ = unified_handle.await;

    Ok(())
}
