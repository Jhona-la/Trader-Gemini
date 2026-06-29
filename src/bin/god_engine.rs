use tokio_tungstenite::connect_async;
use futures_util::StreamExt;
use std::env;
use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use quantum_engine::{parsers, stateful_engine::MarketRegime};
use std::time::Instant;
use std::sync::atomic::{AtomicU32, Ordering, AtomicU64};

use quantum_engine::config::DynamicConfig;

#[derive(Debug, Clone)]
struct PositionInfo {
    side: i32,
    entry_price: f64,
    qty: f64,
    trailing_phase: i32,
    mfe_atr: f64,
    max_pnl_pct: f64,
    trail_stop: f64,
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

    let api_key = env::var("BINANCE_API_KEY").unwrap_or_else(|_| "".to_string());
    let secret_key = env::var("BINANCE_SECRET_KEY").unwrap_or_else(|_| "".to_string());
    
    if api_key.is_empty() || api_key == "test_key" {
        panic!("🚨 [FATAL ERROR] Producción Abortada: Falta BINANCE_API_KEY real en el entorno.");
    }
    if secret_key.is_empty() || secret_key == "test_secret" {
        panic!("🚨 [FATAL ERROR] Producción Abortada: Falta BINANCE_SECRET_KEY real en el entorno.");
    }

    // MAINNET ACTIVATION: is_testnet = false
    let executor = Arc::new(quantum_engine::executor::BinanceWSFuturesExecutor::new(api_key.clone(), secret_key.clone(), false).await);
    let rest_executor = Arc::new(quantum_engine::executor::BinanceRestExecutor::new(api_key.clone(), secret_key.clone(), false));

    let api_key_uds = api_key.clone();
    tokio::spawn(async move {
        quantum_engine::executor::BinanceUserDataStream::start(api_key_uds, false).await;
    });
    
    let dark_router = Arc::new(quantum_engine::dark_alpha_router::DarkAlphaRouter::new());
    
    // Spawn Hyperliquid DEX cascade sniffer
    quantum_engine::dark_alpha_sniffer::spawn_hyperliquid_sniffer(Arc::clone(&dark_router));
    
    // Initialize SQLite WAL for Persistence
    let mut initial_capital = 13.0_f64;
    std::fs::create_dir_all("data").unwrap_or_default();
    if let Ok(conn) = rusqlite::Connection::open("data/state.db") {
        let _ = conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS capital_state (
                 id INTEGER PRIMARY KEY,
                 capital REAL NOT NULL
             );"
        );
        let res: rusqlite::Result<f64> = conn.query_row("SELECT capital FROM capital_state WHERE id = 1", [], |r| r.get(0));
        match res {
            Ok(cap) => initial_capital = cap,
            Err(_) => {
                let _ = conn.execute("INSERT INTO capital_state (id, capital) VALUES (1, ?1)", [initial_capital]);
            }
        }
        println!("💾 [PERSISTENCE] Capital SQLite WAL loaded: ${:.4}", initial_capital);
    }
    
    // Shared Atomic Capital Pool ($13 base limit tracked persistently)
    // Axiom V: Unified Cross-Margin Pool
    let unified_capital = Arc::new(AtomicU64::new(initial_capital.to_bits()));
    
    // Non-blocking SQLite persistence channel
    let (db_tx, mut db_rx) = mpsc::unbounded_channel::<(f64, f64)>();
    tokio::spawn(async move {
        if let Ok(conn) = rusqlite::Connection::open("data/state.db") {
            while let Some((sc, sw)) = db_rx.recv().await {
                let total = sc + sw;
                let _ = conn.execute("UPDATE capital_state SET capital = ?1 WHERE id = 1", [total]);
            }
        }
    });
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
    
    let mut forest_timestamps: std::collections::HashMap<String, std::time::SystemTime> = std::collections::HashMap::new();
    
    // Initial load of all models
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        if let Ok(modified) = meta.modified() {
                            forest_timestamps.insert(file_stem.to_string(), modified);
                            let _ = quantum_engine::ml_inference::NanoForest::load_global(file_stem, path.to_str().unwrap());
                            println!("🧠 Loaded ML Model: {}", file_stem);
                        }
                    }
                }
            }
        }
    }

    tokio::spawn(async move {
        println!("👀 [HOT-RELOAD] Watcher started. Monitoring DNA and ML Weights...");
        let mut last_config_ts = std::fs::metadata("data/dynamic_config.json").and_then(|m| m.modified()).ok();
        
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
                                sl_pct: 0.0015, tp_pct: 0.025,
                                ml_threshold_l: 0.65, ml_threshold_s: 0.65,
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
            
            if let Ok(entries) = std::fs::read_dir("models") {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                            if let Ok(meta) = std::fs::metadata(&path) {
                                if let Ok(modified) = meta.modified() {
                                    let last_ts = forest_timestamps.get(file_stem);
                                    if last_ts != Some(&modified) {
                                        forest_timestamps.insert(file_stem.to_string(), modified);
                                        if quantum_engine::ml_inference::NanoForest::load_global(file_stem, path.to_str().unwrap()).is_ok() {
                                            println!("🔥 [HOT-RELOAD] NanoForest AI Brain hot-swapped for {}!", file_stem);
                                        }
                                    }
                                }
                            }
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
    let rest_exec = Arc::clone(&rest_executor);
    let loop_telemetry_tx = telemetry_tx.clone();
    let dark_router_unified = Arc::clone(&dark_router);
    let unified_handle = tokio::spawn(async move {
        println!("🧠 [UNIFIED CORE] Initialized. Target latency: <500ns.");
        
        // Axiom XIV: Superposición Cuántica - Mentes Aisladas (Scalp vs Swing)
        let mut scalp_engines: std::collections::HashMap<std::sync::Arc<String>, quantum_engine::stateful_engine::StatefulEngine> = std::collections::HashMap::new();
        let mut swing_engines: std::collections::HashMap<std::sync::Arc<String>, quantum_engine::stateful_engine::StatefulEngine> = std::collections::HashMap::new();
        let mut open_positions: std::collections::HashMap<(std::sync::Arc<String>, u8), PositionInfo> = std::collections::HashMap::new();
        let mut symbol_cache: std::collections::HashMap<String, std::sync::Arc<String>> = std::collections::HashMap::new();
        
        let mut msg_count = 0u64;

        while let Some(mut msg) = rx_events.recv().await {
            let start = Instant::now();
            
            let is_trade = msg.contains("\"e\":\"trade\"");
            let is_kline = msg.contains("\"e\":\"kline\"");
            let is_depth = msg.contains("\"e\":\"depthUpdate\"");
            let is_reconnect = msg == "[SYSTEM:RECONNECT]";
            
            if is_reconnect {
                println!("🧹 [AUTO-HEALING] Reconnect signal received. Purging Quantum Engine state to prevent time-glitches...");
                for engine in scalp_engines.values_mut() {
                    engine.reset();
                }
                for engine in swing_engines.values_mut() {
                    engine.reset();
                }
                println!("✅ [AUTO-HEALING] All AI Engines flushed. Entering Warmup Phase (50 ticks).");
                
                // Spawn REST sync here
                let rx_rest = Arc::clone(&rest_exec);
                tokio::spawn(async move {
                    println!("🔄 [REST-SYNC] Fetching truth from Binance API...");
                    if let Ok(positions) = rx_rest.fetch_open_positions().await {
                        let mut open_count = 0;
                        for pos in positions {
                            if let (Some(amt_str), Some(_sym)) = (pos.get("positionAmt").and_then(|v| v.as_str()), pos.get("symbol").and_then(|v| v.as_str())) {
                                if let Ok(amt) = amt_str.parse::<f64>() {
                                    if amt != 0.0 {
                                        open_count += 1;
                                    }
                                }
                            }
                        }
                        println!("✅ [REST-SYNC] Binance reports {} active open positions.", open_count);
                        // Note: For absolute consistency, we would send a message back through a channel to safely mutate open_positions here.
                        // Since open_positions is strictly single-threaded in the loop, we just print the delta for now.
                    }
                });
                continue;
            }
            
            let mut parsed_sym_opt = None;
            let mut current_price = 0.0;
            let mut qty = 0.0;
            let mut is_kline_closed = false;
            let mut event_time = 0i64;
            let mut depth_obi = 0.0;
            let mut depth_micro_div = 0.0;
            
            if is_trade {
                if let Some((e, _, p, q, _, sym)) = parsers::parse_binance_trade(&mut msg) {
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    current_price = p;
                    qty = q;
                }
            } else if is_kline {
                if let Some((e, sym, _, _, _, p, v, c)) = parsers::parse_binance_kline(&mut msg) {
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    current_price = p;
                    qty = v;
                    is_kline_closed = c;
                }
            } else if is_depth {
                if let Some((e, sym, _, bp, bq, ap, aq)) = parsers::parse_binance_depth(&mut msg) {
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    let total_q = bq + aq;
                    if total_q > 0.0 {
                        depth_obi = (bq - aq) / total_q;
                        let microprice = (bp * aq + ap * bq) / total_q;
                        let midprice = (bp + ap) / 2.0;
                        depth_micro_div = if midprice > 0.0 { (microprice - midprice) / midprice } else { 0.0 };
                    }
                }
            }
            
            // FASE 23: QUANTUM LATENCY KILL-SWITCH
            let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
            let latency_ms = now_ms - event_time;
            let mut latency_panic = false;
            
            if event_time > 0 && latency_ms > 800 {
                // If the delta is larger than 800ms, we are operating in the past.
                latency_panic = true;
                println!("⚠️ [LATENCY_PANIC] Delta = {}ms (>800ms limit). Skiping O(1) Scalp execution.", latency_ms);
            }
            
            if let Some(parsed_sym_raw) = parsed_sym_opt {
                let sym_arc = symbol_cache.entry(parsed_sym_raw).or_insert_with_key(|k| std::sync::Arc::new(k.clone())).clone();
                let parsed_sym = sym_arc.as_ref();
                
                let scalp_engine = scalp_engines.entry(sym_arc.clone()).or_insert_with(quantum_engine::stateful_engine::StatefulEngine::new);
                let swing_engine = swing_engines.entry(sym_arc.clone()).or_insert_with(quantum_engine::stateful_engine::StatefulEngine::new);
                
                if is_trade {
                    scalp_engine.process_tick(current_price, qty);
                }
                if is_kline && is_kline_closed {
                    swing_engine.process_tick(current_price, qty);
                }
                if is_depth {
                    scalp_engine.update_macro_features(depth_obi, depth_micro_div, 0.0, event_time as u64);
                    dark_router_unified.ingest_l2_snapshot(0.0, depth_obi, depth_micro_div, event_time as u64);
                }
                
                let scalp_regime = scalp_engine.get_market_regime();
                let swing_regime = swing_engine.get_market_regime();
                
                // --- TENSOR EVALUATION: Horizontes 0 (Scalp) y 1 (Swing) ---
                for horizon in 0..=1u8 {
                    let regime = if horizon == 0 { scalp_regime } else { swing_regime };
                    
                    let (is_active_entry, sl_mult, tp_mult, trail_atr_mult, t_params) = match horizon {
                        0 => (!latency_panic && is_trade && (regime == MarketRegime::Scalping || regime == MarketRegime::Neutral), 0.15, 0.35, 0.005, (0.001, 1.0, 1.5, 2.0, 0.8)),
                        1 => (is_kline && is_kline_closed && regime == MarketRegime::Swing, 0.8, 2.0, 0.01, (0.005, 1.5, 3.0, 4.0, 1.0)),
                        _ => unreachable!(),
                    };
                    
                    let pos_key = (sym_arc.clone(), horizon);
                    let current_sl = f32::from_bits(sl_pct.load(Ordering::Relaxed)) as f64 * sl_mult;
                    let current_tp = f32::from_bits(tp_pct.load(Ordering::Relaxed)) as f64 * tp_mult;
                    
                    // Exit Logic
                    let mut should_remove = false;
                    if let Some(pos) = open_positions.get_mut(&pos_key) {
                        let pnl_pct = if pos.side == 1 { (current_price - pos.entry_price) / pos.entry_price } else { (pos.entry_price - current_price) / pos.entry_price };
                        
                        let pseudo_atr = current_price * trail_atr_mult;
                        let trail_res = quantum_engine::trailing::evaluate_quantum_trailing(
                            pos.side, pos.entry_price, current_price, pseudo_atr, pos.trailing_phase, 
                            pos.mfe_atr, pos.max_pnl_pct, pos.trail_stop,
                            t_params.0, t_params.1, t_params.2, t_params.3, t_params.4
                        );
                        
                        pos.trail_stop = trail_res.stop_price;
                        pos.trailing_phase = trail_res.new_phase;
                        pos.mfe_atr = trail_res.mfe_atr;
                        pos.max_pnl_pct = trail_res.max_pnl_pct;
                        
                        let mut close_pos = trail_res.force_close;
                        let horizon_name = if horizon == 0 { "SCALP" } else { "SWING" };
                        
                        if pnl_pct >= current_tp {
                            println!("💰 [{} CORE] TAKE PROFIT HIT! PnL: {:.2}%", horizon_name, pnl_pct * 100.0);
                            close_pos = true;
                        } else if pnl_pct <= -current_sl || (pos.side == 1 && current_price <= pos.trail_stop) || (pos.side == -1 && current_price >= pos.trail_stop) {
                            println!("🛑 [{} CORE] STOP LOSS / TRAILING STOP HIT! PnL: {:.2}%", horizon_name, pnl_pct * 100.0);
                            close_pos = true;
                        }
                        
                        // Adaptive Regime Exit
                        if horizon == 0 && regime == MarketRegime::Swing && pnl_pct > 0.0 {
                            println!("🔄 [SCALP CORE] MARKET REGIME SHIFTED TO SWING. Taking early profit: {:.2}%", pnl_pct * 100.0);
                            close_pos = true;
                        } else if horizon == 1 && regime == MarketRegime::Scalping && pnl_pct > 0.0 {
                            println!("🔄 [SWING CORE] MARKET REGIME SHIFTED TO SCALPING. Securing profit: {:.2}%", pnl_pct * 100.0);
                            close_pos = true;
                        }

                        if close_pos {
                            let order_side = if pos.side == 1 { "SELL" } else { "BUY" };
                            let pos_side_str = if pos.side == 1 { "LONG" } else { "SHORT" };
                            let req_id = format!("{}_close_{}", horizon_name.to_lowercase(), parsed_sym);
                            exec.place_order(parsed_sym, order_side, pos.qty, pos_side_str, &req_id);
                            
                            let pnl_amount = pos.qty * (current_price - pos.entry_price) * (pos.side as f64);
                            let cap_ref = &unified_capital;
                            let mut curr_cap = f64::from_bits(cap_ref.load(Ordering::Relaxed));
                            curr_cap += pnl_amount;
                            cap_ref.store(curr_cap.to_bits(), Ordering::SeqCst);
                            
                            let unified = f64::from_bits(unified_capital.load(Ordering::Relaxed));
                            let _ = db_tx.send((unified, 0.0));
                            let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::CapitalUpdate(unified));
                            
                            should_remove = true;
                        }
                    }
                    if should_remove {
                        open_positions.remove(&pos_key);
                    }

                    // Entry Logic
                    if !open_positions.contains_key(&pos_key) && is_active_entry {
                        let cap_ref = &unified_capital;
                        let curr_cap = f64::from_bits(cap_ref.load(Ordering::Relaxed));
                        let horizon_name = if horizon == 0 { "SCALP" } else { "SWING" };
                        
                        let mut go_long = false;
                        let mut go_short = false;
                        
                        if horizon == 0 {
                            let features = scalp_engine.get_features();
                            let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::TensorUpdate(features));
                            let signal_prob = quantum_engine::ml_inference::NanoForest::predict_global(&format!("{}_SCALP", parsed_sym), &features);
                            let tl = f32::from_bits(ml_thresh_l.load(Ordering::Relaxed));
                            let ts = f32::from_bits(ml_thresh_s.load(Ordering::Relaxed));
                            
                            if signal_prob > tl { go_long = true; }
                            else if signal_prob < (1.0 - ts) { go_short = true; }
                        } else {
                            let fast = swing_engine.ema_fast;
                            let slow = swing_engine.ema_slow;
                            let tl = f32::from_bits(tech_thresh_l.load(Ordering::Relaxed)) as f64;
                            let ts = f32::from_bits(tech_thresh_s.load(Ordering::Relaxed)) as f64;
                            
                            if fast > slow * (1.0 + tl) { go_long = true; }
                            else if fast < slow * (1.0 - ts) { go_short = true; }
                        }
                        
                        let leverage = f32::from_bits(if horizon == 0 { scalp_lev.load(Ordering::Relaxed) } else { swing_lev.load(Ordering::Relaxed) }) as f64;
                        
                        if go_long || go_short {
                            let atr = if horizon == 0 { scalp_engine.get_atr_pct() } else { swing_engine.get_atr_pct() };
                            if let Some(micro_qty) = quantum_engine::risk::RiskManager::calculate_micro_position_size(parsed_sym, current_price, leverage, curr_cap, atr) {
                                let side_val = if go_long { 1 } else { -1 };
                                let order_side = if go_long { "BUY" } else { "SELL" };
                                let pos_side_str = if go_long { "LONG" } else { "SHORT" };
                                let icon = if horizon == 0 { "⚡" } else { "🚀" };
                                
                                println!("{} [{} CORE] {} Signal (Regime: {:?})! WS Order sent.", icon, horizon_name, pos_side_str, regime);
                                let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LogUpdate(
                                    if go_long { "success".to_string() } else { "error".to_string() }, 
                                    format!("{} {} {} on {}", icon, horizon_name, pos_side_str, parsed_sym)
                                ));
                                
                                let req_id = format!("{}_{}", horizon_name.to_lowercase(), parsed_sym);
                                exec.place_order(parsed_sym, order_side, micro_qty, pos_side_str, &req_id);
                                
                                let trail_offset = current_price * current_sl;
                                let initial_trail = if go_long { current_price - trail_offset } else { current_price + trail_offset };
                                
                                open_positions.insert(pos_key, PositionInfo { 
                                    side: side_val, 
                                    entry_price: current_price, 
                                    qty: micro_qty, 
                                    trailing_phase: 0, 
                                    mfe_atr: 0.0, 
                                    max_pnl_pct: 0.0, 
                                    trail_stop: initial_trail 
                                });
                            }
                        }
                    }
                }
            }
            
            msg_count += 1;
            if msg_count % 100 == 0 {
                let lat = start.elapsed().as_nanos();
                
                let mut scalp_pnl = 0.0;
                let mut swing_pnl = 0.0;
                for (k, v) in open_positions.iter() {
                    let pnl = if v.side == 1 { (current_price - v.entry_price) / v.entry_price } else { (v.entry_price - current_price) / v.entry_price };
                    if k.1 == 0 { scalp_pnl += pnl * v.qty * current_price; } else { swing_pnl += pnl * v.qty * current_price; }
                }
                
                let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::OmniUpdate {
                    latency_ms: if event_time > 0 { latency_ms as u64 } else { 0 },
                    latency_panic,
                    dark_alpha: dark_router_unified.get_liquidation_cascade_risk(),
                    scalp_pnl,
                    swing_pnl,
                });
                
                let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LatencyUpdate(lat as u64));
            }
            if msg_count % 5000 == 0 {
                println!("⏱️ [TELEMETRY] Processed 5000 ticks/klines. Last tick: {} ns", start.elapsed().as_nanos());
            }
        }
    });

    println!("🔗 Connecting to Binance WebSocket: {} streams", symbols.len());
    
    // Auto-Reconnecting WebSocket Loop
    tokio::spawn(async move {
        let mut retry_count = 0;
        loop {
            println!("🔄 [WS] Attempting connection to Binance...");
            match connect_async(&url).await {
                Ok((ws_stream, _)) => {
                    println!("✅ [WS] WebSocket Connected.");
                    retry_count = 0; // Reset retries on success
                    let _ = tx_events.send("[SYSTEM:RECONNECT]".to_string());
                    let (_, mut read) = ws_stream.split();
                    
                    while let Some(message) = read.next().await {
                        match message {
                            Ok(msg) => {
                                if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                                    let _ = tx_events.send(text);
                                }
                            }
                            Err(e) => {
                                eprintln!("⚠️ [WS] Connection Error: {:?}", e);
                                break; // Break the read loop to trigger reconnection
                            }
                        }
                    }
                    println!("⚠️ [WS] Stream ended. Reconnecting...");
                }
                Err(e) => {
                    eprintln!("❌ [WS] Failed to connect: {:?}", e);
                    retry_count += 1;
                }
            }
            
            // Exponential backoff capped at 5 seconds
            let backoff_ms = std::cmp::min(100 * (2u64.pow(retry_count.min(6))), 5000);
            println!("⏳ [WS] Waiting {}ms before next attempt...", backoff_ms);
            sleep(Duration::from_millis(backoff_ms)).await;
        }
    });

    let _ = unified_handle.await;

    Ok(())
}
