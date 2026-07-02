use tokio_tungstenite::connect_async;
use futures_util::StreamExt;
use std::env;
use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use quantum_engine::parsers;

use execution_engine::executor::ExecutionProvider;
use std::time::Instant;
use std::sync::atomic::{AtomicU32, Ordering, AtomicU64};
use redb::{Database, TableDefinition, ReadableTable};
const CAPITAL_TABLE: TableDefinition<u32, f64> = TableDefinition::new("capital_state");

enum ActiveExecutor {
    Real(execution_engine::executor::OrderExecutor),
    Shadow(execution_engine::shadow::ShadowExecutor),
}

impl ExecutionProvider for ActiveExecutor {
    async fn execute_order(
        &self,
        order: &risk_engine::ValidatedOrder,
        symbol: &str,
        current_price: f64,
        step_size: f64,
    ) -> Result<(), String> {
        match self {
            Self::Real(e) => e.execute_order(order, symbol, current_price, step_size).await,
            Self::Shadow(e) => e.execute_order(order, symbol, current_price, step_size).await,
        }
    }
    async fn execute_raw_qty(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        step_size: f64,
    ) -> Result<(), String> {
        match self {
            Self::Real(e) => e.execute_raw_qty(symbol, is_long, quantity, step_size).await,
            Self::Shadow(e) => e.execute_raw_qty(symbol, is_long, quantity, step_size).await,
        }
    }
    async fn execute_limit_order(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        match self {
            Self::Real(e) => e.execute_limit_order(symbol, is_long, quantity, price, step_size, tick_size, client_order_id).await,
            Self::Shadow(e) => e.execute_limit_order(symbol, is_long, quantity, price, step_size, tick_size, client_order_id).await,
        }
    }
    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String> {
        match self {
            Self::Real(e) => e.cancel_order(symbol, client_order_id).await,
            Self::Shadow(e) => e.cancel_order(symbol, client_order_id).await,
        }
    }
    async fn fetch_open_positions(&self) -> Result<Vec<String>, String> {
        match self {
            Self::Real(e) => e.fetch_open_positions().await,
            Self::Shadow(e) => e.fetch_open_positions().await,
        }
    }
    
    fn trigger_kill_switch(&self) {
        match self {
            Self::Real(e) => e.trigger_kill_switch(),
            Self::Shadow(e) => e.trigger_kill_switch(),
        }
    }
}

use quantum_engine::config::DynamicConfig;



#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================");
    println!("🚀 GOD ENGINE - NATIVE RUST ORCHESTRATOR (UNIFIED SCALP/SWING)");
    println!("⚡ Sub-Microsecond Execution Core initialized.");
    println!("🛡️ OS Guardian active: strict resource isolation.");
    println!("========================================================");

    // Initialize OS Guardian: 16 threads (0xFFFF), 8192 MB Memory Limit
    os_guardian::init_guardian(0xFFFF, 8192);
    
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
        streams.push_str(sym);
        streams.push_str("@trade/");
        streams.push_str(sym);
        streams.push_str("@depth5/");
        streams.push_str(sym);
        streams.push_str("@kline_1h");
        if i < symbols.len() - 1 {
            streams.push('/');
        }
    }
    let url = format!("wss://fstream.binance.com/stream?streams={}", streams);

    let (tx_events, rx_events) = crossbeam_channel::unbounded::<Vec<u8>>();

    let api_key = env::var("BINANCE_API_KEY").unwrap_or_else(|_| "".to_string());
    let secret_key = env::var("BINANCE_SECRET_KEY").unwrap_or_else(|_| "".to_string());
    
    let shadow_mode = api_key == "shadow";
    if !shadow_mode && (api_key.is_empty() || api_key == "test_key") {
        panic!("🚨 [FATAL ERROR] Producción Abortada: Falta BINANCE_API_KEY real en el entorno.");
    }
    if !shadow_mode && (secret_key.is_empty() || secret_key == "test_secret") {
        panic!("🚨 [FATAL ERROR] Producción Abortada: Falta BINANCE_SECRET_KEY real en el entorno.");
    }

    // MAINNET ACTIVATION / SHADOW MODE
    let exec = Arc::new(if shadow_mode {
        println!("👻 [SHADOW MODE] Activando ExecutionProvider simulado.");
        ActiveExecutor::Shadow(execution_engine::shadow::ShadowExecutor::new())
    } else {
        ActiveExecutor::Real(execution_engine::executor::OrderExecutor::new(api_key.clone(), secret_key.clone()))
    });
    
    // UDS Removed for native speed optimization
    
    let dark_router = Arc::new(quantum_engine::dark_alpha_router::DarkAlphaRouter::new());
    
    // Spawn Hyperliquid DEX cascade sniffer
    quantum_engine::dark_alpha_sniffer::spawn_hyperliquid_sniffer(Arc::clone(&dark_router));
    
    // Initialize Redb Zero-Copy Persistence
    let mut initial_capital = 13.0_f64;
    std::fs::create_dir_all("data").unwrap_or_default();
    
    let db = Arc::new(Database::create("data/state.redb").expect("CRITICAL: Failed to create redb database"));
    
    if let Ok(write_txn) = db.begin_write() {
        {
            let mut table = write_txn.open_table(CAPITAL_TABLE).unwrap();
            let cap_val = table.get(1).unwrap_or(None).map(|v| v.value());
            if let Some(val) = cap_val {
                initial_capital = val;
            } else {
                let _ = table.insert(1, initial_capital);
            }
        }
        let _ = write_txn.commit();
        println!("💾 [PERSISTENCE] Capital Redb Zero-Copy loaded: ${:.4}", initial_capital);
    }
    
    // Shared Atomic Capital Pool ($13 base limit tracked persistently)
    // Axiom V: Unified Cross-Margin Pool
    let unified_capital = Arc::new(AtomicU64::new(initial_capital.to_bits()));
    
    // Non-blocking Zero-Copy persistence channel
    let (db_tx, mut db_rx) = mpsc::unbounded_channel::<(f64, f64)>();
    let db_clone = Arc::clone(&db);
    tokio::spawn(async move {
        while let Some((sc, sw)) = db_rx.recv().await {
            let total = sc + sw;
            if let Ok(write_txn) = db_clone.begin_write() {
                if let Ok(mut table) = write_txn.open_table(CAPITAL_TABLE) {
                    let _ = table.insert(1, total);
                }
                let _ = write_txn.commit();
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
                            let _ = god_engine_core::ml_inference::NanoForest::load_global(file_stem, path.to_str().unwrap());
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
                                        if god_engine_core::ml_inference::NanoForest::load_global(file_stem, path.to_str().unwrap()).is_ok() {
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
                .args(["run", "--release", "--bin", "evolution_engine"])
                .current_dir(".")
                .spawn()
                .and_then(|mut child| child.wait());
        }
    });
    
    let exec = Arc::clone(&exec);
    let loop_telemetry_tx = telemetry_tx.clone();
    let dark_router_unified = Arc::clone(&dark_router);
    
    let symbols_clone = symbols.clone();
    let mut symbol_to_id: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, sym) in symbols_clone.iter().enumerate() {
        symbol_to_id.insert(sym.to_lowercase(), i);
    }
    
    let rt_handle = tokio::runtime::Handle::current();
    
    let unified_handle = std::thread::Builder::new().stack_size(32 * 1024 * 1024).spawn(move || {
        if let Some(core_ids) = core_affinity::get_core_ids() {
            if core_ids.len() > 1 {
                core_affinity::set_for_current(core_ids[1]);
                println!("🔒 [CPU PINNING] Unified Core pinned to CPU Core {}", core_ids[1].id);
            }
        }
        println!("🧠 [UNIFIED CORE] Initialized. Target latency: <500ns.");
        
        // Axiom XIV: Superposición Cuántica - Mentes Aisladas (Scalp vs Swing)
        let arena_real = Arc::new(quantum_arena::GlobalArena::new(initial_capital));
        let arena_shadow = Arc::new(quantum_arena::GlobalArena::new(initial_capital));
        
        // Reality Physics: Shadow Simulator Penalty
        arena_shadow.config.sim_fee_rate.store(0.0004, Ordering::Relaxed);
        
        if let Ok(genome_str) = std::fs::read_to_string("config_dir/genotypes/active_genome.json") {
            if let Ok(genome_val) = serde_json::from_str::<serde_json::Value>(&genome_str) {
                println!("🧬 [GENOMA] Applying Active Genome to Unified Core...");
                let apply_config = |arena: &Arc<quantum_arena::GlobalArena>| {
                    if let Some(tp) = genome_val["scalp_tp"].as_f64() { arena.config.scalp_tp_base.store(tp, Ordering::Relaxed); }
                    if let Some(sl) = genome_val["scalp_sl"].as_f64() { arena.config.scalp_sl_base.store(sl, Ordering::Relaxed); }
                    if let Some(sw_tp) = genome_val["swing_tp"].as_f64() { arena.config.swing_tp_base.store(sw_tp, Ordering::Relaxed); }
                    if let Some(sw_sl) = genome_val["swing_sl"].as_f64() { arena.config.swing_sl_base.store(sw_sl, Ordering::Relaxed); }
                    if let Some(mc) = genome_val["min_confidence"].as_f64() { 
                        arena.config.scalp_obi_threshold.store(mc, Ordering::Relaxed); 
                        arena.config.ml_threshold_long.store(mc, Ordering::Relaxed);
                        arena.config.ml_threshold_short.store(mc, Ordering::Relaxed);
                    }
                    if let Some(tt) = genome_val["trend_threshold"].as_f64() { arena.config.trend_threshold.store(tt, Ordering::Relaxed); }
                    if let Some(gl) = genome_val["global_leverage"].as_f64() { arena.config.global_leverage.store(gl, Ordering::Relaxed); }
                    if let Some(cs) = genome_val["capital_split_scalp"].as_f64() { arena.config.capital_split_scalp.store(cs, Ordering::Relaxed); }
                };
                apply_config(&arena_real);
                apply_config(&arena_shadow);
            }
        }

        let (tx_real, rx_real) = std::sync::mpsc::channel();
        let (tx_shadow, rx_shadow) = std::sync::mpsc::channel();
        
        let arena_for_darwin = Arc::clone(&arena_real);
        let rt_for_darwin = rt_handle.clone();
        
        // Spawn Auto-Evolucion (Model Watcher)
        rt_for_darwin.spawn(async move {
            println!("🧠 [MODEL WATCHER] Escaneando mutaciones en models/DarkAlpha_BTCUSDT.json cada 10s...");
            let mut last_mtime = std::time::SystemTime::UNIX_EPOCH;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                if let Ok(metadata) = std::fs::metadata("models/DarkAlpha_BTCUSDT.json") {
                    if let Ok(mtime) = metadata.modified() {
                        if mtime > last_mtime {
                            last_mtime = mtime;
                            if let Ok(model) = dark_alpha_engine::DarkAlphaEngine::load_json("models/DarkAlpha_BTCUSDT.json") {
                                println!("🧬 [MODEL WATCHER] Nueva genetica detectada. Desplegando en Zero-Copy...");
                                let _ = tx_real.send(model.clone());
                                let _ = tx_shadow.send(model);
                            }
                        }
                    }
                }
            }
        });
        
        rt_handle.spawn(async move {
            println!("🧬 [DARWIN DAEMON] Initialized. Online Auto-Evolution is ACTIVE.");
            let daemon = god_engine_core::darwin::DarwinDaemon::new(arena_for_darwin);
            loop {
                // Wait for the memory sliding window to accumulate ticks
                tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;
                
                // Spawn a blocking task so we don't stall the async tokio runtime with GA compute
                tokio::task::spawn_blocking({
                    let daemon_clone = god_engine_core::darwin::DarwinDaemon::new(daemon.live_arena.clone());
                    move || {
                        daemon_clone.evolve_online();
                    }
                }).await.unwrap();
            }
        });

        let mut engine_real = god_engine_core::GodEngineCore::new(Arc::clone(&arena_real));
        engine_real.reality.mode = god_engine_core::reality_physics::EngineMode::Optimistic;
        engine_real.set_model_rx(rx_real);

        let mut engine_shadow = god_engine_core::GodEngineCore::new(Arc::clone(&arena_shadow));
        engine_shadow.reality.mode = god_engine_core::reality_physics::EngineMode::HyperRealistic;
        engine_shadow.set_model_rx(rx_shadow);

        let drift_auditor = audit_engine::drift_auditor::DriftAuditor::new(0.05);

        let mut msg_count = 0u64;

        while let Ok(mut msg_bytes) = rx_events.recv() {
            let start = Instant::now();
            
            let is_trade = memchr::memmem::find(&msg_bytes, b"\"e\":\"trade\"").is_some();
            let is_kline = memchr::memmem::find(&msg_bytes, b"\"e\":\"kline\"").is_some();
            let is_depth = memchr::memmem::find(&msg_bytes, b"\"e\":\"depthUpdate\"").is_some();
            let is_reconnect = msg_bytes == b"[SYSTEM:RECONNECT]";
            
            if is_reconnect {
                println!("🧹 [AUTO-HEALING] Reconnect signal received. Purging Quantum Engine state to prevent time-glitches...");
                engine_real.reset_engines();
                engine_shadow.reset_engines();
                println!("✅ [AUTO-HEALING] All AI Engines flushed. Entering Warmup Phase (50 ticks).");
                
                let rx_rest = Arc::clone(&exec);
                rt_handle.spawn(async move {
                    println!("🔄 [REST-SYNC] Fetching truth from Binance API...");
                    if let Ok(positions) = rx_rest.fetch_open_positions().await {
                        println!("✅ [REST-SYNC] Binance reports {} active open positions.", positions.len());
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
            let mut dbp = 0.0;
            let mut dap = 0.0;
            let mut dbq = 0.0;
            let mut daq = 0.0;
            
            let msg_str = unsafe { std::str::from_utf8_unchecked_mut(&mut msg_bytes) };
            
            if is_trade {
                if let Some((e, _, p, q, _, sym)) = parsers::parse_binance_trade(msg_str) {
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    current_price = p;
                    qty = q;
                }
            } else if is_kline {
                if let Some((e, sym, _, _, _, p, v, c)) = parsers::parse_binance_kline(msg_str) {
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    current_price = p;
                    qty = v;
                    is_kline_closed = c;
                }
            } else if is_depth {
                if let Some((e, sym, _, bp, bq, ap, aq)) = parsers::parse_binance_depth(msg_str) {
                    event_time = e;
                    parsed_sym_opt = Some(sym);
                    dbp = bp; dap = ap; dbq = bq; daq = aq;
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
            
            if event_time > 0 && latency_ms > 3000 {
                latency_panic = true;
                // println!("⚠️ [LATENCY_PANIC] Delta = {}ms (>3000ms limit). Skiping O(1) Scalp execution.", latency_ms);
            }
            
            if let Some(parsed_sym) = parsed_sym_opt {
                let coin_id = symbol_to_id.get(&parsed_sym.to_lowercase()).copied().unwrap_or(0);
                
                if current_price <= 0.0 {
                    continue;
                }
                // --- 1. DELEGATE TO UNIFIED GOD ENGINE CORE ---
                let (new_sc, new_sw, closed_sc, closed_sw) = engine_real.process_event(
                    coin_id, is_trade, is_kline_closed, is_depth,
                    current_price, qty, dbp, dap, dbq, daq,
                    depth_obi, depth_micro_div, event_time as u64, latency_panic, &[0.0; 54]
                );

                let (_new_sc_s, _new_sw_s, closed_sc_s, closed_sw_s) = engine_shadow.process_event(
                    coin_id, is_trade, is_kline_closed, is_depth,
                    current_price, qty, dbp, dap, dbq, daq,
                    depth_obi, depth_micro_div, event_time as u64, latency_panic, &[0.0; 54]
                );

                // --- 2. EXECUTE ORDERS ---
                let unified_cap = f64::from_bits(unified_capital.load(Ordering::Relaxed));
                
                if let Some((is_long, pnl, qty)) = closed_sc {
                    let parsed_sym_str = parsed_sym.to_string();
                    let exec_clone = Arc::clone(&exec);
                    let is_long_order = !is_long; // opposite of position to close
                    rt_handle.spawn(async move {
                        let _ = exec_clone.execute_raw_qty(&parsed_sym_str, is_long_order, qty, 0.001).await;
                    });
                    
                    // Auditoría de Drift
                    if let Some((_is_long_s, pnl_s, _qty_s)) = closed_sc_s {
                        let real_trade = audit_engine::drift_auditor::TradeResult {
                            symbol_id: coin_id, is_long, entry_price: 0.0, exit_price: current_price, pnl_pct: pnl, timestamp_ms: event_time as u64,
                        };
                        let shadow_trade = audit_engine::drift_auditor::TradeResult {
                            symbol_id: coin_id, is_long, entry_price: 0.0, exit_price: current_price, pnl_pct: pnl_s, timestamp_ms: event_time as u64,
                        };
                        match drift_auditor.audit_execution(&real_trade, &shadow_trade) {
                            Ok(drift) => println!("🔍 [DRIFT AUDITOR] SCALP Validado. Drift: {:.4}%", drift * 100.0),
                            Err(drift) => {
                                println!("🚨 [DRIFT AUDITOR] CIRCUIT BREAKER! Drift inaceptable: {:.4}%", drift * 100.0);
                                exec.trigger_kill_switch();
                            }
                        }
                    }

                    println!("🛑 [SCALP CORE] CLOSE HIT! PnL: {:.4} (qty: {:.4})", pnl, qty);
                    let _ = db_tx.send((unified_cap, 0.0));
                    let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::CapitalUpdate(unified_cap));
                }
                
                if let Some((is_long, pnl, qty)) = closed_sw {
                    let parsed_sym_str = parsed_sym.to_string();
                    let exec_clone = Arc::clone(&exec);
                    let is_long_order = !is_long; // opposite of position to close
                    rt_handle.spawn(async move {
                        let _ = exec_clone.execute_raw_qty(&parsed_sym_str, is_long_order, qty, 0.001).await;
                    });
                    
                    // Auditoría de Drift para Swing
                    if let Some((_is_long_s, pnl_s, _qty_s)) = closed_sw_s {
                        let real_trade = audit_engine::drift_auditor::TradeResult {
                            symbol_id: coin_id, is_long, entry_price: 0.0, exit_price: current_price, pnl_pct: pnl, timestamp_ms: event_time as u64,
                        };
                        let shadow_trade = audit_engine::drift_auditor::TradeResult {
                            symbol_id: coin_id, is_long, entry_price: 0.0, exit_price: current_price, pnl_pct: pnl_s, timestamp_ms: event_time as u64,
                        };
                        match drift_auditor.audit_execution(&real_trade, &shadow_trade) {
                            Ok(drift) => println!("🔍 [DRIFT AUDITOR] SWING Validado. Drift: {:.4}%", drift * 100.0),
                            Err(drift) => {
                                println!("🚨 [DRIFT AUDITOR] CIRCUIT BREAKER! Drift inaceptable: {:.4}%", drift * 100.0);
                                exec.trigger_kill_switch();
                            }
                        }
                    }

                    println!("🛑 [SWING CORE] CLOSE HIT! PnL: {:.4} (qty: {:.4})", pnl, qty);
                    let _ = db_tx.send((unified_cap, 0.0));
                    let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::CapitalUpdate(unified_cap));
                }
                
                if let Some((is_long, _price, qty)) = new_sc {
                    let parsed_sym_str = parsed_sym.to_string();
                    let exec_clone = Arc::clone(&exec);
                    rt_handle.spawn(async move {
                        let _ = exec_clone.execute_raw_qty(&parsed_sym_str, is_long, qty, 0.001).await;
                    });
                    
                    let side_str = if is_long { "LONG" } else { "SHORT" };
                    println!("⚡ [SCALP CORE] {} Signal! WS Order sent (qty: {:.4}).", side_str, qty);
                    let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LogUpdate(
                        "success".to_string(), 
                        format!("⚡ SCALP {} on {}", side_str, parsed_sym)
                    ));
                }
                
                if let Some((is_long, _price, qty)) = new_sw {
                    let parsed_sym_str = parsed_sym.to_string();
                    let exec_clone = Arc::clone(&exec);
                    rt_handle.spawn(async move {
                        let _ = exec_clone.execute_raw_qty(&parsed_sym_str, is_long, qty, 0.001).await;
                    });
                    
                    let side_str = if is_long { "LONG" } else { "SHORT" };
                    println!("🚀 [SWING CORE] {} Signal! WS Order sent (qty: {:.4}).", side_str, qty);
                    let _ = loop_telemetry_tx.send(quantum_engine::dashboard::TelemetryEvent::LogUpdate(
                        "success".to_string(), 
                        format!("🚀 SWING {} on {}", side_str, parsed_sym)
                    ));
                }
            }
            
            msg_count += 1;
            if msg_count.is_multiple_of(100) {
                let lat = start.elapsed().as_nanos();
                
                let mut scalp_pnl = 0.0;
                let mut swing_pnl = 0.0;
                
                for coin in engine_real.arena.coins.iter() {
                    if coin.positions.scalp_position.is_open() {
                        scalp_pnl += coin.scalp.pnl_unrealized.load(Ordering::Relaxed);
                    }
                    if coin.positions.swing_position.is_open() {
                        swing_pnl += coin.swing.pnl_unrealized.load(Ordering::Relaxed);
                    }
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
            if msg_count.is_multiple_of(5000) {
                println!("⏱️ [TELEMETRY] Processed 5000 ticks/klines. Last tick: {} ns", start.elapsed().as_nanos());
            }
        }
        println!("✅ [UNIFIED CORE] Unified Event Loop safely terminated.");
    }).unwrap();

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
                    let _ = tx_events.send(b"[SYSTEM:RECONNECT]".to_vec());
                    let (_, mut read) = ws_stream.split();
                    
                    while let Some(message) = read.next().await {
                        match message {
                            Ok(msg) => {
                                let _ = tx_events.send(msg.into_data());
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

    let _ = unified_handle.join();

    Ok(())
}
