#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use futures_util::StreamExt;
use quantum_engine::parsers;
use std::env;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

use god_engine_core::orchestrator::PhaseOrchestrator;

use execution_engine::executor::ExecutionProvider;
use redb::{Database, ReadableTable, TableDefinition};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
const CAPITAL_TABLE: TableDefinition<u32, f64> = TableDefinition::new("capital_state");

// Removed ActiveExecutor

use quantum_engine::config::TensorConfig;
use telemetry_engine::telemetry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 🛠️ [BOOTLOADER] Inicializar el entorno desde .env
    dotenvy::dotenv().ok();

    // os_guardian init happens inside init_guardian or similar, we just use the module's time_critical if needed
    // FASE 12: Zero-Latency Telemetry Engine
    telemetry_engine::init_telemetry(100_000);
    std::thread::spawn(|| {
        loop {
            if let Some(msg) = telemetry_engine::pop_telemetry() {
                telemetry_server::telemetry_log!("{}", msg); // Asynchronous non-blocking console write
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    });

    telemetry_engine::telemetry!("========================================================");
    telemetry_server::telemetry_log!(
        "🚀 GOD ENGINE - NATIVE RUST ORCHESTRATOR (UNIFIED SCALP/SWING)"
    );
    telemetry_server::telemetry_log!("⚡ Sub-Microsecond Execution Core initialized.");
    telemetry_server::telemetry_log!("🛡️ OS Guardian active: strict resource isolation.");
    telemetry_server::telemetry_log!("========================================================");

    telemetry_server::telemetry_log!("\n========================================================");
    let args: Vec<String> = env::args().collect();
    // FASE 28 + F5.2: El calentamiento cuántico es obligatorio. Siempre inicia en
    // Demo/Testnet salvo --force-live CON armado humano explícito.
    //
    // PUERTA HUMANA DE MAINNET (Plan Maestro 7.4): --force-live solo activa
    // producción si EXISTE el archivo físico `config_dir/MAINNET_ARMED`
    // (creado a mano por el operador tras la certificación demo) y NO existe
    // STOP_TRADING.LOCK. Un flag de CLI jamás volverá a ser suficiente para
    // tocar capital real — el audit halló mainnet alcanzable tras 120 ticks.
    let requested_live = args.contains(&"--force-live".to_string());
    let mainnet_armed = std::path::Path::new("config_dir/MAINNET_ARMED").exists();
    let emergency_lock = std::path::Path::new("STOP_TRADING.LOCK").exists();
    let is_demo_mode = if requested_live && mainnet_armed && !emergency_lock {
        telemetry_server::telemetry_log!(
            "🔴 [MODO PRODUCCION ARMADO] MAINNET_ARMED presente — fuego real AUTORIZADO por el operador."
        );
        false
    } else if requested_live {
        if emergency_lock {
            telemetry_server::telemetry_log!(
                "🛑 [PUERTA MAINNET] STOP_TRADING.LOCK activo — --force-live IGNORADO. Modo demo."
            );
        } else {
            telemetry_server::telemetry_log!(
                "🛑 [PUERTA MAINNET] --force-live sin config_dir/MAINNET_ARMED — capital real PROHIBIDO hasta certificación (Plan 7.4). Cayendo a DEMO."
            );
        }
        true
    } else {
        true
    };

    let darwin_approved = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let orchestrator = Arc::new(std::sync::RwLock::new(PhaseOrchestrator::new(
        120,
        is_demo_mode,
        Arc::clone(&darwin_approved),
    )));

    if is_demo_mode {
        telemetry_server::telemetry_log!(
            "🔥 [MODO WARMUP/DEMO ACTIVO] El sistema inicia en Paper Trading para simulación."
        );
    } else {
        telemetry_server::telemetry_log!(
            "🔴 [MODO PRODUCCION ACTIVO] El sistema inicia en MAINNET con fuego real."
        );
    }

    // FASE 38: Gestión de Memoria Estricta para Host de 16GB (Cero Swapping)
    // Asignamos 6144 MB (6GB) como límite de JobObject y VirtualLock.
    // Esto garantiza 10GB libres para Windows, erradicando por completo el Page Swap.
    // (Movido a la inicialización de arena_real)

    // ==========================================
    // FASE 1 Y 2 (BOOTLOADER CUÁNTICO INTEGRADO)
    // ==========================================
    let (_ntp_offset_ms, symbols) = god_engine_core::bootloader::SystemDiagnostics::execute_phase_1_and_2(false).await.unwrap_or_else(|e| {
        panic!("⚠️ [CRITICAL] Fallo en la Secuencia de Lanzamiento: {}. El sistema no puede operar a ciegas sin datos de red o símbolos. Abortando.", e);
    });

    quantum_arena::symbols::update_dynamic_universe(symbols.clone());

    let config_bytes = std::fs::read("data/dynamic_config.bin").unwrap_or_default();
    let mut config: TensorConfig = bincode::deserialize(&config_bytes).unwrap_or_else(|_| {
        let config_str =
            std::fs::read_to_string("data/dynamic_config.json").unwrap_or_else(|_| "".to_string());
        let cfg: TensorConfig =
            serde_json::from_str(&config_str).unwrap_or_else(|_| TensorConfig {
                symbols: symbols.clone(),
                is_testnet: false, // We always use mainnet data, paper trading intercepts execution
            });
        if let Ok(encoded) = bincode::serialize(&cfg) {
            let _ = std::fs::write("data/dynamic_config.bin", encoded);
        }
        cfg
    });
    config.symbols = symbols.clone(); // Update config memory

    let (telemetry_tx, _) = tokio::sync::broadcast::channel(100);

    // let dashboard_tx = telemetry_tx.clone(); // Removed unused variable
    // [FASE 23] Deprecated old dashboard in favor of zero-copy telemetry_server
    /*
    tokio::spawn(async move {
        if let Err(e) = quantum_engine::dashboard::start_server(dashboard_tx).await {
            telemetry_server::telemetry_log!("❌ [DASHBOARD] Failed to start server: {}", e);
        }
    });
    */

    // Telemetry Async Formatter (Lock-Free Offload)
    // Telemetry Async Formatter (Lock-Free Offload)
    let (tx_log_worker, rx_log_worker) = crossbeam_channel::bounded::<(bool, bool, usize)>(1000);
    let dash_tx_clone = telemetry_tx.clone();
    let symbols_for_log = symbols.clone();
    std::thread::spawn(move || {
        while let Ok((is_scalp, is_long, coin_id)) = rx_log_worker.recv() {
            let side_str = if is_long { "LONG" } else { "SHORT" };
            let parsed_sym = &symbols_for_log[coin_id];
            if is_scalp {
                let _ = dash_tx_clone.send(telemetry_server::TelemetryEvent::LogUpdate(
                    "success".to_string(),
                    format!("⚡ SCALP {} on {}", side_str, parsed_sym),
                ));
            } else {
                let _ = dash_tx_clone.send(telemetry_server::TelemetryEvent::LogUpdate(
                    "success".to_string(),
                    format!("🚀 SWING {} on {}", side_str, parsed_sym),
                ));
            }
        }
    });

    // Spawn Dynamic Symbol Manager (Top 10 Evolver)
    tokio::spawn(async move {
        quantum_engine::symbol_manager::evolve_symbols_daemon().await;
    });

    let mut streams = String::new();
    for (i, sym) in symbols.iter().enumerate() {
        // Flujo pesado para TODO el universo dinámico (30 activos), permitiendo paralelismo masivo
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
    let streams_str = streams.clone();
    let initial_ws_host = if let Ok(ep) = env::var("BEST_WS_ENDPOINT") {
        ep
    } else {
        let is_env_testnet = env::var("USE_TESTNET")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            == "true";
        if is_env_testnet {
            "stream.binancefuture.com".to_string()
        } else {
            let endpoints = vec![
                "fstream.binance.com",
                "fstream-auth.binance.com",
                "dstream.binance.com",
            ];
            let mut best_host = "fstream.binance.com".to_string();
            let mut best_latency = u128::MAX;

            let mut best_ip = String::new();
            for ep in endpoints {
                if let Ok(addrs) = tokio::net::lookup_host((ep, 443)).await {
                    for addr in addrs {
                        let start = tokio::time::Instant::now();
                        if let Ok(_) = tokio::time::timeout(
                            tokio::time::Duration::from_millis(500),
                            tokio::net::TcpStream::connect(addr),
                        )
                        .await
                        {
                            let latency = start.elapsed().as_millis();
                            if latency < best_latency {
                                best_latency = latency;
                                best_host = ep.to_string();
                                best_ip = addr.ip().to_string();
                            }
                        }
                    }
                }
            }
            if !best_ip.is_empty() {
                telemetry_server::telemetry_log!(
                    "🚀 [LATENCY ACCELERATOR] Selected WS IP {} (Host: {}) with {}ms latency",
                    best_ip,
                    best_host,
                    best_latency
                );
                best_host
            } else {
                best_host
            }
        }
    };
    let ws_url = Arc::new(arc_swap::ArcSwap::from_pointee(format!(
        "wss://{}/stream?streams={}",
        initial_ws_host, streams_str
    )));

    let (tx_ws_control, mut rx_ws_control) = tokio::sync::mpsc::channel::<()>(1);
    let (tx_events, rx_events) = crossbeam_channel::bounded::<Vec<u8>>(250_000);

    // FASE 3A: FETCH CLAVES Y CONEXIÓN API REST PARA CHEQUEO DE COMISIONES Y CAPITAL ANTES DEL WARMUP Y ENTRENAMIENTO
    let mut testnet_key = env::var("TESTNET_API_KEY").unwrap_or_default();
    let mut testnet_secret = env::var("TESTNET_SECRET_KEY").unwrap_or_default();
    let mut mainnet_key = env::var("MAINNET_API_KEY").unwrap_or_default();
    let mut mainnet_secret = env::var("MAINNET_SECRET_KEY").unwrap_or_default();

    // Auto-map TESTNET keys
    if testnet_key.is_empty() {
        testnet_key = env::var("BINANCE_TESTNET_API_KEY").unwrap_or_default();
        testnet_secret = env::var("BINANCE_TESTNET_SECRET_KEY").unwrap_or_default();
    }
    // Auto-map MAINNET keys from .env
    if mainnet_key.is_empty() {
        mainnet_key = env::var("BINANCE_API_KEY").unwrap_or_default();
        mainnet_secret = env::var("BINANCE_SECRET_KEY").unwrap_or_default();
    }

    if testnet_key.is_empty() || mainnet_key.is_empty() {
        telemetry_server::telemetry_log!("⚠️ [WARNING] TESTNET_API_KEY or MAINNET_API_KEY not found. Ensure both are set for seamless transition.");
    }

    let is_env_testnet = env::var("USE_TESTNET")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        == "true";
    let (active_key, active_secret, is_testnet) = if is_env_testnet {
        telemetry_server::telemetry_log!(
            "🌐 [ENV DETECTED] Using BINANCE TESTNET for API connections."
        );
        (testnet_key.clone(), testnet_secret.clone(), true)
    } else {
        telemetry_server::telemetry_log!(
            "🌐 [ENV DETECTED] Using BINANCE MAINNET for API connections."
        );
        (mainnet_key.clone(), mainnet_secret.clone(), false)
    };

    let mut order_executor =
        execution_engine::executor::OrderExecutor::new(active_key, active_secret, is_testnet);
    // If we are on Testnet, we want to hit the Testnet API natively, not intercept locally as Paper Trading
    if is_env_testnet {
        order_executor.set_paper_trading(false);
    }
    let exec = Arc::new(arc_swap::ArcSwap::from_pointee(order_executor));

    let initial_capital = loop {
        match exec.load().fetch_account_balance().await {
            Ok(bal) => {
                telemetry_server::telemetry_log!(
                    "🌍 [OMNI-AWARENESS] API Real Balance Extracted: ${:.4}",
                    bal
                );
                break bal;
            }
            Err(e) => {
                telemetry_server::telemetry_log!("⚠️ [CRITICAL] Failed to fetch balance from API (Testnet/Mainnet): {}. Retrying in 5s to enforce 100% Reality Parity...", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    };

    // Inject Dynamic Capital into Institutional Telemetry Projections
    audit_engine::telemetry::update_dynamic_capital(initial_capital);

    let (live_maker, live_taker) = if let Some(first_sym) = symbols.first() {
        loop {
            match exec.load().fetch_commission_rate(first_sym).await {
                Ok((m, t)) => {
                    telemetry_server::telemetry_log!("🌍 [OMNI-AWARENESS] API Real VIP Fees Extracted: Maker {:.4}%, Taker {:.4}%", m * 100.0, t * 100.0);
                    break (m, t);
                }
                Err(e) => {
                    telemetry_server::telemetry_log!("⚠️ [CRITICAL] Failed to fetch fees from API: {}. Retrying in 5s to enforce 100% Reality Parity...", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }
    } else {
        panic!("⚠️ [CRITICAL] No symbols defined. Cannot extract fees.");
    };

    let initial_genome =
        quantum_arena::genome::SuperGenotype::load_or_baseline(live_maker, live_taker);
    telemetry_server::telemetry_log!(
        "🧬 [INIT] Genesis Genome Loaded. DarwinDaemon is ready to evolve."
    );

    let historical_klines = god_engine_core::bootloader::SystemDiagnostics::execute_phase_3_warmup(
        is_demo_mode,
        &symbols,
    )
    .await;

    let global_learning_rate = 0.001 * (1.0 + initial_genome.quantum_mutation_rate);
    let training_epochs = (10.0 * (1.0 + initial_genome.quantum_mutation_rate)).max(5.0) as usize;

    let _swing_nn = god_engine_core::bootloader::SystemDiagnostics::execute_phase_4_training(
        &symbols,
        &historical_klines,
        global_learning_rate,
        training_epochs,
    )
    .await;

    let dark_router = Arc::new(quantum_engine::dark_alpha_router::DarkAlphaRouter::new());

    // Spawn Hyperliquid DEX cascade sniffer
    quantum_engine::dark_alpha_sniffer::spawn_hyperliquid_sniffer(Arc::clone(&dark_router));

    // El genoma inicial fue validado. Aprobamos la transición al Orchestrator.
    darwin_approved.store(true, Ordering::Relaxed);

    let mut forest_timestamps: std::collections::HashMap<String, std::time::SystemTime> =
        std::collections::HashMap::new();

    // Initial load of all models
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        if let Ok(modified) = meta.modified() {
                            forest_timestamps.insert(file_stem.to_string(), modified);
                            let _ = god_engine_core::ml_inference::NanoForest::load_global(
                                file_stem,
                                path.to_str().unwrap(),
                            );
                            telemetry_server::telemetry_log!("🧠 Loaded ML Model: {}", file_stem);
                        }
                    }
                }
            }
        }
    }

    tokio::spawn(async move {
        telemetry_server::telemetry_log!(
            "👀 [HOT-RELOAD] Watcher started. Monitoring DNA and ML Weights..."
        );
        let mut last_config_ts = std::fs::metadata("data/dynamic_config.json")
            .and_then(|m| m.modified())
            .ok();

        loop {
            sleep(Duration::from_secs(10)).await;

            if let Ok(meta) = std::fs::metadata("data/dynamic_config.json") {
                if let Ok(modified) = meta.modified() {
                    if Some(modified) != last_config_ts {
                        last_config_ts = Some(modified);
                        telemetry_server::telemetry_log!("👀 [HOT-RELOAD] dynamic_config.json changed. Symbols update requires restart.");
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
                                        if god_engine_core::ml_inference::NanoForest::load_global(
                                            file_stem,
                                            path.to_str().unwrap(),
                                        )
                                        .is_ok()
                                        {
                                            telemetry_server::telemetry_log!("🔥 [HOT-RELOAD] NanoForest AI Brain hot-swapped for {}!", file_stem);
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

    // FASE 15: Quantum Leap - Automatic Transition from Demo to Live
    // El Hot-Swap ahora se delega completamente y de forma síncrona y cuántica al PhaseOrchestrator
    // dentro del loop principal, eliminando condiciones de carrera y spaghetti code.

    // Initialize Redb Zero-Copy Persistence and Real Balance

    let ic_clone = initial_capital;
    tokio::spawn(async move {
        telemetry_server::telemetry_log!(
            "⏰ [EVOLUTION-TASK] Scheduled to run genetic algorithm every 6 hours."
        );
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(6 * 3600)).await;
            telemetry_server::telemetry_log!(
                "🧬 [EVOLUTION-TASK] Waking up to evolve genetic config..."
            );
            let _ = std::process::Command::new("cargo")
                .args([
                    "run",
                    "--release",
                    "--bin",
                    "evolution",
                    "--",
                    &ic_clone.to_string(),
                ])
                .current_dir(".")
                .spawn()
                .and_then(|mut child| child.wait());
        }
    });

    std::fs::create_dir_all("data").unwrap_or_default();

    let db = Arc::new(
        Database::create("data/state.redb").expect("CRITICAL: Failed to create redb database"),
    );

    if let Ok(write_txn) = db.begin_write() {
        {
            let mut table = write_txn.open_table(CAPITAL_TABLE).unwrap();
            let cap_val = table.get(1).unwrap_or(None).map(|v| v.value());
            if let Some(_val) = cap_val {
                // DB has a value, but we trust the API value over DB now.
            } else {
                let _ = table.insert(1, initial_capital);
            }
        }
        let _ = write_txn.commit();
        telemetry_server::telemetry_log!(
            "💾 [PERSISTENCE] Capital Redb Zero-Copy loaded: ${:.4}",
            initial_capital
        );
    }

    // Shared Atomic Capital Pool (Dynamically scaled from API extraction)
    // Axiom V: Unified Cross-Margin Pool
    let unified_capital = Arc::new(AtomicU64::new(initial_capital.to_bits()));

    // Non-blocking Zero-Copy persistence channel
    let (db_tx, mut db_rx) = mpsc::channel::<(f64, f64)>(5000);
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

    let exec = Arc::clone(&exec);
    let loop_telemetry_tx = telemetry_tx.clone();
    let dark_router_unified = Arc::clone(&dark_router);

    let symbols_clone = symbols.clone();
    let mut symbol_to_id: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (i, sym) in symbols_clone.iter().enumerate() {
        symbol_to_id.insert(sym.to_lowercase(), i);
    }

    // 🔍 FETCH FORENSE DE POSICIONES ACTIVAS CON BINANCE API
    let mut restored_positions = exec.load().fetch_open_positions().await.unwrap_or_default();
    if orchestrator.read().unwrap().is_demo_mode {
        telemetry_server::telemetry_log!("⚠️ [DEMO MODE] Ignorando reconciliación de posiciones REST para mantener Paper Trading.");
        restored_positions.clear();
    }

    // ── F1.11 MODO HEDGE ─────────────────────────────────────────────────────
    // El motor envía positionSide=LONG/SHORT siempre; si la cuenta está en
    // one-way, TODA orden falla con -4061. Verificar/activar antes de operar.
    if !orchestrator.read().unwrap().is_demo_mode {
        match exec.load().ensure_hedge_mode().await {
            Ok(true) => {
                telemetry_server::telemetry_log!("🔀 [PRE-FLIGHT] Cuenta migrada a modo HEDGE")
            }
            Ok(false) => telemetry_server::telemetry_log!("🔀 [PRE-FLIGHT] Modo HEDGE ya activo"),
            Err(e) => telemetry_server::telemetry_log!(
                "🚨 [PRE-FLIGHT] No se pudo garantizar modo hedge: {} — TODA orden fallará (-4061)",
                e
            ),
        }
    }

    // ── F1.7 RECONCILIACIÓN AL ARRANQUE ──────────────────────────────────────
    // positionRisk (verdad del exchange) diff contra OrderRegistry (F1.5).
    // Directiva: saber SIEMPRE si hay posiciones abiertas antes de operar.
    if !orchestrator.read().unwrap().is_demo_mode {
        match exec.load().fetch_position_risk().await {
            Ok(entries) => {
                let report =
                    execution_engine::reconciliation::reconcile(&entries, &exec.load().registry());
                telemetry_server::telemetry_log!("🧾 [RECONCILIACIÓN ARRANQUE] {}", report.summary);
                for p in &report.open_positions {
                    telemetry_server::telemetry_log!(
                        "   📍 {} {} @ {:.4} (uPnL {:+.4}, liq {:.2}, lev {:.0}x)",
                        p.symbol,
                        p.position_amt,
                        p.entry_price,
                        p.unrealized_pnl,
                        p.liquidation_price,
                        p.leverage
                    );
                }
                for s in &report.suspicious_active_orders {
                    telemetry_server::telemetry_log!("   ⚠️ ORDEN SOSPECHOSA: {}", s);
                }
            }
            Err(e) => {
                telemetry_server::telemetry_log!(
                    "⚠️ [RECONCILIACIÓN ARRANQUE] positionRisk no disponible: {}",
                    e
                );
            }
        }
    }

    // ── F1.6 USER-DATA STREAM (fills en tiempo real) ─────────────────────────
    // ORDER_TRADE_UPDATE → OrderRegistry; ACCOUNT_UPDATE → puente de capital.
    // La adopción de posiciones al arena es dueño el mapa de estado (F4.9);
    // aquí garantizamos que el capital dinámico refleje la verdad del exchange.
    {
        struct CapitalBridgeSink;
        impl execution_engine::user_data_stream::AccountSink for CapitalBridgeSink {
            fn on_capital(&self, usdt: f64) {
                audit_engine::telemetry::update_dynamic_capital(usdt);
            }
            fn on_positions(
                &self,
                positions: &[execution_engine::user_data_stream::RemotePosition],
            ) {
                if !positions.is_empty() {
                    let summary: Vec<String> = positions
                        .iter()
                        .map(|p| format!("{} {}", p.symbol, p.position_amt))
                        .collect();
                    telemetry_server::telemetry_log!(
                        "📡 [USER-DATA] Posiciones vivas: {}",
                        summary.join(", ")
                    );
                }
            }
        }
        let streamer = execution_engine::user_data_stream::UserDataStreamer::new(
            exec.load().client().clone(),
            exec.load().registry(),
        )
        .with_sink(std::sync::Arc::new(CapitalBridgeSink));
        tokio::spawn(async move {
            streamer.start().await;
        });
        telemetry_server::telemetry_log!(
            "🔌 [USER-DATA] Stream privado spawned (fills en tiempo real + capital vivo)"
        );
    }

    let rx_events = rx_events;

    // FASE 15: GLOBAL ROI & PnL TRACKERS (Bifurcación Cuántica)
    let mut scalp_gross_pnl = 0.0;
    let mut scalp_pnl = 0.0;
    let mut scalp_fees = 0.0;
    let mut scalp_trades = 0;
    let mut scalp_wins = 0;

    let mut swing_gross_pnl = 0.0;
    let mut swing_pnl = 0.0;
    let mut swing_fees = 0.0;
    let mut swing_trades = 0;
    let mut swing_wins = 0;

    let rt_handle = tokio::runtime::Handle::current();

    let server_time_ms = match exec.load().fetch_server_time().await {
        Ok(time) => {
            telemetry_server::telemetry_log!(
                "⏱️ [NTP SYNC] Binance Server Time retrieved: {}",
                time
            );
            time
        }
        Err(_) => {
            telemetry_server::telemetry_log!(
                "⚠️ [NTP SYNC FAILED] Falling back to local SystemTime"
            );
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64
        }
    };
    let local_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    let ntp_offset_ms = server_time_ms - local_ms;
    telemetry_server::telemetry_log!(
        "⏱️ [NTP SYNC] Local Time: {}, Offset to Binance: {}ms",
        local_ms,
        ntp_offset_ms
    );

    let num_symbols = symbols.len();

    // ── F4.1: OMNI FEATURES REALES EN PRODUCCIÓN ────────────────────────────
    // El swing NN recibía &[0.0;54] SIEMPRE: el subsistema macro era un
    // fantasma — god_engine jamás creó el OmniState. Ahora: estado vivo +
    // pollers REST ligeros (macro FRED/PAXG + sentiment). Los WS cross-exchange
    // pesados quedan desconectados (documentado) — pero la mitad macro del
    // vector es REAL en vivo, con staleness medible.
    let omni_state_live = Arc::new(data_pipeline::omni_multiplexer::OmniState::new());
    {
        let st = Arc::clone(&omni_state_live);
        tokio::spawn(async move {
            data_pipeline::omni_multiplexer::run_macro_rest_poller(st).await;
        });
        let st = Arc::clone(&omni_state_live);
        tokio::spawn(async move {
            data_pipeline::omni_multiplexer::run_sentiment_onchain_poller(st).await;
        });
        telemetry_server::telemetry_log!(
            "🌐 [OMNI] Features macro REALES cableadas al swing NN (FRED/PAXG + sentiment, 60s/120s)"
        );
    }

    let unified_handle = std::thread::Builder::new().stack_size(32 * 1024 * 1024).spawn({
        let loop_ws_url = Arc::clone(&ws_url);
        let loop_streams_str = streams_str.clone();
        let historical_klines = historical_klines.clone();
        let omni_state_hot = Arc::clone(&omni_state_live);
        move || {
        if let Some(core_ids) = core_affinity::get_core_ids() {
            if core_ids.len() > 1 {
                core_affinity::set_for_current(core_ids[1]);
                telemetry_server::telemetry_log!("🔒 [CPU PINNING] Unified Core pinned to CPU Core {}", core_ids[1].id);
            }
        }
        os_guardian::set_current_thread_time_critical();
        telemetry_server::telemetry_log!("🧠 [UNIFIED CORE] Initialized. Target latency: <500ns.");

        // Axiom XIV: Superposición Cuántica - Mentes Aisladas (Scalp vs Swing)
        let arena_real = Arc::new(quantum_arena::GlobalArena::new(initial_capital));
        let arena_shadow = Arc::new(quantum_arena::GlobalArena::new(initial_capital));
        arena_real.config.live_maker_fee.store(live_maker, Ordering::Relaxed);
        arena_real.config.live_taker_fee.store(live_taker, Ordering::Relaxed);
        arena_shadow.config.live_maker_fee.store(live_maker, Ordering::Relaxed);
        arena_shadow.config.live_taker_fee.store(live_taker, Ordering::Relaxed);

        // FASE 38 & 14: Guardian Activo (Memory Panic)
        // F5.3 — FIX AFINIDAD: 0xFFFF asumía 16 cores — en una máquina de 8 el
        // guardian pedía cores inexistentes. Máscara derivada del hardware REAL.
        let cpu_mask = {
            let cores = core_affinity::get_core_ids().map(|c| c.len()).unwrap_or(1);
            if cores >= 64 {
                usize::MAX
            } else {
                (1usize << cores) - 1
            }
        };
        let memory_per_symbol_mb = 128;
        let dynamic_memory_limit = (num_symbols * memory_per_symbol_mb) + 2048; // Escala dinámica basada en el Universo
        os_guardian::init_guardian(cpu_mask, dynamic_memory_limit, Arc::clone(&arena_real));

        unsafe {
            if os_guardian::memory_compaction::lock_critical_memory(&*arena_real) {
                telemetry_server::telemetry_log!("🔒 [OS-GUARDIAN] arena_real asegurada en RAM física (Zero Swapping)");
            }
            if os_guardian::memory_compaction::lock_critical_memory_slice(&*arena_real.coins) {
                telemetry_server::telemetry_log!("🔒 [OS-GUARDIAN] arena_real.coins (TickRings) asegurados en RAM física");
            }
            if os_guardian::memory_compaction::lock_critical_memory(&*arena_shadow) {
                telemetry_server::telemetry_log!("🔒 [OS-GUARDIAN] arena_shadow asegurada en RAM física (Zero Swapping)");
            }
            if os_guardian::memory_compaction::lock_critical_memory_slice(&*arena_shadow.coins) {
                telemetry_server::telemetry_log!("🔒 [OS-GUARDIAN] arena_shadow.coins (TickRings) asegurados en RAM física");
            }
        }

        // ── F5.2: SISTEMA INMUNE — el kill-switch por fin ARMADO ──────────────
        // Auditoría: arena.kill_switch_active NUNCA se armaba (ningún writer en
        // el camino vivo) y STOP_TRADING.LOCK no lo leía nadie (decorativo).
        // Ahora: cada 5s se vigila (1) STOP_TRADING.LOCK del operador,
        // (2) drawdown vs límite del genoma sobre el pico observado,
        // (3) latencia obsoleta SOSTENIDA (3 strikes ≈ 15s).
        // Al disparar: kill-switch en arena + executor + FLATTEN de todo.
        // LATCH: rearme solo reiniciando el proceso — decisión humana.
        {
            let arena_imm = Arc::clone(&arena_real);
            let exec_imm = Arc::clone(&exec);
            rt_handle.spawn(async move {
                let mut latched = false;
                let mut peak_capital = arena_imm.unified_capital.load(Ordering::Relaxed);
                let mut latency_strikes: u32 = 0;
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    if latched {
                        continue;
                    }

                    // (1) El operador manda: el archivo existe ⇒ parar TODO.
                    let operator_lock = std::path::Path::new("STOP_TRADING.LOCK").exists();

                    // (2) Drawdown sobre pico observado (muestreo 5s).
                    let cap = arena_imm.unified_capital.load(Ordering::Relaxed);
                    if cap > peak_capital {
                        peak_capital = cap;
                    }
                    let dd = if peak_capital > 0.0 {
                        (peak_capital - cap) / peak_capital
                    } else {
                        0.0
                    };
                    let max_dd = arena_imm.config.global_max_drawdown.load(Ordering::Relaxed);
                    let dd_breach = cap > 0.0 && max_dd > 0.0 && dd >= max_dd;

                    // (3) Latencia: 3 muestras consecutivas por encima del umbral.
                    let lat = arena_imm.last_ws_latency_ms.load(Ordering::Relaxed);
                    let lat_thresh = arena_imm
                        .config
                        .latency_ms_panic_threshold
                        .load(Ordering::Relaxed)
                        .max(50.0) as u64;
                    if lat > lat_thresh {
                        latency_strikes += 1;
                    } else {
                        latency_strikes = 0;
                    }
                    let lat_breach = latency_strikes >= 3;

                    if operator_lock || dd_breach || lat_breach {
                        let reason = if operator_lock {
                            "STOP_TRADING.LOCK del operador".to_string()
                        } else if dd_breach {
                            format!(
                                "drawdown {:.1}% >= límite {:.1}%",
                                dd * 100.0,
                                max_dd * 100.0
                            )
                        } else {
                            format!("latencia {}ms > {}ms sostenida", lat, lat_thresh)
                        };
                        telemetry_server::telemetry_log!(
                            "🚨 [SISTEMA INMUNE] ACTIVADO: {}. Kill-switch ARMADO + aplanado total.",
                            reason
                        );
                        arena_imm.kill_switch_active.store(true, Ordering::Relaxed);
                        exec_imm.load().trigger_kill_switch();
                        match exec_imm.load().flatten_all_positions().await {
                            Ok((syms, positions)) => telemetry_server::telemetry_log!(
                                "🧹 [SISTEMA INMUNE] Aplanado: {} símbolos con órdenes canceladas, {} posiciones cerradas. Reinicio manual para rearmar.",
                                syms,
                                positions
                            ),
                            Err(e) => telemetry_server::telemetry_log!(
                                "🚨 [SISTEMA INMUNE] Aplanado FALLÓ: {} — INTERVENCIÓN MANUAL URGENTE.",
                                e
                            ),
                        }
                        latched = true;
                    }
                }
            });
        }

        // Update fees dynamically
        arena_real.config.live_maker_fee.store(live_maker, Ordering::Relaxed);
        arena_real.config.live_taker_fee.store(live_taker, Ordering::Relaxed);
        arena_real.server_time_offset_ms.store(ntp_offset_ms, Ordering::Relaxed);

        // Reality Physics: Shadow Simulator uses identical dynamic fees extracted from exchange
        arena_shadow.config.live_maker_fee.store(live_maker, Ordering::Relaxed);
        arena_shadow.config.live_taker_fee.store(live_taker, Ordering::Relaxed);
        arena_shadow.server_time_offset_ms.store(ntp_offset_ms, Ordering::Relaxed);

        // Phase 17: Apply Genotype Object Directly
        telemetry_server::telemetry_log!("🧬 [GENOMA] Applying Active Genome to Unified Core...");
        initial_genome.apply_to_arena(&arena_real);
        initial_genome.apply_to_arena(&arena_shadow);

        // FASE 5 (ADOPCIÓN DE ESTADO: Recuperar posiciones abiertas tras caída)
        telemetry_server::telemetry_log!("========================================================");
        telemetry_server::telemetry_log!("🔍 [FASE 5] ADOPCIÓN DE ESTADO (RECONCILIACIÓN DE POSICIONES)");
        telemetry_server::telemetry_log!("========================================================");
        if restored_positions.is_empty() {
            telemetry_server::telemetry_log!("🔍 [RECONCILIATION] Cero posiciones abiertas en Binance. Arena inicializada limpia.");
        } else {
            telemetry_server::telemetry_log!("🔍 [RECONCILIATION] Detectadas {} posiciones abiertas en Binance API:", restored_positions.len());
            for pos in &restored_positions {
                telemetry_server::telemetry_log!("   👉 Símbolo activo en exchange: {} (Qty: {})", pos.symbol, pos.qty);
                if let Some(&coin_idx) = symbol_to_id.get(&pos.symbol.to_lowercase()) {
                    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
                    arena_real.coins[coin_idx].positions.swing_position.open(pos.is_long, pos.entry_price, pos.qty, 1.0, now_ms, 0.0, 0.0);
                    telemetry_server::telemetry_log!("   ✅ Posición reconciliada en Arena para {}", pos.symbol);
                }
            }
        }

        let arena_telemetry = Arc::clone(&arena_real);
        let telemetry_tx_for_server = telemetry_tx.clone();
        rt_handle.spawn(async move {
            telemetry_server::start_telemetry_server(arena_telemetry, telemetry_tx_for_server).await;
        });

        let (tx_real, rx_real) = std::sync::mpsc::channel();
        let (tx_shadow, _rx_shadow) = std::sync::mpsc::channel();

        let rt_for_darwin = rt_handle.clone();

        // La evaluación de Warmup en hilo separado fue removida para centralizar la orquestación en el PhaseOrchestrator del motor HFT, evitando colisiones.

        // Spawn Auto-Evolucion (Model Watcher)
        rt_for_darwin.spawn(async move {
            telemetry_server::telemetry_log!("🧠 [MODEL WATCHER] Escaneando mutaciones en models/DarkAlpha_BTCUSDT.json cada 10s...");
            let mut last_mtime = std::time::SystemTime::UNIX_EPOCH;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                if let Ok(metadata) = std::fs::metadata("models/DarkAlpha_BTCUSDT.json") {
                    if let Ok(mtime) = metadata.modified() {
                        if mtime > last_mtime {
                            last_mtime = mtime;
                            if let Ok(model) = dark_alpha_engine::DarkAlphaEngine::load_json("models/DarkAlpha_BTCUSDT.json") {
                                telemetry_server::telemetry_log!("🧬 [MODEL WATCHER] Nueva genetica detectada. Desplegando en Zero-Copy...");
                                let _ = tx_real.send(model.clone());
                                let _ = tx_shadow.send(model);
                            }
                        }
                    }
                }
            }
        });

        let rt_for_gc = rt_handle.clone();
        rt_for_gc.spawn(async move {
            telemetry_server::telemetry_log!("🧹 [OS-GUARDIAN] Inicializando Garbage Collector Estadístico (1H interval)...");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
                telemetry_server::telemetry_log!("🧹 [OS-GUARDIAN] Ejecutando Memory Compaction OS-level...");
                unsafe {
                    os_guardian::memory_compaction::force_working_set_compaction();
                }
            }
        });

        // FASE 33: Iniciar Motor Evolutivo en Tiempo Real (Grafo Silencioso Erradicado)
        let daemon = god_engine_core::darwin::DarwinDaemon::new(Arc::clone(&arena_real));
        rt_for_darwin.spawn(async move {
            telemetry_server::telemetry_log!("🧬 [DARWIN-DAEMON] Iniciando Motor Cuántico Evolutivo...");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                let daemon_clone = god_engine_core::darwin::DarwinDaemon::new(Arc::clone(&daemon.live_arena));
                let _ = tokio::task::spawn_blocking(move || {
                    daemon_clone.evolve_online();
                }).await;
            }
        });

        // The PhaseOrchestrator is injected into the execution context
        let _msg_count: u64 = 0;
        let mut has_transitioned = false;
        let mut engine_real = god_engine_core::GodEngineCore::new(Arc::clone(&arena_real));

        // ── F5.1: ENVOLVENTE KELLY BAYESIANA ───────────────────────────────────
        // El leverage YA NO sale de hardcodes (10/5): emerge del posterior del
        // edge con guard de ruina. Sin evidencia ⇒ f=0 ⇒ NO se opera hasta
        // acumular historial (directriz: la matemática decide, no constantes).
        let mut risk_envelope = risk_engine::kelly_envelope::RiskEnvelope::new();
        let mut avg_win_abs: f64 = 0.0;
        let mut avg_loss_abs: f64 = 0.0;
        let mut trade_count_env: u64 = 0;

        engine_real.reality.mode = god_engine_core::reality_physics::EngineMode::Optimistic;
        engine_real.set_model_rx(rx_real);

        // --- PHASE 3 WARMUP INJECTION ---
        telemetry_server::telemetry_log!("📥 [PHASE 3] Inyectando historial REST K-lines para calentar SwingState...");
        for (_i, sym) in symbols_clone.iter().enumerate() {
            if let Some(klines) = historical_klines.get(sym) {
                // Dummy loop over f64 (klines is Vec<f64>)
                for &k in klines.iter() {
                    engine_real.feature_engines[_i].process_kline(k, k, k, k, 0.0);
                }
                telemetry_server::telemetry_log!("   ✅ Inyectadas {} K-lines (1h) para {}", klines.len(), sym);
            }
        }
        // --------------------------------

        let mut shadow_forest = evolution_engine::random_forest::ShadowForest::new(initial_capital, initial_genome.clone(), 10);

        let _drift_auditor = audit_engine::drift_auditor::DriftAuditor::new(0.05);

        let instant_baseline = std::time::Instant::now();
        let local_ms_now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
        let epoch_baseline_ms = local_ms_now + ntp_offset_ms;

        let mut local_orderbook = quantum_engine::orderbook::OrderBook::new("BTCUSDT".to_string());
        let mut msg_count: u64 = 0;
        let mut consecutive_slow_ticks = 0;

        // FASE 6 (EVENT LOOP)
        god_engine_core::bootloader::SystemDiagnostics::execute_phase_6_hft();

        while let Ok(mut msg_bytes) = rx_events.recv() {
            let start = Instant::now();

            let is_trade = memchr::memmem::find(&msg_bytes, b"\"e\":\"trade\"").is_some();
            let is_kline = memchr::memmem::find(&msg_bytes, b"\"e\":\"kline\"").is_some();
            let is_depth = memchr::memmem::find(&msg_bytes, b"\"e\":\"depthUpdate\"").is_some();
            let is_reconnect = msg_bytes == b"[SYSTEM:RECONNECT]";

            if is_reconnect {
                telemetry_server::telemetry_log!("🧹 [AUTO-HEALING] Reconnect signal received. Purging Quantum Engine state to prevent time-glitches...");
                engine_real.reset_engines();
                local_orderbook.clear();
                telemetry_server::telemetry_log!("✅ [AUTO-HEALING] All AI Engines flushed. Entering Warmup Phase (50 ticks).");

                let rx_rest = Arc::clone(&exec);
                rt_handle.spawn(async move {
                    telemetry_server::telemetry_log!("🔄 [REST-SYNC] Fetching truth from Binance API...");
                    if let Ok(positions) = rx_rest.load().fetch_open_positions().await {
                        telemetry_server::telemetry_log!("✅ [REST-SYNC] Binance reports {} active open positions.", positions.len());
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
                    local_orderbook.update_bid(bp, bq);
                    local_orderbook.update_ask(ap, aq);
                    let total_q = bq + aq;
                    if total_q > 0.0 {
                        depth_obi = (bq - aq) / total_q;
                        let microprice = (bp * aq + ap * bq) / total_q;
                        let midprice = (bp + ap) / 2.0;
                        depth_micro_div = if midprice > 0.0 { (microprice - midprice) / midprice } else { 0.0 };
                    }
                }
            }

            // FASE 23: QUANTUM LATENCY KILL-SWITCH (Optimized via TSC)
            let now_ms = epoch_baseline_ms + instant_baseline.elapsed().as_millis() as i64;
            let latency_ms = now_ms - event_time;
            let mut latency_panic = false;

            let panic_threshold = engine_real.arena.config.latency_ms_panic_threshold.load(std::sync::atomic::Ordering::Relaxed) as i64;

            // Phase 4: Synthetic Volatility Kill Switch
            if event_time > 0 && latency_ms > panic_threshold {
                consecutive_slow_ticks += 1;
            } else if event_time > 0 {
                consecutive_slow_ticks = 0;
            }

            if consecutive_slow_ticks >= 10 {
                latency_panic = true;
                if consecutive_slow_ticks == 10 {
                    telemetry_server::telemetry_log!("🚨 [KILL_SWITCH] Latencia > {}ms detectada por 10 ticks. Volatilidad Sintética activada. SCALP DESACTIVADO.", panic_threshold);
                }
            }

            if event_time > 0 && latency_ms > panic_threshold {
                latency_panic = true;
                telemetry_server::telemetry_log!("⚠️ [LATENCY_PANIC] Delta = {}ms (>{panic_threshold}ms limit). Skiping O(1) Scalp execution.", latency_ms);
            }

            if let Some(parsed_sym) = parsed_sym_opt {
                let coin_id = symbol_to_id.get(&parsed_sym.to_lowercase()).copied().unwrap_or(0);

                // --- SANITY CHECKS (DATA INTEGRITY & NORMALIZATION) ---
                if current_price <= 0.0 || qty < 0.0 || current_price.is_nan() || qty.is_nan() {
                    continue; // Drop corrupt data
                }

                // --- OUTLIER REJECTION (> 15% FLASH CRASH FILTER) ---
                let recent_ticks = engine_real.arena.coins[coin_id].tick_ring.snapshot_recent(1);
                if !recent_ticks.is_empty() {
                    let prev_price = recent_ticks[0].bid_price;
                    if prev_price > 0.0 {
                        let jump_pct = (current_price - prev_price).abs() / prev_price;
                        let max_jump = engine_real.arena.config.flash_crash_jump_pct.load(std::sync::atomic::Ordering::Relaxed);
                        if jump_pct > max_jump {
                            telemetry_server::telemetry_log!("⚠️ [DATA INTEGRITY] Dropping anomalous tick for {}! Jump: {:.2}%", parsed_sym, jump_pct * 100.0);
                            continue;
                        }
                    }
                }
                // --- 1. DELEGATE TO UNIFIED GOD ENGINE CORE ---
                // F4.1: features omni REALES (macro FRED/PAXG + sentiment vivos).
                // Antes: &[0.0; 54] — la NN swing evaluaba ceros en producción.
                let omni_features_hot = omni_state_hot.get_features();
                let (new_sc, new_sw, closed_sc, closed_sw) = engine_real.process_event(
                    coin_id, is_trade, is_kline_closed, is_depth,
                    current_price, qty, dbp, dap, dbq, daq,
                    depth_obi, depth_micro_div, event_time as u64, latency_panic, &omni_features_hot
                );

                shadow_forest.broadcast_tick(
                    coin_id, is_trade, is_kline_closed, is_depth,
                    current_price, qty, dbp, dap, dbq, daq,
                    depth_obi, depth_micro_div, event_time as u64,
                    &engine_real.arena
                );

                // --- 2. EXECUTE ORDERS ---
                let current_phase: god_engine_core::orchestrator::SystemPhase;
                let is_trading_allowed: bool;
                let is_paper_trading: bool;
                {
                    let mut orch = orchestrator.write().unwrap();
                    current_phase = orch.on_tick();
                    is_trading_allowed = orch.is_trading_allowed();
                    is_paper_trading = orch.is_paper_trading();
                }

                let newly_transitioned = is_trading_allowed && !has_transitioned;
                if newly_transitioned {
                    has_transitioned = true;
                }

                if !is_trading_allowed {
                    if msg_count > 0 && msg_count.is_multiple_of(5000) {
                        telemetry_server::telemetry_log!("🔥 [ORCHESTRATOR] Syncing buffers... {} ticks (Fase: {:?}).", msg_count, current_phase);
                    }
                } else {
                    if newly_transitioned {
                        telemetry_server::telemetry_log!("✅ [WARMUP COMPLETE] System state synchronized & Darwin Approved. Transitioning to {:?}", current_phase);

                        let is_env_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
                        let (target_key, target_secret, target_is_testnet) = if is_env_testnet {
                            (testnet_key.clone(), testnet_secret.clone(), true)
                        } else {
                            (mainnet_key.clone(), mainnet_secret.clone(), false)
                        };

                        let mut mainnet_executor = execution_engine::executor::OrderExecutor::new(target_key, target_secret, target_is_testnet);
                        if is_env_testnet {
                            mainnet_executor.set_paper_trading(false);
                        } else {
                            mainnet_executor.set_paper_trading(is_paper_trading);
                        }
                        exec.store(Arc::new(mainnet_executor));

                        let base_ws_url = if is_env_testnet { "wss://stream.binancefuture.com/stream" } else { "wss://fstream.binance.com/stream" };
                        loop_ws_url.store(Arc::new(format!("{}?streams={}", base_ws_url, loop_streams_str)));
                        let _ = tx_ws_control.try_send(());
                        let exec_clone = Arc::clone(&exec);
                        let cap_clone = Arc::clone(&unified_capital);
                        let db_tx_clone = db_tx.clone();
                        let arena_real_clone = Arc::clone(&engine_real.arena);
                        rt_handle.spawn(async move {
                            if let Ok(bal) = exec_clone.load().fetch_account_balance().await {
                                telemetry_server::telemetry_log!("🌍 [TRANSITION] Mainnet API Real Balance Extracted: ${:.4}", bal);
                                cap_clone.store(bal.to_bits(), Ordering::Relaxed);
                                arena_real_clone.config.base_capital.store(bal, Ordering::Relaxed);
                                let _ = db_tx_clone.send((bal, 0.0)).await;
                            }

                            if let Ok(real_positions) = exec_clone.load().fetch_open_positions().await {
                                telemetry_server::telemetry_log!("🔍 [TRUTH-SYNC] Reconciling Mainnet API Positions...");
                                for pos in real_positions {
                                    if pos.qty.abs() > 0.0 {
                                        telemetry_server::telemetry_log!("🚨 [GHOST-DETECTED] Binance has an open position on {}: {:.4} (Long: {}). Alerting for manual or auto reconciliation.", pos.symbol, pos.qty, pos.is_long);
                                    }
                                }
                            }
                        });
                    }

                    let unified_cap = f64::from_bits(unified_capital.load(Ordering::Relaxed));

                    // FASE 15: Internal Netting Engine
                    let mut net_qty: f64 = 0.0;
                    let mut max_leverage = 1;
                    let force_maker = false;
                    let maker_price = current_price;

                    if let Some((is_long, pnl, qty)) = closed_sc {
                        let live_maker_fee = engine_real.arena.config.live_maker_fee.load(Ordering::Relaxed);
                        let live_taker_fee = engine_real.arena.config.live_taker_fee.load(Ordering::Relaxed);
                        let fee = (qty * current_price) * (live_maker_fee + live_taker_fee);
                        let net = pnl - fee;
                        // F5.1: alimentar el posterior del edge con CADA cierre real.
                        trade_count_env += 1;
                        if net >= 0.0 {
                            avg_win_abs = if avg_win_abs == 0.0 { net.abs() } else { avg_win_abs * 0.95 + net.abs() * 0.05 };
                        } else {
                            avg_loss_abs = if avg_loss_abs == 0.0 { net.abs() } else { avg_loss_abs * 0.95 + net.abs() * 0.05 };
                        }
                        risk_envelope.record_trade(net > 0.0, avg_win_abs.max(1e-9), -avg_loss_abs.max(1e-9));
                        scalp_gross_pnl += pnl;
                        scalp_pnl += net;
                        scalp_fees += fee;
                        scalp_trades += 1;
                        if net > 0.0 { scalp_wins += 1; }

                        let roi_post_fees = if (qty * current_price) > 0.0 { (net / (qty * current_price)) * 100.0 } else { 0.0 };
                        telemetry!("🛑 [SCALP CORE] CLOSE HIT! Gross PnL: {:.4} | Net PnL: {:.4} | ROI Post-Fees: {:.4}%", pnl, net, roi_post_fees);
                        let _ = db_tx.try_send((unified_cap, 0.0));
                        let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::CapitalUpdate(unified_cap));
                        let ml_prob = engine_real.arena.coins[coin_id].ml_prob.load(Ordering::Relaxed);
                        let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::TradeClosed {
                            coin_id,
                            trade_type: "SCALP".to_string(),
                            pnl: net,
                            roi_pct: roi_post_fees,
                            duration_ms: 0,
                            ml_prob,
                        });

                        let is_long_order = !is_long;
                        net_qty += if is_long_order { qty } else { -qty };
                    }

                    if let Some((is_long, pnl, qty)) = closed_sw {
                        let live_maker_fee = engine_real.arena.config.live_maker_fee.load(Ordering::Relaxed);
                        let live_taker_fee = engine_real.arena.config.live_taker_fee.load(Ordering::Relaxed);
                        let fee = (qty * current_price) * (live_maker_fee + live_taker_fee);
                        let net = pnl - fee;
                        // F5.1: posterior compartido — todo cierre real alimenta el edge.
                        trade_count_env += 1;
                        if net >= 0.0 {
                            avg_win_abs = if avg_win_abs == 0.0 { net.abs() } else { avg_win_abs * 0.95 + net.abs() * 0.05 };
                        } else {
                            avg_loss_abs = if avg_loss_abs == 0.0 { net.abs() } else { avg_loss_abs * 0.95 + net.abs() * 0.05 };
                        }
                        risk_envelope.record_trade(net > 0.0, avg_win_abs.max(1e-9), -avg_loss_abs.max(1e-9));
                        swing_gross_pnl += pnl;
                        swing_pnl += net;
                        swing_fees += fee;
                        swing_trades += 1;
                        if net > 0.0 { swing_wins += 1; }

                        let roi_post_fees = if (qty * current_price) > 0.0 { (net / (qty * current_price)) * 100.0 } else { 0.0 };
                        telemetry!("🛑 [SWING CORE] CLOSE HIT! Gross PnL: {:.4} | Net PnL: {:.4} | ROI Post-Fees: {:.4}%", pnl, net, roi_post_fees);
                        let _ = db_tx.try_send((unified_cap, 0.0));
                        let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::CapitalUpdate(unified_cap));
                        let ml_prob = engine_real.arena.coins[coin_id].ml_prob.load(Ordering::Relaxed);
                        let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::TradeClosed {
                            coin_id,
                            trade_type: "SWING".to_string(),
                            pnl: net,
                            roi_pct: roi_post_fees,
                            duration_ms: 0,
                            ml_prob,
                        });

                        let is_long_order = !is_long;
                        net_qty += if is_long_order { qty } else { -qty };
                    }

                    let _scalp_entry_price = 0.0;
                    let mut scalp_tp_price = 0.0;
                    let mut scalp_sl_price = 0.0;
                    let mut is_high_confidence_scalp = false;

                    if let Some((is_long, entry_price, qty)) = new_sc {
                        // F5.1: leverage por ENVOLVENTE, no hardcode. El stop del
                        // scalp (piso ATR) define el sizing: L = f_riesgo / stop.
                        let cap_now = engine_real.arena.unified_capital.load(Ordering::Relaxed);
                        let scalp_stop_pct = engine_real
                            .arena
                            .config
                            .scalp_sl_base
                            .load(Ordering::Relaxed)
                            .max(engine_real.feature_engines[coin_id].get_atr_pct() * 1.5)
                            .max(0.0015);
                        let (env_lev, operable) =
                            risk_envelope.max_leverage(cap_now, scalp_stop_pct, 5.0, 1.64, 50.0);
                        // Deadlock resuelto por EXPLORACIÓN: sin historial la envolvente
                        // da f=0 ⇒ sin entradas ⇒ sin evidencia ⇒ bucle eterno. Con
                        // <30 trades cerrados: stakes mínimos (leverage 1) para GENERAR
                        // evidencia real; el riesgo de exploración es ~stop×capital.
                        if risk_envelope.posterior.n() < 30.0 {
                            max_leverage = 1;
                        } else {
                            max_leverage = if operable { env_lev.floor().max(1.0) as u32 } else { 0 };
                        }
                        net_qty += if is_long { qty } else { -qty };
                        let _ = tx_log_worker.try_send((true, is_long, coin_id));

                        let ml_prob = engine_real.arena.coins[coin_id].ml_prob.load(Ordering::Relaxed);
                        if ml_prob > 0.80 || ml_prob < 0.20 {
                            is_high_confidence_scalp = true;
                        }

                        // F1.9b: TP/SL genome-driven para TODA entrada scalp (antes
                        // solo alta confianza). Bases del genoma con piso ATR.
                        let base_tp = engine_real.arena.config.scalp_tp_base.load(Ordering::Relaxed);
                        let base_sl = engine_real.arena.config.scalp_sl_base.load(Ordering::Relaxed);
                        let atr_pct = engine_real.feature_engines[coin_id].get_atr_pct();
                        let scalp_tp = base_tp.max(atr_pct * 2.0).max(0.002);
                        let scalp_sl = base_sl.max(atr_pct * 1.5).max(0.0015);

                        scalp_tp_price = if is_long { entry_price * (1.0 + scalp_tp) } else { entry_price * (1.0 - scalp_tp) };
                        scalp_sl_price = if is_long { entry_price * (1.0 - scalp_sl) } else { entry_price * (1.0 + scalp_sl) };
                    }

                    if let Some((is_long, entry_price, qty)) = new_sw {
                        // F5.1: swing también por envolvente — stop ancho (3×ATR)
                        // ⇒ leverage menor que scalp, emergente del sizing.
                        let cap_now = engine_real.arena.unified_capital.load(Ordering::Relaxed);
                        let swing_stop_pct = engine_real
                            .arena
                            .config
                            .swing_sl_base
                            .load(Ordering::Relaxed)
                            .max(engine_real.feature_engines[coin_id].get_atr_pct() * 3.0)
                            .max(0.003);
                        let (env_lev, operable) =
                            risk_envelope.max_leverage(cap_now, swing_stop_pct, 5.0, 1.64, 50.0);
                        // Exploración (<30 trades): stakes mínimos; luego la envolvente manda.
                        if risk_envelope.posterior.n() < 30.0 {
                            max_leverage = 1;
                        } else {
                            max_leverage = if operable { env_lev.floor().max(1.0) as u32 } else { 0 };
                        }
                        net_qty += if is_long { qty } else { -qty };
                        let _ = tx_log_worker.try_send((false, is_long, coin_id));

                        // F1.9b: swing TAMBIÉN lleva TP/SL genome-driven (gen de
                        // horizonte mayor). Si scalp y swing coinciden en el tick,
                        // el swing domina la protección (horizonte más ancho).
                        let base_tp = engine_real.arena.config.swing_tp_base.load(Ordering::Relaxed);
                        let base_sl = engine_real.arena.config.swing_sl_base.load(Ordering::Relaxed);
                        let atr_pct = engine_real.feature_engines[coin_id].get_atr_pct();
                        let swing_tp = base_tp.max(atr_pct * 4.0);
                        let swing_sl = base_sl.max(atr_pct * 3.0);

                        scalp_tp_price = if is_long { entry_price * (1.0 + swing_tp) } else { entry_price * (1.0 - swing_tp) };
                        scalp_sl_price = if is_long { entry_price * (1.0 - swing_sl) } else { entry_price * (1.0 + swing_sl) };
                    }

                    if net_qty.abs() > 0.0 {
                        let parsed_sym_str = parsed_sym.to_string();
                        let exec_clone = Arc::clone(&exec);
                        let final_is_long = net_qty > 0.0;
                        let final_qty = net_qty.abs();

                        let iceberg_threshold = engine_real.arena.config.iceberg_volume_threshold.load(Ordering::Relaxed);
                        let iceberg_slices = engine_real.arena.config.iceberg_slice_count.load(Ordering::Relaxed).max(2.0);
                        let notional_volume = final_qty * current_price;

                        rt_handle.spawn(async move {
                            // F5.1: la envolvente (LCB sin evidencia, o capital que
                            // no sostiene el riesgo mínimo) dijo NO — sin orden.
                            if max_leverage == 0 {
                                telemetry_engine::telemetry!(
                                    "🛡️ [ENVOLVENTE] Entrada bloqueada: evidencia insuficiente o capital no sostiene el riesgo mínimo (Kelly bayesiano)"
                                );
                                return;
                            }
                            if max_leverage > 1 {
                                let _ = exec_clone.load().set_leverage(&parsed_sym_str, max_leverage).await;
                            }
                            // F1.9b: capturar resultado de la entrada — jamás tragado.
                            let entry_result: Result<(), String>;
                            if force_maker {
                                let mut id_buf = [0u8; 32];
                                id_buf[0..3].copy_from_slice(b"mc_");
                                let micros = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
                                let mut itoa_buf = itoa::Buffer::new();
                                let micros_str = itoa_buf.format(micros);
                                id_buf[3..3 + micros_str.len()].copy_from_slice(micros_str.as_bytes());
                                let order_id = unsafe { std::str::from_utf8_unchecked(&id_buf[..3 + micros_str.len()]) };
                                entry_result = exec_clone.load().execute_maker_chase(&parsed_sym_str, final_is_long, final_qty, maker_price, 0.001, 0.0001, &order_id).await;
                            } else if notional_volume > iceberg_threshold {
                                let iceberg_qty = final_qty / iceberg_slices;
                                telemetry_engine::telemetry!("🧊 [ICEBERG ROUTER] Fragmentando orden institucional ({} USDT) en pedazos de {}...", notional_volume, iceberg_qty);
                                entry_result = exec_clone.load().execute_iceberg_limit(&parsed_sym_str, final_is_long, final_qty, maker_price, iceberg_qty, 0.001, 0.0001, "iceberg_01").await;
                            } else {
                                entry_result = exec_clone.load().execute_raw_qty(&parsed_sym_str, final_is_long, final_qty, 0.001).await;
                            }

                            match entry_result {
                                Err(e) => {
                                    // Entrada fallida: SIN OCO (proteger una posición
                                    // inexistente crearía una posición inversa desnuda).
                                    telemetry_engine::telemetry!(
                                        "❌ [ENTRY] {} rechazada: {}",
                                        parsed_sym_str, e
                                    );
                                }
                                Ok(()) => {
                                    // F1.9b: TODA posición lleva TP/SL genome-driven.
                                    // (antes: solo scalps de alta confianza — el resto naked)
                                    if scalp_tp_price > 0.0 && scalp_sl_price > 0.0 {
                                        let tag = if is_high_confidence_scalp { "🎯 [OCO TENSOR]" } else { "🛡️ [OCO GUARD]" };
                                        telemetry_engine::telemetry!(
                                            "{} Protección para {} (TP: {:.4}, SL: {:.4})",
                                            tag, parsed_sym_str, scalp_tp_price, scalp_sl_price
                                        );
                                        let is_long_close = final_is_long; // si entramos LONG, cerramos con SELL

                                        let mut id_buf = [0u8; 32];
                                        id_buf[0..4].copy_from_slice(b"oco_");
                                        let micros = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
                                        let mut itoa_buf = itoa::Buffer::new();
                                        let micros_str = itoa_buf.format(micros);
                                        id_buf[4..4 + micros_str.len()].copy_from_slice(micros_str.as_bytes());
                                        let base_id = unsafe { std::str::from_utf8_unchecked(&id_buf[..4 + micros_str.len()]) };

                                        // tick/step 0.001 de fallback; F1.0 traerá el tickSize real por símbolo.
                                        if let Err(oco_err) = exec_clone.load().execute_oco_order(&parsed_sym_str, is_long_close, final_qty, scalp_tp_price, scalp_sl_price, 0.001, 0.0001, &base_id).await {
                                            telemetry_engine::telemetry!(
                                                "🚨 [OCO GUARD] {} quedó SIN protección: {} — aplanar manual o alertar",
                                                parsed_sym_str, oco_err
                                            );
                                        }
                                    }
                                }
                            }
                        });

                        telemetry!("⚡ [NETTING ENGINE] Orden Neta enviada a Binance (qty: {:.4}, is_long: {}, force_maker: {}).", final_qty, final_is_long, force_maker);
                    }
                }
            }

            msg_count += 1;
            if msg_count.is_multiple_of(100) {
                let limit_10s = engine_real.arena.config.executor_max_orders_10s.load(Ordering::Relaxed) as usize;
                let limit_w1m = engine_real.arena.config.executor_max_weight_1m.load(Ordering::Relaxed) as usize;
                exec.load().set_rate_limit_thresholds(limit_w1m, limit_10s, 1100);

                let lat = start.elapsed().as_nanos();

                let mut scalp_unrealized_pnl = 0.0;
                let mut swing_unrealized_pnl = 0.0;

                for coin in engine_real.arena.coins.iter() {
                    if coin.positions.scalp_position.is_open() {
                        scalp_unrealized_pnl += coin.scalp.pnl_unrealized.load(Ordering::Relaxed);
                    }
                    if coin.positions.swing_position.is_open() {
                        swing_unrealized_pnl += coin.swing.pnl_unrealized.load(Ordering::Relaxed);
                    }
                }

                let total_trades = scalp_trades + swing_trades;
                let total_wins = scalp_wins + swing_wins;
                let scalp_wr = if scalp_trades > 0 { (scalp_wins as f64 / scalp_trades as f64) * 100.0 } else { 0.0 };
                let swing_wr = if swing_trades > 0 { (swing_wins as f64 / swing_trades as f64) * 100.0 } else { 0.0 };
                let win_rate = if total_trades > 0 { (total_wins as f64 / total_trades as f64) * 100.0 } else { 0.0 };

                // Update Central Profiler for Financial Dashboarding (Zero-Allocation Atomic)
                telemetry_server::profiler::update_financials_atomic(
                    scalp_gross_pnl + swing_gross_pnl,
                    scalp_pnl + swing_pnl,
                    win_rate,
                    f64::from_bits(unified_capital.load(std::sync::atomic::Ordering::Relaxed))
                );

                // FASE 18: Emitir al Zero-Copy Bus (Ring Buffer pre-allocado)
                // Cero locks, cero allocations. 3-5ns delay en lugar de milisegundos.
                let mut payload = [0.0; 6];
                // FASE 19: Métricas Institucionales (Net ROI & Split WR)
                payload[0] = scalp_pnl; // NET PnL Scalp
                payload[1] = swing_pnl; // NET PnL Swing
                payload[2] = scalp_unrealized_pnl;
                payload[3] = swing_unrealized_pnl;
                payload[4] = scalp_wr;
                payload[5] = swing_wr;

                if latency_panic {
                    telemetry_server::zero_copy_bus::GLOBAL_TELEMETRY.emit(
                        telemetry_server::zero_copy_bus::SUBSYSTEM_GOD_ENGINE,
                        telemetry_server::zero_copy_bus::EVT_LATENCY_PANIC,
                        0,
                        [lat as f64, 0.0, 0.0, 0.0, 0.0, 0.0]
                    );
                }

                telemetry_server::zero_copy_bus::GLOBAL_TELEMETRY.emit(
                    telemetry_server::zero_copy_bus::SUBSYSTEM_GOD_ENGINE,
                    telemetry_server::zero_copy_bus::EVT_OMNI_UPDATE_FAST,
                    0,
                    payload
                );

                // Enviar también al ws clásico temporalmente si es estricto, o preferiblemente
                // dejar que un background task procese el ring buffer.
                // Como es refactorización cuántica, delegamos OmniUpdate al background.
                let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::OmniUpdate {
                    latency_ms: if event_time > 0 { latency_ms as u64 } else { 0 },
                    latency_panic,
                    dark_alpha: dark_router_unified.get_liquidation_cascade_risk(),
                    scalp_pnl: scalp_unrealized_pnl,
                    swing_pnl: swing_unrealized_pnl,
                    gross_pnl: scalp_gross_pnl + swing_gross_pnl,
                    net_pnl: scalp_pnl + swing_pnl,
                    win_rate,
                    trade_duration_avg: 0.0,
                });

                let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::LatencyUpdate(lat as u64));

                // FASE 8: Legacy process respawn removed in favor of in-memory Hot-Swap via PhaseOrchestrator.
            }
            if msg_count.is_multiple_of(5000) {
                telemetry_server::telemetry_log!("⏱️ [TELEMETRY] Processed 5000 ticks/klines. Cumulative Fees (Scalp: ${:.4}, Swing: ${:.4}). Last tick: {} ns", scalp_fees, swing_fees, start.elapsed().as_nanos());

                // FASE 12: Cosecha Cuántica en vivo (ShadowForest)
                let (winner, leaderboard) = shadow_forest.harvest_best_genome();
                let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::ShadowLeaderboard(leaderboard));

                if let Some((new_alpha, pnl_gained)) = winner {
                    telemetry!("🧬 [SHADOW FOREST] ¡Cosecha Exitosa! Universo Mutante generó +${:.2} extra. Aplicando Hot-Swap...", pnl_gained);
                    new_alpha.apply_to_arena(&engine_real.arena);
                    shadow_forest.replant(new_alpha.clone());

                    let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::GenomeUpdate(Box::new(new_alpha)));
                } else {
                    // Si no hubo cosecha, enviamos el genoma actual
                    let current_genome = quantum_arena::genome::SuperGenotype::current_from_arena(&engine_real.arena);
                    let _ = loop_telemetry_tx.send(telemetry_server::TelemetryEvent::GenomeUpdate(Box::new(current_genome)));
                }
            }
        }
        telemetry_server::telemetry_log!("✅ [UNIFIED CORE] Unified Event Loop safely terminated.");
        }
    }).unwrap();

    telemetry_server::telemetry_log!(
        "🔗 Connecting to Binance WebSocket: {} streams",
        symbols.len()
    );

    // Auto-Reconnecting WebSocket Loop (Isolated & Pinned)
    std::thread::Builder::new().name("ws-reader".to_string()).spawn(move || {
        if let Some(core_ids) = core_affinity::get_core_ids() {
            if core_ids.len() > 2 {
                core_affinity::set_for_current(core_ids[2]);
                telemetry_server::telemetry_log!("🔒 [CPU PINNING] WS Reader pinned to CPU Core {}", core_ids[2].id);
            }
        }
        os_guardian::set_current_thread_time_critical();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let mut retry_count = 0;
        loop {
            let current_url = ws_url.load().to_string();
            let url = url::Url::parse(&current_url).expect("Invalid WS URL");
            let host = url.host_str().unwrap_or("stream.binancefuture.com");
            let port = url.port_or_known_default().unwrap_or(443);

            // FASE 8: Happy Eyeballs TCP Parallel Connection
            // Resolve IPs dynamically instead of hardcoding AWS endpoints
            let host_port = format!("{}:{}", host, port);
            let mut resolved_addrs = match tokio::net::lookup_host(&host_port).await {
                Ok(addrs) => addrs.collect::<Vec<std::net::SocketAddr>>(),
                Err(e) => {
                    telemetry_server::telemetry_log!("⚠️ [WS] DNS Resolution failed for {}: {}", host_port, e);
                    vec![]
                }
            };

            // Allow manual IP injection via ENV to bypass DNS if desired
            if let Ok(env_ips) = std::env::var("BINANCE_WS_IPS") {
                for ip_str in env_ips.split(',') {
                    if let Ok(addr) = ip_str.trim().parse::<std::net::SocketAddr>() {
                        if !resolved_addrs.contains(&addr) {
                            resolved_addrs.push(addr);
                        }
                    }
                }
            }

            if resolved_addrs.is_empty() {
                // Safe fallback in worst case
                if let Ok(fallback) = format!("{}:{}", host, port).parse::<std::net::SocketAddr>() {
                    resolved_addrs.push(fallback);
                } else {
                    telemetry_server::telemetry_log!("❌ [WS] No IPs could be resolved or parsed. Retrying...");
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    continue;
                }
            }

            telemetry_server::telemetry_log!("🔄 [WS] Parallel TCP Connection Race (Happy Eyeballs) to {} endpoints...", resolved_addrs.len());

            let mut tasks = Vec::new();
            for addr in resolved_addrs.iter() {
                tasks.push(Box::pin(tokio::net::TcpStream::connect(*addr)));
            }

            // For TLS SNI, we still pass the original URL with the hostname (e.g. stream.binancefuture.com)
            // but the underlying TCP stream is connected directly to the fastest raw IP.
            let tcp_stream = match futures_util::future::select_ok(tasks).await {
                Ok((stream, _)) => stream,
                Err(e) => {
                    telemetry_server::telemetry_log!("❌ [WS] Happy Eyeballs Parallel Connect failed on all IPs: {}", e);
                    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                    continue;
                }
            };

            let target_addr = tcp_stream.peer_addr().unwrap();
            telemetry_server::telemetry_log!("✅ [WS] Fast-Lane TCP Connection established to {}...", target_addr);

            // FASE 22: Conexión directa TCP con Zero-Nagle para latencia nula
            let _ = tcp_stream.set_nodelay(true);

            match tokio_tungstenite::client_async_tls(url, tcp_stream).await {
                Ok((ws_stream, _)) => {
                    telemetry_server::telemetry_log!("✅ [WS] WebSocket TLS Connected with TCP_NODELAY.");
                    retry_count = 0; // Reset retries on success
                    let _ = tx_events.send(b"[SYSTEM:RECONNECT]".to_vec());
                    let (_, mut read) = ws_stream.split();

                    loop {
                        tokio::select! {
                            msg_opt = tokio::time::timeout(std::time::Duration::from_secs(5), read.next()) => {
                                match msg_opt {
                                    Ok(Some(Ok(msg))) => {
                                        let _ = tx_events.send(msg.into_data());
                                    }
                                    Ok(Some(Err(e))) => {
                                        telemetry_server::telemetry_log!("⚠️ [WS] Connection Error: {:?}", e);
                                        break;
                                    }
                                    Ok(None) => {
                                        telemetry_server::telemetry_log!("⚠️ [WS] Stream ended.");
                                        break;
                                    }
                                    Err(_) => {
                                        telemetry_server::telemetry_log!("🚨 [WS] Watchdog Timeout: No data received for 5 seconds! Forcing reconnect to prevent Zombie Stream.");
                                        break;
                                    }
                                }
                            }
                            _ = rx_ws_control.recv() => {
                                telemetry_server::telemetry_log!("🔌 [WS] Control signal received. Dropping connection to reconnect to new URL...");
                                break;
                            }
                        }
                    }
                    telemetry_server::telemetry_log!("⚠️ [WS] Reconnecting...");
                }
                Err(e) => {
                    telemetry_server::telemetry_log!("❌ [WS] TLS Handshake failed: {:?}", e);
                    retry_count += 1;
                }
            }

            // Exponential backoff capped at 5 seconds
            let backoff_ms = std::cmp::min(100 * (2u64.pow(retry_count.min(6))), 5000);
            telemetry_server::telemetry_log!("⏳ [WS] Waiting {}ms before next attempt...", backoff_ms);
            sleep(Duration::from_millis(backoff_ms)).await;
        }
        });
    }).unwrap();

    let _ = unified_handle.join();

    Ok(())
}
