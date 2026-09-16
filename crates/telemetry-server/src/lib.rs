use axum::{
    Extension, Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::{Html, IntoResponse, Json},
    routing::get,
};
pub mod flight_recorder;
pub mod forensic_auditor;
pub mod lock_free_logger;
pub mod lockfree_bus;
pub mod macros;
pub mod profiler;
pub mod stream;
pub mod telegram_bot;
pub mod telemetry_mmap;
pub mod tensor_telemetry;
pub mod zero_copy_bus;
pub mod zero_copy_ring;
pub use flight_recorder::{FlightEvent, FlightRecorder};
pub use forensic_auditor::ForensicAuditor;
use quantum_arena::GlobalArena;
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::Ordering;
pub use telegram_bot::TelegramBot;

#[derive(Clone, Serialize, Debug)]
pub enum TelemetryEvent {
    LatencyUpdate(u64),          // Nanoseconds
    LogUpdate(String, String),   // (type, message) e.g., ("info", "Connected...")
    CapitalUpdate(f64),          // Current capital
    TensorUpdate([f32; 12]),     // 12D State Vector (Scalp)
    SwingTensorUpdate(Vec<f32>), // 34D State Vector (Swing)
    OmniUpdate {
        latency_ms: u64,
        latency_panic: bool,
        dark_alpha: f64,
        scalp_pnl: f64,
        swing_pnl: f64,
        gross_pnl: f64,
        net_pnl: f64,
        win_rate: f64,
        trade_duration_avg: f64,
    },
    GenomeUpdate(Box<quantum_arena::genome::SuperGenotype>), // FASE 17: Non-blocking Genome Telemetry
    ShadowLeaderboard(Vec<f64>), // FASE 13: Live competition leaderboard
    TradeClosed {
        coin_id: usize,
        trade_type: String, // "SCALP" | "SWING"
        pnl: f64,
        roi_pct: f64,
        duration_ms: u64,
        ml_prob: f64,
    },
}

// use quantum_arena::genome::SuperGenotype;

#[derive(Serialize)]
struct SystemState {
    tick_counter: u64,
    unified_capital: f64,
    pnl_realized_scalp: f64,
    pnl_gross_scalp: f64,
    pnl_unrealized_scalp: f64,
    win_rate_scalp: f64,
    pnl_realized_swing: f64,
    pnl_gross_swing: f64,
    pnl_unrealized_swing: f64,
    win_rate_swing: f64,
    global_leverage: f64,
    global_max_drawdown: f64,
    ml_prob_avg: f64,
    hurst_avg: f64,
    zombie_count: usize,
    marking_anomalies: u32,
    cpu_usage: f32,
    memory_used_mb: f64,
    total_memory_mb: f64,
    net_roi_pct: f64,
    gross_roi_pct: f64,
    fees_paid: f64,
}

#[derive(Serialize)]
struct CoinState {
    id: usize,
    symbol: String,
    scalp_pnl: f64,
    swing_pnl: f64,
    win_rate: f64,
    ml_prob: f64,
    hurst: f64,
    active_scalp: bool,
    active_swing: bool,
}

/// Inicia el servidor web en background.
/// Escucha en localhost:3000
pub async fn start_telemetry_server(
    arena: Arc<GlobalArena>,
    tx: tokio::sync::broadcast::Sender<TelemetryEvent>,
) {
    profiler::start_profiler_auditor();

    // Iniciar Auditor Forense (SQLite WAL)
    let forensic_auditor = ForensicAuditor::new("data/forensic_audit.db");
    let auditor_rx = tx.subscribe();
    tokio::spawn(async move {
        forensic_auditor.start(auditor_rx).await;
    });

    // Iniciar el Bot de Telegram si las variables de entorno existen
    let telegram_bot = TelegramBot::new();
    if let Some(bot) = telegram_bot {
        let bot_arc = Arc::new(bot);
        let arena_clone = arena.clone();

        tokio::spawn(async move {
            println!("🤖 [TELEGRAM] Bot iniciado y escuchando el motor HFT...");
            let mut last_capital = arena_clone.unified_capital.load(Ordering::Relaxed);

            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await; // Reporte cada 1 hora
                let current_capital = arena_clone.unified_capital.load(Ordering::Relaxed);
                let pnl = current_capital - last_capital;

                let msg = format!(
                    "📊 Reporte Horario\nCapital: ${:.2}\nPnL (1h): ${:.2}",
                    current_capital, pnl
                );
                let _ = bot_arc.send_message(&msg).await;
                last_capital = current_capital;
            }
        });
    }

    let app = Router::new()
        .route("/", get(dashboard_html))
        .route("/api/state", get(get_state))
        .route("/api/coins", get(get_coins))
        .route("/api/tensor/:coin_id", get(get_tensor))
        .route("/api/genome", get(get_genome))
        .route("/ws", get(ws_handler))
        .layer(Extension(tx))
        .with_state(arena);

    // FIX #1416: Bind resiliente con fallback de puertos para evitar pánicos por colisión
    let port_str = std::env::var("TELEMETRY_PORT").unwrap_or_else(|_| "3000".to_string());
    let base_port: u16 = port_str.parse().unwrap_or(3000);

    let mut bound_listener = None;
    let mut final_port = base_port;
    for offset in 0..10 {
        let test_port = base_port + offset;
        let addr = format!("127.0.0.1:{}", test_port);
        match tokio::net::TcpListener::bind(&addr).await {
            Ok(l) => {
                bound_listener = Some(l);
                final_port = test_port;
                break;
            }
            Err(e) => {
                eprintln!(
                    "⚠️ [TELEMETRY] No se pudo vincular en {} ({}). Intentando puerto alternativo...",
                    addr, e
                );
            }
        }
    }

    if let Some(listener) = bound_listener {
        println!(
            "📡 [TELEMETRY] Servidor Táctico iniciado en http://127.0.0.1:{}",
            final_port
        );
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("⚠️ [TELEMETRY] Error en servidor Axum: {}", e);
        }
    } else {
        eprintln!(
            "🛑 [TELEMETRY] No se pudo iniciar el servidor HTTP en ningún puerto del rango {}-{}. Continuando sin servidor web.",
            base_port,
            base_port + 9
        );
    }
}

#[derive(Serialize)]
struct TensorResponse {
    coin_id: usize,
    omni_tensor: Vec<f32>,
}

async fn get_tensor(
    axum::extract::Path(coin_id): axum::extract::Path<usize>,
    State(arena): State<Arc<GlobalArena>>,
) -> Json<TensorResponse> {
    let tensor = if coin_id < arena.coins.len() {
        let coin = &arena.coins[coin_id];
        vec![
            coin.current_price.load(Ordering::Relaxed) as f32,
            coin.current_atr.load(Ordering::Relaxed) as f32,
            coin.ml_prob.load(Ordering::Relaxed) as f32,
            coin.hurst_exponent.load(Ordering::Relaxed) as f32,
            coin.epigenetic_bias.load(Ordering::Relaxed) as f32,
            coin.epigenetic_threshold_modifier.load(Ordering::Relaxed) as f32,
            coin.spot_bid.load(Ordering::Relaxed) as f32,
            coin.spot_ask.load(Ordering::Relaxed) as f32,
            coin.spot_bid_qty.load(Ordering::Relaxed) as f32,
            coin.spot_ask_qty.load(Ordering::Relaxed) as f32,
            coin.agg_buy_vol.load(Ordering::Relaxed) as f32,
            coin.agg_sell_vol.load(Ordering::Relaxed) as f32,
            coin.scalp.win_rate.load(Ordering::Relaxed) as f32,
            coin.scalp.pnl_realized.load(Ordering::Relaxed) as f32,
            coin.swing.win_rate.load(Ordering::Relaxed) as f32,
            coin.swing.pnl_realized.load(Ordering::Relaxed) as f32,
        ]
    } else {
        vec![]
    };
    Json(TensorResponse {
        coin_id,
        omni_tensor: tensor,
    })
}

async fn get_genome(
    State(arena): State<Arc<GlobalArena>>,
) -> Json<quantum_arena::genome::SuperGenotype> {
    let genome = quantum_arena::genome::SuperGenotype::current_from_arena(&arena);
    Json(genome)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Extension(tx): Extension<tokio::sync::broadcast::Sender<TelemetryEvent>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, tx))
}

async fn handle_socket(mut socket: WebSocket, tx: tokio::sync::broadcast::Sender<TelemetryEvent>) {
    let mut rx = tx.subscribe();
    loop {
        let event = match rx.recv().await {
            Ok(e) => e,
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                eprintln!("[WS-HANDLER] Lagged: {} eventos perdidos — continuando", n);
                continue;
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
        };
        #[allow(clippy::collapsible_if)]
        if let Ok(json) = serde_json::to_string(&event) {
            if socket.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    }
}

/// Endpoint JSON O(1): Lee de la Arena y responde en microsegundos
async fn get_state(State(arena): State<Arc<GlobalArena>>) -> Json<SystemState> {
    let mut pnl_realized_scalp = 0.0;
    let mut pnl_gross_scalp = 0.0;
    let mut pnl_unrealized_scalp = 0.0;
    let mut pnl_realized_swing = 0.0;
    let mut pnl_gross_swing = 0.0;
    let mut pnl_unrealized_swing = 0.0;
    let mut win_rate_scalp_sum = 0.0;
    let mut win_rate_swing_sum = 0.0;
    let mut ml_prob_sum = 0.0;
    let mut hurst_sum = 0.0;
    let mut total_zombies = 0;
    let mut active_coins = 0.0;
    let mut active_scalp_coins = 0.0;
    let mut active_swing_coins = 0.0;
    // B3.13 — SANITIZADOR DE MARCADO: contribuciones de unrealized que
    // exceden 2× el capital son marcado local roto (glitch de precio
    // testnet o posición fantasma de rotación — medido +$2.27M en cuenta
    // de $2.2K, 2026-09-15). Se EXCLUYEN de la suma y se cuentan: el
    // veneno silencioso se vuelve contador visible.
    let cap_anchor = arena.unified_capital.load(Ordering::Relaxed).max(1.0);
    let mark_bound = cap_anchor * 2.0;
    let mut marking_anomalies: u32 = 0;

    for coin in arena.coins.iter() {
        let m_realized = coin.metrics.pnl_realized.load(Ordering::Relaxed);
        let m_gross = coin.metrics.pnl_gross.load(Ordering::Relaxed);
        let m_unrealized = coin.metrics.pnl_unrealized.load(Ordering::Relaxed);
        let m_wr = coin.metrics.win_rate.load(Ordering::Relaxed);

        let sc_realized = coin.scalp.pnl_realized.load(Ordering::Relaxed);
        let sw_realized = coin.swing.pnl_realized.load(Ordering::Relaxed);
        let sc_gross = coin.scalp.pnl_gross.load(Ordering::Relaxed);
        let sw_gross = coin.swing.pnl_gross.load(Ordering::Relaxed);
        let sc_unrealized = coin.scalp.pnl_unrealized.load(Ordering::Relaxed);
        let sw_unrealized = coin.swing.pnl_unrealized.load(Ordering::Relaxed);
        let wr_scalp = coin.scalp.win_rate.load(Ordering::Relaxed);
        let wr_swing = coin.swing.win_rate.load(Ordering::Relaxed);

        let ml = coin.ml_prob.load(Ordering::Relaxed); // Phase 22: ml_prob
        let hurst = coin.hurst_exponent.load(Ordering::Relaxed);
        let zombies = coin.scalp.zombie_promotions.load(Ordering::Relaxed)
            + coin.metrics.zombie_promotions.load(Ordering::Relaxed);

        // D-441: Unificar telemetría con coin.metrics
        let eff_sc_realized = if m_realized.abs() > 0.0 && sc_realized == 0.0 {
            m_realized
        } else {
            sc_realized
        };
        let eff_sc_gross = if m_gross.abs() > 0.0 && sc_gross == 0.0 {
            m_gross
        } else {
            sc_gross
        };
        let eff_sc_unrealized = if m_unrealized.abs() > 0.0 && sc_unrealized == 0.0 {
            m_unrealized
        } else {
            sc_unrealized
        };
        let eff_wr_scalp = if m_wr > 0.0 && wr_scalp == 0.0 {
            m_wr
        } else {
            wr_scalp
        };

        pnl_realized_scalp += eff_sc_realized;
        pnl_gross_scalp += eff_sc_gross;
        // B3.13: excluir marcado imposible del agregado y contarlo.
        if eff_sc_unrealized.abs() > mark_bound {
            marking_anomalies += 1;
        } else {
            pnl_unrealized_scalp += eff_sc_unrealized;
        }
        if sw_unrealized.abs() > mark_bound {
            marking_anomalies += 1;
        } else {
            pnl_unrealized_swing += sw_unrealized;
        }
        pnl_realized_swing += sw_realized;
        pnl_gross_swing += sw_gross;
        total_zombies += zombies;

        ml_prob_sum += ml;
        hurst_sum += hurst;

        if eff_sc_realized != 0.0
            || eff_sc_unrealized != 0.0
            || coin.scalp.active_positions.load(Ordering::Relaxed) > 0
            || coin.metrics.active_positions.load(Ordering::Relaxed) > 0
        {
            win_rate_scalp_sum += eff_wr_scalp;
            active_scalp_coins += 1.0;
        }
        if sw_realized != 0.0
            || sw_unrealized != 0.0
            || coin.swing.active_positions.load(Ordering::Relaxed) > 0
        {
            win_rate_swing_sum += wr_swing;
            active_swing_coins += 1.0;
        }
        active_coins += 1.0;
    }

    let avg_win_rate_scalp = if active_scalp_coins > 0.0 {
        win_rate_scalp_sum / active_scalp_coins
    } else {
        0.55
    };
    let avg_win_rate_swing = if active_swing_coins > 0.0 {
        win_rate_swing_sum / active_swing_coins
    } else {
        0.55
    };
    let avg_ml_prob = if active_coins > 0.0 {
        ml_prob_sum / active_coins
    } else {
        0.5
    };
    let avg_hurst = if active_coins > 0.0 {
        hurst_sum / active_coins
    } else {
        0.5
    };

    let total_net_pnl = pnl_realized_scalp + pnl_realized_swing;
    let total_gross_pnl = pnl_gross_scalp + pnl_gross_swing;
    let fees_paid = total_gross_pnl - total_net_pnl;

    let current_cap = arena.unified_capital.load(Ordering::Relaxed);
    let initial_cap = if (current_cap - total_net_pnl) > 0.0 {
        current_cap - total_net_pnl
    } else {
        current_cap.max(1.0)
    };

    let net_roi_pct = (total_net_pnl / initial_cap) * 100.0;
    let gross_roi_pct = (total_gross_pnl / initial_cap) * 100.0;

    let sys_telem = os_guardian::telemetry::get_system_telemetry();

    let state = SystemState {
        tick_counter: arena.tick_counter.load(Ordering::Relaxed),
        unified_capital: current_cap,
        pnl_realized_scalp,
        pnl_gross_scalp,
        pnl_unrealized_scalp,
        win_rate_scalp: avg_win_rate_scalp,
        pnl_realized_swing,
        pnl_gross_swing,
        pnl_unrealized_swing,
        win_rate_swing: avg_win_rate_swing,
        global_leverage: arena.config.global_leverage.load(Ordering::Relaxed),
        global_max_drawdown: arena.config.global_max_drawdown.load(Ordering::Relaxed),
        ml_prob_avg: avg_ml_prob,
        hurst_avg: avg_hurst,
        zombie_count: total_zombies,
        marking_anomalies,
        cpu_usage: sys_telem.cpu_usage,
        memory_used_mb: sys_telem.memory_used_mb,
        total_memory_mb: sys_telem.total_memory_mb,
        net_roi_pct,
        gross_roi_pct,
        fees_paid,
    };
    Json(state)
}

/// Endpoint JSON para estado individual de monedas
async fn get_coins(State(arena): State<Arc<GlobalArena>>) -> Json<Vec<CoinState>> {
    let mut coins_data = Vec::with_capacity(30);
    // Extraemos la lista dinámica actual desde la memoria
    let active_symbols = quantum_arena::symbols::get_active_universe();

    for (i, coin) in arena.coins.iter().enumerate() {
        let symbol_name = if i < active_symbols.len() {
            active_symbols[i].clone()
        } else {
            format!("COIN_{}", i)
        };

        coins_data.push(CoinState {
            id: i,
            symbol: symbol_name,
            scalp_pnl: coin.scalp.pnl_realized.load(Ordering::Relaxed),
            swing_pnl: coin.swing.pnl_realized.load(Ordering::Relaxed),
            win_rate: coin.scalp.win_rate.load(Ordering::Relaxed),
            ml_prob: coin.ml_prob.load(Ordering::Relaxed),
            hurst: coin.hurst_exponent.load(Ordering::Relaxed),
            active_scalp: coin.scalp.active_positions.load(Ordering::Relaxed) > 0,
            active_swing: coin.swing.active_positions.load(Ordering::Relaxed) > 0,
        });
    }
    Json(coins_data)
}

/// UI del Dashboard (Embebida en el binario)
async fn dashboard_html() -> impl IntoResponse {
    let html = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRADER GEMINI V5 - Quantum Telemetry</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@300;600&display=swap');
        
        :root {
            --bg-dark: #07070b;
            --panel-bg: rgba(15, 15, 25, 0.65);
            --neon-green: #00ff88;
            --neon-red: #ff3366;
            --neon-blue: #00f0ff;
            --neon-purple: #b026ff;
            --text-main: #f0f0f5;
            --text-muted: #888899;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(0, 240, 255, 0.04), transparent 30%),
                radial-gradient(circle at 90% 80%, rgba(176, 38, 255, 0.04), transparent 30%),
                radial-gradient(circle at 50% 50%, rgba(0, 255, 136, 0.02), transparent 50%);
        }

        .header {
            padding: 2.5rem;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            background: linear-gradient(180deg, rgba(0,0,0,0.6) 0%, transparent 100%);
            position: relative;
        }

        .header h1 {
            font-family: 'JetBrains Mono', monospace;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -2px;
            background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple), var(--neon-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            text-shadow: 0px 10px 30px rgba(0, 240, 255, 0.2);
        }

        .header p {
            font-size: 1rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 3px;
        }

        .section-title {
            padding: 2rem 2rem 0.5rem 2rem;
            max-width: 1400px;
            margin: 0 auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.2rem;
            color: var(--text-main);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            width: 100%;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }

        .coins-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            padding: 1.5rem 2rem 2rem 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }

        .card {
            background: var(--panel-bg);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease, border-color 0.4s ease;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .coin-card {
            padding: 1.2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(20,20,30,0.8), rgba(10,10,15,0.9));
        }
        
        .coin-card.active-target {
            border-color: rgba(0, 255, 136, 0.5);
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 0 15px rgba(0, 255, 136, 0.2);
        }

        .card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            opacity: 0.5;
        }

        .card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.15);
        }

        .card-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-muted);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .coin-title {
            font-size: 1.2rem;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: var(--text-main);
        }

        .card-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--text-main);
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            text-shadow: 0 0 20px rgba(255,255,255,0.1);
        }
        
        .coin-value { font-size: 1.2rem; }

        .card-unit { font-size: 0.9rem; color: var(--text-muted); font-weight: 400; }

        .positive { color: var(--neon-green); text-shadow: 0 0 15px rgba(0,255,136,0.4); }
        .negative { color: var(--neon-red); text-shadow: 0 0 15px rgba(255,51,102,0.4); }
        .neutral { color: var(--neon-blue); text-shadow: 0 0 15px rgba(0,240,255,0.4); }

        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--neon-green);
            box-shadow: 0 0 10px var(--neon-green);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.5; box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
            70% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
            100% { transform: scale(0.95); opacity: 0.5; box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
        }

        .flash { animation: flash-update 0.4s ease-out; }
        @keyframes flash-update {
            0% { color: #fff; text-shadow: 0 0 20px #fff; transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .live-badge {
            position: absolute; top: 2.5rem; right: 2.5rem;
            display: flex; align-items: center; gap: 0.8rem;
            font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
            color: var(--neon-green);
            border: 1px solid rgba(0, 255, 136, 0.4);
            padding: 0.4rem 1rem; border-radius: 30px;
            background: rgba(0, 255, 136, 0.05);
            backdrop-filter: blur(10px);
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
        }

    </style>
</head>
<body>
    <div class="header">
        <div class="live-badge">
            <div class="status-dot"></div>
            QUANTUM LINK (10 Hz)
        </div>
        <h1>Trader Gemini V5</h1>
        <p>HFT Autonomous Intelligence & Dynamic Asset Router</p>
    </div>

    <div class="section-title">GLOBAL STATE</div>
    <div class="grid">
        <div class="card">
            <div class="card-title">💰 Capital Global</div>
            <div class="card-value neutral" id="val-capital">0.00 <span class="card-unit">USD</span></div>
        </div>
        <div class="card">
            <div class="card-title">📈 PnL Neto (Post-Fees)</div>
            <div class="card-value" id="val-pnl-post">0.00 <span class="card-unit">USD</span></div>
            <div class="card-subtitle" style="font-size: 0.8rem; color: #888;">Bruto: <span id="val-pnl-pre">0.00</span> USD</div>
        </div>
        <div class="card">
            <div class="card-title">🚀 ROI Neto (Post-Fees)</div>
            <div class="card-value" id="val-roi-post">0.00 <span class="card-unit">%</span></div>
            <div class="card-subtitle" style="font-size: 0.8rem; color: #888;">Bruto: <span id="val-roi-pre">0.00</span> %</div>
        </div>
        <div class="card">
            <div class="card-title">💸 Comisiones (Fees)</div>
            <div class="card-value negative" id="val-fees">0.00 <span class="card-unit">USD</span></div>
        </div>
        <div class="card">
            <div class="card-title">🎯 Tasa de Aciertos (Scalp / Swing)</div>
            <div class="card-value neutral" id="val-win-rate">0.0 / 0.0 <span class="card-unit">%</span></div>
        </div>
        <div class="card">
            <div class="card-title">⏱️ Motor HFT</div>
            <div class="card-value neutral" id="val-ticks">0 <span class="card-unit">Events</span></div>
        </div>
        <div class="card">
            <div class="card-title">🧠 Probabilidad Direccional IA</div>
            <div class="card-value neutral" id="val-ai-prob">0.0 <span class="card-unit">%</span></div>
        </div>
        <div class="card">
            <div class="card-title">🧟 Sanidad de Marcado (Zombies / Anomalías)</div>
            <div class="card-value neutral" id="val-marking-health">0 / 0</div>
            <div class="card-subtitle" style="font-size: 0.8rem; color: #888;">B3.13: unrealized excluido del agregado si |v| &gt; 2× capital</div>
        </div>
        <div class="card">
            <div class="card-title">⚖️ Apalancamiento Asintótico</div>
            <div class="card-value neutral" id="val-leverage">-- <span class="card-unit">x</span></div>
        </div>
        <div class="card">
            <div class="card-title">🛡️ OS Guardian (RAM)</div>
            <div class="card-value neutral" id="val-ram">0.0 <span class="card-unit">MB / 6144 MB</span></div>
        </div>
    </div>
    
    <div class="section-title">DYNAMIC ASSET RADAR (TOP TARGETS)</div>
    <div class="coins-grid" id="coins-container">
        <!-- Coins will be injected here -->
    </div>

    <div class="section-title">SHADOW FOREST (QUANTUM LEADERBOARD)</div>
    <div style="max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem; width: 100%;">
        <div class="card" style="padding: 1.5rem; background: rgba(5,5,10,0.8);">
            <div id="leaderboard-container" style="display: flex; gap: 10px; overflow-x: auto; padding-bottom: 10px;">
                <div style="color: var(--text-muted); font-size: 0.9rem;">Awaiting Harvest Event...</div>
            </div>
        </div>
    </div>

    <div class="section-title">PHASE SPACE TOPOLOGY</div>
    <div style="max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem 3rem 2rem; width: 100%;">
        <div class="card" style="padding: 0;">
            <div id="graph-4d" style="width: 100%; height: 500px; border-radius: 20px;"></div>
        </div>
    </div>

    <div class="section-title">NEURAL GENOME 4D TOPOLOGY (SUPERGENOTYPE)</div>
    <div style="max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem 3rem 2rem; width: 100%;">
        <div class="card" style="padding: 0; background: rgba(5,5,10,0.8);">
            <div id="genome-graph" style="width: 100%; height: 600px; border-radius: 20px;"></div>
        </div>
    </div>

    <script>
        const formatNumber = (num, decimals = 2) => Number(num).toFixed(decimals);
        
        // --- WebSockets Telemetry ---
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.GenomeUpdate) {
                    console.log("GenomeUpdate received via WS!");
                    updateGenomeGraph(msg.GenomeUpdate);
                    
                    // Flash effect on body
                    document.body.style.boxShadow = "inset 0 0 100px rgba(0, 255, 136, 0.5)";
                    setTimeout(() => { document.body.style.boxShadow = "none"; }, 500);
                } else if (msg.ShadowLeaderboard) {
                    const lb = msg.ShadowLeaderboard;
                    const container = document.getElementById('leaderboard-container');
                    let htmlStr = '';
                    lb.forEach((pnl, i) => {
                        const pClass = pnl > 0 ? 'positive' : (pnl < 0 ? 'negative' : 'neutral');
                        const isAlpha = i === 0 ? 'border: 1px solid var(--neon-blue); box-shadow: 0 0 10px var(--neon-blue);' : '';
                        htmlStr += `
                            <div style="min-width: 120px; padding: 10px; border-radius: 10px; background: rgba(20,20,30,0.8); text-align: center; ${isAlpha}">
                                <div style="font-size: 0.7rem; color: #888;">${i === 0 ? 'Base Alpha' : 'Mutant ' + i}</div>
                                <div class="${pClass}" style="font-family: monospace; font-size: 1.1rem;">${formatNumber(pnl)}</div>
                            </div>
                        `;
                    });
                    container.innerHTML = htmlStr;
                }
            } catch(e) {
                console.error("WS Parse Error:", e);
            }
        };
        
        const setHtml = (id, html) => {
            const el = document.getElementById(id);
            if(el.innerHTML !== html) {
                el.innerHTML = html;
                el.classList.remove('flash');
                void el.offsetWidth;
                el.classList.add('flash');
            }
        };

        const historyLength = 150;
        let traceData = {
            x: [], y: [], z: [], mode: 'markers+lines',
            marker: {
                size: [], color: [], colorscale: 'Plasma', showscale: true,
                colorbar: { title: 'Win Rate %', font: {color: '#888899'} }
            },
            line: { color: 'rgba(176, 38, 255, 0.4)', width: 3 },
            type: 'scatter3d', name: 'State Trajectory'
        };

        const layout = {
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            scene: {
                xaxis: { title: 'Tick (Time)', color: '#888899', gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: 'Capital (USD)', color: '#888899', gridcolor: 'rgba(255,255,255,0.05)' },
                zaxis: { title: 'Realized PnL (USD)', color: '#888899', gridcolor: 'rgba(255,255,255,0.05)' },
                bgcolor: 'transparent'
            },
            margin: { l: 0, r: 0, b: 0, t: 10 }
        };

        Plotly.newPlot('graph-4d', [traceData], layout, {responsive: true});

        const updateData = async () => {
            try {
                // Fetch Global State
                const res = await fetch('/api/state');
                const data = await res.json();
                
                // Calculate aggregations dynamically
                data.pnl_post_fees = data.pnl_realized_scalp + data.pnl_realized_swing;
                data.pnl_pre_fees = data.pnl_gross_scalp + data.pnl_gross_swing;
                data.total_fees_paid = data.pnl_pre_fees - data.pnl_post_fees;
                data.roi_post_fees = (data.pnl_post_fees / (data.unified_capital - data.pnl_post_fees)) * 100 || 0;
                data.roi_pre_fees = (data.pnl_pre_fees / (data.unified_capital - data.pnl_post_fees)) * 100 || 0;
                
                setHtml('val-capital', formatNumber(data.unified_capital));
                
                const pnlPostClass = data.pnl_post_fees > 0 ? 'positive' : (data.pnl_post_fees < 0 ? 'negative' : 'neutral');
                setHtml('val-pnl-post', `<span class="${pnlPostClass}">${formatNumber(data.pnl_post_fees)}</span> <span class="card-unit">USD</span>`);
                setHtml('val-pnl-pre', formatNumber(data.pnl_pre_fees));

                const roiPostClass = data.roi_post_fees > 0 ? 'positive' : (data.roi_post_fees < 0 ? 'negative' : 'neutral');
                setHtml('val-roi-post', `<span class="${roiPostClass}">${formatNumber(data.roi_post_fees, 1)}</span> <span class="card-unit">%</span>`);
                setHtml('val-roi-pre', formatNumber(data.roi_pre_fees, 1));
                
                setHtml('val-fees', `${formatNumber(data.total_fees_paid)} <span class="card-unit">USD</span>`);
                
                const wrClass = data.win_rate_scalp > 0.5 ? 'positive' : (data.win_rate_scalp < 0.4 ? 'negative' : 'neutral');
                setHtml('val-win-rate', `<span class="${wrClass}">${formatNumber(data.win_rate_scalp * 100, 1)} / ${formatNumber(data.win_rate_swing * 100, 1)}</span> <span class="card-unit">%</span>`);
                
                setHtml('val-ticks', `${data.tick_counter} <span class="card-unit">Events</span>`);
                
                let aiProbClass = data.ml_prob_avg > 0.6 ? 'positive' : (data.ml_prob_avg < 0.4 ? 'negative' : 'neutral');
                setHtml('val-ai-prob', `<span class="${aiProbClass}">${formatNumber(data.ml_prob_avg * 100, 1)}</span> <span class="card-unit">%</span>`);

                // B3.13 — el sanitizador expone marking_anomalies: sin esta
                // tarjeta el contador era un veneno silencioso (serializado
                // pero invisible). Zombies y anomalías de marcado en vivo.
                const anomalies = data.marking_anomalies || 0;
                const zombies = data.zombie_count || 0;
                const healthClass = anomalies > 0 ? 'negative' : (zombies > 0 ? 'neutral' : 'positive');
                setHtml('val-marking-health', `<span class="${healthClass}">${zombies} / ${anomalies}</span>`);

                setHtml('val-leverage', `${formatNumber(data.global_leverage, 0)} <span class="card-unit">x</span>`);
                
                const ramPercent = (data.memory_used_mb / data.total_memory_mb) * 100;
                const ramClass = ramPercent > 80 ? 'negative' : (ramPercent > 60 ? 'neutral' : 'positive');
                setHtml('val-ram', `<span class="${ramClass}">${formatNumber(data.memory_used_mb, 1)}</span> <span class="card-unit">MB / ${formatNumber(data.total_memory_mb, 0)} MB</span>`);

                // 4D Graph Update
                traceData.x.push(data.tick_counter);
                traceData.y.push(data.unified_capital);
                traceData.z.push(data.pnl_realized_scalp);
                traceData.marker.color.push(data.win_rate_scalp * 100);
                traceData.marker.size.push(Math.max(4, data.global_leverage * 1.5));

                if (traceData.x.length > historyLength) {
                    traceData.x.shift(); traceData.y.shift(); traceData.z.shift();
                    traceData.marker.color.shift(); traceData.marker.size.shift();
                }
                Plotly.react('graph-4d', [traceData], layout);
                
                // Fetch Coins State
                const coinsRes = await fetch('/api/coins');
                const coinsData = await coinsRes.json();
                
                // Sort by ML Prob deviation from 0.5 (highest absolute prediction signal)
                let activeCoins = coinsData.sort((a, b) => Math.abs(b.ml_prob - 0.5) - Math.abs(a.ml_prob - 0.5)).slice(0, 12);
                
                const container = document.getElementById('coins-container');
                let htmlStr = '';
                
                activeCoins.forEach(coin => {
                    // Use the dynamic symbol extracted from the backend
                    const symbol = coin.symbol || `COIN_${coin.id}`;
                    
                    const totalPnl = coin.scalp_pnl + coin.swing_pnl;
                    const pClass = totalPnl > 0 ? 'positive' : (totalPnl < 0 ? 'negative' : 'neutral');
                    const probClass = coin.ml_prob > 0.55 ? 'positive' : (coin.ml_prob < 0.45 ? 'negative' : 'neutral');
                    const isActive = coin.active_scalp || coin.active_swing ? 'active-target' : '';
                    
                    htmlStr += `
                        <div class="card coin-card ${isActive}">
                            <div class="coin-title">${symbol}</div>
                            <div style="font-size: 0.8rem; color: #888; margin-bottom: 0.5rem;">IA Prob: <span class="${probClass}">${formatNumber(coin.ml_prob * 100, 1)}%</span></div>
                            <div class="card-value coin-value ${pClass}">${formatNumber(coin.scalp_pnl)} <span class="card-unit">USD</span></div>
                        </div>
                    `;
                });
                
                container.innerHTML = htmlStr;

            } catch (err) { console.error("Error fetching state:", err); }
        };

        // --- Genome Graph Logic ---
        let network = null;
        let nodesDataset = new vis.DataSet([]);
        let edgesDataset = new vis.DataSet([]);
        let previousGenome = null;

        function initGenomeGraph() {
            const container = document.getElementById('genome-graph');
            const data = { nodes: nodesDataset, edges: edgesDataset };
            const options = {
                nodes: {
                    shape: 'dot', font: { color: '#fff', size: 12 },
                    borderWidth: 2, shadow: true
                },
                edges: {
                    color: { color: 'rgba(255,255,255,0.1)' },
                    smooth: { type: 'continuous' }
                },
                physics: {
                    forceAtlas2Based: { gravitationalConstant: -50, centralGravity: 0.01, springLength: 100, springConstant: 0.08 },
                    maxVelocity: 50, solver: 'forceAtlas2Based', timestep: 0.35, stabilization: { iterations: 150 }
                },
                interaction: { hover: true, tooltipDelay: 200 }
            };
            network = new vis.Network(container, data, options);
        }
        
        function updateGenomeGraph(genome) {
            const categories = {
                risk: { color: { background: '#ff3366', border: '#ff0044' } },
                ml: { color: { background: '#b026ff', border: '#8000ff' } },
                micro: { color: { background: '#00f0ff', border: '#0099ff' } },
                macro: { color: { background: '#ffaa00', border: '#cc7700' } },
                general: { color: { background: '#00ff88', border: '#00cc66' } }
            };
            
            // Map keys to categories
            const getCategory = (key) => {
                if (key.includes('drawdown') || key.includes('kelly') || key.includes('sl') || key.includes('leverage')) return 'risk';
                if (key.includes('ml') || key.includes('hurst') || key.includes('volatility') || key.includes('probability')) return 'ml';
                if (key.includes('obi') || key.includes('ofi') || key.includes('spread') || key.includes('depth')) return 'micro';
                if (key.includes('trend') || key.includes('range') || key.includes('tp') || key.includes('atr')) return 'macro';
                return 'general';
            };

            const nodes = [];
            const edges = [];
            
            // Core central node
            nodes.push({ id: 'core', label: 'SuperGenotype\\nCore', size: 30, color: { background: '#ffffff', border: '#aaaaaa' } });

            Object.entries(genome).forEach(([key, val]) => {
                const cat = getCategory(key);
                // Pulse if changed
                let size = 15;
                if (previousGenome && previousGenome[key] !== val) {
                    size = 25; // Bump size
                }
                
                nodes.push({
                    id: key,
                    label: key,
                    title: `Value: ${val}`,
                    size: size,
                    color: categories[cat].color
                });
                edges.push({ from: 'core', to: key });
                
                // Add some cross edges
                if (cat === 'risk' && key.includes('kelly')) {
                    edges.push({ from: key, to: 'global_leverage', color: { color: 'rgba(255,51,102,0.3)'} });
                }
            });

            nodesDataset.update(nodes);
            edgesDataset.update(edges);
            
            previousGenome = genome;
        }

        initGenomeGraph();

        setInterval(updateData, 100);
        updateData();
    </script>
</body>
</html>
    "#;
    Html(html)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_system_state_serialization() {
        let state = SystemState {
            tick_counter: 100,
            unified_capital: 13.0,
            pnl_realized_scalp: 0.5,
            pnl_gross_scalp: 0.6,
            pnl_unrealized_scalp: 0.1,
            win_rate_scalp: 0.85,
            pnl_realized_swing: 1.0,
            pnl_gross_swing: 1.2,
            pnl_unrealized_swing: 0.2,
            win_rate_swing: 0.75,
            global_leverage: 10.0,
            global_max_drawdown: 0.02,
            ml_prob_avg: 0.78,
            hurst_avg: 0.65,
            zombie_count: 0,
            marking_anomalies: 0,
            cpu_usage: 15.0,
            memory_used_mb: 250.0,
            total_memory_mb: 16384.0,
            net_roi_pct: 11.5,
            gross_roi_pct: 13.8,
            fees_paid: 0.03,
        };

        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("unified_capital"));
        assert!(json.contains("13.0"));
    }

    #[test]
    fn test_telemetry_event_trade_closed_serialization() {
        let ev = TelemetryEvent::TradeClosed {
            coin_id: 1,
            trade_type: "SCALP".to_string(),
            pnl: 0.45,
            roi_pct: 3.46,
            duration_ms: 12000,
            ml_prob: 0.82,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("SCALP"));
        assert!(json.contains("0.45"));
    }

    #[test]
    fn test_telemetry_event_omni_and_coin_state_serialization() {
        let omni = TelemetryEvent::OmniUpdate {
            latency_ms: 12,
            latency_panic: false,
            dark_alpha: 0.88,
            scalp_pnl: 2.5,
            swing_pnl: 4.1,
            gross_pnl: 6.6,
            net_pnl: 6.55,
            win_rate: 0.72,
            trade_duration_avg: 45.0,
        };
        let omni_json = serde_json::to_string(&omni).unwrap();
        assert!(omni_json.contains("dark_alpha"));
        assert!(omni_json.contains("0.88"));

        let coin = CoinState {
            id: 0,
            symbol: "BTCUSDT".to_string(),
            scalp_pnl: 1.2,
            swing_pnl: 2.3,
            win_rate: 0.80,
            ml_prob: 0.75,
            hurst: 0.62,
            active_scalp: true,
            active_swing: false,
        };
        let coin_json = serde_json::to_string(&coin).unwrap();
        assert!(coin_json.contains("BTCUSDT"));
        assert!(coin_json.contains("active_scalp"));
    }

    /// B3.13 — SANITIZADOR DE /api/state: un marcado imposible (|unrealized|
    /// > 2× capital) se EXCLUYE del agregado y se cuenta en
    /// marking_anomalies; uno sano pasa íntegro. El campo además debe estar
    /// serializado (el dashboard lo consume).
    #[tokio::test]
    async fn sanitizer_marca_y_excluye_unrealized_imposible() {
        let arena = std::sync::Arc::new(quantum_arena::GlobalArena::new(2_200.0));

        // Coin 0: marcado roto (glitch medido: +$2.27M en cuenta de $2.2K).
        arena.coins[0]
            .metrics
            .pnl_unrealized
            .store(2_270_000.0, Ordering::Relaxed);
        // Coin 1: marcado sano.
        arena.coins[1]
            .metrics
            .pnl_unrealized
            .store(15.0, Ordering::Relaxed);

        let state = get_state(axum::extract::State(arena)).await;

        // El veneno quedó fuera del agregado; el sano pasó.
        assert!((state.pnl_unrealized_scalp - 15.0).abs() < 1e-6);
        // Contado, no silenciado.
        assert_eq!(state.marking_anomalies, 1);
        // Serialización expone el contador (wiring al dashboard).
        let json = serde_json::to_string(&state.0).unwrap();
        assert!(json.contains("marking_anomalies"));
        assert!(json.contains("zombie_count"));
    }
}
