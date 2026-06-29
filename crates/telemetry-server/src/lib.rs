use axum::{
    extract::State,
    response::{Html, IntoResponse, Json},
    routing::get,
    Router,
};
use quantum_arena::GlobalArena;
use serde::Serialize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Serialize)]
struct SystemState {
    tick_counter: u64,
    unified_capital: f64,
    pnl_realized_scalp: f64,
    pnl_unrealized_scalp: f64,
    win_rate_scalp: f64,
    global_leverage: f64,
    global_max_drawdown: f64,
}

/// Inicia el servidor web en background.
/// Escucha en localhost:3000
pub async fn start_telemetry_server(arena: Arc<GlobalArena>) {
    let app = Router::new()
        .route("/", get(dashboard_html))
        .route("/api/state", get(get_state))
        .with_state(arena);

    println!("📡 [TELEMETRY] Servidor Táctico iniciado en http://127.0.0.1:3000");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    axum::serve(listener, app).await.unwrap();
}

/// Endpoint JSON O(1): Lee de la Arena y responde en microsegundos
async fn get_state(State(arena): State<Arc<GlobalArena>>) -> Json<SystemState> {
    let state = SystemState {
        tick_counter: arena.tick_counter.load(Ordering::Relaxed),
        unified_capital: arena.unified_capital.load(Ordering::Relaxed),
        pnl_realized_scalp: arena.scalp.pnl_realized.load(Ordering::Relaxed),
        pnl_unrealized_scalp: arena.scalp.pnl_unrealized.load(Ordering::Relaxed),
        win_rate_scalp: arena.scalp.win_rate.load(Ordering::Relaxed),
        global_leverage: arena.config.global_leverage.load(Ordering::Relaxed),
        global_max_drawdown: arena.config.global_max_drawdown.load(Ordering::Relaxed),
    };
    Json(state)
}

/// UI del Dashboard (Embebida en el binario)
async fn dashboard_html() -> impl IntoResponse {
    let html = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRADER GEMINI V5 - Telemetry</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@300;600&display=swap');
        
        :root {
            --bg-dark: #0a0a0f;
            --panel-bg: rgba(20, 20, 30, 0.7);
            --neon-green: #00ff88;
            --neon-red: #ff3366;
            --neon-blue: #00f0ff;
            --text-main: #e0e0e0;
            --text-muted: #888899;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(0, 255, 136, 0.03), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(0, 240, 255, 0.03), transparent 25%);
        }

        .header {
            padding: 2rem;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            background: linear-gradient(180deg, rgba(0,0,0,0.8) 0%, transparent 100%);
        }

        .header h1 {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(90deg, var(--neon-blue), var(--neon-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
        }
        .header p {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }

        .card {
            background: var(--panel-bg);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            opacity: 0.3;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border-color: rgba(255,255,255,0.1);
        }

        .card-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-muted);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            text-shadow: 0 0 20px rgba(255,255,255,0.1);
        }

        .card-unit {
            font-size: 1rem;
            color: var(--text-muted);
            font-weight: 400;
        }

        .positive { color: var(--neon-green); text-shadow: 0 0 15px rgba(0,255,136,0.3); }
        .negative { color: var(--neon-red); text-shadow: 0 0 15px rgba(255,51,102,0.3); }
        .neutral { color: var(--neon-blue); text-shadow: 0 0 15px rgba(0,240,255,0.3); }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--neon-green);
            box-shadow: 0 0 10px var(--neon-green);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.5; }
        }

        /* Utilidades de Flash al actualizar */
        .flash {
            animation: flash-update 0.3s ease-out;
        }
        @keyframes flash-update {
            0% { color: #fff; text-shadow: 0 0 10px #fff; }
            100% { }
        }

        .live-badge {
            position: absolute;
            top: 2rem;
            right: 2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--neon-green);
            border: 1px solid rgba(0, 255, 136, 0.3);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            background: rgba(0, 255, 136, 0.05);
        }

    </style>
</head>
<body>
    <div class="live-badge">
        <div class="status-dot"></div>
        LIVE (10 Hz)
    </div>

    <div class="header">
        <h1>Trader Gemini V5</h1>
        <p>HFT Quantum Telemetry Interface</p>
    </div>

    <div class="grid">
        <!-- Capital -->
        <div class="card">
            <div class="card-title">💰 Capital Global</div>
            <div class="card-value neutral" id="val-capital">
                0.00 <span class="card-unit">USD</span>
            </div>
        </div>

        <!-- PnL Realizado -->
        <div class="card">
            <div class="card-title">📈 PnL Realizado (Scalp)</div>
            <div class="card-value" id="val-pnl-realized">
                0.00 <span class="card-unit">USD</span>
            </div>
        </div>

        <!-- PnL Flotante -->
        <div class="card">
            <div class="card-title">🌊 PnL Flotante (Scalp)</div>
            <div class="card-value" id="val-pnl-unrealized">
                0.00 <span class="card-unit">USD</span>
            </div>
        </div>

        <!-- Leverage -->
        <div class="card">
            <div class="card-title">⚡ Apalancamiento IA</div>
            <div class="card-value neutral" id="val-leverage">
                0.0 <span class="card-unit">X</span>
            </div>
        </div>

        <!-- Win Rate -->
        <div class="card">
            <div class="card-title">🎯 Tasa de Aciertos</div>
            <div class="card-value neutral" id="val-win-rate">
                0.0 <span class="card-unit">%</span>
            </div>
        </div>

        <!-- Latency / Ticks -->
        <div class="card">
            <div class="card-title">⏱️ Motor HFT</div>
            <div class="card-value neutral" id="val-ticks">
                0 <span class="card-unit">Ticks (Eventos)</span>
            </div>
        </div>
    </div>

    <script>
        const formatNumber = (num, decimals = 2) => Number(num).toFixed(decimals);
        const setHtml = (id, html) => {
            const el = document.getElementById(id);
            if(el.innerHTML !== html) {
                el.innerHTML = html;
                el.classList.remove('flash');
                void el.offsetWidth; // trigger reflow
                el.classList.add('flash');
            }
        };

        const updateData = async () => {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                
                setHtml('val-capital', `${formatNumber(data.unified_capital)} <span class="card-unit">USD</span>`);
                
                const pnlClass = data.pnl_realized_scalp > 0 ? 'positive' : (data.pnl_realized_scalp < 0 ? 'negative' : 'neutral');
                setHtml('val-pnl-realized', `<span class="${pnlClass}">${formatNumber(data.pnl_realized_scalp)}</span> <span class="card-unit">USD</span>`);
                
                const flClass = data.pnl_unrealized_scalp > 0 ? 'positive' : (data.pnl_unrealized_scalp < 0 ? 'negative' : 'neutral');
                setHtml('val-pnl-unrealized', `<span class="${flClass}">${formatNumber(data.pnl_unrealized_scalp)}</span> <span class="card-unit">USD</span>`);
                
                setHtml('val-leverage', `${formatNumber(data.global_leverage, 1)} <span class="card-unit">x</span>`);
                setHtml('val-win-rate', `${formatNumber(data.win_rate_scalp * 100, 1)} <span class="card-unit">%</span>`);
                setHtml('val-ticks', `${data.tick_counter} <span class="card-unit">Eventos</span>`);

            } catch (err) {
                console.error("Error fetching state:", err);
            }
        };

        // Poll at 10Hz (100ms)
        setInterval(updateData, 100);
        updateData();
    </script>
</body>
</html>
    "#;
    Html(html)
}
