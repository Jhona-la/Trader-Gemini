"""
🖥️ TRADER GEMINI - DASHBOARD DE CONTROL EXPERTO
=================================================

PROFESSOR METHOD:
- QUÉ: Dashboard profesional de alta densidad para monitoreo de trading.
- POR QUÉ: Para visualizar métricas, señales y estado del bot en tiempo real.
- PARA QUÉ: Control experto sin necesidad de terminal.
- CÓMO: Streamlit con pestañas, KPIs, gráficos Plotly y streaming de eventos.
- CUÁNDO: Se ejecuta en paralelo con el bot.
- DÓNDE: http://localhost:8501
- QUIÉN: Trader/Operador del sistema.
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import deque

# ==============================================================================
# FIX: Agregar directorio raíz del proyecto al PATH de Python
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.analytics import AnalyticsEngine
from utils.session_manager import SessionManager
from core.api_manager import get_api_manager, APIManager
from core.api_manager import get_api_manager, APIManager
from config import Config
from utils.statistics_pro import StatisticsPro

# ==============================================================================
# PAGE CONFIG - EXPERT MODE
# ==============================================================================
st.set_page_config(
    page_title="Trader Gemini | Expert Terminal",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# DARK PROFESSIONAL STYLE
# ==============================================================================
st.markdown("""
<style>
    .stApp { background-color: #0a0e14; color: #e0e0e0; }
    
    /* Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .kpi-value { font-size: 24px; font-weight: 700; color: #58a6ff; }
    .kpi-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
    .kpi-positive { color: #3fb950 !important; }
    .kpi-negative { color: #f85149 !important; }
    .kpi-warning { color: #d29922 !important; }
    
    /* Environment Badge */
    .env-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .env-prod { background: #238636; color: white; }
    .env-demo { background: #1f6feb; color: white; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d;
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* Log area */
    .log-container {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Consolas', monospace;
        font-size: 12px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'environment' not in st.session_state:
    st.session_state.environment = 'PROD'
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = deque(maxlen=100)
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'api_manager_started' not in st.session_state:
    st.session_state.api_manager_started = False
if 'equity_live' not in st.session_state:
    st.session_state.equity_live = 0.0

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_data_dir(mode: str = "futures", env: str = "PROD") -> str:
    """Get data directory with fallback to project root."""
    candidate_paths = [
        f"dashboard/data/{mode}",
        f"data/{mode}",
        f"dashboard/data",
        "data"
    ]
    
    for path in candidate_paths:
        full_path = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            return path
            
    # Default fallback
    return f"dashboard/data/{mode}"

def load_live_status(data_dir: str) -> dict:
    """Load current bot status from JSON."""
    paths = [
        os.path.join(data_dir, "live_status.json"),
        os.path.join(data_dir, "status.json")
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                pass
    return None

def load_historical_status(data_dir: str, tail: int = 500) -> pd.DataFrame:
    """Load historical status for charts."""
    path = os.path.join(data_dir, "status.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Normalization: Lowercase headers to avoid mismatch
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Validate required columns
        if 'total_equity' not in df.columns or 'timestamp' not in df.columns:
            return pd.DataFrame()
        return df.tail(tail)
    except:
        return pd.DataFrame()

def load_trades(data_dir: str) -> pd.DataFrame:
    """Load trade history."""
    path = os.path.join(data_dir, "trades.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

def load_recent_logs(limit: int = 50) -> list:
    """Load recent log entries from JSON log file."""
    today = datetime.now().strftime("%Y%m%d")
    log_path = f"logs/bot_{today}.json"
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()[-limit:]
        logs = []
        for line in lines:
            try:
                logs.append(json.loads(line.strip()))
            except:
                pass
        return logs
    except:
        return []

def load_health_logs(limit: int = 20) -> list:
    """Load CI-HMA health logs."""
    log_path = "logs/health_log.json"
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()[-limit:]
        logs = []
        for line in lines:
            try:
                logs.append(json.loads(line.strip()))
            except:
                pass
        return logs
    except:
        return []

def list_sessions(data_dir: str, limit: int = 10) -> list:
    """List available sessions sorted by latest activity."""
    # Start with LIVE session
    sessions = [{"id": "LIVE", "label": "🟢 Live Session", "path": None, "timestamp": float('inf')}]
    
    found_sessions = []
    sessions_dir = os.path.join(data_dir, "sessions")
    
    if os.path.exists(sessions_dir):
        for date_dir in os.listdir(sessions_dir):
            date_path = os.path.join(sessions_dir, date_dir)
            if os.path.isdir(date_path):
                for run_dir in os.listdir(date_path):
                    run_path = os.path.join(date_path, run_dir)
                    info_file = os.path.join(run_path, "session_info.json")
                    if os.path.exists(info_file):
                        try:
                            # Get info and mtime
                            with open(info_file, 'r') as f:
                                info = json.load(f)
                            mtime = os.path.getmtime(info_file)
                            
                            icon = "🟢" if info.get("status") == "RUNNING" else "⚪"
                            found_sessions.append({
                                "id": info.get("session_id", run_dir),
                                "label": f"{icon} {info.get('session_id', run_dir)[:16]}...",
                                "path": run_path,
                                "timestamp": mtime
                            })
                        except:
                            pass
                            
    # Sort found sessions by timestamp descending (newest first)
    found_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Combine
    sessions.extend(found_sessions)
    return sessions[:limit]

def format_kpi(value, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
    """Format KPI value with proper handling of None/NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{prefix}{value:,.{decimals}f}{suffix}"

def get_kpi_class(value, thresholds: tuple = (0, 0)) -> str:
    """Get CSS class based on value thresholds."""
    if value is None:
        return ""
    if value > thresholds[1]:
        return "kpi-positive"
    elif value < thresholds[0]:
        return "kpi-negative"
    return ""

# ==============================================================================
# SIDEBAR - CONTROL PANEL
# ==============================================================================
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 24px;'>🤖 GEMINI TERMINAL</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Environment Selector
    st.subheader("🌐 Environment")
    env_options = ["PROD", "DEMO"]
    env_idx = env_options.index(st.session_state.environment)
    new_env = st.selectbox(
        "API Context",
        env_options,
        index=env_idx,
        help="Switch between Production and Demo Binance APIs"
    )
    if new_env != st.session_state.environment:
        st.session_state.environment = new_env
        st.rerun()
    
    # Environment Badge
    badge_class = "env-prod" if st.session_state.environment == "PROD" else "env-demo"
    badge_text = "🔴 PRODUCTION" if st.session_state.environment == "PROD" else "🔵 DEMO MODE"
    st.markdown(f'<div class="env-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mode & Session
    st.subheader("📊 Data Source")
    mode = st.selectbox("Market Mode", ["futures", "spot"], index=0)
    data_dir = get_data_dir(mode, st.session_state.environment)
    
    sessions = list_sessions(data_dir)
    session_labels = [s["label"] for s in sessions]
    selected_session_idx = st.selectbox("Session", range(len(session_labels)), format_func=lambda x: session_labels[x])
    selected_session = sessions[selected_session_idx] if sessions else None
    
    # Determine active data directory
    if selected_session and selected_session.get("path"):
        active_data_dir = selected_session["path"]
        st.info(f"📂 Viewing: {selected_session['id'][:16]}")
    else:
        active_data_dir = data_dir
    
    st.markdown("---")
    
    # Controls
    st.subheader("⚙️ Controls")
    st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
    refresh_rate = st.slider("Refresh Rate (sec)", 2, 10, 3)
    
    if st.button("🔄 Refresh Now", type="primary"):
        st.rerun()
    
    st.markdown("---")
    
    # Emergency
    st.subheader("🚨 Emergency")
    if st.button("💀 KILL SWITCH", type="secondary"):
        kill_path = os.path.join(data_dir, "kill_switch.txt")
        with open(kill_path, "w") as f:
            f.write("STOP")
        st.error("⚠️ KILL SIGNAL SENT!")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# ==============================================================================
# LOAD DATA WITH API MANAGER FAILOVER
# ==============================================================================

# Initialize API Manager and start heartbeat
api_mgr = get_api_manager()
if not st.session_state.api_manager_started:
    is_prod = st.session_state.environment == 'PROD'
    api_mgr.start_heartbeat_worker(interval=5, is_prod=is_prod)
    st.session_state.api_manager_started = True

with st.spinner("Loading data..."):
    # Try to get live data from API, fallback to cache
    try:
        is_prod = st.session_state.environment == 'PROD'
        # Strict 3s timeout for dashboard responsiveness
        live_data = api_mgr.get_account_balance(is_prod=is_prod)
        
        # Priority: Real Live API > Local File > Cached Live
        if live_data and live_data.get('total_equity', 0) > 0:
             st.session_state.equity_live = live_data.get('total_equity', 0)
        elif status and status.get('total_equity', 0) > 0:
             st.session_state.equity_live = status.get('total_equity', 0)
             
    except Exception as e:
        # Fallback to cache if API strictly fails
        live_data = api_mgr.load_cached_status()
    
    # Local file data
    status = load_live_status(data_dir) or live_data
    equity = status.get('total_equity', 0) if status else 0
    positions = status.get('positions', {}) if status else {}
    
    history = load_historical_status(active_data_dir)
    trades = load_trades(active_data_dir)
    analytics = AnalyticsEngine.calculate_metrics(history)
    win_stats = AnalyticsEngine.calculate_winrate_details(trades)
    recent_logs = load_recent_logs(50)
    
    # Calculate Expectancy (Phase 5)
    # Priority: Pre-calculated in status > Calculate on-the-fly
    if status and 'performance_metrics' in status:
        expectancy_stats = status['performance_metrics']
    else:
        expectancy_stats = AnalyticsEngine.calculate_expectancy(trades)
    
    # Get connection status
    conn_emoji, conn_text, conn_type = api_mgr.get_connection_badge()
    api_status = api_mgr.get_status_summary()

# ==============================================================================
# HEADER - ENVIRONMENT & STATUS
# ==============================================================================
header_col1, header_col2, header_col3, header_col4 = st.columns([2, 1, 1, 1])
with header_col1:
    st.markdown(f"## Trading Terminal — {mode.upper()}")
with header_col2:
    # Dynamic Status Badge based on API connection
    if conn_type == "success":
        st.success(f"API: {conn_text}")
    elif conn_type == "warning":
        st.warning(f"API: {conn_text}")
    else:
        st.error(f"API: {conn_text}")
        
    # Heartbeat (The Pulse) - Phase 5/6
    last_hb_str = status.get('last_heartbeat')
    if last_hb_str:
        try:
            # Parse ISO with UTC
            last_ts = datetime.fromisoformat(last_hb_str.replace('Z', '+00:00'))
            latency = (datetime.now(timezone.utc) - last_ts).total_seconds()
            
            if latency < 10:
                st.success(f"🟢 PULSE: {int(latency)}s")
            elif latency < 60:
                st.warning(f"🟡 LAG: {int(latency)}s")
            else:
                st.error(f"🔴 DEAD: {int(latency)}s")
        except:
             st.error("🔴 PULSE: ERROR")
    else:
        st.info("⚪ STARTING...")
with header_col3:
    latency = api_status.get('last_latency_ms', 0)
    st.metric("Latency", f"{latency}ms")
with header_col4:
    env_color = "#238636" if st.session_state.environment == "PROD" else "#1f6feb"
    st.markdown(f"<div style='text-align: right;'><span style='background: {env_color}; padding: 4px 12px; border-radius: 20px; font-size: 12px;'>{st.session_state.environment}</span></div>", unsafe_allow_html=True)

# ==============================================================================
# ACCOUNT HEALTH & RISK (Phase 6)
# ==============================================================================
if active_data_dir:
    integrity_path = os.path.join(active_data_dir, "integrity.json")
    try:
        current_equity = st.session_state.equity_live if st.session_state.equity_live > 0 else equity
        integrity_data = {
            "timestamp_epoch": time.time(),
            "timestamp_iso": datetime.utcnow().isoformat(),
            "displayed_equity": current_equity,
            "session_id": selected_session['id'] if selected_session else "LIVE"
        }
        # Atomic write manually or just standard write (less critical if partial read occassionally)
        with open(integrity_path, 'w') as f:
            json.dump(integrity_data, f)
    except Exception as e:
        pass

st.markdown("---")
# MARGIN & RISK METRICS
maint_margin = status.get('maint_margin', 0)
margin_balance = status.get('margin_balance', 0)
unrealized_pnl = status.get('unrealized_pnl', 0)

if margin_balance > 0:
    margin_ratio = (maint_margin / margin_balance) * 100
else:
    margin_ratio = 0.0

# Effective Leverage
if equity > 0:
    # Sum absolute position values (notional)
    total_notional = sum([abs(p['quantity']) * p.get('current_price', 0) for p in positions.values()])
    effective_leverage = total_notional / equity
else:
    effective_leverage = 0.0

# Liquidation Warning
if margin_ratio > 80:
    st.error(f"🚨 DANGER: LIQUIDATION RISK HIGH ({margin_ratio:.1f}%)")
elif margin_ratio > 50:
    st.warning(f"⚠️ CAUTION: MARGIN RATIO {margin_ratio:.1f}%")

risk_cols = st.columns([1, 1, 2])
with risk_cols[0]:
    un_pnl_color = "red" if unrealized_pnl < 0 else "green"
    st.markdown(f"**Unrealized PnL**: <span style='color:{un_pnl_color}'>${unrealized_pnl:+.2f}</span>", unsafe_allow_html=True)
with risk_cols[1]:
    st.metric("Effective Leverage", f"{effective_leverage:.2f}x")
with risk_cols[2]:
    st.caption(f"Margin Ratio: {margin_ratio:.1f}%")
    st.progress(margin_ratio / 100.0 if margin_ratio <= 100 else 1.0)

st.markdown("---")

kpi_cols = st.columns(8)  # Increased from 7 to 8 for Expectancy KPI

# 1. Total Equity
equity = status.get('total_equity', 0) if status else 0
with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">${equity:,.2f}</div>
        <div class="kpi-label">Total Equity</div>
    </div>
    """, unsafe_allow_html=True)

# 2. Daily PnL
daily_pnl = status.get('realized_pnl', 0) if status else 0
pnl_class = "kpi-positive" if daily_pnl > 0 else "kpi-negative" if daily_pnl < 0 else ""
with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value {pnl_class}">${daily_pnl:+,.2f}</div>
        <div class="kpi-label">Daily PnL</div>
    </div>
    """, unsafe_allow_html=True)

# 3. Sharpe Ratio
sharpe = analytics.get('sharpe', 0)
sharpe_class = "kpi-positive" if sharpe > 1.5 else "kpi-warning" if sharpe > 0.5 else "kpi-negative"
with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value {sharpe_class}">{sharpe:.2f}</div>
        <div class="kpi-label">Sharpe Ratio</div>
    </div>
    """, unsafe_allow_html=True)

# 4. Win Rate
win_rate = win_stats.get('global_winrate', 0)
wr_class = "kpi-positive" if win_rate > 55 else "kpi-warning" if win_rate > 45 else "kpi-negative"
with kpi_cols[3]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value {wr_class}">{win_rate:.1f}%</div>
        <div class="kpi-label">Win Rate</div>
    </div>
    """, unsafe_allow_html=True)

# 5. Max Drawdown
max_dd = analytics.get('max_drawdown', 0)
dd_class = "kpi-positive" if max_dd < 2 else "kpi-warning" if max_dd < 5 else "kpi-negative"
with kpi_cols[4]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value {dd_class}">{max_dd:.2f}%</div>
        <div class="kpi-label">Max Drawdown</div>
    </div>
    """, unsafe_allow_html=True)

# 6. Profit Factor
pf = win_stats.get('profit_factor', 0)
pf_class = "kpi-positive" if pf > 1.5 else "kpi-warning" if pf > 1 else "kpi-negative"
with kpi_cols[5]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value {pf_class}">{pf:.2f}</div>
        <div class="kpi-label">Profit Factor</div>
    </div>
    """, unsafe_allow_html=True)

# 7. Total Trades
# 7. Total Trades
total_trades = win_stats.get('total_trades', 0)
with kpi_cols[6]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{total_trades}</div>
        <div class="kpi-label">Total Trades</div>
    </div>
    """, unsafe_allow_html=True)

# 8. Expectancy (New Phase 5 KPI)
exp_val = expectancy_stats.get('expectancy', 0)
exp_class = "kpi-positive" if exp_val > 0 else "kpi-negative" if exp_val < 0 else "kpi-warning"
exp_display = f"${exp_val:.2f}" if expectancy_stats.get('status') != 'INSUFFICIENT_DATA' else "Calc..."

with kpi_cols[7]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value {exp_class}">{exp_display}</div>
        <div class="kpi-label">Expectancy</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN TABS
# ==============================================================================
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Monitor en Vivo",
    "📈 Análisis de Desempeño", 
    "🔄 Control Reversiones",
    "🏥 System Health", # Phase 6
    "📝 Logs del Sistema"
])

# ------------------------------------------------------------------------------
# TAB 1: LIVE MONITOR
# ------------------------------------------------------------------------------
with tab1:
    col_chart, col_positions = st.columns([3, 1])
    
    with col_chart:
        st.subheader("💹 Equity Curve & Drawdown")
        
        if not history.empty and 'total_equity' in history.columns and 'timestamp' in history.columns:
            try:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Equity Curve", "Drawdown %")
                )
                
                # Equity Curve
                fig.add_trace(
                    go.Scatter(
                        x=history['timestamp'],
                        y=history['total_equity'],
                        name='Equity',
                        line=dict(color='#3fb950', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(63,185,80,0.1)'
                    ),
                    row=1, col=1
                )
                
                # Drawdown
                dd_series = AnalyticsEngine.calculate_drawdown_series(history['total_equity'])
                dd = dd_series * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=history['timestamp'],
                        y=dd,
                        name='Drawdown',
                        line=dict(color='#f85149', width=1),
                        fill='tozeroy',
                        fillcolor='rgba(248,81,73,0.2)'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    height=450,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(showgrid=True, gridcolor='#21262d')
                fig.update_yaxes(showgrid=True, gridcolor='#21262d')
                
                st.plotly_chart(fig, key="equity_dd_chart")
            except Exception as e:
                st.error(f"⚠️ Chart Error: {e}")
                st.caption("Debug Info (Head):")
                st.write(history.head())
        elif history.empty:
            # Placeholder Chart
            fig = go.Figure()
            fig.add_annotation(
                text="🔍 Scanning Binance History...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="#8b949e")
            )
            fig.update_layout(
                template="plotly_dark",
                height=450,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, key="equity_placeholder")
        else:
            st.info("📊 Waiting for trading data...")
    
    with col_positions:
        st.subheader("📋 Open Positions")
        
        if status and status.get('positions'):
            positions = status['positions']
            for sym, pos in positions.items():
                qty = pos.get('quantity', 0)
                avg_price = pos.get('avg_price', 0)
                current_price = pos.get('current_price', avg_price)
                
                if qty != 0:
                    # Calculate unrealized PnL
                    if qty > 0:  # Long
                        upnl = (current_price - avg_price) * qty
                        side = "🟢 LONG"
                    else:  # Short
                        upnl = (avg_price - current_price) * abs(qty)
                        side = "🔴 SHORT"
                    
                    pnl_color = "#3fb950" if upnl > 0 else "#f85149"
                    
                    st.markdown(f"""
                    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                        <div style="font-weight: 600; font-size: 14px;">{sym}</div>
                        <div style="font-size: 12px; color: #8b949e;">{side}</div>
                        <div style="font-size: 11px; color: #8b949e;">Qty: {abs(qty):.4f}</div>
                        <div style="font-size: 11px; color: #8b949e;">Entry: ${avg_price:.4f}</div>
                        <div style="font-size: 14px; color: {pnl_color}; font-weight: 600;">PnL: ${upnl:+.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No open positions")

# ------------------------------------------------------------------------------
# TAB 2: PERFORMANCE ANALYSIS
# ------------------------------------------------------------------------------
with tab2:
    col_header, col_filter = st.columns([3, 1])
    
    with col_header:
        st.subheader("📈 Performance Metrics")
        
    with col_filter:
        show_reversals = st.toggle("🔄 Only Reversals (Phase 5)", value=False)
    
    # Filter Logic
    analysis_trades = trades.copy()
    if show_reversals:
        if 'is_reverse' in analysis_trades.columns:
            # Check if we have reversal trades
            rev_trades = analysis_trades[analysis_trades['is_reverse'] == True]
            if not rev_trades.empty:
                analysis_trades = rev_trades
                st.caption(f"Showing {len(analysis_trades)} reversal trades")
            else:
                st.info("ℹ️ No Phase 5 operations recorded yet.")
                analysis_trades = pd.DataFrame() # Clear data to show 0s
        else:
            st.warning("⚠️ No reversal column found (Old Data)")
            analysis_trades = pd.DataFrame()
            
    # Re-calculate metrics for display if filtered
    if show_reversals:
        # We need to re-calc based on filtered trades
        # Note: 'analytics' dict comes from 'history' (equity curve), which is hard to filter by trade type directly
        # So we focus on Win Rate and Expectancy which come from 'trades'
        
        filtered_win_stats = AnalyticsEngine.calculate_winrate_details(analysis_trades)
        filtered_expectancy = AnalyticsEngine.calculate_expectancy(analysis_trades)
        
        # Override global stats for display (local scope)
        win_stats = filtered_win_stats
        expectancy_stats = filtered_expectancy
        
        st.info("ℹ️ Metrics above are now filtered for Reversal Trades only.")

    # Calculate Friction Analysis (Phase 5)
    friction_stats = AnalyticsEngine.calculate_friction(analysis_trades)
    
    # Check for False Edge Alert
    if friction_stats.get('false_edge'):
        st.error("⚠️ CRITICAL ALERT: FALSE EDGE DETECTED! Your strategy wins often (>55%) but loses money on Friction (Expectancy < 0). Reduce Fees or increase Avg Win.")
    elif friction_stats.get('friction_pct', 0) > 20:
        st.warning(f"⚠️ High Friction Alert: {friction_stats.get('friction_pct')}% of your Gross Profit is eaten by Fees/Slippage.")

    perf_cols = st.columns(4)
    
    with perf_cols[0]:
        st.metric("Sharpe Ratio", f"{analytics.get('sharpe', 0):.2f}")
        st.metric("Sortino Ratio", f"{analytics.get('sortino', 0):.2f}")
    
    with perf_cols[1]:
        st.metric("Win Rate", f"{win_stats.get('global_winrate', 0):.1f}%")
        st.metric("Profit Factor", f"{win_stats.get('profit_factor', 0):.2f}")
    
    with perf_cols[2]:
        st.metric("Max Drawdown", f"{analytics.get('max_drawdown', 0):.2f}%")
        st.metric("Volatility", f"{analytics.get('volatility', 0):.2f}%")
    
    with perf_cols[3]:
        friction_val = friction_stats.get('friction_pct', 0)
        fr_class = "kpi-positive" if friction_val < 10 else "kpi-warning" if friction_val < 30 else "kpi-negative"
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value {fr_class}">{friction_val:.1f}%</div>
            <div class="kpi-label">Friction Impact</div>
            <div style="font-size: 10px; color: #8b949e;">Fees: ${friction_stats.get('total_fees', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value {fr_class}">{friction_val:.1f}%</div>
            <div class="kpi-label">Friction Impact</div>
            <div style="font-size: 10px; color: #8b949e;">Fees: ${friction_stats.get('total_fees', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # 🏆 STRATEGY LEADERBOARD (Phase 6)
    # ----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🏆 Strategy Competition Leaderboard")
    
    if 'strategy_id' in trades.columns and not trades.empty:
        # Group by strategy
        strategies = trades.groupby('strategy_id')
        leaderboard_data = []
        
        for strat_id, strat_trades in strategies:
            if len(strat_trades) < 1: continue
            
            # Helper to safely get stats
            try:
                # Calculate Stress Metrics (Phase 6)
                stress = {}
                if len(strat_trades) >= 20:
                    # Extract PnL % for MC
                    # Assuming we can derive pnl_pct for simple simulation
                    # If pnl_pct missing, approximate via pnl/entry_price*qty
                    pnl_returns = []
                    for _, t in strat_trades.iterrows():
                        try:
                            # Try to infer return %
                            if t['entry_price'] > 0 and t['quantity'] > 0:
                                cost = t['entry_price'] * t['quantity']
                                ret = t['net_pnl'] / cost
                                pnl_returns.append(ret)
                        except:
                            pass
                            
                    if len(pnl_returns) > 10:
                        paths = StatisticsPro.generate_monte_carlo_paths(pnl_returns, n_sims=1000)
                        stress = StatisticsPro.calculate_stress_metrics(paths)

                s_wr = AnalyticsEngine.calculate_winrate_details(strat_trades)
                s_exp = AnalyticsEngine.calculate_expectancy(strat_trades)
                
                leaderboard_data.append({
                    "Strategy": strat_id,
                    "Net PnL": strat_trades['net_pnl'].sum(),
                    "Trades": len(strat_trades),
                    "Win Rate": f"{s_wr.get('global_winrate', 0):.1f}%",
                    "Profit Factor": f"{s_wr.get('profit_factor', 0):.2f}",
                    "Expectancy": f"${s_exp.get('expectancy', 0):.4f}",
                    "Kelly %": f"{s_exp.get('kelly_percent', 0):.1f}%",
                    "Stress Score": f"{stress.get('stress_score', 0):.0f}" if stress else "N/A",
                    "PoR": f"{stress.get('por', 0):.1f}%" if stress else "N/A"
                })
            except Exception as e:
                continue
                
        if leaderboard_data:
            lb_df = pd.DataFrame(leaderboard_data).sort_values("Net PnL", ascending=False)
            
            # Format visual
            st.dataframe(
                lb_df,
                column_config={
                    "Net PnL": st.column_config.NumberColumn(format="$%.2f"),
                    "Stress Score": st.column_config.ProgressColumn(
                        "Stress Score (0-100)",
                        min_value=0,
                        max_value=100,
                        format="%d"
                    ),
                    "PoR": "Risk of Ruin %"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # ----------------------------------------------------------------------
            # 🎲 MONTE CARLO PROJECTION (Fan Chart)
            # ----------------------------------------------------------------------
            st.markdown("---")
            st.subheader("🎲 Monte Carlo Projection (Phase 6)")
            st.caption("Stochastic projection of 1,000 future equity paths based on historical returns.")
            
            # Use top strategy or overall trades
            if len(trades) >= 20: 
                # Extract global PnL %
                global_returns = []
                for _, t in trades.iterrows():
                     if t['entry_price'] > 0 and t['quantity'] > 0:
                         cost = t['entry_price'] * t['quantity']
                         ret = t['net_pnl'] / cost
                         global_returns.append(ret)
                
                if len(global_returns) > 10:
                    sim_paths = StatisticsPro.generate_monte_carlo_paths(global_returns, n_sims=1000, n_period=50)
                    
                    # Calculate Percentiles (P10, P50, P90)
                    p10 = np.percentile(sim_paths, 10, axis=0)
                    p50 = np.percentile(sim_paths, 50, axis=0)
                    p90 = np.percentile(sim_paths, 90, axis=0)
                    x_axis = list(range(len(p50)))
                    
                    fig_mc = go.Figure()
                    
                    # Fan Chart
                    fig_mc.add_trace(go.Scatter(
                        x=x_axis, y=p90,
                        mode='lines',
                        line=dict(width=0),
                        name='Optimistic (P90)',
                        showlegend=False
                    ))
                    
                    fig_mc.add_trace(go.Scatter(
                        x=x_axis, y=p10,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 255, 255, 0.1)',
                        name='Pessimistic (P10)'
                    ))
                    
                    fig_mc.add_trace(go.Scatter(
                        x=x_axis, y=p50,
                        mode='lines',
                        line=dict(color='#3fb950', width=2),
                        name='Median Projection'
                    ))
                    
                    # Ruin Line
                    fig_mc.add_hline(y=50, line_dash="drive", line_color="red", annotation_text="Limit of Ruin (-50%)")
                    
                    fig_mc.update_layout(
                        template="plotly_dark",
                        height=400,
                        title="Equity Curve Projection (Next 50 Trades)",
                        yaxis_title="projected Equity (Normalized)",
                        margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_mc, key="mc_fan_chart")
                    
            else:
                st.info("ℹ️ Need at least 20 trades to run Monte Carlo Simulation.")
                
        else:
            st.info("Waiting for strategy data...")
    else:
        st.info("ℹ️ Strategy Leaderboard requires 'strategy_id' in trades.csv (Phase 6 update)")

    st.markdown("---")
    st.subheader("🧮 Expectancy Engine (Phase 5)")
    
    if expectancy_stats.get('status') == 'INSUFFICIENT_DATA':
        st.warning("⚠️ Insufficient data for Expectancy Analysis (Min 10 trades required)")
    elif expectancy_stats:
        exp_cols = st.columns(4)
        
        # Expectancy Metric
        exp_val = expectancy_stats.get('expectancy', 0)
        exp_color = "kpi-positive" if exp_val > 0 else "kpi-negative"
        with exp_cols[0]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value {exp_color}">${exp_val:.4f}</div>
                <div class="kpi-label">Expectancy / Trade</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Kelly Criterion
        kelly = expectancy_stats.get('kelly_percent', 0)
        kelly_color = "kpi-positive" if kelly > 0 else "kpi-warning"
        with exp_cols[1]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value {kelly_color}">{kelly:.1f}%</div>
                <div class="kpi-label">Kelly Suggestion</div>
            </div>
            """, unsafe_allow_html=True)
            
        # R/R Ratio
        rr = expectancy_stats.get('reward_risk', 0)
        with exp_cols[2]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{rr:.2f}</div>
                <div class="kpi-label">Reward/Risk Ratio</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Avg Win/Loss
        avg_w = expectancy_stats.get('avg_win', 0)
        avg_l = expectancy_stats.get('avg_loss', 0)
        with exp_cols[3]:
             st.markdown(f"""
            <div class="kpi-card" style="font-size: 10px;">
                <div style="color: #3fb950; font-weight: bold;">Avg Win: ${avg_w:.2f}</div>
                <div style="color: #f85149; font-weight: bold;">Avg Loss: ${avg_l:.2f}</div>
                <div class="kpi-label">Averages</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("📜 Trade History")
    
    if not trades.empty:
        # Display recent trades
        display_cols = [c for c in ['datetime', 'symbol', 'direction', 'quantity', 'price', 'pnl'] if c in trades.columns]
        if display_cols:
            st.dataframe(
                trades[display_cols].sort_values('datetime', ascending=False).head(20),
                hide_index=True
            )
    else:
        st.info("No trade history available")

# ------------------------------------------------------------------------------
# TAB 3: REVERSE CONTROL (Phase 5)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🔄 Intelligent Reverse Operations (Phase 5)")
    
    st.markdown("""
    **Configuration:**
    - Max Flips per Day: `{}`
    - Min Signal Strength: `{}`
    - Cooldown Period: `{} seconds`
    """.format(
        getattr(Config, 'FLIP_MAX_DAILY_COUNT', 1),
        getattr(Config, 'FLIP_MIN_SIGNAL_STRENGTH', 0.8),
        getattr(Config, 'FLIP_COOLDOWN_SECONDS', 300)
    ))
    
    st.markdown("---")
    
    # Reverse signals from logs
    st.subheader("📊 Recent REVERSE Signals")
    
    reverse_logs = [log for log in recent_logs if 'REVERSE' in log.get('message', '').upper()]
    if reverse_logs:
        for log in reverse_logs[-5:]:
            st.markdown(f"""
            <div style="background: #161b22; border-left: 3px solid #d29922; padding: 8px 12px; margin-bottom: 4px; font-size: 12px;">
                <span style="color: #8b949e;">{log.get('timestamp', '')}</span>
                <span style="color: #e0e0e0;">{log.get('message', '')}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No REVERSE signals in recent logs")

# ------------------------------------------------------------------------------
# TAB 4: SYSTEM HEALTH (Phase 6 CI-HMA)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🏥 System Health & Integrity (CI-HMA)")
    
    health_logs = load_health_logs(50)
    
    if health_logs:
        h_col1, h_col2 = st.columns([3, 1])
        
        with h_col1:
            # Latency Chart
            timestamps = [h['timestamp'][11:19] for h in health_logs]
            latencies = [h.get('ui_latency_sec', 0) for h in health_logs]
            
            fig_h = go.Figure()
            fig_h.add_trace(go.Bar(
                x=timestamps, 
                y=latencies,
                name="UI Latency (s)",
                marker_color=['#f85149' if l > 5 else '#3fb950' for l in latencies]
            ))
            fig_h.update_layout(
                title="End-to-End Latency (UI Lag)",
                template="plotly_dark",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_h, key="health_latency_chart")
            
        with h_col2:
            # Stats
            ok_count = sum(1 for h in health_logs if h['status'] == "OK")
            fail_count = len(health_logs) - ok_count
            
            st.metric("Health Checks (last 50)", len(health_logs))
            st.metric("Sync OK", ok_count, delta_color="normal")
            if fail_count > 0:
                st.metric("Sync Errors", fail_count, delta_color="inverse")
            
            last_check = health_logs[-1]
            st.info(f"Last Check: {last_check['status']}")
            
        st.markdown("### 📋 Health Check Log")
        
        # Table
        h_df = pd.DataFrame(health_logs)
        st.dataframe(
            h_df[['timestamp', 'status', 'api_balance', 'file_balance', 'ui_balance', 'notes']].sort_values('timestamp', ascending=False),
            hide_index=True,
            use_container_width=True
        )
            
    else:
        st.info("ℹ️ Waiting for Supervisor Agent logs...")
        st.caption("The HealthSupervisor runs in the backend every 60s.")

# ------------------------------------------------------------------------------
# TAB 5: SYSTEM LOGS
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("📝 System Logs")
    
    log_level_filter = st.selectbox("Log Level", ["ALL", "INFO", "WARNING", "ERROR"], index=0)
    
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    
    if recent_logs:
        for log in reversed(recent_logs[-30:]):
            level = log.get('level', 'INFO')
            
            # Filter by level
            if log_level_filter != "ALL" and level != log_level_filter:
                continue
            
            # Color by level
            level_colors = {
                'DEBUG': '#8b949e',
                'INFO': '#58a6ff',
                'WARNING': '#d29922',
                'ERROR': '#f85149',
                'CRITICAL': '#ff7b72'
            }
            color = level_colors.get(level, '#e0e0e0')
            
            timestamp = log.get('timestamp', '')[:19]
            message = log.get('message', '')[:100]
            
            st.markdown(f"""
            <div style="font-family: monospace; font-size: 11px; margin-bottom: 2px;">
                <span style="color: #8b949e;">{timestamp}</span>
                <span style="color: {color}; font-weight: 600;">[{level}]</span>
                <span style="color: #e0e0e0;">{message}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No logs available. Logs will appear when the bot starts generating events.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# AUTO REFRESH
# ==============================================================================
if st.session_state.auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
