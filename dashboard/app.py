import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
from datetime import datetime

st.set_page_config(page_title="Trader Gemini Dashboard", layout="wide", page_icon="🚀")

import sys
import os

# Add project root to sys.path to allow importing config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import Config
    # Use the exact same DATA_DIR as the bot
    DATA_DIR = Config.DATA_DIR
    is_futures = Config.BINANCE_USE_FUTURES
except ImportError:
    # Fallback if config cannot be imported (should not happen if run correctly)
    st.error("⚠️ Could not import Config. Please run from project root.")
    DATA_DIR = "dashboard/data"
    is_futures = False

LIVE_STATUS_PATH = os.path.join(DATA_DIR, "live_status.json")
TRADES_PATH = os.path.join(DATA_DIR, "trades.csv")
STATUS_PATH = os.path.join(DATA_DIR, "status.csv")

mode_label = "FUTURES (20x) 🚀" if is_futures else "SPOT 🛡️"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.info(f"Mode: **{mode_label}**")
    auto_refresh = st.checkbox("Enable Auto-Refresh (5s)", value=True)
    show_history = st.checkbox("Load Full History", value=True)
    
    if st.button("🔄 Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("Trader Gemini v2.1 (Aggressive)")

# --- MAIN LAYOUT ---
st.title(f"Trader Gemini Monitor")

# Load Live Data
def load_live_data():
    try:
        if os.path.exists(LIVE_STATUS_PATH):
            with open(LIVE_STATUS_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        pass
    return None

live_data = load_live_data()

# Status Banner
if live_data:
    status = live_data.get('status', 'ONLINE')
    regime = live_data.get('regime', 'UNKNOWN')
    last_update = live_data.get('timestamp', 'Unknown')
    
    if status == 'OFFLINE':
        st.error(f"🔴 **BOT OFFLINE** (Last seen: {last_update})")
    else:
        st.success(f"🟢 **BOT ONLINE** | Regime: **{regime}** | Last Update: {last_update}")

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    equity = live_data.get('total_equity', 0)
    cash = live_data.get('cash', 0)
    realized = live_data.get('realized_pnl', 0)
    unrealized = live_data.get('unrealized_pnl', 0)
    
    col1.metric("💰 Total Equity", f"${equity:,.2f}", delta=f"${equity - 10000:,.2f}")
    col2.metric("💵 Cash Available", f"${cash:,.2f}")
    col3.metric("✅ Realized PnL", f"${realized:,.2f}")
    col4.metric("⏳ Unrealized PnL", f"${unrealized:,.2f}")

else:
    st.warning("⚠️ Waiting for bot data...")

# --- TABS ---
tab_live, tab_analytics, tab_history = st.tabs(["⚡ Live Monitor", "📊 Analytics & Strategy", "📜 Trade History"])

# === TAB 1: LIVE MONITOR ===
with tab_live:
    if live_data:
        # Active Positions
        st.subheader("Active Positions")
        positions = live_data.get('positions', {})
        
        active_pos = []
        for symbol, data in positions.items():
            if data['quantity'] != 0:
                current_val = data['quantity'] * data['current_price']
                cost_basis = data['quantity'] * data['avg_price']
                pnl = current_val - cost_basis
                pnl_pct = (pnl / cost_basis) * 100 if cost_basis != 0 else 0
                
                active_pos.append({
                    'Symbol': symbol,
                    'Side': 'LONG' if data['quantity'] > 0 else 'SHORT',
                    'Size': f"{abs(data['quantity']):.4f}",
                    'Entry': data['avg_price'],
                    'Current': data['current_price'],
                    'Value': current_val,
                    'PnL ($)': pnl,
                    'PnL (%)': pnl_pct
                })
        
        if active_pos:
            df_pos = pd.DataFrame(active_pos)
            
            def color_pnl(val):
                color = '#2ecc71' if val > 0 else '#e74c3c' if val < 0 else 'gray'
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(
                df_pos.style.format({
                    'Entry': '${:,.4f}',
                    'Current': '${:,.4f}',
                    'Value': '${:,.2f}',
                    'PnL ($)': '${:,.2f}',
                    'PnL (%)': '{:,.2f}%'
                }).applymap(color_pnl, subset=['PnL ($)', 'PnL (%)']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("💤 No active positions. Waiting for signals...")
            
    # Session Chart (Fast)
    st.subheader("Session Performance")
    @st.cache_data(ttl=10)
    def load_session_data():
        try:
            if os.path.exists(STATUS_PATH):
                df = pd.read_csv(STATUS_PATH, usecols=['timestamp', 'total_equity'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.tail(500)
        except: pass
        return pd.DataFrame()

    session_df = load_session_data()
    if not session_df.empty:
        fig = px.area(session_df, x='timestamp', y='total_equity', title="Equity Curve (Last 8 Hours)")
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# === TAB 2: ANALYTICS ===
with tab_analytics:
    st.header("Strategy Performance")
    
    # Calculate from trades.csv (more reliable)
    @st.cache_data(ttl=5)
    def load_strategy_performance():
        try:
            if os.path.exists(TRADES_PATH):
                df = pd.read_csv(TRADES_PATH)
                
                if 'direction' not in df.columns or 'symbol' not in df.columns:
                    return {}
                    
                strategy_pnl = {}
                
                for symbol in df['symbol'].unique():
                    symbol_trades = df[df['symbol'] == symbol].sort_values('datetime') if 'datetime' in df.columns else df[df['symbol'] == symbol]
                    buy_queue = []
                    
                    for _, row in symbol_trades.iterrows():
                        if row['direction'] == 'BUY':
                            strat_id = row.get('strategy_id', 'Unknown')
                            buy_queue.append((row['quantity'], row['price'], strat_id))
                        elif row['direction'] == 'SELL':
                            remaining_sell = row['quantity']
                            sell_price = row['price']
                            
                            while remaining_sell > 0 and buy_queue:
                                buy_qty, buy_price, strat_id = buy_queue[0]
                                match_qty = min(buy_qty, remaining_sell)
                                pnl = match_qty * (sell_price - buy_price)
                                
                                if strat_id not in strategy_pnl:
                                    strategy_pnl[strat_id] = {'pnl': 0, 'wins': 0, 'losses': 0}
                                
                                strategy_pnl[strat_id]['pnl'] += pnl
                                if pnl > 0:
                                    strategy_pnl[strat_id]['wins'] += 1
                                else:
                                    strategy_pnl[strat_id]['losses'] += 1
                                
                                if match_qty == buy_qty:
                                    buy_queue.pop(0)
                                else:
                                    buy_queue[0] = (buy_qty - match_qty, buy_price, strat_id)
                                
                                remaining_sell -= match_qty
                
                return strategy_pnl
        except Exception as e:
            st.error(f"Error: {e}")
        return {}
    
    strat_perf = load_strategy_performance()
    
    if strat_perf:
        strat_data = []
        for strat, data in strat_perf.items():
            strat_name = {
                '1': 'Technical', '2': 'Statistical', '3': 'ML',
                1: 'Technical', 2: 'Statistical', 3: 'ML'
            }.get(strat, f'Strategy {strat}')
            
            strat_data.append({
                'Strategy': strat_name, 
                'PnL': data['pnl'],
                'Wins': data['wins'],
                'Losses': data['losses']
            })
        
        df_strat = pd.DataFrame(strat_data).sort_values('PnL', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                df_strat.style.format({'PnL': '${:,.2f}'}),
                use_container_width=True,
                hide_index=True
            )
            
        with col2:
            fig_bar = px.bar(df_strat, x='Strategy', y='PnL', color='PnL', 
                             color_continuous_scale='RdYlGn', title="PnL by Strategy")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No strategy performance data available yet.")

# === TAB 3: HISTORY ===
with tab_history:
    if show_history:
        st.header("Trade History")
        
        @st.cache_data(ttl=5)
        def load_trades():
            if os.path.exists(TRADES_PATH):
                df = pd.read_csv(TRADES_PATH)
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df.sort_values('datetime', ascending=False)
            return pd.DataFrame()
            
        trades_df = load_trades()
        
        if not trades_df.empty:
            fills = trades_df[trades_df['type'] == 'FILL'].copy()
            
            if not fills.empty:
                if 'fill_cost' not in fills.columns:
                    fills['fill_cost'] = fills['quantity'] * fills['price']
                
                if 'strategy_id' not in fills.columns:
                    fills['strategy_id'] = 'Unknown'

                fills['pnl_usd'] = 0.0
                fills['pnl_pct'] = 0.0
                
                for symbol in fills['symbol'].unique():
                    symbol_trades = fills[fills['symbol'] == symbol].copy()
                    buy_queue = []
                    
                    for idx, row in symbol_trades.iterrows():
                        if row['direction'] == 'BUY':
                            buy_queue.append((row['quantity'], row['price'], idx))
                        elif row['direction'] == 'SELL':
                            remaining_sell = row['quantity']
                            sell_price = row['price']
                            total_pnl = 0
                            
                            while remaining_sell > 0 and buy_queue:
                                buy_qty, buy_price, buy_idx = buy_queue[0]
                                match_qty = min(buy_qty, remaining_sell)
                                pnl = match_qty * (sell_price - buy_price)
                                total_pnl += pnl
                                
                                if match_qty == buy_qty:
                                    buy_queue.pop(0)
                                else:
                                    buy_queue[0] = (buy_qty - match_qty, buy_price, buy_idx)
                                
                                remaining_sell -= match_qty
                            
                            if row['quantity'] > 0:
                                fills.at[idx, 'pnl_usd'] = total_pnl
                                avg_cost = row['fill_cost'] - total_pnl
                                if avg_cost > 0:
                                    fills.at[idx, 'pnl_pct'] = (total_pnl / avg_cost) * 100

                display_cols = ['datetime', 'symbol', 'direction', 'quantity', 'price', 'fill_cost', 'pnl_usd', 'pnl_pct', 'strategy_id']
                display_df = fills[display_cols].copy()
                
                display_df['price'] = display_df['price'].apply(lambda x: f'${x:,.4f}')
                display_df['fill_cost'] = display_df['fill_cost'].apply(lambda x: f'${x:,.2f}')
                display_df['pnl_usd'] = display_df['pnl_usd'].apply(lambda x: f'${x:,.2f}' if x != 0 else '-')
                display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f'{x:+.2f}%' if x != 0 else '-')
                
                display_df.columns = ['Fecha', 'Símbolo', 'Dirección', 'Cantidad', 'Precio', 'Costo', 'PnL USD', 'PnL %', 'Estrategia']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No filled trades yet.")
        else:
            st.info("No trade history found.")

# Auto Refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()
