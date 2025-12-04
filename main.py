import time
import asyncio
import queue
import signal
import sys
import os
from datetime import datetime
import pandas as pd

from config import Config
from core.engine import Engine
from data.binance_loader import BinanceData
# from data.ibkr_loader import IBKRData  # Disabled - crypto-only focus
from strategies.technical import TechnicalStrategy
from strategies.statistical import StatisticalStrategy
from strategies.ml_strategy import MLStrategy
from strategies.pattern import PatternStrategy # NEW
from core.portfolio import Portfolio
from core.events import OrderEvent, SignalEvent, MarketEvent
from risk.risk_manager import RiskManager
from execution.binance_executor import BinanceExecutor
from data.sentiment_loader import SentimentLoader
from utils.logger import logger

def close_all_positions(portfolio, executor, crypto_symbols):
    """
    Emergency close all open positions.
    Called when bot is stopped (Ctrl+C).
    """
    logger.info("🛑 Shutdown signal received. Closing all open positions...")
    
    closed_count = 0
    for symbol, pos in portfolio.positions.items():
        if pos['quantity'] != 0:
            # Determine direction: SELL for longs, BUY for shorts
            if pos['quantity'] > 0:
                direction = 'SELL'
                quantity = pos['quantity']
            else:
                direction = 'BUY'
                quantity = abs(pos['quantity'])
            
            logger.info(f"  Closing {symbol}: {direction} {quantity} @ Market")
            
            # Create market order to close
            order = OrderEvent(symbol, 'MARKET', quantity, direction)
            
            # Execute immediately
            try:
                executor.execute_order(order)
                closed_count += 1
            except ccxt.NetworkError as e:
                logger.warning(f"  ⚠️  Network error closing {symbol}: {e}")
                logger.warning(f"      Continuing with other positions...")
            except ccxt.ExchangeError as e:
                logger.warning(f"  ⚠️  Exchange error closing {symbol}: {e}")
            except Exception as e:
                logger.error(f"  ⚠️  Unexpected error closing {symbol}: {e}")
    
    if closed_count > 0:
        logger.info(f"✅ Closed {closed_count} position(s). Waiting 2 seconds for settlement...")
        time.sleep(2)
    else:
        logger.info("✅ No open positions to close.")
    
    logger.info("="*50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("="*50)
    
    # Calculate Session Stats
    total_equity = portfolio.get_total_equity()
    pnl = total_equity - portfolio.initial_capital
    pnl_pct = (pnl / portfolio.initial_capital) * 100
    
    logger.info(f"💰 Final Equity:   ${total_equity:,.2f}")
    logger.info(f"📈 Total PnL:      ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    logger.info(f"💵 Cash Balance:   ${portfolio.current_cash:,.2f}")
    logger.info("-" * 50)
    
    # Show recent trades if any
    try:
        if os.path.exists(portfolio.csv_path):
            df = pd.read_csv(portfolio.csv_path)
            if not df.empty:
                logger.info(f"📝 Total Trades:   {len(df)}")
                logger.info("\nLast 5 Trades:")
                logger.info("\n" + df.tail(5)[['symbol', 'direction', 'quantity', 'price']].to_string(index=False))
    except FileNotFoundError:
        logger.info("No trade history file found (this is normal for new sessions)")
    except pd.errors.EmptyDataError:
        logger.warning("Trade history file is empty")
    except KeyError as e:
        logger.error(f"Trade history file missing required columns: {e}")
    except Exception as e:
        logger.error(f"Could not load trade history: {e}")
    
    # FORCE SAVE: Write final dashboard snapshot
    logger.info("💾 Saving session data...")
    try:
        # Save final status snapshot
        status_data = {
            'timestamp': [datetime.now()],
            'total_equity': [total_equity],
            'cash': [portfolio.current_cash],
            'realized_pnl': [portfolio.realized_pnl],
            'unrealized_pnl': [pnl - portfolio.realized_pnl],
            'positions': [str(portfolio.positions).replace("'", '"')]
        }
        df_status = pd.DataFrame(status_data)
        
        status_path = os.path.join(Config.DATA_DIR, "status.csv")
        if os.path.exists(status_path):
            df_status.to_csv(status_path, mode='a', header=False, index=False)
        else:
            df_status.to_csv(status_path, index=False)
        
        logger.info("✅ Dashboard data saved")
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"⚠️  Could not save dashboard data: File system error - {e}")
    except pd.errors.ParserError as e:
        logger.error(f"⚠️  Could not save dashboard data: CSV format error - {e}")
    except Exception as e:
        logger.error(f"⚠️  Could not save dashboard data: {e}")
    
    # CLEANUP: Free memory
    logger.info("🧹 Cleaning up memory...")
    try:
        # Clear large data buffers (helps OS reclaim memory faster)
        if hasattr(portfolio, 'positions'):
            portfolio.positions.clear()
            
        # Close Database Connection
        if hasattr(portfolio, 'db'):
            portfolio.db.close()
            logger.info("✅ Database connection closed.")
            
        # UPDATE DASHBOARD STATUS TO OFFLINE
        import json
        live_status = {
            'timestamp': str(datetime.now()),
            'status': 'OFFLINE',
            'total_equity': total_equity,
            'cash': portfolio.current_cash,
            'realized_pnl': pnl,
            'unrealized_pnl': 0.0,
            'positions': {},
            'regime': 'OFFLINE'
        }
        
        live_status_path = os.path.join(Config.DATA_DIR, "live_status.json")
        with open(live_status_path, "w") as f:
            json.dump(live_status, f)
            
        logger.info("✅ Memory cleanup complete & Dashboard set to OFFLINE")
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"⚠️  Cleanup error: File access issue - {e}")
    except json.JSONDecodeError as e:
        logger.error(f"⚠️  Cleanup error: JSON formatting issue - {e}")
    except Exception as e:
        logger.error(f"⚠️  Cleanup error: {e}")
        
    logger.info("="*50)
    logger.info("🔒 Shutdown complete. All data saved.")

async def main():
    import argparse
    
    # 0. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description='Trader Gemini Bot')
    parser.add_argument('--mode', type=str, choices=['spot', 'futures'], help='Trading mode: spot or futures')
    args = parser.parse_args()
    
    # 1. Override Configuration based on CLI Argument
    if args.mode:
        os.environ['BOT_MODE'] = args.mode # Set env var for logger
        if args.mode == 'futures':
            logger.info("🔵 MODE: FUTURES (Override from CLI)")
            Config.BINANCE_USE_FUTURES = True
        elif args.mode == 'spot':
            logger.info("🟡 MODE: SPOT (Override from CLI)")
            Config.BINANCE_USE_FUTURES = False
            
    # 2. Dynamic Data Directory & Pairs Update (Crucial for Dual Terminal)
    if Config.BINANCE_USE_FUTURES:
        Config.DATA_DIR = "dashboard/data/futures"
        Config.TRADING_PAIRS = Config.CRYPTO_FUTURES_PAIRS
        logger.info(f"📂 Data Directory: {Config.DATA_DIR}")
    else:
        Config.DATA_DIR = "dashboard/data/spot"
        Config.TRADING_PAIRS = Config.CRYPTO_SPOT_PAIRS
        logger.info(f"📂 Data Directory: {Config.DATA_DIR}")
        
    # Ensure directory exists
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR, exist_ok=True)

    logger.info("Starting Trader Gemini...")
    
    # NOTE: status.csv is NOT deleted on startup to preserve historical dashboard data
    # The file will continuously grow with historical snapshots
    
    # 3. Create Event Queue
    events_queue = queue.Queue()
    
    # 4. Initialize Data Handlers
    # Binance (Crypto)
    crypto_symbols = Config.TRADING_PAIRS
    binance_data = BinanceData(events_queue, crypto_symbols)
    
    # Start WebSocket in background
    socket_task = asyncio.create_task(binance_data.start_socket())
    logger.info("🚀 WebSocket Task Started")
    
    # IBKR (Disabled - Crypto-only focus for better win rate)
    # ibkr_symbols = Config.STOCKS + Config.FOREX
    # ibkr_data = IBKRData(events_queue, ibkr_symbols)
    
    # Sentiment Analysis
    sentiment_loader = SentimentLoader()
    sentiment_loader.start_background_thread() # Start non-blocking updates
    
    # Market Regime Detector
    from core.market_regime import MarketRegimeDetector
    regime_detector = MarketRegimeDetector()
    logger.info("[OK] Market Regime Detector initialized")
    
    # 3. Initialize Strategies (CRYPTO ONLY)
    # A. Technical Strategy (RSI)
    tech_strategy_crypto = TechnicalStrategy(binance_data, events_queue)
    
    # B. Statistical Strategy (Pairs Trading - Crypto only)
    # Note: Portfolio is initialized later, so we might need to pass it after init or reorder
    # Let's reorder: Initialize Portfolio FIRST, then Strategies.
    
    # 4. Initialize Portfolio & Risk (Moved UP)
    # FIX: Pass dynamic paths from Config so it uses 'futures' folder when in Futures mode
    portfolio = Portfolio(
        initial_capital=10000.0,
        csv_path=os.path.join(Config.DATA_DIR, "trades.csv"),
        status_path=os.path.join(Config.DATA_DIR, "status.csv")
    )
    
    # Now we can initialize strategies that need portfolio
    stat_strategy_crypto = StatisticalStrategy(binance_data, events_queue, portfolio=portfolio, pair=('ETH/USDT', 'BTC/USDT'))
    
    # C. Pattern Recognition Strategy (Candlestick Patterns)
    pattern_strategy = PatternStrategy(binance_data, events_queue)
    
    # D. Machine Learning Strategy (XGBoost + RF Ensemble)
    ml_strategies = []
    
    # Crypto ML Strategies only (with Sentiment)
    logger.info(f"Initializing ML Strategies for {len(crypto_symbols)} crypto pairs...")
    for symbol in crypto_symbols:
        ml_strat = MLStrategy(binance_data, events_queue, symbol=symbol, sentiment_loader=sentiment_loader)
        ml_strategies.append(ml_strat)
    
    # CRASH RECOVERY: Restore state from SQLite Database
    logger.info("🔄 Checking for previous session state in Database...")
    if portfolio.restore_state_from_db():
        logger.info("✅ Local state restored from DB. Now syncing with Binance...")
    else:
        logger.warning("⚠️ Failed to load local state from DB. Starting fresh.")
        
    risk_manager = RiskManager(max_concurrent_positions=5, portfolio=portfolio)

    # 5. Initialize Engine
    engine = Engine(events_queue)
    engine.register_data_handler(binance_data)
    # engine.register_data_handler(ibkr_data)  # Disabled - crypto only
    
    # 6. Register Execution Handler
    binance_executor = BinanceExecutor(events_queue, portfolio=portfolio)
    engine.register_execution_handler(binance_executor)
    
    # SYNC PORTFOLIO WITH BINANCE (Balance + Positions)
    # This replaces the local JSON state restoration
    binance_executor.sync_portfolio_state(portfolio)
    
    # Priority 1: ML Strategies (most sophisticated - regime aware)
    for strat in ml_strategies:
        engine.register_strategy(strat)
        
    # Priority 2: Technical Strategy (Trend Following & Breakouts)
    engine.register_strategy(tech_strategy_crypto)
    
    # Priority 3: Statistical Strategy (Pairs Trading)
    engine.register_strategy(stat_strategy_crypto)
    
    # Priority 4: Pattern Strategy (Candlestick Reversals)
    engine.register_strategy(pattern_strategy)
    
    logger.info(f"[OK] Registered {len(ml_strategies) + 3} strategies in the Engine.")
    
    engine.register_risk_manager(risk_manager)
    
    # Graceful Shutdown Setup (AFTER binance_executor is created)
    def signal_handler(sig, frame):
        close_all_positions(portfolio, binance_executor, crypto_symbols)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # 9. Run Loop
    # PERIODIC SYNC: Track time for balance synchronization
    import time as time_module
    last_sync_time = time_module.time()
    SYNC_INTERVAL = 3600  # 60 minutes in seconds
    
    try:
        while True:
            # PERIODIC BALANCE SYNC (Every 60 minutes)
            # This prevents drift due to Funding Fees in Futures
            current_time = time_module.time()
            if current_time - last_sync_time >= SYNC_INTERVAL:
                logger.info("🔄 Periodic Balance Sync (60min interval)...")
                binance_executor.sync_portfolio_state(portfolio)
                last_sync_time = current_time
            
            # Update Data (Crypto only)
            binance_data.update_bars()
            # ibkr_data.update_bars()  # Disabled - crypto only
            
            # Trigger strategies by sending MARKET event
            market_event = MarketEvent()
            events_queue.put(market_event)
            
            # Update Portfolio Prices (Crucial for Trailing Stops & PnL)
            for symbol in crypto_symbols:
                try:
                    bars = binance_data.get_latest_bars(symbol, n=1)
                    if bars:
                        # Use latest close (which is current price for incomplete bar)
                        price = bars[-1]['close'] 
                        portfolio.update_market_price(symbol, price)
                except:
                    pass
            
            # REGIME DETECTION: Detect market regime for BTC (market leader)
            try:
                bars_1m_btc = binance_data.get_latest_bars('BTC/USDT', n=50)
                bars_1h_btc = binance_data.get_latest_bars_1h('BTC/USDT', n=200)
                if len(bars_1m_btc) >= 50 and len(bars_1h_btc) >= 200:
                    market_regime = regime_detector.detect_regime('BTC/USDT', bars_1m_btc, bars_1h_btc)
                    
                    # UPDATE RISK MANAGER: Enable dynamic cooldowns based on regime
                    risk_manager.current_regime = market_regime
                    
                    # Log regime every 10 loops to avoid spam
                    if hasattr(risk_manager, 'loop_count'):
                        risk_manager.loop_count += 1
                    else:
                        risk_manager.loop_count = 0
                    
                    if risk_manager.loop_count % 10 == 0:
                        advice = regime_detector.get_regime_advice(market_regime)
                        logger.info(f"📊 Market Regime: {market_regime} - {advice['description']}")
                    
                    # Store regime for strategies to access
                    risk_manager.current_regime = market_regime
                else:
                    risk_manager.current_regime = 'RANGING'  # Default
            except:
                risk_manager.current_regime = 'RANGING'  # Safe default
            
            # Check Trailing Stops & Take Profit Levels
            stop_signals = risk_manager.check_stops(portfolio, binance_data)
            for sig in stop_signals:
                logger.info(f"Risk Manager: Triggering Stop Loss for {sig.symbol}")
                events_queue.put(sig)
            
            # Check for Manual Close Signals from Dashboard
            manual_close_path = "dashboard/data/manual_close.txt"
            if os.path.exists(manual_close_path):
                try:
                    with open(manual_close_path, 'r') as f:
                        manual_symbol = f.read().strip()
                    
                    if manual_symbol and manual_symbol in portfolio.positions:
                        if portfolio.positions[manual_symbol]['quantity'] > 0:
                            logger.info(f"📲 Dashboard: Manual close requested for {manual_symbol}")
                            # Create EXIT signal
                            manual_close_signal = SignalEvent("DASHBOARD", manual_symbol, datetime.now(), 'EXIT', strength=1.0)
                            events_queue.put(manual_close_signal)
                    
                    # Delete the file after reading
                    os.remove(manual_close_path)
                except FileNotFoundError:
                    pass  # File was already removed, ignore
                except PermissionError as e:
                    logger.error(f"Error processing manual close: Permission denied - {e}")
                except Exception as e:
                    logger.error(f"Error processing manual close: {e}")
            
            # Process Events
            while not events_queue.empty():
                event = events_queue.get()
                engine.process_event(event)
            
            # --- DASHBOARD UPDATE (SMART CONDITIONAL LOGGING) ---
            # Only save to CSV when something important happens:
            # 1. Positions changed (opened/closed)
            # 2. Equity moved > $10
            # 3. Every 60 seconds (backup for equity curve)
            try:
                # Calculate totals
                total_equity = portfolio.get_total_equity()
                cash = portfolio.current_cash
                realized_pnl = portfolio.realized_pnl
                unrealized_pnl = total_equity - portfolio.initial_capital - realized_pnl
                
                # Track state changes
                if not hasattr(portfolio, '_last_log_time'):
                    portfolio._last_log_time = 0
                    portfolio._last_equity = total_equity
                    portfolio._last_positions_count = 0
                    portfolio._last_dashboard_update = 0  # NEW: Separate timer for dashboard
                
                # Count open positions
                open_positions = sum(1 for pos in portfolio.positions.values() if pos['quantity'] != 0)
                
                # Determine if we should log TO CSV
                should_log = False
                log_reason = ""
                
                # Reason 1: Position count changed
                if open_positions != portfolio._last_positions_count:
                    should_log = True
                    log_reason = f"Positions: {portfolio._last_positions_count} → {open_positions}"
                
                # Reason 2: Equity moved > $10
                elif abs(total_equity - portfolio._last_equity) > 10:
                    should_log = True
                    log_reason = f"Equity: ${portfolio._last_equity:.2f} → ${total_equity:.2f}"
                
                # Reason 3: 60 seconds elapsed (backup)
                elif time.time() - portfolio._last_log_time > 60:
                    should_log = True
                    log_reason = "60s interval (backup)"
                
                if should_log:
                    # Format positions for CSV
                    positions_str = str(portfolio.positions).replace("'", '"')
                    
                    # Create DataFrame
                    status_data = {
                        'timestamp': [datetime.now()],
                        'total_equity': [total_equity],
                        'cash': [cash],
                        'realized_pnl': [realized_pnl],
                        'unrealized_pnl': [unrealized_pnl],
                        'positions': [positions_str]
                    }
                    
                    # 1. Append to CSV (Historical Data)
                    df_status = pd.DataFrame(status_data)
                    status_path = os.path.join(Config.DATA_DIR, "status.csv")
                    
                    if os.path.exists(status_path):
                        df_status.to_csv(status_path, mode='a', header=False, index=False)
                    else:
                        df_status.to_csv(status_path, index=False)
                    
                    # Update tracking variables
                    portfolio._last_log_time = time.time()
                    portfolio._last_equity = total_equity
                    portfolio._last_positions_count = open_positions
                    
                    # Print log reason (for debugging)
                    logger.info(f"📊 CSV Logged: {log_reason}")
                
                # 2. Write to JSON (Real-Time Data) - UPDATE EVERY 10 SECONDS (Independent)
                # BUG FIX: This was inside the "if should_log" block, causing stale dashboard data
                current_time_dashboard = time.time()
                if current_time_dashboard - portfolio._last_dashboard_update >= 10:
                    import json
                    live_status = {
                        'timestamp': str(datetime.now()),
                        'status': 'ONLINE', # Explicitly set online
                        'total_equity': total_equity,
                        'cash': cash,
                        'realized_pnl': realized_pnl,
                        'unrealized_pnl': unrealized_pnl,
                        'positions': portfolio.positions,
                        'regime': risk_manager.current_regime if hasattr(risk_manager, 'current_regime') else 'UNKNOWN'
                    }
                    
                    live_status_path = os.path.join(Config.DATA_DIR, "live_status.json")
                    with open(live_status_path, "w") as f:
                        json.dump(live_status, f)
                    
                    portfolio._last_dashboard_update = current_time_dashboard
                        
                    # 3. Save Market Data for Charts (NEW)
                    # We save data for: Active Positions + BTC/USDT (Market Leader)
                    market_data = {}
                    symbols_to_chart = list(portfolio.positions.keys())
                    
                    # Always include BTC/USDT for reference
                    if 'BTC/USDT' not in symbols_to_chart:
                        symbols_to_chart.append('BTC/USDT')
                    
                    for sym in symbols_to_chart:
                        # Get latest 100 bars
                        # Note: binance_data might use different keys if not normalized? 
                        # But we initialized it with Config.TRADING_PAIRS which has BTC/USDT
                        bars = binance_data.get_latest_bars(sym, n=100)
                        if bars:
                            # Convert datetime objects to string for JSON serialization
                            serializable_bars = []
                            for b in bars:
                                bar_copy = b.copy()
                                bar_copy['datetime'] = str(b['datetime'])
                                serializable_bars.append(bar_copy)
                            market_data[sym] = serializable_bars
                        else:
                            # print(f"DEBUG: No bars found for {sym}")
                            pass
                    
                    market_data_path = os.path.join(Config.DATA_DIR, "market_data.json")
                    with open(market_data_path, "w") as f:
                        json.dump(market_data, f)
                    
                    # print(f"DEBUG: Saved market data for {len(market_data)} symbols to {market_data_path}")
                        
            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"Error updating dashboard CSV: File system error - {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Error updating dashboard CSV: JSON error - {e}")
            except KeyError as e:
                logger.error(f"Error updating dashboard CSV: Missing data key - {e}")
            except Exception as e:
                logger.error(f"Error updating dashboard CSV: {e}")
            # ------------------------
            
            # Sleep to respect API limits but allow faster reaction
            # FIXED: Reduced from 5s to 1s to prevent signal staleness (TTL 10s)
            # Sleep to respect API limits but allow faster reaction
            # FIXED: Reduced from 5s to 1s to prevent signal staleness (TTL 10s)
            await asyncio.sleep(1) 
            
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Stopping Trader Gemini...")
        
        # Stop WebSocket
        if binance_data:
            await binance_data.stop_socket()
            
        close_all_positions(portfolio, binance_executor, crypto_symbols)
        engine.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass # Handled inside main
