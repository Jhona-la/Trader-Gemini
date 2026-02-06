import os
import sys
import time
import asyncio
import signal
try:
    import ujson as json
except ImportError:
    import json
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Dict
from dataclasses import dataclass
import numpy as np

# Imports locales
from config import Config
from core.engine import Engine
from data.binance_loader import BinanceData
from data.sentiment_loader import SentimentLoader
from strategies.ml_strategy import UniversalEnsembleStrategy as MLStrategy  # ← UNIVERSAL ENSEMBLE FOR ALL SYMBOLS
from core.portfolio import Portfolio
from core.events import OrderEvent, SignalEvent
from core.enums import OrderSide, OrderType
from core.market_regime import MarketRegimeDetector
from core.order_manager import OrderManager
from core.market_scanner import MarketScanner
from core.strategy_selector import StrategySelector
from execution.binance_executor import BinanceExecutor
from utils.logger import logger
from utils.session_manager import init_session_manager, get_session_manager
from utils.health_supervisor import start_health_supervisor, _supervisor as health_sup # CI-HMA (Phase 6)
from core.data_handler import get_data_handler  # For Dashboard persistence
from utils.reloader import init_hot_reload, get_hot_reload_manager  # Hot Reload System
from utils.heartbeat import get_heartbeat

# ==================== NUEVAS CLASES ====================

@dataclass
class TradeRecord:
    """Registro individual de trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    strategy: str = ""
    signal_strength: float = 0.0
    position_side: str = "LONG"
    closed: bool = False
    
    @property
    def duration_seconds(self):
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return (datetime.now(timezone.utc) - self.entry_time).total_seconds()


class PerformanceTracker:
    """Track avanzado de performance (asíncrono)."""
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.trades: List[TradeRecord] = []
        self.open_trades: Dict[str, TradeRecord] = {}
        self.start_time = datetime.now(timezone.utc)
        
    def open_trade(self, symbol: str, price: float, quantity: float, 
                   strategy: str, side: str = "LONG") -> str:
        """Registrar apertura de trade."""
        trade_id = f"{symbol}_{int(time.time())}"
        trade = TradeRecord(
            symbol=symbol,
            entry_time=datetime.now(timezone.utc),
            entry_price=price,
            quantity=quantity,
            strategy=strategy,
            position_side=side
        )
        self.open_trades[trade_id] = trade
        return trade_id
    
    def close_trade(self, trade_id: str, exit_price: float, fees: float = 0.0):
        """Cerrar trade y calcular P&L."""
        if trade_id not in self.open_trades:
            return None
        
        trade = self.open_trades[trade_id]
        trade.exit_time = datetime.now(timezone.utc)
        trade.exit_price = exit_price
        trade.fees = fees
        trade.closed = True
        
        # Calcular P&L
        if trade.position_side == "LONG":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity - fees
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity - fees
        
        trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
        
        # Mover a historial
        self.trades.append(trade)
        del self.open_trades[trade_id]
        
        # Actualizar capital
        self.current_capital += trade.pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        logger.info(f"📊 Trade {trade_id}: P&L ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
        return trade
    
    def get_statistics(self) -> Dict:
        """Obtener estadísticas completas."""
        if not self.trades:
            return None
        
        closed_trades = [t for t in self.trades if t.closed]
        if not closed_trades:
            return None
        
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        total_fees = sum(t.fees for t in closed_trades)
        
        stats = {
            'total_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(closed_trades) if closed_trades else 0,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'net_pnl': total_pnl - total_fees,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown': ((self.peak_capital - self.current_capital) / self.peak_capital) * 100,
            'avg_trade_duration': np.mean([t.duration_seconds for t in closed_trades]) if closed_trades else 0,
            'roi': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'runtime_hours': (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        }
        
        if wins:
            stats['avg_win'] = np.mean([t.pnl for t in wins])
            stats['largest_win'] = max(t.pnl for t in wins)
        if losses:
            stats['avg_loss'] = np.mean([t.pnl for t in losses])
            stats['largest_loss'] = min(t.pnl for t in losses)
        
        # Sharpe Ratio (simplificado)
        returns = [t.pnl_pct for t in closed_trades]
        if len(returns) > 1 and np.std(returns) > 0:
            stats['sharpe_ratio'] = np.mean(returns) / np.std(returns)
        else:
            stats['sharpe_ratio'] = 0
        
        return stats
    
    def print_summary(self):
        """Imprimir resumen de performance."""
        stats = self.get_statistics()
        if not stats:
            logger.info("📭 No trades executed yet")
            return
        
        logger.info("\n" + "="*60)
        logger.info("📈 PERFORMANCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Capital: ${self.initial_capital:.2f} → ${stats['current_capital']:.2f} "
                   f"(ROI: {stats['roi']:+.2f}%)")
        logger.info(f"Trades: {stats['total_trades']} "
                   f"(W:{stats['wins']} L:{stats['losses']} | WR: {stats['win_rate']*100:.1f}%)")
        logger.info(f"Net P&L: ${stats['net_pnl']:.2f} | Fees: ${stats['total_fees']:.2f}")
        logger.info(f"Sharpe: {stats['sharpe_ratio']:.2f} | Drawdown: {stats['drawdown']:.2f}%")
        logger.info(f"Avg Duration: {stats['avg_trade_duration']/60:.1f} min")
        logger.info(f"Runtime: {stats['runtime_hours']:.1f} hours")
        logger.info("="*60 + "\n")


class SessionFilter:
    """Filtro por sesiones de mercado."""
    @staticmethod
    def is_active_session() -> bool:
        """Verificar si estamos en sesión activa."""
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour
        
        # Sesiones (ajustar según Config)
        london_active = 8 <= hour_utc < 17  # 8 AM - 5 PM UTC
        ny_active = 13 <= hour_utc < 22     # 1 PM - 10 PM UTC
        
        return london_active or ny_active
    
    @staticmethod
    def get_session_name() -> str:
        """Obtener nombre de sesión actual."""
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour
        
        if 8 <= hour_utc < 17 and 13 <= hour_utc < 17:
            return "LONDON_NY_OVERLAP"
        elif 8 <= hour_utc < 17:
            return "LONDON"
        elif 13 <= hour_utc < 22:
            return "NEW_YORK"
        return "ASIA/CLOSED"


class ScalpingOptimizer:
    """Optimizador específico para scalping."""
    def __init__(self, portfolio: Portfolio, risk_manager: RiskManager):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.last_trade_time = {}
        self.trade_counts = {}
        self.max_trades_per_hour = 12  # Límite conservador
        
    def can_trade(self, symbol: str) -> bool:
        """Verificar si podemos hacer otro trade en este símbolo."""
        now = time.time()
        last_trade = self.last_trade_time.get(symbol, 0)
        
        # Esperar mínimo 5 minutos entre trades en mismo símbolo
        if now - last_trade < 300:
            return False
        
        # Límite de trades por hora
        hour_key = f"{symbol}_{datetime.now().hour}"
        if self.trade_counts.get(hour_key, 0) >= self.max_trades_per_hour:
            return False
        
        return True
    
    def record_trade(self, symbol: str):
        """Registrar que se ejecutó un trade."""
        now = time.time()
        self.last_trade_time[symbol] = now
        
        hour_key = f"{symbol}_{datetime.now().hour}"
        self.trade_counts[hour_key] = self.trade_counts.get(hour_key, 0) + 1


async def meta_brain_loop(selector: StrategySelector):
    """
    Background loop for the Sovereign Meta-Brain.
    Re-evaluates strategy performance every 2 hours.
    """
    logger.info("🧠 Sovereign Meta-Brain Active. Monitoring strategy health...")
    while True:
        try:
            selector.update_strategy_rankings()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"❌ Meta-Brain Loop Error: {e}")
        
        await asyncio.sleep(7200) # 2 Hours

async def global_regime_loop(detector: MarketRegimeDetector, data_handler: BinanceData, risk_manager: RiskManager, portfolio: Portfolio):
    """
    Sovereign Market Context Loop (Phase 8.1).
    Aggregates sentiment across all active symbols to calculate market breadth.
    """
    logger.info("📡 Regime Orchestrator: Monitoring Market Breadth Context...")
    while True:
        try:
            # 1. Get Active Symbols
            active_symbols = data_handler.symbol_list
            context_data = {}
            
            # 2. Gather MTF data for all symbols in the basket
            # PROFESSOR: Analizamos 'enjambre' para no depender solo de BTC.
            for symbol in active_symbols:
                bars_1m = data_handler.get_latest_bars(symbol, n=100, timeframe='1m')
                bars_5m = data_handler.get_latest_bars(symbol, n=50, timeframe='5m')
                bars_1h = data_handler.get_latest_bars(symbol, n=50, timeframe='1h')
                
                if bars_1m:
                    context_data[symbol] = {
                        '1m': bars_1m,
                        '5m': bars_5m,
                        '1h': bars_1h
                    }
            
            if context_data:
                # 3. Calculate Sovereign Context (Breadth)
                breadth = detector.calculate_market_context(context_data)
                
                # 4. Broadcast to Risk Manager & Portfolio
                risk_manager.update_global_regime(breadth['sentiment'])
                portfolio.global_regime_data = breadth  # Richer data for Dashboard
                portfolio.global_regime = breadth['sentiment'] # Compatibility
            else:
                logger.warning("⏳ Regime Orchestrator: Waiting for market history...")
                
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"❌ Regime Orchestrator Error: {e}")
            await asyncio.sleep(60)

async def order_manager_loop(manager: OrderManager):
    """
    Phase 9: Anti-Liquidity Sniping Loop.
    Runs every second to monitor and cancel stale limit orders.
    """
    logger.info("📡 Order Manager: Active Order Lifecycle Protection enabled.")
    while True:
        try:
            await manager.monitor_lifecycle()
            await asyncio.sleep(1) # Check every second
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"❌ Order Manager Error: {e}")
            await asyncio.sleep(5)

async def market_adaptive_loop(engine: Engine, data_handler: BinanceData, scanner: MarketScanner, 
                              portfolio: Portfolio, risk_manager: RiskManager, sentiment_loader: SentimentLoader,
                              events_queue: Any):
    """
    Background loop that periodically re-evaluates the best coins to trade.
    """
    logger.info("🧬 Starting Adaptive Market Optimizer...")
    
    # Weight settings
    ADAPTIVE_INTERVAL = 4 * 3600 # Every 4 hours
    
    while True:
        try:
            # 1. Scan for Top Performers
            top_symbols = scanner.get_top_ranked_symbols(limit=20) 
            if not top_symbols:
                await asyncio.sleep(300)
                continue
                
            # 2. Identify Changes
            current_symbols = data_handler.symbol_list
            to_add = [s for s in top_symbols if s not in current_symbols]
            
            # 3. Handle Retirements (Safety First)
            to_remove = []
            for s in current_symbols:
                if s not in top_symbols:
                    # SAFETY: Do NOT remove if we have an active position
                    pos = portfolio.positions.get(s, {'quantity': 0})
                    if pos['quantity'] == 0:
                        to_remove.append(s)
                    else:
                        logger.info(f"⏳ Postponing removal of {s} due to active position.")
            
            if to_add or to_remove:
                logger.info(f"✨ Adaptive Swap: Adding {to_add}, Removing {to_remove}")
                
                # A. Unregister old strategies
                for s in to_remove:
                    engine.unregister_strategy(s)
                
                # B. Update Data Layer subscriptions
                new_list = [s for s in current_symbols if s not in to_remove] + to_add
                await data_handler.update_symbol_list(new_list)
                
                # C. Wait for new symbols history to load (simple delay)
                if to_add:
                    logger.info("📡 Waiting 30s for new symbol history...")
                    await asyncio.sleep(30)
                
                # D. Register new strategies
                for s in to_add:
                    try:
                        is_leader = ('BTC' in s)
                        ml_strat = MLStrategy(
                            data_provider=data_handler,
                            events_queue=events_queue,
                            symbol=s,
                            lookback=Config.Strategies.ML_LOOKBACK_BARS,
                            sentiment_loader=sentiment_loader,
                            portfolio=portfolio,
                            risk_manager=risk_manager if is_leader else None
                        )
                        engine.register_strategy(ml_strat)
                    except Exception as e:
                        logger.error(f"Failed to spawn adaptive strategy for {s}: {e}")

                logger.info("✅ Adaptive Swap Complete.")
            
            # Sleep until next scan
            await asyncio.sleep(ADAPTIVE_INTERVAL)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Market Adaptive Loop Error: {e}")
            await asyncio.sleep(600)


# ==================== MAIN ACTUALIZADO ====================

async def main():
    # 0. ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='Trader Gemini Bot - Scalping Optimized')
    parser.add_argument('--mode', type=str, choices=['spot', 'futures', 'scalping'], 
                       default='futures', help='Trading mode (Exclusive: futures)')
    parser.add_argument('--capital', type=float, default=15.0, help='Initial capital in USD')
    parser.add_argument('--symbols', type=str, default=None, help='Specific symbols to trade (comma-separated)')
    args = parser.parse_args()
    
    # 1. SETUP CONFIG
    if args.symbols:
        # Phase 6 Fix: Standardize on SYMBOL/USDT format
        Config.TRADING_PAIRS = [s.strip().upper() if '/' in s else f"{s.strip().upper()[:3]}/{s.strip().upper()[3:]}" for s in args.symbols.split(",")]
        # Refined slash injection for variable base lengths (e.g. BTCUSDT, DOGEUSDT)
        Config.TRADING_PAIRS = []
        for s in args.symbols.split(","):
            s_clean = s.strip().upper().replace("/", "")
            if s_clean.endswith("USDT"):
                Config.TRADING_PAIRS.append(f"{s_clean[:-4]}/USDT")
            else:
                Config.TRADING_PAIRS.append(s.strip().upper())
        logger.info(f"📊 FILTERED SYMBOLS: {Config.TRADING_PAIRS}")

    if args.mode == 'scalping':
        logger.info("🎯 MODE: SCALPING OPTIMIZED")
        Config.BINANCE_USE_FUTURES = False  # Scalping en spot para empezar
        Config.DATA_DIR = "dashboard/data/scalping"
        if not args.symbols:
            Config.TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT']  # Pares principales
        Config.INITIAL_CAPITAL = args.capital
        Config.MAX_CONCURRENT_POSITIONS = 1  # Una posición a la vez para scalping
        Config.POSITION_SIZE_PCT = 0.3  # 30% del capital por trade
    elif args.mode == 'futures':
        Config.BINANCE_USE_FUTURES = True
        Config.DATA_DIR = "dashboard/data/futures"
        Config.INITIAL_CAPITAL = args.capital
        if not args.symbols:
            Config.TRADING_PAIRS = Config.CRYPTO_FUTURES_PAIRS
    else:
        Config.BINANCE_USE_FUTURES = False
        Config.DATA_DIR = "dashboard/data/spot"
        Config.INITIAL_CAPITAL = args.capital
        if not args.symbols:
            Config.TRADING_PAIRS = Config.CRYPTO_SPOT_PAIRS

    
    # Ensure directories
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # 1.5. SESSION MANAGEMENT (Phase 6)
    session_mgr = init_session_manager(Config.DATA_DIR)
    session_id = session_mgr.start_session(
        mode=args.mode,
        symbols=Config.TRADING_PAIRS,
        initial_capital=Config.INITIAL_CAPITAL
    )
    
    # 2. PERFORMANCE TRACKER
    performance = PerformanceTracker(initial_capital=Config.INITIAL_CAPITAL)
    
    logger.info(f"🚀 STARTING TRADER GEMINI [Mode: {args.mode} | Capital: ${Config.INITIAL_CAPITAL}]")
    
    # 3. CORE INITIALIZATION
    import queue
    import threading
    from core.market_scanner import MarketScanner
    from strategies.technical import HybridScalpingStrategy as TechnicalStrategy
    
    # Events Queue (Thread-Safe)
    events_queue = queue.Queue()
    
    # 3.1. PRE-INITIALIZATION DISCOVERY (ELITE PROTOCOL)
    # Instantiate a temporary data handler just for scanning
    temp_loader = BinanceData(events_queue, ["BTC/USDT"]) # Minimal symbols for fast connect
    scanner = MarketScanner(temp_loader)
    
    if not args.symbols:
        logger.info("🔭 [Elite Protocol] Performing autonomous market discovery...")
        top_20 = scanner.get_top_ranked_symbols(limit=20)
        if top_20:
            Config.TRADING_PAIRS = top_20
            logger.info(f"💎 Elite Basket Selected: {len(top_20)} symbols.")
        else:
            logger.warning("⚠️ Discovery yielded no results, using default futures pairs.")
            Config.TRADING_PAIRS = Config.CRYPTO_FUTURES_PAIRS
    
    # Cleanup temp scanner resources
    await temp_loader.shutdown()
    
    # 3.2. REAL DATA HANDLER (With Elite Basket)
    data_handler = BinanceData(events_queue, Config.TRADING_PAIRS)
    
    # 3.2.1. PORTFOLIO & RISK (Restored)
    portfolio = Portfolio(
        initial_capital=Config.INITIAL_CAPITAL,
        csv_path=f"{Config.DATA_DIR}/trades.csv",
        status_path=f"{Config.DATA_DIR}/status.csv"
    )
    risk_manager = RiskManager(
        max_concurrent_positions=getattr(Config, 'MAX_CONCURRENT_POSITIONS', 3),
        portfolio=portfolio
    )

    # 3.3. DATA WARMING BARRIER
    # Wait for parallel workers to fetch enough history for ML
    logger.info("📡 [Elite Protocol] Warming up data for universal training...")
    warming = True
    start_warm = time.time()
    while warming and (time.time() - start_warm < 120): # Max 2 min wait
        ready_count = 0
        with data_handler._data_lock:
            for s in Config.TRADING_PAIRS:
                if len(data_handler.latest_data.get(s, [])) >= 500:
                    ready_count += 1
        
        if ready_count >= len(Config.TRADING_PAIRS):
            logger.info("✅ All elite symbols warmed up.")
            warming = False
        elif (time.time() - start_warm) > 5:
            logger.info(f"⏳ Warming progress: {ready_count}/{len(Config.TRADING_PAIRS)} symbols ready...")
            await asyncio.sleep(5)
        else:
            await asyncio.sleep(1)
    
    # Sentiment Engine
    sentiment_loader = SentimentLoader()
    sentiment_loader.start_background_thread()
    
    # Executor
    # FIXED: Pass actual portfolio instance, not Config class
    executor = BinanceExecutor(events_queue, portfolio=portfolio)
    
    # Engine
    engine = Engine(events_queue)
    engine.register_data_handler(data_handler)
    engine.register_portfolio(portfolio)
    engine.register_risk_manager(risk_manager)
    engine.register_execution_handler(executor)
    
    # SYNC PORTFOLIO WITH BINANCE
    # CRITICAL: This ensures we see manually opened positions or positions from previous run
    logger.info("🔄 Syncing initial portfolio state with Binance...")
    try:
        executor.sync_portfolio_state(portfolio)
    except Exception as e:
        logger.error(f"❌ Failed to sync initial portfolio state: {e}")
    
    # Strategies
    strategies = []
    
    # ML Strategy (one per symbol)
    # EXPANDED: Analyze ALL symbols in Config
    for symbol in Config.TRADING_PAIRS:

        try:
            # ORCHESTRATION (Phase 12): Only BTC drives Global Regime
            is_leader = ('BTC' in symbol)
            
            ml_strat = MLStrategy(
                data_provider=data_handler,
                events_queue=events_queue,
                symbol=symbol,
                lookback=Config.Strategies.ML_LOOKBACK_BARS,
                sentiment_loader=sentiment_loader,
                portfolio=portfolio,
                risk_manager=risk_manager if is_leader else None
            )
            strategies.append(ml_strat)
            engine.register_strategy(ml_strat)
        except Exception as e:
            logger.warning(f"Could not init ML Strategy for {symbol}: {e}")
    
    # Sniper Strategy
    try:
        sniper = SniperStrategy(data_handler, events_queue, executor, portfolio)
        strategies.append(sniper)
        engine.register_strategy(sniper)
    except Exception as e:
        logger.warning(f"Could not init Sniper Strategy: {e}")
    
    # Technical Strategy
    try:
        tech = TechnicalStrategy(data_handler, events_queue)
        strategies.append(tech)
        engine.register_strategy(tech)
    except Exception as e:
        logger.warning(f"Could not init Technical Strategy: {e}")
    
    logger.info(f"[OK] Registered {len(strategies)} strategies in the Engine.")
    
    # 3.5. START CI-HMA SUPERVISOR (Phase 6)
    supervisor = start_health_supervisor()
    logger.info("🩺 CI-HMA Health Supervisor started in background.")

    # Scalping Optimizer (optional)
    scalping_optimizer = ScalpingOptimizer(portfolio, risk_manager)
    
    # --- GRACEFUL SHUTDOWN SYSTEM (Rule 2.1) ---
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    shutdown_requested = False  # Track if shutdown was requested
    
    def signal_handler(signum=None, frame=None):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            logger.info("🛑 Shutdown signal received (SIGINT/SIGTERM)...")
            loop.call_soon_threadsafe(shutdown_event.set)
    
    # Register OS signals (Windows-compatible)
    import sys
    if sys.platform == 'win32':
        # Windows: Use signal.signal directly
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        # Unix: Use asyncio signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                signal.signal(sig, signal_handler)
    
    # Start WebSocket in background
    ws_task = asyncio.create_task(data_handler.start_socket())
    
    # --- HOT RELOAD SYSTEM ---
    hot_reload = init_hot_reload(engine=engine, strategies_path="strategies")
    hot_reload.start()
    
    # --- ADAPTIVE SCANNER & META BRAIN ---
    scanner = MarketScanner(data_handler)
    selector = StrategySelector(portfolio=portfolio, data_provider=data_handler)
    
    # Link to Risk Manager
    if risk_manager:
        risk_manager.strategy_selector = selector
        
    adaptive_task = asyncio.create_task(market_adaptive_loop(
        engine, data_handler, scanner, portfolio, risk_manager, sentiment_loader, events_queue
    ))
    
    meta_task = asyncio.create_task(meta_brain_loop(selector))
    
    # 3.4. REGIME ORCHESTRATOR
    regime_detector = MarketRegimeDetector()
    regime_task = asyncio.create_task(global_regime_loop(regime_detector, data_handler, risk_manager, portfolio))
    
    # PHASE 9: ORDER MANAGER
    order_manager = OrderManager(executor)
    executor.order_manager = order_manager
    engine.register_order_manager(order_manager)
    order_task = asyncio.create_task(order_manager_loop(order_manager))
    
    # 4. MAIN EVENT LOOP
    loop_count = 0
    last_summary_time = time.time()
    last_heartbeat = time.time()
    last_reconcile_time = time.time() # Phase 13
    
    logger.info(f"🔍 DEBUG: shutdown_event is_set={shutdown_event.is_set()}")
    logger.info(" Starting main event loop...")
    logger.info("💡 Press Ctrl+C to stop gracefully...")
    
    while not shutdown_event.is_set():
        try:
            loop_count += 1
            now = time.time()
            
            # Update bars from REST (fallback if WS slow)
            if loop_count % 60 == 0:  # Every ~60 seconds
                try:
                    data_handler.update_bars()
                    # Heartbeat pulse
                    get_heartbeat().pulse(metadata={
                        "loop_count": loop_count,
                        "equity": portfolio.get_total_equity() if portfolio else 0
                    })
                except Exception as e:
                    logger.error(f"Error updating bars/heartbeat: {e}")
            
            # --- HOT RELOAD: Process pending updates ---
            if hot_reload and hot_reload.is_active:
                reload_results = hot_reload.process_pending_reloads()
                for result in reload_results:
                    if result.success:
                        logger.info(f"🔥 [HOT_RELOAD] Applied: {result.module_name} ({result.latency_ms:.1f}ms)")
            
            # Process events
            processed = 0
            while not events_queue.empty() and processed < 50:
                try:
                    event = events_queue.get_nowait()
                    engine.process_event(event)
                    processed += 1
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
            
            # Risk Manager: Check stops
            if loop_count % 10 == 0:
                try:
                    stop_signals = risk_manager.check_stops(portfolio, data_handler)
                    for sig in stop_signals:
                        events_queue.put(sig)
                except Exception as e:
                    logger.error(f"Error checking stops: {e}")
            
            # Update risk manager equity
            if loop_count % 30 == 0:
                try:
                    equity = portfolio.get_total_equity()
                    risk_manager.update_equity(equity)
                except:
                    pass
            
            # RECONCILIATION (Phase 13): Periodic Sync every 60m
            if now - last_reconcile_time >= 3600:
                try:
                    logger.info("♻️ [Auto-Reconcile] Syncing state with Binance...")
                    executor.sync_portfolio_state(portfolio)
                    last_reconcile_time = now
                except Exception as e:
                    logger.error(f"Reconcile failed: {e}")
            
            # Market Intelligence Heartbeat
            if now - last_heartbeat >= 60:
                equity = portfolio.get_total_equity()
                open_pos_symbols = [s for s, p in portfolio.positions.items() if p['quantity'] != 0]
                open_pos = len(open_pos_symbols)
                logger.info(f"💓 Heartbeat | Equity: ${equity:.2f} | Pos: {open_pos} | Loop: {loop_count}")

                logger.info(f"[PORTFOLIO_STATUS] Active: {open_pos_symbols} | Bal: ${equity:.2f}")

                
                # --- MARKET COMMENTARY ---
                analysis_info = []
                # engine.strategies is a list
                for s in engine.strategies:
                    if hasattr(s, 'analysis_stats') and s.analysis_stats['total'] > 0:
                        stats = s.analysis_stats
                        analysis_info.append(f"{s.symbol}: {stats['total']} analyzed")
                
                if analysis_info:
                    logger.info(f"🧬 Market State: Analyzing {len(analysis_info)} pairs... [M1 Scalping Engine Active]")
                    # Mostrar top pairs analizados
                    logger.info(f"📊 Activity: {', '.join(analysis_info[:3])}...")
                
                last_heartbeat = now
                
                # 3.5. HISTORICAL PERSISTENCE (Phase 6)
                try:
                    status_packet = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'total_equity': equity,
                        'available_balance': portfolio.current_cash,
                        'realized_pnl': portfolio.realized_pnl,
                        'unrealized_pnl': portfolio.unrealized_pnl
                    }
                    get_data_handler().append_status_log(portfolio.status_path, status_packet)
                except Exception as e:
                    logger.error(f"Persistence Error: {e}")
            
            # Performance summary
            if now - last_summary_time >= 1800:  # Every 30 minutes
                performance.print_summary()
                last_summary_time = now
            
            # Use wait_for with timeout to allow Ctrl+C to interrupt on Windows
            try:
                # OPTIMIZED: 0.1s for faster signal response (Rule 2.4 Fix)
                await asyncio.wait_for(asyncio.sleep(0.1), timeout=0.2)
            except asyncio.TimeoutError:
                pass
            
        except KeyboardInterrupt:
            logger.info("🛑 Ctrl+C detected, initiating shutdown...")
            shutdown_event.set()
            break
        except Exception as e:
            logger.error(f"❌ Loop error: {e}", exc_info=True)
            await asyncio.sleep(5)
    
    # 5. GRACEFUL SHUTDOWN (Rule 2.4)
    logger.info(f"🛑 Initiating clean stop... (Reason: signal={shutdown_event.is_set()})")
    
    # A. Stop Engine loops
    engine.stop()
    
    # B. Stop Adaptive Tasks
    if 'adaptive_task' in locals():
        adaptive_task.cancel()
    if 'meta_task' in locals():
        meta_task.cancel()
    if 'regime_task' in locals():
        regime_task.cancel()
        
    # C. Close WebSocket & Data Sessions (with timeout to prevent hanging)
    ws_task.cancel()
    try:
        # Use timeout to prevent indefinite blocking
        await asyncio.wait_for(data_handler.shutdown(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("⚠️ Data handler shutdown timed out, forcing close")
    except Exception as e:
        logger.debug(f"Data handler cleanup: {e}")
    
    try:
        await asyncio.wait_for(asyncio.shield(ws_task), timeout=3.0)
    except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as e:
        logger.debug(f"WS Task cleanup: {e}")
    
    # C. Close Portfolio & Database
    portfolio.close() # NEW: Closes DB handler
    performance.print_summary()
    
    # C1. Stop Sentiment Engine
    if 'sentiment_loader' in locals():
        sentiment_loader.stop()
        logger.info("📰 Sentiment Engine stopped.")

    # C2. Stop Health Supervisor

    if supervisor:
        supervisor.stop()
        logger.info("🩺 Health Supervisor stopped.")
    
    # C3. Stop Hot Reload System
    if hot_reload:
        hot_reload.stop()
        logger.info("🔥 Hot Reload System stopped.")
    
    # D. Close Session with Summary (Phase 6)
    session_mgr = get_session_manager()
    if session_mgr:
        session_mgr.end_session({
            'total_trades': len(performance.trades),
            'pnl': performance.current_capital - performance.initial_capital,
            'winning_trades': len([t for t in performance.trades if t.pnl > 0]),
            'losing_trades': len([t for t in performance.trades if t.pnl < 0]),
        })
    
    logger.info("👋 Bot stopped gracefully")


if __name__ == "__main__":
    print("""
    ⚠️  IMPORTANTE PARA SCALPING:
    
    1. CAPITAL INICIAL: ${} (ajustable/dinámico)
    2. MÁXIMO {} POSICIÓN(ES) CONCURRENTES
    3. TAMAÑO DE POSICIÓN: {:.0%} del capital (Micro/Small)
    4. SESIONES ACTIVAS: Londres (8-17 UTC) y NY (13-22 UTC)
    5. COOLDOWN: {} min entre trades en mismo símbolo
    6. RIESGO: {:.1%} por trade (Max Risk)
    
    🎯 OBJETIVO: ${} → $500+ en fases de crecimiento
    📊 MÉTRICAS MÍNIMAS (Adaptive):
       - Min Profit Net > {:.2%}
       - Min R:R > {}:1
       - Max Drawdown < 5%
    
    ¡Éxito! 🚀
    """.format(
        getattr(Config, 'INITIAL_CAPITAL', 15.0),
        getattr(Config, 'MAX_CONCURRENT_POSITIONS', 1),
        getattr(Config, 'POSITION_SIZE_MICRO_ACCOUNT', 0.40),
        getattr(Config, 'COOLDOWN_PERIOD_SECONDS', 300) / 60,
        getattr(Config, 'MAX_RISK_PER_TRADE', 0.01),
        getattr(Config, 'INITIAL_CAPITAL', 15.0),
        getattr(Config, 'MIN_PROFIT_AFTER_FEES', 0.003),
        getattr(Config, 'MIN_RR_RATIO', 1.5)
    ))
    
    # Audit keys before starting (Phase 6 Fix)
    logger.info("🛠️  AUDIT: Checking Configuration...")
    demo_key = os.getenv('BINANCE_DEMO_API_KEY')
    if Config.BINANCE_USE_DEMO:
        if demo_key:
            logger.info(f"✅ Demo Key Loaded: {demo_key[:6]}...{demo_key[-4:]} (Active)")
        else:
            logger.error("❌ Demo Mode Enabled but Demo Key NOT Found!")
    else:
        # In Real/Futures mode, we don't need to spam about Demo keys
        real_key = os.getenv('BINANCE_API_KEY')
        if real_key:
            logger.info(f"✅ Production Key Loaded: {real_key[:6]}...{real_key[-4:]}")
        
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Bot detenido por el usuario")
    except Exception as e:
        logger.critical(f"💥 ERROR FATAL: {e}", exc_info=True)
        sys.exit(1)