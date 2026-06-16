import os

# ═══════════════════════════════════════════════════════════════
# HARDWARE UNLOCK (RYZEN 7 5700U OPTIMIZATION - LIVE)
# QUÉ: Fuerza a C++ y Python a usar los 16 hilos del CPU
# POR QUÉ: Reduce drásticamente la latencia de inferencia en Producción.
# ═══════════════════════════════════════════════════════════════
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

try:
    import torch
    torch.set_num_threads(16)
    torch.set_grad_enabled(False)  # Producción tampoco entrena, solo infiere.
except ImportError:
    pass

import sys
import time
import asyncio
import signal

# 🧪 GOD-MODE PRE-FLIGHT AUDIT (Institutional Protocol - Level 0)
# This MUST be the first thing to run before any other imports or logic.
try:
    # Use a local-style import to avoid complexity before audit
    import importlib.util
    spec = importlib.util.spec_from_file_location("pre_flight", "core/pre_flight.py")
    pre_flight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre_flight)
    pre_flight.SystemPreFlight.launch_audit()
except Exception as e:
    print(f"\n🚨 [CRITICAL] God-Mode Audit Bootstrap Failed: {e}")
    sys.exit(1)

# PROTOCOL METAL-CORE OMEGA: Phase 2 (Global Orjson Patch)
# Monkey-patch standard json to use orjson (via FastJson wrapper)
# This forces libraries like 'python-binance' and 'ccxt' to use our high-perf parser.
try:
    import json
    from utils.fast_json import FastJson
    
    # ⚡ CRITICAL: Patching
    json.loads = FastJson.loads
    json.dumps = FastJson.dumps
    json.load = FastJson.load
    json.dump = FastJson.dump
    
    # Also patch ujson if present (common in sensitive libraries)
    import sys
    sys.modules['ujson'] = FastJson
    sys.modules['json'] = FastJson # Aggressive
    
    print("🚀 [Metal-Core] Global ORJSON Monkey-Patch Applied.")
except ImportError as e:
    print(f"⚠️ [Metal-Core] FastJson Patch Failed: {e}")
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
from risk.risk_manager import RiskManager
from strategies.omni_strategy import OmniStrategy # FASE 32: BINOMIO PERFECTO EN PRODUCCION
from core.micro_awareness import MicroAccountAwareness
from core.events import OrderEvent, SignalEvent
from core.enums import OrderSide, OrderType
from core.market_regime import MarketRegimeDetector
from core.order_manager import OrderManager
from core.market_scanner import MarketScanner
from core.strategy_selector import StrategySelector
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from execution.binance_executor import BinanceExecutor
# FASE 30: C++ / Cython Execution Bridge
try:
    from execution.binance_executor_c import FastBinanceExecutor # Asumiendo que se compiló como .pyd/.so
    CYTHON_EXECUTION_AVAILABLE = True
except ImportError:
    CYTHON_EXECUTION_AVAILABLE = False
    
from core.neural_bridge import neural_bridge
from utils.telemetry import telemetry  # Phase 99: Fleet Telemetry
from utils.efficacy_tracker import efficacy_tracker  # Phase 99: RL Feedback
from data.user_stream import UserDataStreamListener  # Phase 99: Manual Close Detection
from core.world_awareness import world_awareness
from utils.session_manager import init_session_manager, get_session_manager
from utils.health_supervisor import start_health_supervisor, _supervisor as health_sup # CI-HMA (Phase 6)
from core.data_handler import get_data_handler  # For Dashboard persistence
from utils.reloader import init_hot_reload, get_hot_reload_manager  # Hot Reload System
from utils.heartbeat import get_heartbeat
from utils.cpu_affinity import CPUManager # Phase 29: CPU Affinity
from utils.network_optimizer import patch_sockets # Phase 31: TCP NoDelay
from utils.timer_resolution import set_high_resolution_timer # Phase 32: Timer Resolution
from utils.dns_cache import cache_dns_lookups # Phase 34: DNS Cache
from utils.shared_memory import SharedStateManager # Phase 36: Memory Mapping
from utils.ntp_monitor import NTPSync # Phase 37: NTP Sync

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


# [FORENSIC CLEANUP] PerformanceTracker REMOVED
# QUÉ: Clase eliminada — duplicaba Portfolio.get_statistics()
# POR QUÉ: Nunca estaba conectada al pipeline real de fills.
#   Portfolio es la SINGLE SOURCE OF TRUTH para trades y PnL.
# CUÁNDO: Auditoría Forense de Límites de Trades (2026-04-20)


# [FORENSIC CLEANUP] SessionFilter + ScalpingOptimizer REMOVED
# QUÉ: Clases eliminadas — SessionFilter nunca se usaba (crypto 24/7),
#   ScalpingOptimizer se instanciaba pero NUNCA se invocaba en el event loop.
# POR QUÉ: Código muerto que añadía complejidad sin funcionalidad.
#   ScalpingOptimizer duplicaba lógica ya cubierta por CooldownManager.
# CUÁNDO: Auditoría Forense de Límites de Trades (2026-04-20)


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
                bars_15m = data_handler.get_latest_bars(symbol, n=50, timeframe='15m')
                bars_1h = data_handler.get_latest_bars(symbol, n=50, timeframe='1h')
                
                if bars_1m is not None and len(bars_1m) > 0:
                    context_data[symbol] = {
                        '1m': bars_1m,
                        '5m': bars_5m,
                        '15m': bars_15m,
                        '1h': bars_1h
                    }
            
            if context_data and len(context_data) > 0:
        # 3. Calculate Sovereign Context (Breadth)
                breadth = detector.calculate_market_context(context_data)
                
                # --- AEGIS-ULTRA: DYNAMIC CORRELATION (Contagion Guard) ---
                # Build returns matrix from 1m contexts
                try:
                    from strategies.stat_arb import StatArbEngine
                    # Extract 1m closes into a matrix [Time, Asset]
                    # This requires alignment. Simplification: Take last 50 returns
                    returns_list = []
                    valid_symbols = []
                    
                    for s, d in context_data.items():
                        c = d.get('1m', [])
                        if len(c) >= 50:
                            closes = c['close'].astype(np.float64)[-50:]
                            rets = np.diff(closes) / closes[:-1]
                            returns_list.append(rets)
                            valid_symbols.append(s)
                            
                    if len(returns_list) >= 2:
                        # Stack arrays (Shape: [n_assets, n_samples] -> Transpose to [n_samples, n_assets])
                        # Truncate to min length just in case
                        min_len = min(len(r) for r in returns_list)
                        aligned_rets = np.array([r[-min_len:] for r in returns_list]).T
                        
                        corr_matrix = StatArbEngine.calculate_correlation_matrix(aligned_rets)
                        avg_corr = StatArbEngine.get_systemic_risk(corr_matrix)
                        
                        breadth['fleet_correlation'] = avg_corr
                        
                        if avg_corr > 0.85:
                            logger.warning(f"☢️ [AEGIS] HIGH SYSTEMIC RISK: Fleet Correlation {avg_corr:.2f} > 0.85")
                            breadth['contagion_risk'] = True
                        else:
                            breadth['contagion_risk'] = False
                            
                except Exception as e:
                    logger.error(f"Correlation Matrix Logic Error: {e}")
                    breadth['fleet_correlation'] = 0.0
                    breadth['contagion_risk'] = False
                
                # 4. Broadcast to Risk Manager & Portfolio
                risk_manager.update_global_regime(breadth['sentiment'])
                
                # Pass extensive breadth data including correlation
                portfolio.global_regime_data = breadth 
                portfolio.global_regime = breadth['sentiment'] 
            else:
                logger.warning("⏳ Regime Orchestrator: Waiting for market history...")
                
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"❌ Regime Orchestrator Error: {e}")
            await asyncio.sleep(60)

    # NOTE: Sovereign context calculation is done via MarketRegimeDetector.calculate_market_context
    # as defined in core/market_regime.py. Do not use local duplicate.
async def order_manager_loop(manager):
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
                
                # D. Register new ML strategies (DUAL HORIZON)
                for s in to_add:
                    try:
                        is_leader = ('BTC' in s)
                        # Scalping Engine
                        from strategies.ml_strategy import UniversalEnsembleStrategy as MLStrategy
                        ml_strat_scalp = MLStrategy(
                            data_provider=data_handler,
                            events_queue=events_queue,
                            symbol=s,
                            lookback=Config.Strategies.ML_LOOKBACK_BARS,
                            sentiment_loader=sentiment_loader,
                            portfolio=portfolio,
                            risk_manager=risk_manager if is_leader else None,
                            horizon="SCALPING"
                        )
                        engine.register_strategy(ml_strat_scalp)
                        
                        # Swing Engine
                        ml_strat_swing = MLStrategy(
                            data_provider=data_handler,
                            events_queue=events_queue,
                            symbol=s,
                            lookback=Config.Strategies.ML_LOOKBACK_BARS,
                            sentiment_loader=sentiment_loader,
                            portfolio=portfolio,
                            risk_manager=None,
                            horizon="SWING"
                        )
                        ml_strat_swing.strategy_id += "_SWING"
                        engine.register_strategy(ml_strat_swing)
                    except Exception as e:
                        logger.error(f"Failed to spawn adaptive ML strategy for {s}: {e}")

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
            Config.TRADING_PAIRS = Config.CRYPTO_FUTURES_PAIRS  # [FORENSIC] Sync with backtest
        Config.INITIAL_CAPITAL = args.capital
        # [FORENSIC FIX] Removed MAX_CONCURRENT_POSITIONS=1 override
        # POR QUÉ: Conflicto con Config.MAX_CONCURRENT_POSITIONS=3.
        #   God Mode necesita 3+ posiciones concurrentes para scalping multi-symbol.
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
    
    # 2. PERFORMANCE TRACKER (Delegated to Portfolio - Single Source of Truth)
    # [FORENSIC CLEANUP] PerformanceTracker removed — Portfolio.get_statistics() is the SSOT
    
    logger.info(f"🚀 STARTING TRADER GEMINI [Mode: {args.mode} | Capital: ${Config.INITIAL_CAPITAL}]")
    
    # 2.1. CPU AFFINITY OPTIMIZATION (Phase 29)
    try:
        CPUManager.pin_process()       # Bind to Performance Cores
        CPUManager.set_priority("HIGH") # Phase 35: Process Priority (HIGH for safety, REALTIME is risky)
        patch_sockets()                # Phase 31: TCP_NODELAY
        set_high_resolution_timer()    # Phase 32: 1ms Timer
        cache_dns_lookups()            # Phase 34: DNS Cache
        NTPSync.sync_time()            # Phase 37: Initial Time Check
        NTPSync.start_background_monitor() # Phase 37: Background Loop
    except Exception as e:
        logger.warning(f"Failed to apply System Optimizations: {e}")
    
    # 2.2. METRICS EXPORTER (Phase 53)
    from utils.metrics_exporter import metrics
    metrics.start_server(port=8000)
    
    # 3. CORE INITIALIZATION
    import queue
    import threading
    from core.market_scanner import MarketScanner
    from strategies.omni_strategy import OmniStrategy
    from strategies.technical import HybridScalpingStrategy as TechnicalStrategy
    from strategies.vacuum_sniper import VacuumSniperStrategy
    from strategies.asymmetric_mm import AsymmetricMMStrategy
    from strategies.cvd_sniper import CVDSniperStrategy
    from strategies.dark_pool_surfer import DarkPoolSurferStrategy
    
    # Events Queue (Thread-Safe)
    events_queue = queue.Queue()
    
    # 3.1. PRE-INITIALIZATION DISCOVERY (ELITE PROTOCOL)
    # Instantiate the data handler with a minimal payload for fast connect
    data_handler = BinanceData(events_queue, ["BTC/USDT"])
    scanner = MarketScanner(data_handler)
    
    if not args.symbols:
        logger.info("🔭 [Elite Protocol] Performing autonomous market discovery...")
        top_20 = scanner.get_top_ranked_symbols(limit=26) # FULL BASKET FOR GOD-MODE
        if top_20:
            Config.TRADING_PAIRS = top_20
            logger.info(f"💎 Elite Basket Selected: {len(top_20)} symbols.")
        else:
            logger.warning("⚠️ Discovery yielded no results, using default futures pairs.")
            Config.TRADING_PAIRS = Config.CRYPTO_FUTURES_PAIRS
            
    # Update data handler with the selected elite basket (downloads history in background)
    await data_handler.update_symbol_list(Config.TRADING_PAIRS)
    
    # 🛡️ SOVEREIGN CONTEXT MEMORY (Phase 3 Mutación)
    from core.sovereign_memory import ZmqKVServer
    memory_server = ZmqKVServer(port=5557)
    memory_server_task = asyncio.create_task(memory_server.start())
    # Esperar a que el puerto se abra antes de crear el cliente
    await asyncio.sleep(0.1)
    
    portfolio = Portfolio(
        initial_capital=Config.INITIAL_CAPITAL,
        csv_path=f"{Config.DATA_DIR}/trades.csv",
        status_path=f"{Config.DATA_DIR}/status.csv"
    )
    portfolio.data_provider = data_handler
    risk_manager = RiskManager(
        max_concurrent_positions=getattr(Config, 'MAX_CONCURRENT_POSITIONS', 3),
        portfolio=portfolio
    )

    # 3.2.2. MICRO ACCOUNT AWARENESS (Phase 1: Real Wallet sync)
    micro_awareness = MicroAccountAwareness()

    # 3.3. DATA WARMING BARRIER
    # Wait for parallel workers to fetch enough history for ML
    logger.info("📡 [Elite Protocol] Warming up data for universal training...")
    warming = True
    start_warm = time.time()
    while warming and (time.time() - start_warm < 120): # Max 2 min wait
        ready_count = 0
        with data_handler._data_lock:
            for s in Config.TRADING_PAIRS:
                b1m = data_handler.buffers_1m.get(s)
                if b1m is not None and b1m.size >= 500:
                    ready_count += 1
        
        if ready_count >= len(Config.TRADING_PAIRS):
            logger.info("✅ All elite symbols warmed up.")
            warming = False
        elif (time.time() - start_warm) > 5:
            logger.info(f"⏳ Warming progress: {ready_count}/{len(Config.TRADING_PAIRS)} symbols ready...")
            await asyncio.sleep(5)
        else:
            await asyncio.sleep(1)
    
    # Sentiment Engine — Phase 8: NLP Ensemble (FinBERT + CryptoBERT)
    # [Phase 6 Audit] Old TextBlob-based SentimentLoader was disabled.
    # [Phase 8] Replaced with institutional-grade HuggingFace NLP ensemble.
    # Models load lazily on first RSS poll (5 min cycle). Features decay
    # exponentially to 0.0 if no fresh news arrives.
    from data.news_sentiment_nlp import news_sentiment
    news_sentiment.start_background()
    sentiment_loader = None  # Legacy param — FeatureEngineering now uses news_sentiment singleton directly
    logger.info("📰 [Phase 8] News Sentiment NLP Engine activated (FinBERT + CryptoBERT).")
    
    # ═══════════════════════════════════════════════════════════════
    # MUTACIÓN 2: ZERO-MQ IPC ARCHITECTURE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════
    from core.zmq_bus import ZmqPullNode, ZmqPushNode
    engine_pull_node = ZmqPullNode(bind_port=5555)            # Engine listens here
    executor_push_node = ZmqPushNode(target_port=5556)        # Engine pushes orders here
    
    executor_pull_node = ZmqPullNode(bind_port=5556)          # Executor listens here
    engine_push_node = ZmqPushNode(target_port=5555)          # Executor pushes fills here
    loader_push_node = ZmqPushNode(target_port=5555)          # Loader pushes market events here
    
    data_handler.zmq_push = loader_push_node
    
    # Executor - Phase 36: Paper Trading Mock Executor Injection
    print("DEBUG: Instanciando Executor...")
    try:
        if Config.BINANCE_USE_TESTNET or getattr(Config, 'BINANCE_USE_DEMO', False):
            print("⚡ [SYSTEM] MOCK EXECUTOR ACTIVATED (Phase 36 Paper Trading)")
            from execution.mock_executor import MockExecutor
            executor = MockExecutor(leverage=Config.BINANCE_LEVERAGE)
            # Inject required attributes mock executor doesn't natively handle via __init__
            executor.events_queue = events_queue
            executor.portfolio = portfolio
            executor.data_provider = data_handler
            executor.micro_awareness = micro_awareness
        else:
            try:
                if CYTHON_EXECUTION_AVAILABLE and not Config.BINANCE_USE_TESTNET and not Config.BINANCE_USE_DEMO:
                    # Fast Path for Live Trading
                    logger.info("🚀 [C++ Bridge] Initializing FastBinanceExecutor (Cython/C++) for Hot Path Execution.")
                    executor = FastBinanceExecutor(Config.BINANCE_API_KEY, Config.BINANCE_API_SECRET, testnet=False)
                else:
                    executor = BinanceExecutor(events_queue, portfolio=portfolio, data_provider=data_handler, micro_awareness=micro_awareness)
            except Exception as e:
                executor = BinanceExecutor(events_queue, portfolio=portfolio, data_provider=data_handler, micro_awareness=micro_awareness)
        
        executor.zmq_pull = executor_pull_node
        executor.zmq_push = engine_push_node
        print("DEBUG: Executor instanciado exitosamente.")
    except Exception as e:
        print(f"DEBUG CRITICAL FAIL en BinanceExecutor: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Engine
    engine = Engine(events_queue)
    engine.zmq_pull = engine_pull_node
    engine.zmq_push = executor_push_node
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
    
    # ════════════════════════════════════════════════════════════════
    # FORENSIC-V47: INTEGRAL MODE — ALL SYMBOLS ALWAYS ACTIVE
    # QUÉ: Removido el override de LEAN_MODE que reducía a 3 pares.
    # POR QUÉ: Crear paridad total con backtest (21 símbolos).
    #   La Dead Zone de comisiones + consenso ponderado manejan el filtrado.
    # PARA QUÉ: El sistema opera integralmente en 21 monedas.
    # ════════════════════════════════════════════════════════════════
    logger.info(f"🎯 [INTEGRAL MODE] Operating with {len(Config.TRADING_PAIRS)} symbols (Full basket)")
    
    # 🎯 FASE 30: UNIVERSAL STRATEGY REGISTRY
    # Reemplaza la inicialización manual hardcodeada. Carga TODAS las estrategias
    # de la carpeta strategies/ automáticamente usando el UniversalStrategyAdapter.
    from core.strategy_registry import UniversalStrategyRegistry
    
    # Preparamos las dependencias globales que las estrategias puedan pedir
    global_dependencies = {
        'data_provider': data_handler,
        'events_queue': events_queue,
        'portfolio': portfolio,
        'executor': execution,
        'risk_manager': risk_manager,
        'sentiment_loader': sentiment_loader
    }

    try:
        # Instanciamos TODAS las estrategias detectadas para SCALPING
        global_dependencies['horizon'] = "SCALPING"
        all_scalp_strats = UniversalStrategyRegistry.create_all(**global_dependencies)
        for strat in all_scalp_strats:
            engine.register_strategy(strat)
            strategies.append(strat)
        logger.info(f"✅ Registradas {len(all_scalp_strats)} estrategias universales para [SCALPING]")

        # Instanciamos TODAS las estrategias detectadas para SWING
        global_dependencies['horizon'] = "SWING"
        all_swing_strats = UniversalStrategyRegistry.create_all(**global_dependencies)
        for strat in all_swing_strats:
            # Añadir subfijo para evitar colisión de IDs en logs/métricas
            if hasattr(strat, 'strategy_id'):
                strat.strategy_id += "_SWING"
            engine.register_strategy(strat)
            strategies.append(strat)
        logger.info(f"✅ Registradas {len(all_swing_strats)} estrategias universales para [SWING]")
        
    except Exception as e:
        logger.error(f"❌ Error al registrar estrategias con UniversalRegistry: {e}")
        
        
    # ════════════════════════════════════════════════════════════════
    # NOTA: Las estrategias específicas por símbolo (como MLStrategy que necesita un `symbol`)
    # pueden requerir instanciación iterativa si el Adapter universal no sabe multiplicarlas.
    # El UniversalAdapter ya intenta pasar `symbol='ALL'` por defecto si lo piden.
    # Para estrategias que sí o sí necesitan instanciarse por cada symbol, 
    # deberían refactorizarse para escuchar a todos los símbolos (ej. Omni-Symbol approach),
    # o bien registrarse localmente dentro de UniversalAdapter en futuras versiones.
    # Por ahora mantenemos la inicialización por símbolo solo para MLStrategy si es estrictamente necesario,
    # pero OmniStrategy/TechStrategy/Wyckoff/OrderFlow operan a nivel de DataProvider global.
    # ════════════════════════════════════════════════════════════════

    
    logger.info(f"[OK] Registered {len(strategies)} strategies in the Engine.")
    
    # 3.5. START CI-HMA SUPERVISOR (Phase 6)
    supervisor = start_health_supervisor()
    logger.info("🩺 CI-HMA Health Supervisor started in background.")

    # Scalping Optimizer (optional)
    # [FORENSIC CLEANUP] ScalpingOptimizer removed — CooldownManager handles anti-overtrading

    # 🧟 ZOMBIE FEATURE INTEGRATION: Activate latent AI and Evolution modules
    logger.info("🧟 [PHASE 2] Initializing Dormant Zombie Modules...")
    
    # 🌑 DARK ALPHA LAYER
    try:
        from core.dark_alpha_worker import dark_alpha_worker
        from core.mempool_worker import mempool_worker
        dark_alpha_worker.start()
        mempool_worker.start()
    except Exception as e:
        logger.warning(f"Could not init DarkAlpha or Mempool Worker: {e}")

    try:
        from core.world_awareness import world_awareness
        logger.info("✅ WorldAwareness activated.")
    except Exception as e:
        logger.warning(f"Could not init WorldAwareness: {e}")

    try:
        from core.sovereign_oracle import SovereignOracle
        engine.sovereign_oracle = SovereignOracle()
        logger.info("✅ SovereignOracle activated.")
    except Exception as e:
        logger.warning(f"Could not init SovereignOracle: {e}")

    try:
        from core.multiverse_simulator import MultiverseSimulator
        engine.multiverse = MultiverseSimulator()
        logger.info("✅ MultiverseSimulator activated.")
    except Exception as e:
        logger.warning(f"Could not init MultiverseSimulator: {e}")

    try:
        from core.shadow_darwin import ShadowDarwin
        engine.shadow_darwin = ShadowDarwin()
        logger.info("✅ ShadowDarwin activated.")
    except Exception as e:
        logger.warning(f"Could not init ShadowDarwin: {e}")

    try:
        from core.swarm_correlator import SwarmCorrelator
        engine.swarm = SwarmCorrelator()
        logger.info("✅ SwarmCorrelator activated.")
    except Exception as e:
        logger.warning(f"Could not init SwarmCorrelator: {e}")

    try:
        from core.neural_bridge import NeuralBridge
        engine.neural_bridge = NeuralBridge()
        logger.info("✅ NeuralBridge activated.")
    except Exception as e:
        logger.warning(f"Could not init NeuralBridge: {e}")
    
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
    regime_detector = MarketRegimeDetector(events_queue=events_queue)
    regime_task = asyncio.create_task(global_regime_loop(regime_detector, data_handler, risk_manager, portfolio))
    
    # 🌌 MUTACIÓN 4: QUANTUM ROUTER INJECTION
    from core.quantum_router import QuantumRouter
    quantum_router = QuantumRouter(portfolio, selector, regime_detector)
    engine.quantum_router = quantum_router
    logger.info("🌌 [QUANTUM] Adaptive Meta-Orchestrator Online.")
    
    # PHASE 9/41: ORDER MANAGER
    order_manager = OrderManager(executor, data_provider=data_handler)
    executor.order_manager = order_manager
    engine.register_order_manager(order_manager)
    order_task = asyncio.create_task(order_manager_loop(order_manager))
    
    # MUTACIÓN 2: START EXECUTOR ZMQ LOOP
    executor_zmq_task = asyncio.create_task(executor.start_zmq_loop())
    
    # PHASE 99: USER DATA STREAM (Manual Close Detection)
    # [Phase 6 Audit] DISABLED to prevent conflict with execution/user_data_stream.py
    # user_stream = UserDataStreamListener(
    #     order_manager=order_manager,
    #     portfolio=portfolio,
    #     data_provider=data_handler,
    #     executor=executor,
    #     engine=engine,
    #     efficacy_tracker=efficacy_tracker
    # )
    # user_stream_task = asyncio.create_task(user_stream.start())
    
    # Phase 36: Shared Memory Manager
    shm_path = os.path.join(Config.DATA_DIR, "live_status.dat")
    shm_manager = SharedStateManager(shm_path)

    # 4. MAIN EVENT LOOP ORCHESTRATION (SUPREMO-V3)
    logger.info("⚡ [SUPREMO-V3] Orchestrating Concurrent Async Tasks...")
    
    # Send Rich Startup Notification to Telegram
    try:
        from utils.notifier import Notifier
        _startup_mode = "PRODUCTION"
        if Config.BINANCE_USE_TESTNET:
            _startup_mode = "PAPER"
        elif Config.BINANCE_USE_DEMO:
            _startup_mode = "PAPER"
        Notifier.send_system_startup(_startup_mode, {
            'trading_mode': args.mode,
            'capital': Config.INITIAL_CAPITAL,
            'leverage': Config.BINANCE_LEVERAGE,
            'symbols_count': len(Config.TRADING_PAIRS),
            'strategies_count': len(strategies),
            'testnet': Config.BINANCE_USE_TESTNET,
            'demo': Config.BINANCE_USE_DEMO,
            'max_drawdown': Config.Risk.MAX_DRAWDOWN,
            'tp_scalp': Config.Horizons.Scalping.get('tp_pct', 0.006),
            'sl_scalp': Config.Horizons.Scalping.get('sl_pct', 0.0075),
            'symbols_list': Config.TRADING_PAIRS,
        })
    except Exception as e:
        logger.warning(f"Could not send Telegram startup alert: {e}")
        
    # 4.1. Paper Trading Bootstrap (Fase 30)
    # QUÉ: Cargar datos históricos MASIVOS antes de iniciar el motor
    # POR QUÉ: Las estrategias basadas en EMAs largas o ML fallan si no tienen `lookback` en Paper Trading.
    if args.mode == "paper" or Config.BINANCE_USE_TESTNET or Config.BINANCE_USE_DEMO:
        logger.info("⏳ [BOOTSTRAP] Iniciando Warmup histórico de Paper Trading...")
        try:
            # 5 iteraciones * ~100 velas (dependiendo de data_handler) para llenar los buffers
            for _ in range(5):
                await data_handler.update_bars_async()
                await asyncio.sleep(1)
            logger.info("✅ [BOOTSTRAP] Warmup completado. Buffers llenos.")
        except Exception as e:
            logger.error(f"❌ Error en Warmup de Paper Trading: {e}")

    # 4.2. Initialize Engine Task & User Stream
    engine_task = asyncio.create_task(engine.start())
    user_stream_task = asyncio.create_task(executor.user_stream.start())
    
    # 4.1.5 Liquidation Sniper Websockets (Phase 1 Power) - DISABLED (Redundant with start_socket)
    
    # 4.2. Background Task for Metrics & Heartbeat
    loop_count = 0
    last_heartbeat = time.time()
    last_summary_time = time.time()
    last_shm_update = time.time()
    last_pulse_time = time.time()  # Strategy Pulse tracker
    last_leaderboard_time = time.time() # Strategy Leaderboard tracker
    PULSE_INTERVAL_SECONDS = 900   # Every 15 minutes
    
    async def metrics_heartbeat_loop():
        nonlocal loop_count, last_heartbeat, last_summary_time, last_shm_update, last_pulse_time, last_leaderboard_time
        while not shutdown_event.is_set():
            try:
                now = time.time()
                loop_count += 1
                
                # Phase 36: Shared Memory Update (1s)
                if now - last_shm_update > 1.0:
                    current_equity = portfolio.get_total_equity()
                    shm_manager.write_state({
                        "timestamp": now, "equity": current_equity,
                        "active_positions": len(portfolio.positions), "mode": args.mode
                    })
                    last_shm_update = now
                
                # Heartbeat & Data Update (60s)
                if now - last_heartbeat > 60:
                    await data_handler.update_bars_async()
                    get_heartbeat().pulse(metadata={"loop_count": loop_count, "equity": portfolio.get_total_equity()})
                    
                    equity = portfolio.get_total_equity()
                    open_pos = len([s for s, p in portfolio.positions.items() if p['quantity'] != 0])
                    logger.info(f"💓 Heartbeat | Equity: ${equity:.2f} | Pos: {open_pos} | Events: {engine.metrics['processed_events']}")
                    
                    # Phase 99: Fleet Telemetry Display
                    telemetry_output = telemetry.render(portfolio, data_handler)
                    logger.info(telemetry_output)
                    
                    metrics.update(portfolio=portfolio, engine=engine, queue_size=engine.events.qsize())
                    metrics.update_health(risk_manager)
                    
                    # Phase 1: Real-Time Wallet Balance Sync
                    try:
                        real_balance = await asyncio.to_thread(executor.fetch_real_balance)
                        portfolio.current_cash = real_balance
                        if micro_awareness:
                            micro_awareness.update_balance(real_balance)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to sync wallet balance: {e}")
                        
                    last_heartbeat = now

                # ═══════════════════════════════════════════════════════════════
                # STRATEGY PULSE: Every 15 min, report system state via Telegram
                # QUÉ: Reporte periódico de estado cuando no hay trades.
                # POR QUÉ: El usuario necesita saber qué está pasando.
                # PARA QUÉ: Visibilidad total, incluso cuando el mercado está quiet.
                # ═══════════════════════════════════════════════════════════════
                if now - last_pulse_time > PULSE_INTERVAL_SECONDS:
                    try:
                        from utils.notifier import Notifier
                        _equity = portfolio.get_total_equity()
                        _open_pos = [(s, p) for s, p in portfolio.positions.items() if p.get('quantity', 0) != 0]
                        _open_symbols = [s for s, _ in _open_pos]
                        
                        # Get BTC price
                        _btc_price = 0
                        try:
                            _btc_bars = data_handler.get_latest_bars('BTC/USDT', n=1)
                            if _btc_bars and len(_btc_bars) > 0:
                                _btc_price = _btc_bars[-1].get('close', 0)
                        except Exception:
                            pass
                        
                        # Get market regime
                        _regime = 'UNKNOWN'
                        if risk_manager and hasattr(risk_manager, 'current_regime'):
                            _regime = risk_manager.current_regime
                        
                        # Session stats from portfolio
                        _stats = portfolio.get_statistics() or {}
                        
                        # Strategy signal counts
                        _strat_status = []
                        for strat in strategies[:16]:  # Cap at 16 to avoid message explosion
                            _sname = getattr(strat, 'strategy_id', strat.__class__.__name__)
                            _shorizon = getattr(strat, 'horizon', 'SCALPING')
                            _ssignals = getattr(strat, 'signal_count', 0)
                            _strat_status.append({
                                'name': _sname,
                                'horizon': _shorizon,
                                'signals_emitted': _ssignals,
                            })
                        
                        Notifier.send_strategy_pulse({
                            'equity': _equity,
                            'initial_capital': Config.INITIAL_CAPITAL,
                            'open_positions': len(_open_pos),
                            'open_symbols': _open_symbols,
                            'events_processed': engine.metrics.get('processed_events', 0),
                            'signals_generated': engine.metrics.get('strategy_executions', 0),
                            'signals_rejected': engine.metrics.get('discarded_events', 0),
                            'avg_latency_ms': engine.metrics.get('avg_latency_ms', 0),
                            'market_regime': _regime,
                            'btc_price': _btc_price,
                            'strategies_status': _strat_status,
                            'session_trades': _stats.get('total_trades', 0),
                            'session_wins': _stats.get('winning_trades', 0),
                            'session_losses': _stats.get('losing_trades', 0),
                            'minutes_since_last_trade': (now - last_summary_time) / 60,
                        })
                    except Exception as e:
                        logger.error(f"Strategy Pulse Error: {e}")
                    last_pulse_time = now
                    
                # ═══════════════════════════════════════════════════════════════
                # STRATEGY LEADERBOARD: Every 4 hours (14400s), report Top 5 Strategies
                # ═══════════════════════════════════════════════════════════════
                if now - last_leaderboard_time > 14400:
                    try:
                        from utils.notifier import Notifier
                        Notifier.send_strategy_leaderboard(portfolio.strategy_performance, title_prefix="LIVE (4H)")
                    except Exception as e:
                        logger.error(f"Strategy Leaderboard Error: {e}")
                    last_leaderboard_time = now
                
                # Performance Summary (30m) — via Portfolio SSOT
                if now - last_summary_time > 1800:
                    stats = portfolio.get_statistics()
                    if stats and stats.get('total_trades', 0) > 0:
                        logger.info(f"📊 [30m] Trades: {stats['total_trades']} | WR: {stats.get('win_rate', 0)*100:.1f}% | Equity: ${portfolio.get_total_equity():.2f}")
                    last_summary_time = now
                    
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics Loop Error: {e}")
                await asyncio.sleep(5)

    heartbeat_task = asyncio.create_task(metrics_heartbeat_loop())
    
    # 5. WAIT FOR SHUTDOWN
    logger.info("🚀 System Operational. Monitoring for signals...")
    
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    
    try:
        # Wait for shutdown event or any critical task to fail
        done, pending = await asyncio.wait(
            [shutdown_task, engine_task, ws_task, regime_task, order_task, heartbeat_task, user_stream_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Check if a task failed
        for task in done:
            if task != shutdown_task and task.exception():
                logger.critical(f"💥 Task failure detected: {task.exception()}")
                shutdown_event.set()

    except Exception as e:
        import traceback
        logger.error(f"Orchestration Error: {e}\n{traceback.format_exc()}")
        shutdown_event.set()
    
    # 6. GRACEFUL SHUTDOWN (SUPREMO-V3)
    logger.info("🛑 Initiating clean stop...")
    
    # Signal Engine to stop
    engine.stop()
    
    # Cancel all background tasks
    tasks = [ws_task, adaptive_task, meta_task, regime_task, order_task, heartbeat_task, engine_task, user_stream_task, memory_server_task]
    for task in tasks:
        task.cancel()
        
    memory_server.stop()
    
    # Wait for tasks to clean up
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Data Layer Cleanup
    try:
        await asyncio.wait_for(data_handler.shutdown(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("⚠️ Data handler shutdown timed out")
    
    # Performance & Session Closure — via Portfolio SSOT
    portfolio.close()
    final_stats = portfolio.get_statistics()
    if final_stats:
        logger.info(f"📊 [FINAL] Trades: {final_stats.get('total_trades', 0)} | WR: {final_stats.get('win_rate', 0)*100:.1f}% | Equity: ${portfolio.get_total_equity():.2f}")
    else:
        logger.info("📭 No trades executed during this session.")
    
    if sentiment_loader is not None:
        sentiment_loader.stop()
    if supervisor:
        supervisor.stop()
    if hot_reload:
        hot_reload.stop()
        
    session_mgr = get_session_manager()
    if session_mgr:
        session_mgr.end_session({
            'total_trades': final_stats.get('total_trades', 0) if final_stats else 0,
            'pnl': portfolio.get_total_equity() - Config.INITIAL_CAPITAL
        })
    
    # Neural Bridge Cleanup (SharedMemory)
    neural_bridge.cleanup()
    
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
        # PROTOCOL METAL-CORE OMEGA: Nano-Latency Loop (Phase 1)
        if sys.platform == 'win32':
            try:
                import winloop
                winloop.install()
                logger.info("🚀 [Metal-Core] winloop activated (Windows Nano-Latency).")
            except ImportError:
                # Fallback to Proactor (Standard High Perf)
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                logger.warning("⚠️ [Metal-Core] winloop not found. Using WindowsProactor.")
        else:
            try:
                import uvloop
                # uvloop.install() replaces the default policy
                uvloop.install() 
                logger.info("🚀 [Metal-Core] uvloop activated (Unix Nano-Latency).")
            except ImportError:
                logger.warning("⚠️ [Metal-Core] uvloop not found. Using default asyncio.")
                
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Bot detenido por el usuario")
    except Exception as e:
        logger.critical(f"💥 ERROR FATAL: {e}", exc_info=True)
        sys.exit(1)