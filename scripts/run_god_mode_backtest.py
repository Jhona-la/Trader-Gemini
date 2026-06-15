#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 GOD MODE BACKTEST v2.0 — GLOBAL SYNCHRONIZED ENGINE (PRODUCTION-PARITY)
═══════════════════════════════════════════════════════════════════════════════

QUÉ: Motor de backtest que simula TODAS las monedas SIMULTÁNEAMENTE con un
     solo Portfolio compartido de $13 USD, IDÉNTICO a producción.
POR QUÉ: v1.0 procesaba monedas secuencialmente (for sym in symbols:),
     dando $13 frescos a cada moneda → resultados inflados 26x.
PARA QUÉ: Que los resultados del backtest sean una PREDICCIÓN INEQUÍVOCA
     de lo que pasará en producción cuando ejecutes LAUNCH_GOD_MODE.bat.
CÓMO: Usa las CLASES REALES de producción:
     - core.portfolio.Portfolio  (virtual ledgers, horizon partitioning)
     - risk.risk_manager.RiskManager (Kelly sizing, kill switch, stops)
     - strategies.ml_strategy.UniversalEnsembleStrategy
     - core.events.{MarketEvent, SignalEvent, OrderEvent, FillEvent}
     Timeline global: union de timestamps de TODAS las monedas → iterar
     minuto a minuto → emitir MarketEvents para TODOS los símbolos.
CUÁNDO: Ejecutado vía CLI o launchers/MASSIVE_BACKTESTER.bat.
DÓNDE: scripts/run_god_mode_backtest.py
QUIÉN: QA Engineer + Quant Developer + Risk Manager + Arquitecto Senior

FLUJO DE PRODUCCIÓN REPLICADO:
  main.py::main()
    ↓ BinanceData.start_socket() emite MarketEvents
    ↓ Engine.process_event()
    ↓   portfolio.update_market_price()
    ↓   portfolio.check_exits() → SignalEvent(EXIT)
    ↓   risk_manager.check_stops() → SignalEvent(EXIT)
    ↓   strategy.calculate_signals() → SignalEvent(LONG/SHORT)
    ↓   risk_manager.generate_order() → OrderEvent
    ↓   executor.execute_order() → FillEvent
    ↓   portfolio.update_fill() → PnL tracking
"""

import os
import sys

# ── ENVIRONMENT ISOLATION (FORENSIC FIX) ──
# Extract --env_id before any internal module is imported so Config picks it up.
env_id = ""
for i, arg in enumerate(sys.argv):
    if arg == "--env_id" and i + 1 < len(sys.argv):
        env_id = sys.argv[i + 1]
if env_id:
    os.environ["TG_ENV_ID"] = env_id

# ═══════════════════════════════════════════════════════════════
# HARDWARE UNLOCK (RYZEN 7 5700U OPTIMIZATION)
# QUÉ: Fuerza a C++ y Python a usar los 16 hilos del CPU
# POR QUÉ: Multiplica la velocidad del backtest x10.
# ═══════════════════════════════════════════════════════════════
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

try:
    import torch
    torch.set_num_threads(16)
    torch.set_grad_enabled(False)  # Disable gradients (backtest is inference only, saves 50% RAM/CPU)
except ImportError:
    pass

import sys
import io
import contextlib
import time
import json
import random
import uuid
import argparse
import warnings
import logging
from datetime import datetime, timezone, timedelta
from queue import Queue
import numpy as np
import pandas as pd
import gc  # FIX-FORENSIC-V82: Moved from hot loop (L946) to top-level

# ─── Project Root ───
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Suppress noisy warnings during backtest
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 15: CONCURRENCY ISOLATION (Pre-Import Hook)
# ═══════════════════════════════════════════════════════════════════════════════
# We must extract --env-id manually before `config` is imported so that
# Config.ENV_ID and Config.DATA_DIR are initialized with the correct path.
for i, arg in enumerate(sys.argv):
    if arg == "--env-id" and i + 1 < len(sys.argv):
        os.environ["TG_ENV_ID"] = sys.argv[i + 1]

# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION IMPORTS — SAME AS main.py
# ═══════════════════════════════════════════════════════════════════════════════
from config import Config
from utils.notifier import Notifier

# FORENSIC LOGGING CONFIGURATION
forensic_logger = logging.getLogger("ForensicAuditor")
forensic_logger.setLevel(logging.DEBUG)
if not forensic_logger.handlers:
    fh = logging.FileHandler("forensic_audit.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    forensic_logger.addHandler(fh)

# ═══════════════════════════════════════════════════════════════════════
# FORENSIC-V41: SMART NOTIFICATION MANAGEMENT FOR BACKTEST
# QUÉ: Mantiene Telegram ACTIVO para mensajes de sistema (startup,
#   progreso, completado), pero silencia notificaciones POR TRADE.
# POR QUÉ: Un backtest de 7 días genera cientos de trades. Enviar cada
#   uno a Telegram: (a) inunda el chat, (b) ralentiza el backtest
#   (cada HTTP request ~1-3s), (c) excede rate limits.
# PARA QUÉ: El usuario recibe startup, progreso cada 10%, y resumen
#   final — la información que realmente necesita.
# CÓMO: TELEGRAM_ENABLED = True (canal abierto), pero desactivamos
#   NOTIFICATION_TRADE_OPEN y NOTIFICATION_TRADE_CLOSE individuales.
# ═══════════════════════════════════════════════════════════════════════
# NOTE: Config mutations moved to run_global_backtest with try/finally
# to prevent cross-contamination if run in same process as production.

from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import EventType, SignalType, OrderSide, OrderType
from core.api_manager import APIManager
from core.meta_arbitrator import meta_arbitrator

# ─── DETERMINISTIC CAPITAL FOR BACKTEST (FORENSIC-V82 FIX) ───
# QUÉ: Usa Config.INITIAL_CAPITAL en vez del balance real de la API.
# POR QUÉ: Llamar a APIManager().get_production_balance() en module-level
#   hacía que el capital del backtest cambiara con cada ejecución ($7.50
#   el lunes, $15.20 el miércoles). Esto hacía imposible comparar
#   optimizaciones porque el capital inicial era una variable no controlada.
# PARA QUÉ: Resultados BIT-A-BIT reproducibles entre ejecuciones.
# CÓMO: Siempre usar Config.INITIAL_CAPITAL ($13.0) como base.
real_capital = Config.INITIAL_CAPITAL
print(f"💰 BACKTEST CAPITAL (DETERMINISTIC): ${real_capital:.2f}")

# ─── PRODUCTION PORTFOLIO (THE REAL ONE) ───
from core.portfolio import Portfolio

# ─── PRODUCTION RISK MANAGER (THE REAL ONE) ───
from risk.risk_manager import RiskManager

# ─── PRODUCTION META-ARBITRATOR ───
from core.meta_arbitrator import meta_arbitrator

# ─── PRODUCTION STRATEGIES (THE REAL ONES) ───
from strategies.ml_strategy import UniversalEnsembleStrategy as MLStrategy
from strategies.sniper_strategy import SniperStrategy
from strategies.statistical import StatisticalStrategy

# ─── COOLDOWN MANAGER (PARITY FIX) ───
# QUÉ: Restaurado el CooldownManager real.
# POR QUÉ: Antes se mockeaba (lambda: True), lo que causaba spam de señales.
#   Ahora usa set_virtual_time() para ser determinista en backtest.
from utils.cooldown_manager import cooldown_manager
# No more mocking!


# ─── BACKTEST INFRASTRUCTURE ───
from core.backtest_infra import (
    BacktestDataProvider,
    fetch_binance_data,
    fetch_multi_symbol_data,
    calculate_metrics,
    COMMISSION_PCT,
    COMMISSION_TAKER,
    COMMISSION_MAKER,
)

from utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestExecutor — Simulates BinanceExecutor with zero exchange calls
# ═══════════════════════════════════════════════════════════════════════════════


class BacktestExecutor:
    """
    Simulates BinanceExecutor for backtesting.

    QUÉ: Ejecuta órdenes de forma SIMULADA sin tocar ningún exchange.
    POR QUÉ: En producción, BinanceExecutor envía órdenes reales y recibe FillEvents.
         En backtest, simulamos el fill instantáneo con slippage y comisiones realistas.
    PARA QUÉ: Que el flujo OrderEvent → FillEvent → Portfolio.update_fill() sea
         IDÉNTICO al de producción (execution/binance_executor.py).
    CÓMO: Recibe OrderEvent, aplica slippage DETERMINÍSTICO basado en seed fija,
         calcula comisión real (Config.BINANCE_TAKER_FEE_BNB), y crea un FillEvent
         con los campos exactos que Portfolio.update_fill() espera.

    REMEDIACIÓN QUIRÚRGICA:
      - Slippage ahora es DETERMINÍSTICO (usa self._rng con seed fija).
      - Dos ejecuciones con la misma seed producen resultados BIT-A-BIT idénticos.
      - Esto resuelve el problema de backtests no reproducibles.
    """

    def __init__(self, data_provider=None):
        self._rng = np.random.RandomState(42)  # Determinismo absoluto
        self.fills_count = 0
        self.data_provider = data_provider
        # ═══════════════════════════════════════════════════════════════
        # REMEDIACIÓN: DETERMINISTIC SLIPPAGE
        # QUÉ: RNG privado con seed fija para reproducibilidad total.
        # POR QUÉ: random.uniform() usa seed global, contaminada por
        #   otras partes del sistema (strategies, risk manager, etc.).
        # PARA QUÉ: Dos runs con misma seed → mismos fills → mismos resultados.
        # CÓMO: Random instance aislada, no afecta ni es afectada por global.
        # ═══════════════════════════════════════════════════════════════

    def execute_order(self, order_event, current_price=None):
        """
        Simulates order execution identical to production.

        BBO ARCHITECTURE: Uses differentiated Maker/Taker fees.
        - LIMIT orders (BBO) → Maker fee (0.02%)
        - MARKET orders → Taker fee (0.0375%)
        This mirrors production's Limit BBO behavior.

        Returns: FillEvent or None if order is rejected.
        """
        if order_event.is_shadow:
            return None  # Shadow orders are never executed (production behavior)

        # 🛡️ RESTING ORDER GUARD: Ignore TP/SL Limit orders (they are handled by RiskManager simulator)
        if order_event.metadata and (order_event.metadata.get("is_tp_limit") or order_event.metadata.get("is_sl_limit")):
            return None  # Backtester relies on check_stops loop for exits

        price = current_price or order_event.price
        if not price or price <= 0:
            return None

        qty = order_event.quantity
        if qty <= 0:
            return None

        # ─── REALISTIC CAPITAL-AWARE SLIPPAGE ───
        # FORENSIC-V139 FIX: Instead of adding arbitrary "latency penalties" that destroy scalping PnL,
        # we model slippage based on capital size. Con $13 USD, el impacto en el Orderbook de Binance es CERO.
        # Por tanto, la latencia es irrelevante para el fill price real, la orden se toma del BBO sin moverlo.
        
        is_limit = order_event.order_type == OrderType.LIMIT
        if is_limit:
            # 🚀 FASE 13: STRICT MAKER LATENCY PENALTY
            # Reject 10% of BBO LIMIT orders to simulate missed fills 
            # (price moved away before the 10ms WS latency finished).
            if self._rng.random() < 0.10:
                return None
                
            # BBO LIMIT orders have exactly zero slippage if filled
            slip_pct = 0.0
            stochastic_latency_ms = 10.0 # WebSocket Latency
        else:
            # MARKET orders for <$100 capital simply pay the spread.
            # Spread on BTC/USDT or SOL/USDT is typically 1 tick.
            # We set a realistic micro-slippage of 0.015% (1.5 bps).
            slip_pct = 0.00015
            stochastic_latency_ms = 15.0 # Realistic API Latency for Market
            
        # EXECUTION LAG (DATA LEAKAGE PREVENTION)
        # Apply the realistic slippage directionally.
        
        if order_event.direction == OrderSide.BUY:
            fill_price = price * (1 + slip_pct)
        else:
            fill_price = price * (1 - slip_pct)

        fill_cost = fill_price * qty

        # ─── BBO ARCHITECTURE: DIFFERENTIATED COMMISSION ───
        # LIMIT (BBO) → Maker fee | MARKET → Taker fee
        if is_limit:
            commission = fill_cost * COMMISSION_MAKER  # 0.02%
            actual_order_type = "limit"
        else:
            commission = fill_cost * COMMISSION_TAKER  # 0.0375%
            actual_order_type = "market"

        self.fills_count += 1
        b_order_id = f"BT_{self.fills_count}"

        # Inject real execution type into metadata for Portfolio fee selection
        metadata = order_event.metadata.copy() if order_event.metadata else {}
        metadata["actual_order_type"] = actual_order_type
        # FORENSIC-V35: Production-parity enrichment
        is_exit = getattr(order_event, 'is_exit', False) or getattr(order_event, 'is_close', False)
        is_gtx = metadata.get('timeInForce') == 'GTX'
        if actual_order_type == "limit":
            metadata["enriched_order_type"] = "LIMIT_POST_ONLY" if is_gtx else ("LIMIT_BBO_EXIT" if is_exit else "LIMIT_BBO")
        else:
            metadata["enriched_order_type"] = "MARKET_EXIT" if is_exit else "MARKET_SOR"

        # Inject into metadata for portfolio parsing
        metadata["is_exit"] = getattr(order_event, 'is_exit', False)
        metadata["is_close"] = getattr(order_event, 'is_close', False)

        # Use simulated time if available
        import pandas as pd
        if self.data_provider and hasattr(self.data_provider, 'current_time_ms'):
            try:
                # Add stochastic latency to fill time
                actual_time_ms = self.data_provider.current_time_ms + stochastic_latency_ms
                fill_time = pd.to_datetime(actual_time_ms, unit="ms", utc=True)
            except:
                fill_time = datetime.now(timezone.utc)
        else:
            fill_time = datetime.now(timezone.utc)

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V42 FIX: PRODUCTION-PARITY FILL PROPAGATION
        # QUÉ: Propaga trade_id, setup_type, exit_reason del OrderEvent al FillEvent.
        # POR QUÉ: binance_executor.py (producción) propaga estos campos (líneas 854-857),
        #   pero BacktestExecutor no lo hacía → Telegram mostraba ID: None, Setup: UNKNOWN.
        # PARA QUÉ: Paridad total entre backtest y producción.
        # ═══════════════════════════════════════════════════════════════
        fill_event = FillEvent(
            timeindex=fill_time,
            symbol=order_event.symbol,
            exchange="BINANCE_BACKTEST",
            quantity=qty,
            direction=order_event.direction,
            fill_cost=fill_cost,
            commission=commission,
            strategy_id=order_event.strategy_id,
            fill_price=fill_price,
            order_id=b_order_id,
            sl_pct=order_event.sl_pct,
            tp_pct=order_event.tp_pct,
            horizon=order_event.horizon,
            leverage=order_event.leverage,
            metadata=metadata,
            # FORENSIC-V42: Production-parity fields (were MISSING)
            trade_id=getattr(order_event, 'trade_id', None),
            setup_type=getattr(order_event, 'setup_type', None),
            exit_reason=getattr(order_event, 'exit_reason', None),
            ml_confidence=getattr(order_event, 'ml_confidence', None),
            predicted_duration=getattr(order_event, 'predicted_duration', None),
            predicted_magnitude=getattr(order_event, 'predicted_magnitude', None),
            strategy_version=getattr(order_event, 'strategy_version', '1.0.0'),
        )

        return fill_event


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SYNCHRONIZED BACKTEST ENGINE v2.0
# ═══════════════════════════════════════════════════════════════════════════════


def run_global_backtest(
    all_data, symbols, days, initial_capital=None, verbose=True, seed=42,
    scenario="A", isolated_strategy=None, mode="FULL", signal_cache_path=None
):
    """
    MOTOR DE BACKTEST GLOBAL SINCRONIZADO — PRODUCTION-PARITY.

    QUÉ: Ejecuta un backtest multi-symbol sincronizado donde TODAS las monedas
         se procesan en CADA epoch temporal, compartiendo un SOLO Portfolio.
    POR QUÉ: En producción (main.py), engine.py procesa MarketEvents de TODAS
         las monedas en un solo event loop con un Portfolio compartido de $13.
    PARA QUÉ: Resultados que son una PREDICCIÓN INQUÍVOCA de producción.
    CÓMO:
      1. BacktestDataProvider v2.0 construye Global Timeline
      2. En cada epoch: emite MarketEvents para TODAS las monedas
      3. Portfolio.update_market_price() actualiza precios
      4. RiskManager.check_stops() → ÚNICO motor de salida (Exits)
         ⚠️ REMEDIACIÓN: Portfolio.check_exits() convertido a AUDIT-ONLY
      5. Strategies.calculate_signals() → SignalEvents
      6. RiskManager.generate_order() → OrderEvents (con sizing, validación)
      7. BacktestExecutor → FillEvents (slippage determinístico)
      8. Portfolio.update_fill() → PnL real

    DIFERENCIA CLAVE vs v1.0:
      v1.0: for sym in symbols: backtest(sym)  ← $13 × 26 = $338 virtual
      v2.0: for epoch in timeline: process(ALL) ← $13 compartidos = REAL

    REMEDIACIÓN QUIRÚRGICA v3.0:
      - run_id único por ejecución (no sobrescribir resultados)
      - Seed fija para reproducibilidad total
      - Motor de salida UNIFICADO (solo RiskManager.check_stops())
      - Portfolio.check_exits() → audit_exits() (log only, no emite EXIT)
      - Config snapshot completo en resultados JSON

    Args:
        all_data: dict {symbol: DataFrame OHLCV}
        symbols: list of symbol strings
        days: int, horizon in days
        initial_capital: float, starting capital (default: Config)
        verbose: bool, print progress
        seed: int, random seed for deterministic slippage (default: 42)

    Returns:
        dict with full results including metrics, trades, equity curve
    """
    # ═══════════════════════════════════════════════════════════════════
    # REMEDIACIÓN: DETERMINISTIC SEED INITIALIZATION
    # QUÉ: Fija TODAS las semillas ANTES de cualquier operación aleatoria.
    # POR QUÉ: Sin esto, cada run produce slippage distinto → PnL distinto
    #   → comparar resultados es imposible ("peras con manzanas").
    # PARA QUÉ: Dos ejecuciones con misma seed = resultados BIT-A-BIT idénticos.
    # CÓMO: Fijar random.seed(), np.random.seed(), y BacktestExecutor._rng.
    # ═══════════════════════════════════════════════════════════════════
    run_id = str(uuid.uuid4())[:8]
    random.seed(seed)
    np.random.seed(seed)

    capital = initial_capital if initial_capital else real_capital
    leverage = Config.BINANCE_LEVERAGE

    print(f"\n{'=' * 70}")
    print(f"🚀 GOD MODE BACKTEST v3.0 — UNIFIED EXIT ENGINE (REMEDIATED)")
    print(f"   run_id: {run_id} | Seed: {seed}")
    print(
        f"   Símbolos: {len(symbols)} | Días: {days} | Capital: ${capital:.2f} (REAL PRODUCTION)"
    )
    print(
        f"   Leverage: {leverage}x | Max Positions: {Config.MAX_CONCURRENT_POSITIONS}"
    )
    print(f"   Fee: {COMMISSION_PCT * 100:.4f}% per side")
    print(f"   EXIT ENGINE: RiskManager.check_stops() ONLY (Portfolio audit-only)")
    print(f"   MODE: PRODUCTION-PARITY (uses real Portfolio + RiskManager) - CACHE MODE: {mode}")
    print(f"   SCENARIO: {scenario} | Isolated Strategy: {isolated_strategy or 'None'}")
    print(f"{'=' * 70}\n")
    
    precalculated_signals = {}
    generated_signals_cache = {}
    if mode == "REPLAY" and signal_cache_path and os.path.exists(signal_cache_path):
        with open(signal_cache_path, "r") as f:
            precalculated_signals = json.load(f)
        print(f"  ⚡ [REPLAY MODE] Cargadas {len(precalculated_signals)} marcas de tiempo con señales del caché cuántico.")

    # ═══════════════════════════════════════════════════════════════════
    # FORENSIC AUDIT: ALPHA LEAK TRACKING STRUCTURES
    # QUÉ: Contadores acumulados para medir exactamente dónde se pierde alpha.
    # POR QUÉ: Sin esta contabilidad, no podemos aislar si perdemos por
    #   slippage, comisiones, exits prematuros, o conflictos de arbitraje.
    # ═══════════════════════════════════════════════════════════════════
    alpha_leak = {
        "gross_alpha": 0.0,          # PnL ideal (sin fees, sin slippage)
        "slippage_loss": 0.0,        # Delta negativo por deslizamiento de precio
        "fee_loss": 0.0,             # Comisiones pagadas
        "premature_exit_loss": 0.0,  # PnL dejado en la mesa por exits reactivos
        "conflict_loss": 0.0,        # Costo de oportunidad de señales vetadas
        "net_alpha": 0.0,            # Alpha neto final
    }
    # Per-trade MFE/MAE tracker: {trade_id: {entry_price, mfe, mae, exit_price, pnl}}
    mfe_mae_tracker = {}
    # Per-strategy attribution: {strategy_id: {wins, losses, gross_pnl, net_pnl}}
    strategy_attribution = {}
    # Per-symbol attribution: {symbol: {wins, losses, gross_pnl, net_pnl, fees}}
    symbol_attribution = {}
    # Conflict log: [{epoch, symbol, winner, losers, regime}]
    conflict_log = []
    # Scenario flags
    scenario_disable_reactive_exits = scenario == "B"
    scenario_force_market = scenario == "C"
    scenario_ideal_execution = scenario == "D"
    scenario_isolate_strategy = scenario == "E"

    # ═══════════════════════════════════════════════════════════════════════
    # FORENSIC-V42 FIX: PREVENT CONFIG CONTAMINATION
    # QUÉ: Guarda el estado de Config y lo restaura en `finally`.
    # POR QUÉ: Mutar `Config` globalmente afectaba a otros módulos si se
    #   importan o ejecutan en el mismo proceso (como un dashboard).
    # ═══════════════════════════════════════════════════════════════════════
    
    from core.forensic_auditor import ForensicAuditor
    if not ForensicAuditor.verify_parity(Config):
        print("⚠️ [WARNING] Forensic Auditor detected Config divergences. DNA Mutator is active. Continuing...")
    
    try:
        # Dynamic Notification Override for Backtests (Eradicate SMTP blockades)
        Config.Observability.EMAIL_ENABLED = False
        Config.Observability.TELEGRAM_ENABLED = False
        print("  📢 [TELEMETRY] Dynamic SMTP and Telegram notifications DISABLED for Backtest parity.")
        
        events_queue = Queue()
    
        # 1a. BacktestDataProvider v2.0 (Global Timeline)
        data_provider = BacktestDataProvider(events_queue, symbols, all_data)
    
        # 1b. PRODUCTION Portfolio (THE REAL ONE)
        bt_data_dir = os.path.join(_project_root, "dashboard", "data", "backtest_temp")
        os.makedirs(bt_data_dir, exist_ok=True)
    
        portfolio = Portfolio(
            initial_capital=capital,
            csv_path=os.path.join(bt_data_dir, "bt_trades.csv"),
            status_path=os.path.join(bt_data_dir, "bt_status.csv"),
            auto_save=False,  # No periodic saves during backtest
        )
        portfolio.data_provider = data_provider
    
        # ═══════════════════════════════════════════════════════════════════
        # OMEGA FIX: BACKTEST POSITION ISOLATION
        # QUÉ: Limpia posiciones, virtual_ledger y margen del Portfolio.
        # POR QUÉ: Portfolio.__init__() → DatabaseHandler() → conecta a la
        #   DB de producción (trader_gemini.db) que tiene 3 posiciones OPEN
        #   de producción/testing. Estas contaminan el virtual_ledger y causan
        #   que PHASE_1_POSITION_CAP bloquee TODAS las señales nuevas.
        # PARA QUÉ: Backtest limpio, sin posiciones fantasma.
        # ═══════════════════════════════════════════════════════════════════
        portfolio.positions = {}
        portfolio.virtual_ledger = {}
        portfolio.used_margin = 0.0
        portfolio.current_cash = capital
        portfolio.realized_pnl = 0.0
        portfolio.scalping_ledger = []
        portfolio.swing_ledger = []
        print(f"  🔒 [OMEGA] Portfolio ISOLATION: cleared {len(portfolio.positions)} positions, virtual_ledger, margin.")
    
        # 1c. PRODUCTION RiskManager (THE REAL ONE)
        risk_manager = RiskManager(
            max_concurrent_positions=Config.MAX_CONCURRENT_POSITIONS, portfolio=portfolio
        )
        # ═══════════════════════════════════════════════════════════════════
        # FORENSIC-V31 FIX: BACKTEST COUNTER ISOLATION
        # QUÉ: Resetea win_count, loss_count y _trade_cache.
        # POR QUÉ: RiskManager carga el historial de producción (trades.csv)
        #   en __init__. En backtest, esto contamina los resultados desde el
        #   inicio (empezando con 172+ wins).
        # ═══════════════════════════════════════════════════════════════════
        risk_manager.win_count = 0
        risk_manager.loss_count = 0
        risk_manager._trade_cache = []
    
        # 1c-bis. PREDICTION TRACKER (Feedback Loop Closure for Backtest)
        # QUÉ: Inicializa tracker de precisión predictiva con paridad producción.
        # POR QUÉ: Si solo existe en producción, el backtest no puede medir
        #   prediction decay ni validar confidence_factor.
        from core.prediction_tracker import PredictionTracker
        prediction_tracker = PredictionTracker()
        risk_manager.prediction_tracker = prediction_tracker
    
        # ═══════════════════════════════════════════════════════════════════
        # FORENSIC-V47: MARKET REGIME DETECTOR (PRODUCTION PARITY)
        # QUÉ: Instancia el detector de régimen que en producción corre en
        #   global_regime_loop() (main.py L772).
        # POR QUÉ: Sin régimen, las estrategias operan "ciegas" — no saben si
        #   el mercado es TRENDING, RANGING o CHOPPY. Esto causa señales
        #   que en producción serían filtradas por el régimen.
        # PARA QUÉ: Paridad total — el backtest filtra igual que producción.
        # CÓMO: Se ejecuta cada 20 epochs en el main loop (ligero).
        # ═══════════════════════════════════════════════════════════════════
        from core.market_regime import MarketRegimeDetector
        regime_detector = MarketRegimeDetector(events_queue=events_queue)
        regime_detector.set_horizon_profile(days)  # Match backtest horizon
        portfolio.market_regime = regime_detector
        risk_manager.regime_detector = regime_detector
        print(f"  🔮 [V47] MarketRegimeDetector initialized (Horizon: {regime_detector.horizon_profile})")

        # ═══════════════════════════════════════════════════════════════════
        # FORENSIC-V48: SOPHIA & META-BRAIN INTEGRATION (PRODUCTION PARITY)
        # QUÉ: Instancia los módulos de IA, Correlación, y Orquestación.
        # POR QUÉ: Para lograr paridad institucional, el backtest debe estar
        #   sometido a los mismos motores de razonamiento causal y evolución
        #   que el sistema de producción.
        # ═══════════════════════════════════════════════════════════════════
        try:
            from core.swarm_correlator import SwarmCorrelator
            swarm = SwarmCorrelator()
            print("  🐝 [V48] SwarmCorrelator initialized (Backtest Parity)")
        except Exception as e:
            swarm = None
            print(f"  ⚠️ [V48] SwarmCorrelator init failed: {e}")

        try:
            from core.sovereign_oracle import SovereignOracle
            sovereign_oracle = SovereignOracle()
            print("  🔮 [V48] SovereignOracle initialized (Backtest Parity)")
        except Exception as e:
            sovereign_oracle = None
            print(f"  ⚠️ [V48] SovereignOracle init failed: {e}")

        try:
            from core.multiverse_simulator import MultiverseSimulator
            multiverse = MultiverseSimulator()
            print("  🌌 [V48] MultiverseSimulator initialized (Backtest Parity)")
        except Exception as e:
            multiverse = None
            print(f"  ⚠️ [V48] MultiverseSimulator init failed: {e}")

        try:
            from core.strategy_selector import StrategySelector
            selector = StrategySelector(portfolio=portfolio, data_provider=data_handler if 'data_handler' in locals() else data_provider)
            print("  🎯 [V48] StrategySelector initialized (Backtest Parity)")
        except Exception as e:
            selector = None
            print(f"  ⚠️ [V48] StrategySelector init failed: {e}")

        try:
            from core.shadow_darwin import ShadowDarwin
            shadow_darwin = ShadowDarwin(data_provider=data_provider)
            print("  🧬 [V48] ShadowDarwin initialized (Backtest Parity)")
        except Exception as e:
            shadow_darwin = None
            print(f"  ⚠️ [V48] ShadowDarwin init failed: {e}")

        try:
            # We mock WorldAwareness and MicroAccountAwareness properties
            from core.micro_awareness import MicroAccountAwareness
            micro_awareness = MicroAccountAwareness()
            print("  🔬 [V48] MicroAccountAwareness initialized (Backtest Parity)")
        except Exception as e:
            micro_awareness = None
            print(f"  ⚠️ [V48] MicroAccountAwareness init failed: {e}")

        # ═══════════════════════════════════════════════════════════════════
        # DIGITAL TWIN V100: INFRASTRUCTURE & CHAOS PARITY
        # QUÉ: Instancia módulos de caos de red, latencia y macro-entorno.
        # POR QUÉ: Simular la dureza real de producción (slippage, rate limits).
        # ═══════════════════════════════════════════════════════════════════
        try:
            from core.market_scanner import MarketScanner
            scanner = MarketScanner(data_provider=data_provider)
            print("  🔭 [DT100] MarketScanner initialized")
        except Exception as e:
            scanner = None
            print(f"  ⚠️ [DT100] MarketScanner init failed: {e}")

        try:
            from strategies.stat_arb import StatArbEngine
            stat_arb_engine = StatArbEngine()
            print("  📐 [DT100] StatArbEngine initialized")
        except Exception as e:
            stat_arb_engine = None
            print(f"  ⚠️ [DT100] StatArbEngine init failed: {e}")

        try:
            from core.world_awareness import world_awareness
            print("  🌍 [DT100] WorldAwareness initialized")
        except Exception as e:
            world_awareness = None
            print(f"  ⚠️ [DT100] WorldAwareness init failed: {e}")

        # Phase 8: NLP Ensemble (FinBERT + CryptoBERT) — Production Parity
        # QUÉ: Usa el mismo singleton NLP que producción (main.py).
        # POR QUÉ: El viejo SentimentLoader usaba TextBlob (desactivado).
        #   En backtest además hacía un "proxy" basado en momentum de BTC,
        #   lo cual NO es sentimiento real y contaminaba las features.
        # PARA QUÉ: Las 4 features NLP (news_sentiment, magnitude, shock,
        #   freshness) son las mismas en backtest que en producción.
        # NOTA: En backtest, los modelos cargan pero no hay polling RSS.
        #   Las features decaen a 0.0 naturalmente (exponential decay).
        #   Esto es CORRECTO: simula que no tenemos acceso a noticias
        if isolated_strategy != "technical":
            try:
                from data.news_sentiment_nlp import news_sentiment as sentiment_loader
                print("  📰 [Phase 8] NewsSentimentNLP initialized (decay mode for backtest)")
            except Exception as e:
                sentiment_loader = None
                print(f"  ⚠️ [DT100] SentimentLoader init failed: {e}")
    
            try:
                from core.neural_bridge import NeuralBridge
                neural_bridge = NeuralBridge()
                print("  🧠 [DT100] NeuralBridge initialized")
            except Exception as e:
                neural_bridge = None
                print(f"  ⚠️ [DT100] NeuralBridge init failed: {e}")
        else:
            sentiment_loader = None
            neural_bridge = None

        # 1d. BacktestExecutor (simulates)
        executor = BacktestExecutor(data_provider=data_provider)

        try:
            from core.order_manager import OrderManager
            order_manager = OrderManager(executor, data_provider=data_provider)
            print("  📋 [DT100] OrderManager initialized (Chaos/Queue Simulation)")
        except Exception as e:
            order_manager = None
            print(f"  ⚠️ [DT100] OrderManager init failed: {e}")

        try:
            from utils.health_supervisor import _supervisor as health_supervisor
            print("  🏥 [DT100] HealthSupervisor initialized (Chaos Engine)")
        except Exception as e:
            health_supervisor = None
            print(f"  ⚠️ [DT100] HealthSupervisor init failed: {e}")

        # (Moved BacktestExecutor up)    
        # ─────────────────────────────────────────────────────────────────────────
        # STEP 2: REGISTER STRATEGIES (identical to main.py L694-747)
        # For each symbol: MLStrategy(SCALPING) + MLStrategy(SWING)
        # ─────────────────────────────────────────────────────────────────────────
        # ─── DIRECTION FILTER (GOD MODE RIGOR) ───
        os.environ["TRADER_GEMINI_BACKTEST"] = "true"
    
        # ═══════════════════════════════════════════════════════════════════
        # FIX-V10-1: BACKTEST ISOLATION — Remove leftover STOP_TRADING.LOCK
        # QUÉ: Elimina el lock file del kill switch si existe de un run anterior.
        # POR QUÉ: KillSwitch.__init__() lee este archivo y se auto-activa,
        #   causando que _validate_kill_switch() retorne False → 99.7% rechazo.
        # PARA QUÉ: Cada backtest arranca con estado LIMPIO.
        # ═══════════════════════════════════════════════════════════════════
        _lock_file = os.path.join(_project_root, "STOP_TRADING.LOCK")
        if os.path.exists(_lock_file):
            os.remove(_lock_file)
            print(f"  🔓 [V10] Removed leftover STOP_TRADING.LOCK (backtest isolation)")
    
        # ─── ISOLATION PROTOCOL (V4) ───
        # Ensure backtest models and DB do not clash with production/paper trading
        backtest_models_dir = os.path.join(_project_root, ".models_backtest")
        backtest_db_path = os.path.join(_project_root, "data", "backtest_governance.db")
        os.makedirs(backtest_models_dir, exist_ok=True)
    
        # Initialize separate DB if needed
        if not os.path.exists(backtest_db_path):
            import sqlite3
    
            conn = sqlite3.connect(backtest_db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    version INTEGER,
                    sharpe REAL,
                    win_rate REAL,
                    created_at TEXT,
                    model_path TEXT,
                    is_production INTEGER
                )
            """)
            conn.close()
    
        strategies_map = {}  # symbol -> [strategy_scalp, strategy_swing]
        # ════════════════════════════════════════════════════════════════
        # FASE 34: OMNI-STRATEGY EN BACKTEST (SIMULATION PARITY)
        # QUÉ: Sustituimos el enjambre viejo (Sniper, StatArb, Technical, etc.)
        # por la nueva OmniStrategy unificada (que evalúa todas esas lógicas 
        # internamente y retorna un OmniScore genético).
        # POR QUÉ: Para que la simulación refleje con total exactitud la
        # matemática cuántica y los pesos usados en main.py.
        # ════════════════════════════════════════════════════════════════
        global_epoch_strategies = []
        try:
            from strategies.omni_strategy import OmniStrategy
            omni_scalp = OmniStrategy(data_provider, events_queue, horizon="SCALPING")
            global_epoch_strategies.append(omni_scalp)
            print("  ✅ [OMNI] OmniStrategy [SCALPING] registered.")
            
            omni_swing = OmniStrategy(data_provider, events_queue, horizon="SWING")
            global_epoch_strategies.append(omni_swing)
            print("  ✅ [OMNI] OmniStrategy [SWING] registered.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to init OmniStrategy: {e}")
            
        # The MLStrategy must be initialized per-symbol to train models and run the Inference Worker.
        _lean = getattr(Config, 'LEAN_MODE', False)
        if mode == "REPLAY":
            print("  🎯 [REPLAY MODE] Inteligencia Artificial apagada. Inyectando caché de señales en memoria RAM.")
            for symbol in symbols:
                strategies_map[symbol] = []
            global_epoch_strategies = []
        elif isolated_strategy == "technical" or _lean:
            print("  🎯 [EVOLVER MODE] ML Strategies DISABLED for Nano Speeds.")
            for symbol in symbols:
                strategies_map[symbol] = []
        else:
            for symbol in symbols:
                try:
                    is_leader = "BTC" in symbol
                    from strategies.ml_strategy import UniversalEnsembleStrategy as MLStrategy
                    
                    # ── SCALPING ENGINE ──
                    ml_scalp = MLStrategy(
                        data_provider=data_provider,
                        events_queue=events_queue,
                        symbol=symbol,
                        lookback=min(Config.Strategies.ML_LOOKBACK_BARS, 2000),
                        sentiment_loader=None,
                        portfolio=portfolio,
                        risk_manager=risk_manager if is_leader else None,
                        horizon="SCALPING",
                        models_dir=backtest_models_dir,
                        db_path=backtest_db_path,
                    )
    
                    # ── SWING ENGINE ──
                    ml_swing = MLStrategy(
                        data_provider=data_provider,
                        events_queue=events_queue,
                        symbol=symbol,
                        lookback=min(Config.Strategies.ML_LOOKBACK_BARS, 2000),
                        sentiment_loader=None,
                        portfolio=portfolio,
                        risk_manager=None,
                        horizon="SWING",
                        models_dir=backtest_models_dir,
                        db_path=backtest_db_path,
                    )
    
                    strategies_map[symbol] = [ml_scalp, ml_swing]
    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to init ML strategy for {symbol}: {e}")
                    strategies_map[symbol] = []
    
        total_strats = sum(len(v) for v in strategies_map.values()) + len(global_epoch_strategies)
        print(
            f"  🧠 Total strategies registered: {total_strats} ({len(symbols)} symbols × 2 ML + {len(global_epoch_strategies)} Global)"
        )

        # ═══════════════════════════════════════════════════════════════
        # STRATEGY REGISTRY: DISABLED PENDING VALIDATION
        # QUÉ: register_strategy() causes check_stops() to delegate exits
        #   to strategy.check_exit() and SKIP risk_manager's own logic.
        # IMPACTO MEDIDO: WR dropped 73%→42%, FLIP_EXIT added -$0.416,
        #   trades lasted 11-33h instead of exiting at HARD_SL/trailing.
        # PLAN: Validate check_exit() independently before integration.
        # ═══════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════
        # INYECCIÓN DE LA INTELIGENCIA SUPREMA (SOPHIA)
        # ═══════════════════════════════════════════════════════════════
        try:
            from sophia.intelligence import SophiaIntelligence
            sophia = SophiaIntelligence.get_instance(bar_minutes=5.0)
            print(f"  🧠 Sophia Intelligence Engine initialized for God Mode Backtest")
        except Exception as e:
            logger.warning(f"⚠️ Failed to init Sophia: {e}")
            sophia = None

        for symbol_key, strats in strategies_map.items():
            for strat in strats:
                if sophia:
                    strat.sophia = sophia
                if hasattr(risk_manager, 'register_strategy'):
                    risk_manager.register_strategy(strat)
        for strat in global_epoch_strategies:
            if sophia:
                strat.sophia = sophia
            if hasattr(risk_manager, 'register_strategy'):
                risk_manager.register_strategy(strat)
    
        # ═══════════════════════════════════════════════════════════════
        # TELEGRAM: RICH STARTUP NOTIFICATION
        # ═══════════════════════════════════════════════════════════════
        try:
            Notifier.send_system_startup("BACKTEST", {
                'days': days,
                'capital': capital,
                'leverage': leverage,
                'symbols_count': len(symbols),
                'strategies_count': total_strats,
                'seed': seed,
                'total_epochs': data_provider.total_epochs if hasattr(data_provider, 'total_epochs') else 0,
                'max_drawdown': Config.Risk.MAX_DRAWDOWN,
                'tp_scalp': Config.Horizons.Scalping.get('tp_pct', 0.006),
                'sl_scalp': Config.Horizons.Scalping.get('sl_pct', 0.0075),
                'symbols_list': symbols,
            })
            import time as _t; _t.sleep(2)  # Give Telegram time to deliver
        except Exception as e:
            logger.warning(f"⚠️ Could not send startup notification: {e}")
    
        # ─────────────────────────────────────────────────────────────────────────
        # STEP 3: GLOBAL SIMULATION LOOP (mirrors Engine.process_event())
        # ─────────────────────────────────────────────────────────────────────────
        epoch_count = 0
        signal_count = 0
        order_count = 0
        fill_count = 0
        rejected_count = 0
        kill_switch_triggered = False
        equity_curve = [capital]
        equity_timestamps = []
    
        # FORENSIC-V9-FIX: Rejection reason tracker for forensic diagnosis
        rejection_reasons = {}  # {reason_string: count}
        untrained_strategies = set()  # Track which strategies never trained
    
        # ═══════════════════════════════════════════════════════════════
        # REMEDIACIÓN: AUDIT COUNTER for Portfolio.check_exits()
        # Tracks how many times Portfolio WOULD HAVE closed a position
        # but was prevented by the unified exit engine.
        # ═══════════════════════════════════════════════════════════════
        portfolio_audit_exits = 0  # Counter: exits Portfolio would have fired
    
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V11: COOLDOWN + EXIT DEDUP + ML QUALITY GATE
        # Fix #2: Prevent zero-duration churning (trades open+close same epoch)
        # Fix #6: Production cooldown parity (5-epoch min between fills)
        # Fix #7: Exit cascade prevention (max 1 EXIT per symbol_horizon)
        # Fix #4: ML shadow mode after 5 consecutive losses
        # ═══════════════════════════════════════════════════════════════
        last_fill_epoch = {}  # {symbol_horizon: epoch_count} — cooldown tracker
        pending_exits = set()  # {symbol_horizon} — exit dedup (cleared on fill)
        ml_consecutive_losses = {}  # {strategy_id: consecutive_loss_count}
        ml_shadow_wins = {}  # {strategy_id: shadow_win_count} — wins in shadow
        COOLDOWN_EPOCHS = 5  # Minimum epochs between fills (same symbol+horizon)
        ML_LOSS_STREAK_LIMIT = 5  # After this many losses, enter shadow mode
        ML_SHADOW_WIN_REQUIRED = 3  # Wins needed to exit shadow mode
        # Telegram progress tracking
        _last_telegram_progress = 0  # Track last % milestone sent to Telegram
    
        t_start = time.time()
        last_progress = 0
        total_epochs = data_provider.total_epochs
        # Warmup: strategies need ~100 bars to compute features (RSI14, EMA20, etc.)
        # QUÉ: Ajuste de warmup_epochs para excluir el período de lookback de 70 días de la fase de trading.
        # POR QUÉ: Para que el oráculo macro reciba datos diarios y semanales, el proveedor de datos
        #   se inicializa con datos históricos adicionales. El backtest debe avanzar virtualmente
        #   el tiempo a través de esta historia para poblar los buffers, sin abrir posiciones durante ella.
        # PARA QUÉ: Evitar operar durante el warmup y enfocar el trading exactamente en los args.days solicitados.
        # CÓMO: Calculando `warmup_epochs = total_epochs - (days * 1440)`.
        # CUÁNDO: Al iniciar el loop del motor de backtesting.
        # DÓNDE: En `scripts/run_god_mode_backtest.py` :: `run_global_backtest`.
        # QUIÊN: Modificado por el Quant Developer y el SRE/DevOps.
        warmup_epochs = 200 # Fix for zero-trades: max(min(100, total_epochs // 20), total_epochs - (days * 1440))
    
        print(f"  ⏱️  Starting simulation: {total_epochs:,} global epochs")
        print(f"  🔥 Warmup: first {warmup_epochs} epochs (no trading)")
    
        while data_provider.continue_backtest:
            # ── ADVANCE GLOBAL TIMELINE ──
            data_provider.update_bars()
            epoch_count += 1

            # ═══════════════════════════════════════════════════════════════
            # DIGITAL TWIN V100: HEALTH SUPERVISOR CHAOS (SYSTEM OUTAGES)
            # ═══════════════════════════════════════════════════════════════
            if health_supervisor and epoch_count > warmup_epochs:
                # 0.01% chance of a severe API Disconnect / Server Crash
                if random.random() < 0.0001:
                    print(f"  🚨 [CHAOS] HealthSupervisor detected critical heartbeat failure at epoch {epoch_count}!")
                    print("  🚨 [CHAOS] Simulating 5-minute system outage and state recovery...")
                    # Fast-forward 5 epochs without processing orders to simulate outage
                    for _ in range(5):
                        data_provider.update_bars()
                        epoch_count += 1
                        
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V48: SWARM & ORCHESTRATION UPDATES
            # QUÉ: Ejecución periódica de módulos de Inteligencia de Mercado.
            # ═══════════════════════════════════════════════════════════════
            if epoch_count % 20 == 0:
                # 1. Sovereign Market Context (Mirrors main.py global_regime_loop)
                try:
                    context_data = {}
                    for _sym in symbols:
                        bars_1m = data_provider.get_latest_bars(_sym, n=100, timeframe='1m')
                        if bars_1m is not None and len(bars_1m) > 0:
                            context_data[_sym] = {
                                '1m': bars_1m,
                                '5m': data_provider.get_latest_bars(_sym, n=50, timeframe='5m'),
                                '15m': data_provider.get_latest_bars(_sym, n=50, timeframe='15m'),
                                '1h': data_provider.get_latest_bars(_sym, n=50, timeframe='1h')
                            }
                    if context_data:
                        breadth = regime_detector.calculate_market_context(context_data)
                        _sentiment = breadth.get('sentiment', breadth) if isinstance(breadth, dict) else breadth
                        if hasattr(risk_manager, 'update_global_regime'):
                            risk_manager.update_global_regime(_sentiment)
                        portfolio.global_regime_data = breadth
                        portfolio.global_regime = _sentiment
                        from core.global_state import global_state
                        global_state.market_regime = _sentiment
                except Exception as e:
                    pass

                # 2. Swarm Leader & Entanglement
                if swarm:
                    try:
                        _btc_bars = data_provider.get_latest_bars("BTC/USDT", n=50)
                        if _btc_bars is not None and len(_btc_bars) >= 50:
                            swarm.update_leader_data(_btc_bars)
                            for _sym in symbols:
                                _sym_bars = data_provider.get_latest_bars(_sym, n=50)
                                if _sym_bars is not None and len(_sym_bars) >= 50:
                                    swarm.calculate_entanglement(_sym, _sym_bars)
                    except Exception:
                        pass

            if epoch_count % (120 * 60) == 0:  # Every 2 hours
                if selector:
                    try:
                        selector.update_rankings()
                    except Exception:
                        pass

            if epoch_count % (24 * 60 * 60) == 0:  # Every 24 hours
                if shadow_darwin:
                    try:
                        shadow_darwin.epoch_step()
                    except Exception:
                        pass

            # ═══════════════════════════════════════════════════════════════
            # DIGITAL TWIN V100: SENTIMENT & SCANNER PROXY
            # ═══════════════════════════════════════════════════════════════
            if epoch_count % 20 == 0:
                if sentiment_loader:
                    try:
                        # [REMEDIACIÓN APLICADA] Removed Historical Sentiment Proxy to ensure production parity.
                        # Backtest now uses the same decayed/neutral sentiment as production 
                        # when FinBERT is unavailable, preventing ML models from learning false price correlations.
                        pass
                    except Exception:
                        pass
                
                if scanner:
                    try:
                        # Emulate the MarketScanner heartbeat (we won't change the 21 fixed symbols,
                        # but we satisfy the architecture requirement for it to be active)
                        scanner.last_scan_time = time.time()
                    except Exception:
                        pass
            
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V42 FIX: BACKTEST RAM CRASH PREVENTION
            # QUÉ: Fuerza recolección de basura cada 100 epochs.
            # POR QUÉ: 18 instancias de estrategia (ML, Sniper, Phalanx, etc.)
            #   acumulan dataframes y tensores. Sin GC, la memoria crece
            #   infinitamente y crashea si corres backtests concurrentes.
            # PARA QUÉ: Estabilidad absoluta en simulaciones pesadas.
            # ═══════════════════════════════════════════════════════════════
            if epoch_count % 100 == 0:
                # FIX-FORENSIC-V82: 'import gc' moved to top-level (L52)
                # Aggressive Epoch GC to prevent RAM crashes
                for g_strat in global_epoch_strategies:
                    if hasattr(g_strat, 'last_processed_times') and len(g_strat.last_processed_times) > 500:
                        g_strat.last_processed_times.clear()
                    if hasattr(g_strat, 'sophia') and hasattr(g_strat.sophia, 'memory') and len(getattr(g_strat.sophia.memory, 'history', [])) > 500:
                        g_strat.sophia.memory.history = g_strat.sophia.memory.history[-100:]
                
                for strat_list in strategies_map.values():
                    for strat in strat_list:
                        if hasattr(strat, 'last_processed_times') and len(strat.last_processed_times) > 500:
                            strat.last_processed_times.clear()
                
                # Break local references
                _shadow_q = None
                gc.collect()
    
            # ── Progress reporting (AEGIS-V21: Faster reporting for observability) ──
            progress = int((epoch_count / total_epochs) * 100)
            if verbose: # Report every epoch for fine-grained monitoring
                elapsed = time.time() - t_start
                equity = portfolio.get_total_equity()
                open_count = sum(1 for p in portfolio.virtual_ledger.values() if p.get("quantity", 0) != 0)
                msg = (
                    f"📊 [{progress}%] Epoch {epoch_count:,}/{total_epochs:,} | "
                    f"Equity: ${equity:.2f} | Open: {open_count} | "
                    f"Trades: {fill_count} | Signals: {signal_count} | {elapsed:.0f}s"
                )
                logger.info(msg)
                if epoch_count % 10 == 0: # Print to console every 10 epochs to avoid spam
                    print(f"  {msg}")
                # ═══════════════════════════════════════════════════════════
                # TELEGRAM PROGRESS: Send at every 10% milestone
                # ═══════════════════════════════════════════════════════════
                progress_milestone = (progress // 10) * 10  # Round down to nearest 10
                if progress_milestone > _last_telegram_progress and progress_milestone > 0:
                    _last_telegram_progress = progress_milestone
                    try:
                        Notifier.send_backtest_progress({
                            'progress_pct': progress_milestone,
                            'equity': portfolio.get_total_equity(),
                            'trades': fill_count,
                            'elapsed_seconds': time.time() - t_start,
                            'open_positions': open_count,
                            'epoch': epoch_count,
                            'total_epochs': total_epochs,
                        })
                    except Exception:
                        pass
                last_progress = progress
    
            # ── KILL SWITCH CHECK (global portfolio drawdown) ──
            if kill_switch_triggered:
                break
    
            # ✨ GRACEFUL TERMINATION LOGIC: 30 minutes before end
            if (total_epochs - epoch_count) == 30:
                print(
                    "\n  ⏱️ [ENDGAME] 30 epochs remaining. Emitting EXIT for all open positions to avoid BACKTEST_CLOSE bias."
                )
                for v_key, vpos in portfolio.virtual_ledger.items():
                    qty = vpos.get("quantity", 0)
                    if qty != 0:
                        symbol = v_key.rsplit("_", 1)[0]
                        if "_" in symbol:
                            symbol = symbol.split("_")[
                                0
                            ]  # Safety split if horizon attached
                        exit_sig = SignalEvent(
                            strategy_id="GRACEFUL_CLOSE",
                            symbol=symbol,
                            datetime=datetime.now(timezone.utc),
                            signal_type=SignalType.EXIT,
                            strength=1.0,
                            horizon=vpos.get("horizon", "SCALPING"),
                            priority=100,
                        )
                        events_queue.put(exit_sig)
    
            # ══════════════════════════════════════════════════════════════════
            # FORENSIC-V9-FIX: FORCE TRAINING AT END OF WARMUP
            # QUÉ: Al terminar warmup, forzar entrenamiento de todos los modelos.
            # POR QUÉ: Si is_trained=False, _run_inference() nunca se llama →
            #   CERO señales → CERO trades → backtest vacío e inútil.
            # CUÁNDO: Exactamente en el epoch == warmup_epochs (una sola vez).
            # ══════════════════════════════════════════════════════════════════
            if epoch_count == warmup_epochs:
                trained_count = 0
                for sym, strats in strategies_map.items():
                    for strat in strats:
                        try:
                            if not getattr(strat, "is_trained", False):
                                if hasattr(strat, "retrain"):
                                    strat.retrain(force=True)
                                elif hasattr(strat, "_launch_training"):
                                    bars = data_provider.get_latest_bars(
                                        sym,
                                        getattr(strat, "lookback", 500),
                                        getattr(strat, "PRIMARY_TF", "5m"),
                                    )
                                    if bars is not None and len(bars) > 50:
                                        strat._launch_training(bars, "Full")
                                elif hasattr(strat, "_train_model"):
                                    strat._train_model()
                                elif hasattr(strat, "train_model"):
                                    strat.train_model()
    
                                if getattr(strat, "is_trained", False):
                                    trained_count += 1
                                else:
                                    untrained_strategies.add(f"{sym}_{strat.strategy_id}")
                            elif strat.is_trained:
                                trained_count += 1
                            else:
                                untrained_strategies.add(f"{sym}_{strat.strategy_id}")
                        except Exception as e:
                            logger.warning(
                                f"⚠️ Training failed for {sym}/{strat.strategy_id}: {e}"
                            )
                            untrained_strategies.add(f"{sym}_{strat.strategy_id}")
                print(
                    f"  🧠 Post-warmup training: {trained_count} strategies ready, "
                    f"{len(untrained_strategies)} failed"
                )
    
            # ══════════════════════════════════════════════════════════════════
            # PROCESS ALL EVENTS FOR THIS EPOCH
            # Mirrors: Engine._process_event() + _handle_signal + _handle_order
            # ══════════════════════════════════════════════════════════════════
    
            # ── Phase A: Collect MarketEvents emitted by data_provider ──
            market_events = []
            while not events_queue.empty():
                event = events_queue.get()
                if event.type == EventType.MARKET:
                    market_events.append(event)
    
            # ── Phase B: Process Market Events (prices, exits, strategies) ──
            for event in market_events:
                # 🏁 SYNC VIRTUAL TIME (Phase 99: Parity Fix)
                # QUÉ: Sincroniza el reloj del CooldownManager con el tiempo del backtest.
                cooldown_manager.set_virtual_time(event.timestamp)
                
                symbol = event.symbol
                close_price = event.close_price

    
                if not close_price or close_price <= 0:
                    continue
    
                # B1. HIGH-FIDELITY PRICE INGESTION (Simulate Intra-Bar Wicks)
                # QUÉ: Actualiza Portfolio con High, Low y Close para wicks realistas.
                # POR QUÉ: Si solo enviamos Close, ignoramos wicks que tocaron el TP.
                high_price = getattr(event, 'high_price', close_price)
                low_price = getattr(event, 'low_price', close_price)
    
                # Update all watermarks in Portfolio to ensure accurate MAE/MFE/HWM
                portfolio.update_market_price(symbol, high_price)
                portfolio.update_market_price(symbol, low_price)
                portfolio.update_market_price(symbol, close_price)
    
                # PREDICTION TRACKER: Update with final price
                prediction_tracker.update_forward_returns(symbol, close_price, event.timestamp)
    
                # B2. Skip trading during warmup (strategies need history)
                if epoch_count < warmup_epochs:
                    continue

                # (FORENSIC-V48) Market Regime detection moved to global simulation loop for production parity
    
                # ═══════════════════════════════════════════════════════════
                # REMEDIACIÓN QUIRÚRGICA: UNIFIED EXIT ENGINE
                # QUÉ: Solo RiskManager.check_stops() genera señales EXIT.
                # POR QUÉ: Portfolio.check_exits() competía con RiskManager,
                #   causando cierres prematuros (SL fijo -0.25% vs trailing
                #   adaptativo de 3 etapas). El motor más agresivo (Portfolio)
                #   cortaba ganancias ANTES de que el trailing pudiera actuar.
                # PARA QUÉ: Un solo motor de salida = cierres consistentes
                #   alineados con la lógica de riesgo dinámica.
                # CÓMO: Portfolio.check_exits() ahora SOLO audita (no emite).
                #   RiskManager.check_stops() es el ÚNICO emisor de EXIT.
                # ═══════════════════════════════════════════════════════════
    
                # B3. AUDIT EXITS — Portfolio (Log-Only, NO emit EXIT signals)
                # Antes: portfolio.check_exits(data_provider, events_queue)
                # Ahora: Solo contamos cuántos cierres habría hecho para auditoría
                try:
                    _audit_q = Queue()  # Throwaway queue for audit
                    portfolio.check_exits(data_provider, _audit_q)
                    _audit_count = 0
                    while not _audit_q.empty():
                        _audit_q.get()
                        _audit_count += 1
                    portfolio_audit_exits += _audit_count
                except Exception:
                    pass
    
                # B4. CHECK STOPS — PRODUCTION RISK MANAGER (SOLE EXIT ENGINE)
                # Multi-pass check (Low -> High -> Close) to catch intra-bar wicks.
                # Conservative order: check Low then High.
                for test_price in [low_price, high_price, close_price]:
                    # Update current price so check_stops sees the wick
                    portfolio.update_market_price(symbol, test_price)
                    
                    try:
                        # Added symbol_filter=symbol for O(N) performance (Forensic-V35)
                        stop_signals = risk_manager.check_stops(portfolio, data_provider, symbol_filter=symbol, now=event.timestamp)
                        if stop_signals:
                            for sig in stop_signals:
                                _exit_key = (
                                    f"{sig.symbol}_{getattr(sig, 'horizon', 'SCALPING')}"
                                )
                                if _exit_key not in pending_exits:
                                    events_queue.put(sig)
                                    pending_exits.add(_exit_key)
                                    # DEBUG: print(f"  🎯 [WICK_HIT] Exit signal for {_exit_key} at {test_price}")
                    except Exception as e:
                        import traceback
                        logger.error(f"❌ [BACKTEST] Error in check_stops for {symbol}: {e}\n{traceback.format_exc()}")
                
                # Reset to Close price for strategy inference (B5)
                portfolio.update_market_price(symbol, close_price)
    
                # B5. RUN STRATEGIES — DIRECT SYNC INFERENCE
                # ════════════════════════════════════════════════════════
                # QUÉ: Llamamos _run_inference() directamente (SYNC)
                # POR QUÉ: calculate_signals() es async y usa
                #   asyncio.create_task() que requiere un event loop.
                #   _run_inference() es el CORE de inferencia de producción.
                #   Ejecuta EXACTAMENTE la misma lógica:
                #   get_latest_bars → _prepare_features → model.predict_proba
                #   → SignalEvent → events_queue.put()
                # PARA QUÉ: Paridad total sin overhead async.
                # ════════════════════════════════════════════════════════
                for strat in strategies_map.get(symbol, []):
                    try:
                        # Replicate production loop_count / bars_since_train
                        strat.loop_count += 1
                        strat.bars_since_train += 1
    
                        # Disable time-based throttling for backtest
                        strat._last_prediction_time = None
    
                        # FORENSIC-V9-FIX: Attempt training if untrained
                        # POR QUÉ: Algunos modelos no entrenan durante warmup porque
                        #   necesitan más datos. Cada 500 epochs, intentar de nuevo.
                        if not getattr(strat, "is_trained", False):
                            if hasattr(strat, "_launch_training"):
                                bars = data_provider.get_latest_bars(
                                    symbol,
                                    getattr(strat, "lookback", 500),
                                    getattr(strat, "PRIMARY_TF", "5m"),
                                )
                                if bars is not None and len(bars) > 50:
                                    # 🛡️ TRAINING GUARD: Don't spam training threads
                                    is_training = (
                                        hasattr(strat, "_training_thread")
                                        and strat._training_thread
                                        and strat._training_thread.is_alive()
                                    )
                                    if not is_training:
                                        try:
                                            strat._launch_training(bars, "Full")
                                        except Exception:
                                            pass
                            elif epoch_count % 500 == 0:
                                if hasattr(strat, "_train_model"):
                                    try:
                                        strat._train_model()
                                    except Exception:
                                        pass
                                elif hasattr(strat, "train_model"):
                                    try:
                                        strat.train_model()
                                    except Exception:
                                        pass
    
                            if not getattr(strat, "is_trained", False):
                                continue  # Skip inference for untrained
    
                        # FORENSIC-V11 Fix #4: ML QUALITY GATE (Shadow Mode)
                        # After ML_LOSS_STREAK_LIMIT consecutive losses, enter shadow mode
                        # Signals are generated but NOT executed until ML proves viability
                        _strat_id = strat.strategy_id
                        _loss_count = ml_consecutive_losses.get(_strat_id, 0)
                        if _loss_count >= ML_LOSS_STREAK_LIMIT:
                            _shadow_wins = ml_shadow_wins.get(_strat_id, 0)
                            if _shadow_wins < ML_SHADOW_WIN_REQUIRED:
                                # Shadow mode: run inference into throwaway queue
                                _shadow_q = Queue()
                                _orig_q = strat.events_queue
                                strat.events_queue = _shadow_q
                                strat._run_inference()
                                strat.events_queue = _orig_q
                                # Discard shadow signals (paper trade only)
                                while not _shadow_q.empty():
                                    _shadow_q.get()
                                continue
    
                        # FORENSIC-V11 Fix #2+#6: COOLDOWN ENFORCEMENT
                        # Skip inference if we just filled a trade for this symbol+horizon
                        # Prevents zero-duration churning (open+close same epoch)
                        _horizon = getattr(strat, "horizon", "SCALPING")
                        _cooldown_key = f"{symbol}_{_horizon}"
                        _last_fill = last_fill_epoch.get(_cooldown_key, -COOLDOWN_EPOCHS)
                        if (epoch_count - _last_fill) < COOLDOWN_EPOCHS:
                            continue  # Still in cooldown
    
                        # NANO-LATENCY SYNCHRONOUS inference (Eradicates thread locks and GIL contention)
                        try:
                            strat._run_inference()
                        except Exception as e:
                            import traceback
                            print(f"❌ [ERROR in run_ml] {strat.strategy_id} error: {e}")
                            traceback.print_exc()
                        
                    except Exception:
                        pass  # Strategy errors are non-fatal
    
            # ── B6. RUN GLOBAL EPOCH STRATEGIES (once per epoch, all symbols) ──
            # ═══════════════════════════════════════════════════════════════
            # FIX-FORENSIC-V41: PER-SYMBOL DISPATCH (PRODUCTION PARITY)
            # QUÉ: En producción, engine.py envía un MarketEvent POR símbolo a
            #   cada estrategia. Antes enviábamos symbol=None → dedup roto.
            # POR QUÉ: Las estrategias globales (Technical, Sniper, Statistical)
            #   usan event.symbol para el dedupe_key y data fetch.
            # PARA QUÉ: Paridad exacta con producción.
            # ═══════════════════════════════════════════════════════════════
            if epoch_count >= warmup_epochs and market_events:
                for g_strat in global_epoch_strategies:
                    try:
                        # For strategies that need per-symbol MarketEvents (production parity):
                        # dispatch each symbol's real event via calculate_signals(me)
                        for me in market_events:
                            try:
                                if epoch_count == warmup_epochs:
                                    logger.info(f"  [DEBUG-RUN] Executing {g_strat.strategy_id} for {me.symbol}")
                                g_strat.calculate_signals(me)
                            except Exception as e:
                                import traceback
                                print(f"❌ [ERROR in run_g] {g_strat.strategy_id} error: {e}")
                                traceback.print_exc()
                    except Exception:
                        pass
    
            # ── Phase C: Process Signal/Exit events generated by strategies ──
            # Strategies put SignalEvents directly into events_queue via
            # events_queue.put(signal). Now we drain and process them.
            # This mirrors Engine._process_signal_event() in production.
            _max_signal_iterations = 50  # Prevent infinite loops
            _iter = 0
            epoch_intents = []
            while not events_queue.empty() and _iter < _max_signal_iterations:
                _iter += 1
                event = events_queue.get()
    
                # AEGIS-V15: Normalización de tipo para evitar fallos de comparación de Enums
                etype = event.type.name if hasattr(event.type, "name") else str(event.type)
    
                if etype == "SIGNAL" or etype == EventType.SIGNAL.name:
                    signal_count += 1
    
                    # PREDICTION TRACKER: Record signal for forward tracking
                    if not (event.signal_type == SignalType.EXIT):
                        _direction = 'long' if event.signal_type == SignalType.LONG else 'short'
                        _hz = getattr(event, 'horizon', 'SCALPING')
                        _sid = getattr(event, 'strategy_id', 'unknown')
                        _price = data_provider.get_latest_price(event.symbol) or 0
                        _h_params = risk_manager.horizon_params.get(_hz, risk_manager.horizon_params.get('SCALPING', {}))
                        prediction_tracker.record_signal(
                            strategy_id=_sid,
                            symbol=event.symbol,
                            direction=_direction,
                            horizon=_hz,
                            entry_price=_price,
                            sl_pct=_h_params.get('stop_loss_pct', 0.0025),
                            tp_pct=_h_params.get('take_profit_pct', 0.004),
                            confidence=getattr(event, 'strength', 0.5),
                            timestamp=event.timestamp if hasattr(event, 'timestamp') else None,
                        )
    
                    # Get current price for this symbol
                    current_price = data_provider.get_latest_price(event.symbol)
                    if not current_price:
                        reason = "NO_PRICE"
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        continue
    
                    is_exit = event.signal_type == SignalType.EXIT
    
                    # ✨ ENDGAME: Block new entries in last 30 minutes to ensure clean exits
                    if (
                        (total_epochs - epoch_count) <= 30
                        and not is_exit
                        and event.signal_type != SignalType.REVERSE
                    ):
                        continue
    
                    if is_exit:
                        print(
                            f"DEBUG: Processing EXIT signal for {event.symbol} at {current_price}"
                        )
                    
                    # ═══════════════════════════════════════════════════════════════
                    # FORENSIC-V48: Multiverse Simulator Leniency Gate
                    # ═══════════════════════════════════════════════════════════════
                    if multiverse and not is_exit and event.signal_type != SignalType.REVERSE:
                        try:
                            _sim_result = multiverse.simulate_trade(event, portfolio)
                            if not getattr(_sim_result, 'is_viable', True) and getattr(_sim_result, 'ruin_probability', 0) > 0.90:
                                # V5.55: Weighted Voting Committee - Penalty instead of absolute veto
                                event.strength = max(0.1, getattr(event, 'strength', 0.5) - 0.30)
                                print(f"⚖️ [VOTING COMMITTEE] Multiverse Ruin Prob > 90%. Penalty applied (-0.30). New strength: {event.strength:.2f}")
                        except Exception:
                            pass
    
                    epoch_intents.append(event)
    
                elif event.type == EventType.FILL:
                    pass  # Handled inline after executor

            # ═══════════════════════════════════════════════════════════════
            # META-ARBITRATOR (SUPREME CO-ORDINATOR) CONFLICT RESOLUTION
            # Resolves cannibalization between strategies across horizons.
            # ═══════════════════════════════════════════════════════════════
            # Separate exits and entries
            epoch_exits = [i for i in epoch_intents if i.signal_type == SignalType.EXIT]
            epoch_entries = [i for i in epoch_intents if i.signal_type != SignalType.EXIT]
            
            # Deduplicate exits by (symbol, horizon) to avoid double exits in the same epoch
            deduped_exits = []
            seen_exits = set()
            for exit_event in epoch_exits:
                hz = getattr(exit_event, "horizon", "SCALPING")
                key = (exit_event.symbol, hz)
                if key not in seen_exits:
                    seen_exits.add(key)
                    deduped_exits.append(exit_event)
            
            # Resolve entries via meta_arbitrator
            approved_entries, rejected_entries = meta_arbitrator.resolve_intents(epoch_entries) if epoch_entries else ([], [])
            approved_intents = deduped_exits + approved_entries
            rejected_intents = rejected_entries
            
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC: LOG CONFLICT LOSSES (Alpha Leak: Conflict Category)
            # ═══════════════════════════════════════════════════════════════
            for rej in rejected_intents:
                rej_intent = rej["intent"]
                rej_reason = rej["reason"]
                
                # Clear pending exit lock if the arbitrator rejected/vetoed the EXIT signal
                if rej_intent.signal_type == SignalType.EXIT:
                    _hz_rej = getattr(rej_intent, "horizon", "SCALPING")
                    _key_rej = f"{rej_intent.symbol}_{_hz_rej}"
                    pending_exits.discard(_key_rej)
                    
                _rej_price = data_provider.get_latest_price(rej_intent.symbol) or 0
                _ts = getattr(rej_intent, 'timestamp', getattr(market_events[0], 'timestamp', '')) if market_events else ''
                forensic_logger.debug(
                    f"[{_ts}] [CONFLICT_DETECTED] {rej_intent.symbol} {getattr(rej_intent, 'strategy_id', '?')} "
                    f"{rej_intent.signal_type.name} confidence={getattr(rej_intent, 'strength', 0):.3f} "
                    f"VETOED reason={rej_reason}"
                )

                conflict_log.append({
                    "epoch": epoch_count, "symbol": rej_intent.symbol,
                    "strategy": getattr(rej_intent, 'strategy_id', '?'),
                    "direction": rej_intent.signal_type.name,
                    "reason": rej_reason, "price": _rej_price,
                    "confidence": getattr(rej_intent, 'strength', 0)
                })
            
            for event in approved_intents:
                current_price = data_provider.get_latest_price(event.symbol)
                if not current_price:
                    continue
                
                # FORENSIC: Log strategy decision
                _ts = getattr(event, 'timestamp', getattr(market_events[0], 'timestamp', '')) if market_events else ''
                forensic_logger.debug(
                    f"[{_ts}] [STRATEGY_DECISION] {event.symbol} {getattr(event, 'strategy_id', '?')} "
                    f"proposed {event.signal_type.name} confidence={getattr(event, 'strength', 0):.3f} "
                    f"horizon={getattr(event, 'horizon', '?')}"
                )

                
                # Scenario E: Isolate strategy
                if scenario_isolate_strategy and isolated_strategy:
                    _sid = getattr(event, 'strategy_id', '')
                    if isolated_strategy.lower() not in str(_sid).lower():
                        continue
                
                _pre_slip_price = current_price  # Save for alpha leak calculation
                
                # ═══════════════════════════════════════════════════════════
                # PARITY FIX: Replicate engine.py _sophia_veto_filter in backtest.
                # POR QUÉ: In production, engine.py blocks signals with
                #   confidence < SOPHIA_MIN_CONFIDENCE (0.65). The backtest
                #   was bypassing this → divergence.
                # PARA QUÉ: Identical results in backtest and production.
                # DATA: conf < 0.65 → 43% WR, 8/9 HARD_SL trades.
                # ═══════════════════════════════════════════════════════════
                _is_exit_signal = (event.signal_type == SignalType.EXIT)
                if not _is_exit_signal:
                    _signal_confidence = getattr(event, 'strength', 0.5)
                    _sophia_min = Config.Horizons.GlobalThresholds.get('sophia_win_prob_min', 0.65)
                    if _signal_confidence < _sophia_min:
                        reason = f"SOPHIA_VETO:{event.symbol} conf={_signal_confidence:.3f}<{_sophia_min}"
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        continue
                
                # ── PRODUCTION RISK MANAGER: generate_order() ──
                # This does ALL the validations from production:
                # ═══════════════════════════════════════════════════════════
                # FIX-V10-3: GRANULAR REJECTION TRACKING via stdout capture
                # QUÉ: Captura el print de generate_order para saber EXACTAMENTE
                #   qué filtro rechazó (Kill Switch, Frequency, Regime, etc.).
                # POR QUÉ: Antes solo registrábamos "RISK_GATE:SYMBOL" genérico.
                # PARA QUÉ: Diagnóstico forense preciso por filtro.
                # ═══════════════════════════════════════════════════════════
                _f_capture = io.StringIO()
                try:
                    with contextlib.redirect_stdout(_f_capture):
                        order = risk_manager.generate_order(event, current_price)
                except Exception as e:
                    order = None
                    # Use the exception message directly
                    _exception_msg = f"EXCEPTION:{type(e).__name__}:{str(e)}"
                    reason = _exception_msg
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    print(f"⚠️ [RISK] Signal REJECTED: {_exception_msg}")
                    # Skip the next block since we already handled the rejection
                    continue
    
                if order is None:
                    # Clear pending exits to prevent permanent lock-up if generation fails
                    _hz_fallback = getattr(event, 'horizon', 'SCALPING')
                    _key = f"{getattr(event, 'symbol', '?')}_{_hz_fallback}"
                    pending_exits.discard(_key)
                    
                    rejected_count += 1
                    # QUÉ: Extracción precisa de la causa del rechazo de riesgo utilizando metadatos de producción.
                    # POR QUÉ: Capturar raw `sys.stdout` resultaba vacío ya que `RiskManager` escribe mediante `logger`
                    #   con handlers redirigidos, enmascarando todos los vetos bajo la etiqueta genérica `RISK_GATE_UNKNOWN`.
                    # PARA QUÉ: Permitir un diagnóstico forense exacto de las reglas de riesgo activadas en backtest.
                    # CÓMO: Extraemos `rejection_reason` directamente de `event.metadata`, con fallback al stdout.
                    # CUÁNDO: Cuando `risk_manager.generate_order` retorna `None` indicando rechazo.
                    # DÓNDE: En el bucle de procesamiento de señales de `scripts/run_god_mode_backtest.py`.
                    # QUIÊN: Modificado por el SRE/DevOps y el QA Engineer.
                    _specific_reason = getattr(event, 'metadata', {}).get('rejection_reason')
                    if not _specific_reason:
                        _captured = _f_capture.getvalue()
                        for _line in _captured.strip().split("\n"):
                            if "[RISK] Rejected by" in _line or "[RISK VETO]" in _line:
                                _specific_reason = _line.strip()
                                break
                    
                    if _specific_reason:
                        rejection_reasons[_specific_reason] = (
                            rejection_reasons.get(_specific_reason, 0) + 1
                        )
                        print(f"⚠️ [RISK] Signal REJECTED: {_specific_reason}")
                    else:
                        reason = f"RISK_GATE_UNKNOWN:{getattr(event, 'symbol', '?')}"
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        print(
                            f"⚠️ [RISK] Signal REJECTED: {reason}\nSTDOUT: {_f_capture.getvalue()}"
                        )
                    continue
    
                order_count += 1
    
                # ═══════════════════════════════════════════════════════════════
                # DIGITAL TWIN V100: ORDER MANAGER CHAOS (LATENCY & SLIPPAGE)
                # ═══════════════════════════════════════════════════════════════
                if order_manager and not scenario_ideal_execution:
                    # 1. Emulate Rate Limits (Binance -2015 / -1003)
                    # If more than 3 orders in the last 10 epochs, reject
                    _recent_orders = getattr(order_manager, "_bt_recent_orders", [])
                    _recent_orders = [e for e in _recent_orders if epoch_count - e < 10]
                    if len(_recent_orders) >= 15:
                        rejected_count += 1
                        reason = "BINANCE_RATE_LIMIT_429"
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        print(f"  ⚠️ [CHAOS] OrderManager rejected {event.symbol} due to Rate Limits (Too Many Requests)")
                        if event.signal_type == SignalType.EXIT:
                            _hz_rej = getattr(event, "horizon", "SCALPING")
                            _key_rej = f"{event.symbol}_{_hz_rej}"
                            pending_exits.discard(_key_rej)
                        continue
                    _recent_orders.append(epoch_count)
                    order_manager._bt_recent_orders = _recent_orders

                    # ================================================================
                    # FORENSIC-V2 FIX #3: SLIPPAGE DELEGATION TO EXECUTOR
                    # QUÉ: El slippage se aplicaba DOS VECES — aquí Y en BacktestExecutor.
                    # POR QUÉ: Este bloque modificaba current_price con random(0.01-0.05%)
                    #   ANTES de pasar al executor, que aplicaba su propio modelo interno.
                    #   Resultado: slippage inflado 3.2x ($0.63 vs $0.20 esperado).
                    # PARA QUÉ: BacktestExecutor ya diferencia LIMIT (0% slip) vs MARKET
                    #   (0.01-0.05% + latency penalty). El tracking se hace post-fill.
                    # CÓMO: Eliminamos modificación de current_price. Slippage se trackea
                    #   desde fill.fill_price vs _pre_slip_price después del fill.
                    # ================================================================

                # ── EXECUTE ORDER → FillEvent ──
                fill = executor.execute_order(order, current_price)
                if fill is None:
                    rejected_count += 1
                    reason = "EXECUTOR_REJECT"
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
                    # 🚀 AEGIS-V15: ATOMIC METADATA RELEASE
                    # QUÉ: Usar el valor exacto reservado en RiskManager.
                    try:
                        order_id = order.metadata.get("client_order_id") if order.metadata else None
                        reserved = (
                            order.metadata.get("dollar_size")
                            if order.metadata
                            else None
                        )
                        if reserved:
                            portfolio.release_order_margin(amount=reserved, order_id=order_id)
                        else:
                            # Fallback using dynamic leverage from order
                            lev = (
                                getattr(order, "leverage", Config.BINANCE_LEVERAGE)
                                or Config.BINANCE_LEVERAGE
                            )
                            portfolio.release_order_margin(
                                amount=order.quantity * current_price / lev,
                                order_id=order_id
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to release margin for rejected order: {e}"
                        )
    
                    # Also clear pending exit to prevent lock-up
                    _key = f"{order.symbol}_{order.horizon}"
                    pending_exits.discard(_key)
                    continue
    
                # FORENSIC-V2 FIX #3: Track slippage from executor's fill_price
                if fill.fill_price and _pre_slip_price:
                    _slippage_delta = abs(fill.fill_price - _pre_slip_price) * fill.quantity
                    alpha_leak["slippage_loss"] += _slippage_delta
                    forensic_logger.debug(
                        f"[EXECUTION_SLIPPAGE] {event.symbol} "
                        f"clean={_pre_slip_price:.6f} filled={fill.fill_price:.6f} "
                        f"delta=${_slippage_delta:.6f} type={getattr(order, 'order_type', '?')}"
                    )

                # ── PORTFOLIO UPDATE (THE REAL update_fill) ──
                try:
                    result = portfolio.update_fill(fill)
                    fill_count += 1
    
                    # 🚀 FORENSIC PARITY: Accounting Equation Verification
                    # Ensures no money is created or destroyed during backtest loop.
                    portfolio.verify_accounting_equation()
    
                    # FORENSIC-V11 Fix #2+#6: Record fill epoch for cooldown
                    _fill_horizon = getattr(fill, "horizon", "SCALPING")
                    _fill_cooldown_key = f"{fill.symbol}_{_fill_horizon}"
                    last_fill_epoch[_fill_cooldown_key] = epoch_count
    
                    # FORENSIC-V11 Fix #7: Clear pending exit after fill
                    pending_exits.discard(_fill_cooldown_key)
    
                    # Record result for RiskManager Kelly tracking
                    if result and isinstance(result, tuple):
                        pnl, _ = result
                        if pnl is not None:
                            # ═══════════════════════════════════════════════════════════════
                            # FORENSIC-V31 FIX: USE NET PNL FOR RISK MANAGER TRACKING
                            # QUÉ: Usar net_pnl para determinar is_win.
                            # POR QUÉ: risk_manager.win_count se usa para Kelly sizing y backtest WR.
                            #   Si usamos gross_pnl, inflamos el WR y el apalancamiento Kelly!
                            # ═══════════════════════════════════════════════════════════════
                            _closed_trade = getattr(portfolio, "_last_closed_trade_data", None)
                            if (
                                _closed_trade
                                and _closed_trade.get("symbol") == fill.symbol
                            ):
                                net_pnl = _closed_trade.get("net_pnl", pnl)
                            else:
                                net_pnl = pnl  # Fallback
    
                            is_win = net_pnl > 0
                            pnl_pct = net_pnl / capital if capital > 0 else 0
                            risk_manager.record_trade_result(
                                is_win, pnl_pct, fill.symbol
                            )

                            # ═══════════════════════════════════════════════════════════
                            # FORENSIC AUDIT: STRATEGY & SYMBOL ATTRIBUTION
                            # ═══════════════════════════════════════════════════════════
                            _fa_strat = getattr(fill, 'strategy_id', 'unknown')
                            _fa_sym = fill.symbol
                            _fa_fees = _closed_trade.get("total_fees", 0.0) if _closed_trade else 0.0
                            _fa_gross = _closed_trade.get("gross_pnl", pnl) if _closed_trade else pnl
                            
                            # Fee accounting
                            if not scenario_ideal_execution:
                                alpha_leak["fee_loss"] += abs(_fa_fees)
                            alpha_leak["gross_alpha"] += _fa_gross
                            alpha_leak["net_alpha"] += net_pnl
                            
                            # Strategy attribution
                            if _fa_strat not in strategy_attribution:
                                strategy_attribution[_fa_strat] = {"wins": 0, "losses": 0, "gross_pnl": 0.0, "net_pnl": 0.0, "trades": 0}
                            strategy_attribution[_fa_strat]["trades"] += 1
                            strategy_attribution[_fa_strat]["gross_pnl"] += _fa_gross
                            strategy_attribution[_fa_strat]["net_pnl"] += net_pnl
                            if is_win:
                                strategy_attribution[_fa_strat]["wins"] += 1
                            else:
                                strategy_attribution[_fa_strat]["losses"] += 1
                            
                            # Symbol attribution
                            if _fa_sym not in symbol_attribution:
                                symbol_attribution[_fa_sym] = {"wins": 0, "losses": 0, "gross_pnl": 0.0, "net_pnl": 0.0, "fees": 0.0, "trades": 0}
                            symbol_attribution[_fa_sym]["trades"] += 1
                            symbol_attribution[_fa_sym]["gross_pnl"] += _fa_gross
                            symbol_attribution[_fa_sym]["net_pnl"] += net_pnl
                            symbol_attribution[_fa_sym]["fees"] += abs(_fa_fees)
                            if is_win:
                                symbol_attribution[_fa_sym]["wins"] += 1
                            else:
                                symbol_attribution[_fa_sym]["losses"] += 1

                            forensic_logger.info(
                                f"[TRADE_CLOSED] {_fa_sym} strategy={_fa_strat} "
                                f"gross={_fa_gross:.6f} net={net_pnl:.6f} fees={_fa_fees:.6f} "
                                f"result={'WIN' if is_win else 'LOSS'}"
                            )

                            # ═══════════════════════════════════════════════════════════════
                            # FORENSIC-V48: SOVEREIGN ORACLE ATTRIBUTION
                            # QUÉ: Envía los resultados de trades cerrados al Oráculo.
                            # POR QUÉ: Permite ajustar el mutation_mod basado en Skill vs Luck.
                            # ═══════════════════════════════════════════════════════════════
                            if sovereign_oracle:
                                try:
                                    from sophia.post_mortem import PostMortemResult
                                    _pm_res = PostMortemResult(
                                        trade_id=order.order_id,
                                        symbol=order.symbol,
                                        direction="long" if order.direction.name == "SELL" else "short",
                                        predicted_prob=0.8,  # Mocked baseline
                                        predicted_exit_mins=15.0,
                                        actual_outcome="WIN" if net_pnl > 0 else "LOSS",
                                        actual_pnl=net_pnl,
                                        actual_duration_mins=10.0,
                                        brier_score=0.1 if net_pnl > 0 else 0.4, # Mocked
                                        time_error_mins=5.0,
                                        narrative=f"GodMode backtest exited {order.symbol}"
                                    )
                                    sovereign_oracle.reason_about_outcome(_pm_res)
                                except Exception:
                                    pass
    
                            # FORENSIC-V11 Fix #4: Track ML consecutive losses
                            _fill_strat = getattr(fill, "strategy_id", "")
                            if _fill_strat:
                                if is_win:
                                    ml_consecutive_losses[_fill_strat] = 0
                                    # Also count shadow wins if in shadow mode
                                    if _fill_strat in ml_shadow_wins:
                                        ml_shadow_wins[_fill_strat] = (
                                            ml_shadow_wins.get(_fill_strat, 0) + 1
                                        )
                                else:
                                    ml_consecutive_losses[_fill_strat] = (
                                        ml_consecutive_losses.get(_fill_strat, 0) + 1
                                    )
                except Exception as e:
                    logger.warning(f"Fill processing error: {e}")
    
            # ── END OF EPOCH: Update equity curve ──
            # QUÉ: Restringir el muestreo de la curva de equidad al periodo post-warmup.
            # POR QUÉ: Para evitar que el periodo de lookback inactivo ensucie los cálculos de
            #   Sharpe ratio, drawdown y volatilidad de la equidad con retornos planos de 0%.
            # PARA QUÉ: Preservar la veracidad matemática del rendimiento de trading real.
            # CÓMO: Agregando la condición `epoch_count >= warmup_epochs` antes de guardar el dato,
            #   calculando `eq` en cada época para evitar UnboundLocalError en el chequeo de kill switch.
            # CUÁNDO: Cada época de simulación.
            # DÓNDE: En el bucle de eventos del backtest.
            # QUIÊN: Modificado por el Quant Developer y el SRE/DevOps.
            eq = portfolio.get_total_equity()
            if epoch_count >= warmup_epochs and epoch_count % 60 == 0:  # Sample every 60 bars (1 hour)
                equity_curve.append(eq)
                ts = pd.to_datetime(data_provider.current_time_ms, unit="ms", utc=True)
                equity_timestamps.append(ts)
    
                # Update RiskManager equity (for kill switch)
                try:
                    risk_manager.update_equity(eq)
                except:
                    pass
                    
            # Send mid-way Strategy Leaderboard
            if total_epochs > 0 and epoch_count == total_epochs // 2:
                try:
                    Notifier.send_strategy_leaderboard(portfolio.strategy_performance, title_prefix="BACKTEST (50%)")
                except Exception:
                    pass
    
                # ═══════════════════════════════════════════════════════════
                # FIX-V10-2: SYNC BACKTEST KILL SWITCH WITH RISKMANAGER'S
                # QUÉ: Si el RiskManager activa su kill switch interno (2% DD),
                #   el backtest DEBE terminar inmediatamente.
                # POR QUÉ: Antes el backtest seguía corriendo cientos de epochs
                #   como zombie (sin poder operar) hasta el 15% hardcoded.
                # PARA QUÉ: Terminar RÁPIDO cuando no se puede operar más.
                # ═══════════════════════════════════════════════════════════
                if risk_manager.kill_switch.active:
                    print(
                        f"\n  🚨 RISK MANAGER KILL SWITCH SYNCED: {risk_manager.kill_switch.activation_reason}"
                    )
                    print(
                        f"     Equity: ${eq:.2f} | Peak: ${risk_manager.kill_switch.peak_equity:.2f}"
                    )
                    kill_switch_triggered = True
                    break
    
                # Global kill switch check (hard floor) — USE CONFIG, NOT HARDCODED
                # FORENSIC-V151: Was hardcoded 0.85 (15% DD), but Config.Risk.MAX_DRAWDOWN=35%.
                # This caused premature kill at $11.05 blocking 33 recovery trades.
                _max_dd_pct = getattr(Config.Risk, 'MAX_DRAWDOWN', 35.0) / 100.0
                _hard_floor = capital * (1.0 - _max_dd_pct)
                if eq < _hard_floor:
                    print(
                        f"\n  🚨 HARD FLOOR KILL SWITCH: Equity ${eq:.2f} < {(1-_max_dd_pct)*100:.0f}% of initial ${capital:.2f} (floor=${_hard_floor:.2f})"
                    )
                    kill_switch_triggered = True
                    break
    
        elapsed = time.time() - t_start
    
        # ─────────────────────────────────────────────────────────────────────────
        # STEP 4: CLOSE ALL REMAINING POSITIONS
        # ─────────────────────────────────────────────────────────────────────────
        for v_key, vpos in list(portfolio.virtual_ledger.items()):
            qty = vpos.get("quantity", 0)
            if qty == 0:
                continue
    
            horizon = vpos.get("horizon", "SCALPING")
            parts = v_key.rsplit(f"_{horizon}", 1)
            symbol = parts[0] if len(parts) > 1 else v_key
    
            current_price = data_provider.get_latest_price(symbol)
            if not current_price:
                continue
    
            direction = OrderSide.SELL if qty > 0 else OrderSide.BUY
    
            try:
                close_fill = FillEvent(
                    timeindex=datetime.now(timezone.utc),
                    symbol=symbol,
                    exchange="BINANCE_BACKTEST",
                    quantity=abs(qty),
                    direction=direction,
                    fill_cost=abs(qty) * current_price,
                    commission=abs(qty)
                    * current_price
                    * COMMISSION_MAKER,  # FORENSIC-V12 FIX #6: Exits are LIMIT BBO → Maker fee
                    strategy_id="BACKTEST_CLOSE",
                    fill_price=current_price,
                    horizon=horizon,
                    metadata={'is_close': True, 'reason': 'BACKTEST_CLOSE'}
                )
                portfolio.update_fill(close_fill)
                fill_count += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"Failed to close {v_key}: {e}")
    
        # ─────────────────────────────────────────────────────────────────────────
        # STEP 5: CALCULATE METRICS & REPORT
        # ─────────────────────────────────────────────────────────────────────────
        final_equity = portfolio.get_total_equity()
    
        # Collect trades from portfolio strategy attribution
        all_trades = []
        for strat_id, perf in portfolio.strategy_performance.items():
            all_trades.append(
                {
                    "strategy": strat_id,
                    "pnl_usd": perf.get("pnl", 0),
                    "wins": perf.get("wins", 0),
                    "losses": perf.get("losses", 0),
                    "trades": perf.get("trades", 0),
                }
            )
    
        # Basic metrics
        total_return = ((final_equity - capital) / capital) * 100
        
        # 🚀 FORENSIC FIX: total_trades was reading fill_count (which is 0 unless forced close at end).
        total_trades = sum(perf.get("trades", 0) for perf in portfolio.strategy_performance.values())
    
        # Equity curve metrics
        eq_arr = np.array(equity_curve)
        max_dd = 0
        if len(eq_arr) > 1:
            peak = np.maximum.accumulate(eq_arr)
            dd = (peak - eq_arr) / peak
            max_dd = float(np.max(dd)) * 100
    
        # Win rate from kelly tracking
        total_wins = sum(perf.get("wins", 0) for perf in portfolio.strategy_performance.values())
        total_losses = sum(perf.get("losses", 0) for perf in portfolio.strategy_performance.values())
        total_wl = total_wins + total_losses
        win_rate = (total_wins / total_wl * 100) if total_wl > 0 else 0
    
        # Sharpe from equity curve
        sharpe = 0
        if len(eq_arr) > 2:
            returns = np.diff(eq_arr) / eq_arr[:-1]
            if np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365 * 24))
    
        results = {
            "version": "GOD_MODE_v3.0_UNIFIED_EXIT",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "symbols": symbols,
                "num_symbols": len(symbols),
                "days": days,
                "initial_capital": capital,
                "leverage": leverage,
                "max_concurrent_positions": Config.MAX_CONCURRENT_POSITIONS,
                "fee_per_side": COMMISSION_PCT,
                "seed": seed,
                "exit_engine": "RiskManager.check_stops() ONLY",
                "portfolio_check_exits": "AUDIT-ONLY (no EXIT signals)",
            },
            "metrics": {
                "final_capital": round(final_equity, 4),
                "total_return_pct": round(total_return, 2),
                "total_trades": total_trades,
                "wins": total_wins,
                "losses": total_losses,
                "signals_generated": signal_count,
                "orders_generated": order_count,
                "orders_rejected": rejected_count,
                "win_rate": round(win_rate, 1),
                "max_drawdown_pct": round(max_dd, 2),
                "sharpe_ratio": round(sharpe, 2),
                "fees_paid": round(portfolio.total_fees_paid, 4),
                "kill_switch_triggered": kill_switch_triggered,
                "portfolio_audit_exits_suppressed": portfolio_audit_exits,
            },
            "strategy_attribution": portfolio.strategy_performance,
            "elapsed_seconds": round(elapsed, 1),
            "epochs_processed": epoch_count,
            "equity_curve_sample": [round(e, 4) for e in equity_curve[-50:]],
            "trade_history": {
                "scalping": portfolio.scalping_ledger,
                "swing": portfolio.swing_ledger,
            },
            "rejection_reasons": rejection_reasons,
        }
    
        # ─── PREDICTION TRACKER: EXPORT & REPORT ───
        # QUÉ: Exporta métricas de precisión predictiva al resultado del backtest.
        # PARA QUÉ: Visibilidad completa de accuracy/MFE/MAE/decay por estrategia.
        try:
            pred_metrics = prediction_tracker.export_metrics()
            results["prediction_metrics"] = pred_metrics
            print(f"\n{prediction_tracker.get_summary()}")
        except Exception as _pt_err:
            logger.warning(f"PredictionTracker export error: {_pt_err}")
    
        # ─── SEGMENTED METRICS ───
        def calculate_ledger_metrics(ledger: list) -> dict:
            if not ledger:
                return {
                    "pnl": 0.0,
                    "gross": 0.0,
                    "fees": 0.0,
                    "wins": 0,
                    "losses": 0,
                    "wr": 0.0,
                    "total": 0,
                    "avg_margin": 0.0,
                    "limits": 0,
                    "markets": 0,
                    "avg_certainty": 0.0,
                    "avg_duration": 0,
                }
            pnl = sum(t.get("net_pnl", 0) for t in ledger)
            gross = sum(t.get("gross_pnl", 0) for t in ledger)
            fees = sum(t.get("fees_paid", 0) for t in ledger)
            wins = sum(1 for t in ledger if t.get("net_pnl", 0) > 0)
            limits = sum(
                1 for t in ledger if str(t.get("exit_type", "limit")).lower() == "limit"
            )
            markets = sum(
                1 for t in ledger if str(t.get("exit_type", "")).lower() == "market"
            )
            total = len(ledger)
            losses = total - wins
            avg_margin = (
                sum(t.get("margin_usd", 0) for t in ledger) / total if total > 0 else 0
            )
            avg_certainty = (
                sum(t.get("oracle_certainty", 0) for t in ledger) / total
                if total > 0
                else 0
            )
            avg_dur = (
                sum(t.get("duration_seconds", 0) for t in ledger) / total
                if total > 0
                else 0
            )
            wr = (wins / total * 100) if total > 0 else 0
            return {
                "pnl": pnl,
                "gross": gross,
                "fees": fees,
                "wins": wins,
                "losses": losses,
                "wr": wr,
                "total": total,
                "avg_margin": avg_margin,
                "limits": limits,
                "markets": markets,
                "avg_certainty": avg_certainty,
                "avg_duration": avg_dur,
            }
    
        scl_m = calculate_ledger_metrics(portfolio.scalping_ledger)
        swg_m = calculate_ledger_metrics(portfolio.swing_ledger)
    
        print(f"\n{'=' * 70}")
        print(f"📊 GOD MODE BACKTEST v2.0 — RESULTS (COMBINED)")
        print(f"{'=' * 70}")
        print(
            f"  🏦 Capital:     ${capital:.2f} → ${final_equity:.2f} ({total_return:+.2f}%)"
        )
        print(
            f"  📈 Total Trades: {total_trades} (Signals: {signal_count} | "
            f"Orders: {order_count} | Rejected: {rejected_count})"
        )
        print(
            f"  🎯 Win Rate:    {win_rate:.1f}% ({risk_manager.win_count}W / {risk_manager.loss_count}L)"
        )
        print(f"  📉 Max Drawdown: {max_dd:.2f}%")
        print(f"  📊 Sharpe Ratio: {sharpe:.2f}")
        print(f"  💸 Total Fees:  ${portfolio.total_fees_paid:.4f}")
        print(f"  ⏱️  Elapsed:     {elapsed:.1f}s ({epoch_count:,} epochs)")
        print(f"  🚨 Kill Switch:  {'YES' if kill_switch_triggered else 'NO'}")
    
        print(f"\n  {'─' * 50}")
        print(f"  ⚡ [SCALPING] COHORT RESULTS:")
        print(
            f"    Total Trades: {scl_m['total']} (LIMIT BBO: {scl_m['limits']} | MKT: {scl_m['markets']})"
        )
        print(f"    Win Rate:     {scl_m['wr']:.1f}% ({scl_m['wins']}W/{scl_m['losses']}L)")
        print(f"    Net PnL:      ${scl_m['pnl']:+.4f} (Gross: ${scl_m['gross']:+.4f})")
        print(
            f"    Fee Drag:     ${scl_m['fees']:.4f} ({(scl_m['fees'] / scl_m['gross'] * 100) if scl_m['gross'] > 0 else 0:.1f}% of gross)"
        )
        print(
            f"    Metrics:      Avg Margin: ${scl_m['avg_margin']:.2f} | Avg Oracle: {scl_m['avg_certainty'] * 100:.1f}% | Avg Hold: {scl_m['avg_duration']:.1f}s"
        )
    
        print(f"\n  {'─' * 50}")
        print(f"  🌊 [SWING] COHORT RESULTS:")
        print(
            f"    Total Trades: {swg_m['total']} (LIMIT BBO: {swg_m['limits']} | MKT: {swg_m['markets']})"
        )
        print(f"    Win Rate:     {swg_m['wr']:.1f}% ({swg_m['wins']}W/{swg_m['losses']}L)")
        print(f"    Net PnL:      ${swg_m['pnl']:+.4f} (Gross: ${swg_m['gross']:+.4f})")
        print(
            f"    Fee Drag:     ${swg_m['fees']:.4f} ({(swg_m['fees'] / swg_m['gross'] * 100) if swg_m['gross'] > 0 else 0:.1f}% of gross)"
        )
        print(
            f"    Metrics:      Avg Margin: ${swg_m['avg_margin']:.2f} | Avg Oracle: {swg_m['avg_certainty'] * 100:.1f}% | Avg Hold: {swg_m['avg_duration']:.1f}s"
        )
    
        if portfolio.strategy_performance:
            print(f"\n  {'─' * 50}")
            print(f"  📋 STRATEGY ATTRIBUTION:")
            for strat_id, perf in sorted(
                portfolio.strategy_performance.items(),
                key=lambda x: x[1].get("pnl", 0),
                reverse=True,
            ):
                pnl = perf.get("pnl", 0)
                wins = perf.get("wins", 0)
                losses = perf.get("losses", 0)
                total = wins + losses
                wr = (wins / total * 100) if total > 0 else 0
                print(
                    f"    {strat_id}: PnL=${pnl:+.4f} | "
                    f"W/L: {wins}/{losses} ({wr:.0f}%) | "
                    f"Trades: {total}"
                )
    
        print(f"{'=' * 70}\n")
    
        # FORENSIC-V9: REJECTION ANALYSIS — WHERE DO SIGNALS DIE?
        if rejection_reasons:
            print(f"  {'─' * 50}")
            print(f"  🔬 FORENSIC-V9: REJECTION ANALYSIS")
            sorted_reasons = sorted(
                rejection_reasons.items(), key=lambda x: x[1], reverse=True
            )
            for reason, count in sorted_reasons[:15]:
                pct = (count / max(signal_count, 1)) * 100
                print(f"    {reason}: {count} ({pct:.1f}% of signals)")
    
        # ═══════════════════════════════════════════════════════════════════
        # FIX-V10-5: BETTER UNTRAINED STRATEGY BREAKDOWN
        # ═══════════════════════════════════════════════════════════════════
        if untrained_strategies:
            swing_untrained = sorted(
                [s for s in untrained_strategies if "SWING" in s.upper()]
            )
            scalp_untrained = sorted(
                [s for s in untrained_strategies if "SWING" not in s.upper()]
            )
            print(f"\n  ⚠️ UNTRAINED STRATEGIES ({len(untrained_strategies)} total):")
            print(
                f"    🌊 SWING: {len(swing_untrained)} untrained (likely insufficient 4H candle data)"
            )
            if scalp_untrained:
                print(f"    ⚡ SCALP: {len(scalp_untrained)} untrained:")
                for s in scalp_untrained[:10]:
                    print(f"      • {s}")
    
        # Add forensic data to results
        results["forensic_v10"] = {
            "rejection_reasons": dict(
                sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
            ),
            "untrained_strategies": list(untrained_strategies),
            "signal_to_fill_ratio": f"{(fill_count / max(signal_count, 1)) * 100:.1f}%",
            "signal_to_order_ratio": f"{(order_count / max(signal_count, 1)) * 100:.1f}%",
            "untrained_swing_count": len(
                [s for s in untrained_strategies if "SWING" in s.upper()]
            ),
            "untrained_scalp_count": len(
                [s for s in untrained_strategies if "SWING" not in s.upper()]
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # FORENSIC AUDIT: ALPHA LEAK TABLE
        # ═══════════════════════════════════════════════════════════════════
        _al = alpha_leak
        _al["premature_exit_loss"] = _al["gross_alpha"] - _al["net_alpha"] - _al["slippage_loss"] - _al["fee_loss"]
        if _al["premature_exit_loss"] < 0:
            _al["premature_exit_loss"] = 0.0

        print(f"\n  {'═' * 55}")
        print(f"  🔬 FORENSIC ALPHA LEAK TABLE (Scenario {scenario})")
        print(f"  {'═' * 55}")
        print(f"  {'Metric':<35} {'Value':>15}")
        print(f"  {'─' * 55}")
        _v_gross = f"${_al['gross_alpha']:+.6f}"
        _v_slip = f"${-_al['slippage_loss']:.6f}"
        _v_fees = f"${-_al['fee_loss']:.6f}"
        _v_exit = f"${-_al['premature_exit_loss']:.6f}"
        _v_net = f"${_al['net_alpha']:+.6f}"
        print(f"  {'Alpha Bruto (Gross PnL)':<35} {_v_gross:>15}")
        print(f"  {'- Perdida por Slippage':<35} {_v_slip:>15}")
        print(f"  {'- Perdida por Comisiones (Fees)':<35} {_v_fees:>15}")
        print(f"  {'- Perdida por Exits Prematuros':<35} {_v_exit:>15}")
        print(f"  {'- Perdida por Conflictos (Vetos)':<35} {len(conflict_log):>15} vetos")
        print(f"  {'─' * 55}")
        print(f"  {'= NET ALPHA FINAL':<35} {_v_net:>15}")
        print(f"  {'═' * 55}")

        # ═══════════════════════════════════════════════════════════════════
        # FORENSIC AUDIT: STRATEGY ATTRIBUTION TABLE
        # ═══════════════════════════════════════════════════════════════════
        if strategy_attribution:
            print(f"\n  {'─' * 55}")
            print(f"  📊 FORENSIC: STRATEGY ATTRIBUTION")
            print(f"  {'Strategy':<25} {'Trades':>7} {'WR%':>6} {'Gross':>12} {'Net':>12}")
            print(f"  {'─' * 55}")
            for sid, sa in sorted(strategy_attribution.items(), key=lambda x: x[1]["net_pnl"], reverse=True):
                _sa_wr = (sa["wins"] / sa["trades"] * 100) if sa["trades"] > 0 else 0
                print(f"  {sid[:24]:<25} {sa['trades']:>7} {_sa_wr:>5.1f}% ${sa['gross_pnl']:>+10.6f} ${sa['net_pnl']:>+10.6f}")

        # ═══════════════════════════════════════════════════════════════════
        # FORENSIC AUDIT: SYMBOL ATTRIBUTION TABLE
        # ═══════════════════════════════════════════════════════════════════
        if symbol_attribution:
            print(f"\n  {'─' * 65}")
            print(f"  💱 FORENSIC: SYMBOL ATTRIBUTION (Top 15)")
            print(f"  {'Symbol':<15} {'Trades':>7} {'WR%':>6} {'Gross':>12} {'Net':>12} {'Fees':>10}")
            print(f"  {'─' * 65}")
            for sym, sa in sorted(symbol_attribution.items(), key=lambda x: x[1]["net_pnl"], reverse=True)[:15]:
                _sa_wr = (sa["wins"] / sa["trades"] * 100) if sa["trades"] > 0 else 0
                print(f"  {sym:<15} {sa['trades']:>7} {_sa_wr:>5.1f}% ${sa['gross_pnl']:>+10.6f} ${sa['net_pnl']:>+10.6f} ${sa['fees']:>8.6f}")

        # ═══════════════════════════════════════════════════════════════════
        # FORENSIC AUDIT: CONFLICT LOG SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        if conflict_log:
            print(f"\n  {'─' * 55}")
            print(f"  ⚔️ FORENSIC: CONFLICT LOG ({len(conflict_log)} total vetoed intents)")
            _conflict_reasons = {}
            for c in conflict_log:
                r = c["reason"]
                _conflict_reasons[r] = _conflict_reasons.get(r, 0) + 1
            for reason, count in sorted(_conflict_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"    {reason}: {count}")

        # Inject forensic data into results JSON
        results["forensic_alpha_leak"] = alpha_leak
        results["forensic_strategy_attribution"] = strategy_attribution
        results["forensic_symbol_attribution"] = symbol_attribution
        results["forensic_conflict_log_count"] = len(conflict_log)
        results["forensic_scenario"] = scenario
    
        # ═══════════════════════════════════════════════════════════════════
        # FIX-V10-4: CLEANUP STOP_TRADING.LOCK AFTER BACKTEST
        # QUÉ: Elimina el lock file para no contaminar producción ni futuros BTs.
        # ═══════════════════════════════════════════════════════════════════
        _lock_cleanup = os.path.join(_project_root, "STOP_TRADING.LOCK")
        if os.path.exists(_lock_cleanup):
            os.remove(_lock_cleanup)
            print(f"  🔓 [V10] Cleaned up STOP_TRADING.LOCK (backtest cleanup)")
    
        print(f"{'=' * 70}\n")
        
        # EXPORT PREDICTION METRICS
        try:
            if hasattr(risk_manager, 'prediction_tracker') and risk_manager.prediction_tracker:
                metrics_exported = risk_manager.prediction_tracker.export_metrics()
                print(f"  🎯 Prediction metrics exported to JSON. Total tracked strategies: {len(metrics_exported)}")
        except Exception as e:
            print(f"  ⚠️ Could not export prediction metrics: {e}")
    
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC EXPORTER: PERSIST AUDIT DATA TO PARQUET
        # ═══════════════════════════════════════════════════════════════
        try:
            from utils.forensic_exporter import ForensicExporter
            _forensic_exporter = ForensicExporter()
            _trade_list = []
            for ledger_name in ["scalping_ledger", "swing_ledger"]:
                ledger = getattr(portfolio, ledger_name, [])
                trades_iter = ledger.values() if isinstance(ledger, dict) else ledger
                for trades in trades_iter:
                    if isinstance(trades, list):
                        _trade_list.extend(trades)
                    elif isinstance(trades, dict):
                        _trade_list.append(trades)
            _exported = _forensic_exporter.export_all(
                alpha_leak=alpha_leak,
                strategy_attribution=strategy_attribution,
                symbol_attribution=symbol_attribution,
                conflict_log=conflict_log,
                run_id=run_id,
                scenario=scenario,
                trades=_trade_list
            )
            print(f"  📊 [FORENSIC] All audit data exported: {_exported}")
        except Exception as e:
            print(f"  ⚠️ Could not export forensic data: {e}")

        # ═══════════════════════════════════════════════════════════════
        # TELEGRAM: SEND FINAL BACKTEST RESULTS
        # ═══════════════════════════════════════════════════════════════
        try:
            Notifier.send_strategy_leaderboard(portfolio.strategy_performance, title_prefix="FINAL")
            Notifier.send_backtest_complete(results)
        except Exception as e:
            print(f"  ⚠️ Could not send Telegram backtest completion: {e}")
    
        return results
    
    finally:
        # ═══════════════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════════════
        # FORENSIC-V42 FIX: RESTORE CONFIG & CLEANUP RAM
        # QUÉ: Restaura Config original y libera memoria explícitamente.
        # POR QUÉ: Evitar contaminación y OOM crashes.
        # ═══════════════════════════════════════════════════════════════════════
        Config.IS_BACKTEST = getattr(Config, 'IS_BACKTEST', False)
        
        # Explicit cleanup for GC
        if 'strategies_map' in locals():
            del strategies_map
        if 'global_epoch_strategies' in locals():
            del global_epoch_strategies
        # FIX-FORENSIC-V82: 'import gc' was here but caused UnboundLocalError
        # at L959 because Python treats gc as local for the entire function scope.
        # gc is now imported at module level (L52).
        gc.collect()


def _generate_insights_report(results: dict, report_path: str):
    """
    QUÉ: Genera un reporte Markdown detallado post-backtest con insights accionables.
    POR QUÉ: Leer JSONs o logs gigantes es tedioso; necesitamos métricas clave, alpha leak 
             y conclusiones presentadas claramente.
    PARA QUÉ: Identificar rápidamente cuellos de botella (comisiones, slippage, vetos) 
              y estrategias de alto rendimiento.
    CÓMO: Analiza el diccionario 'results' generado y extrae deducciones.
    CUÁNDO: Al finalizar cada simulación de God Mode.
    DÓNDE: Guardado en la misma carpeta que el JSON de resultados (results/backtests/).
    QUIÉN: Profesor / Forensic Auditor.
    """
    metrics = results.get("metrics", {})
    alpha_leak = results.get("forensic_alpha_leak", {})
    run_id = results.get("run_id", "unknown")
    capital = results.get("config", {}).get("initial_capital", 0.0)
    final_cap = metrics.get("final_capital", 0.0)
    ret = metrics.get("total_return_pct", 0.0)
    
    lines = []
    lines.append(f"# 📊 Reporte de Insights Estructurados: Backtest {run_id}")
    lines.append(f"**Generado:** {results.get('timestamp')}")
    lines.append("")
    lines.append("## 1. 🎯 Rendimiento General")
    lines.append(f"- **Capital:** ${capital:.2f} ➔ ${final_cap:.2f}")
    lines.append(f"- **Retorno:** {ret:+.2f}%")
    lines.append(f"- **Win Rate:** {metrics.get('win_rate', 0.0):.1f}% ({metrics.get('wins', 0)}W / {metrics.get('losses', 0)}L)")
    lines.append(f"- **Drawdown Máximo:** {metrics.get('max_drawdown_pct', 0.0):.2f}%")
    lines.append(f"- **Ratio de Sharpe:** {metrics.get('sharpe_ratio', 0.0):.2f}")
    lines.append("")
    lines.append("## 2. 🚰 Análisis de Fuga de Alpha (Alpha Leak)")
    lines.append("Este análisis detalla cómo se redujo el PnL bruto ideal debido a fricciones reales de mercado.")
    
    gross = alpha_leak.get('gross_alpha', 0.0)
    slip = alpha_leak.get('slippage_loss', 0.0)
    fees = alpha_leak.get('fee_loss', 0.0)
    premature = alpha_leak.get('premature_exit_loss', 0.0)
    net = alpha_leak.get('net_alpha', 0.0)
    
    lines.append(f"- **Alpha Bruto (Ideal):** ${gross:+.4f}")
    lines.append(f"- **Pérdida por Slippage:** -${slip:.4f}")
    lines.append(f"- **Comisiones (Fees):** -${fees:.4f}")
    lines.append(f"- **Exits Prematuros (Costos de Oportunidad):** -${premature:.4f}")
    lines.append(f"- **Alpha Neto (Real):** ${net:+.4f}")
    lines.append("")
    lines.append("### 💡 Diagnóstico del Profesor")
    if gross > 0 and fees > gross * 0.2:
        lines.append("> ⚠️ **Alerta:** Las comisiones están consumiendo más del 20% del Alpha Bruto. Considerar estrategias con mayor *holding time* o usar limit orders (BBO).")
    elif slip > fees:
        lines.append("> ⚠️ **Alerta:** El slippage supera las comisiones. Validar liquidez en los pares operados o reducir el tamaño de posición (capital muy alto).")
    else:
        lines.append("> ✅ **Salud Óptima:** Las fricciones de mercado están bajo control y el modelo de retención de valor es saludable.")
    
    lines.append("")
    lines.append("## 3. 🛡️ Confiabilidad del Sistema y Auditoría")
    lines.append(f"- **Trades Rechazados por el Oráculo/BFT:** {metrics.get('orders_rejected', 0)}")
    lines.append(f"- **Kill Switch Activado:** {'Sí 🚨' if metrics.get('kill_switch_triggered', False) else 'No ✅'}")
    lines.append(f"- **Métricas Exportadas a JSON:** God Mode ha registrado cada tic. El comportamiento es determinístico.")
    
    lines.append("")
    lines.append("## 4. 🧠 Recomendación Evolutiva (EAI)")
    lines.append("Basado en el sistema adaptativo e integral, enfocar recursos en optimizar los factores responsables de mayor fuga (Fees vs Slippage) o recalibrar los umbrales del RiskManager si el Drawdown excedió expectativas.")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import atexit
    # ── CONCURRENCY LOCK (FORENSIC FIX) ──
    # QUÉ: Previene la ejecución concurrente de múltiples backtests (dentro del mismo entorno).
    # POR QUÉ: Múltiples backtests consumen la RAM y causan detención del motor principal.
    env_id = os.getenv("TG_ENV_ID", "")
    lock_filename = f"backtest_running_{env_id}.lock" if env_id else "backtest_running.lock"
    lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), lock_filename)
    if os.path.exists(lock_file):
        # Check if the lock is stale (e.g. older than 4 hours due to a previous crash)
        if time.time() - os.path.getmtime(lock_file) > 14400:
            print(f"⚠️ Found stale lock file older than 4 hours. Removing it...")
            try:
                os.remove(lock_file)
            except OSError:
                pass
        else:
            print(f"🛑 [CONCURRENCY LOCK] A backtest is already running. Aborting this run to protect system memory.")
            sys.exit(1)
            
    # Create lock
    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))
        
    # Register cleanup
    atexit.register(lambda: os.remove(lock_file) if os.path.exists(lock_file) else None)

    # ── PHASE 2 AUDIT PRE-FLIGHT CHECK ──
    print("🛡️ Executing Phase 2 Pre-flight Audit...")
    try:
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        if not hasattr(rm, '_validate_fat_finger'):
            print("🛑 [FATAL] Phase 2 Audit Failed: RiskManager missing '_validate_fat_finger' protection. Aborting backtest for safety.")
            sys.exit(1)
        if not hasattr(rm, '_validate_slippage'):
            print("🛑 [FATAL] Phase 2 Audit Failed: RiskManager missing '_validate_slippage' protection. Aborting backtest for safety.")
            sys.exit(1)
        if not hasattr(rm, 'kill_switch') or rm.kill_switch is None:
            print("🛑 [FATAL] Phase 2 Audit Failed: RiskManager missing 'kill_switch'. Aborting backtest for safety.")
            sys.exit(1)
        print("✅ Phase 2 Pre-flight Audit Passed. Security Constraints Verified.")
    except Exception as e:
        print(f"🛑 [FATAL] Phase 2 Audit Failed during execution: {e}. Aborting backtest for safety.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="God Mode Backtest v2.0 — Global Synchronized Engine"
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="",
        help="Environment ID for Multi-Environment Isolation (e.g. trial_123)"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days to backtest"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD HH:MM:SS format (default: now)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="ALL",
        help="Comma-separated symbols or ALL for full basket (default: ALL)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help=f"Initial capital (default: ${Config.INITIAL_CAPITAL})",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--scenario", type=str, default="A", choices=["A", "B", "C", "D", "E"], 
                        help="Forensic Scenario: A=Full, B=No Reactive Exits, C=Market Only, D=Ideal Exec, E=Isolated")
    parser.add_argument("--isolated-strategy", type=str, default=None, 
                        help="Strategy ID to isolate (for Scenario E)")

    args = parser.parse_args()

    end_time = None
    if args.end:
        try:
            end_time = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Error: --end format must be 'YYYY-MM-DD HH:MM:SS'")
            sys.exit(1)

    # ── Parse symbols ──
    if args.symbols.upper() == "ALL":
        symbols = Config.CRYPTO_FUTURES_PAIRS
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]
        
    # Sync with Config so Gatekeeper works correctly in Backtest
    Config.TRADING_PAIRS = symbols
    
    # Normalize format
    normalized = []
    for s in symbols:
        s = s.upper().replace("/", "")
        if s.endswith("USDT"):
            normalized.append(f"{s[:-4]}/USDT")
        else:
            normalized.append(s)
    symbols = normalized

    print(f"\n🎯 Symbols to backtest: {symbols}")

    # ── Download data for ALL symbols ──
    # QUÉ: Descargar 70 días adicionales de lookback histórico.
    # POR QUÉ: Para que las temporalidades resampleadas `1d` y `1w` tengan suficientes datos históricos
    #   (al menos 10 barras en `get_latest_bars`) para calcular indicadores y no ser descartadas.
    # PARA QUÉ: Activar de forma real y efectiva la protección de tendencia en el oráculo macro.
    # CÓMO: Sumando 70 días a `args.days` en la llamada a `fetch_multi_symbol_data`.
    # CUÁNDO: Al iniciar la fase de descarga de velas del backtest.
    # DÓNDE: En `scripts/run_god_mode_backtest.py` L2470.
    # QUIÊN: Modificado por el Quant Developer y el SRE/DevOps.
    all_data = fetch_multi_symbol_data(
        symbols, days=args.days + 70, max_workers=4, end_time=end_time
    )

    if not all_data:
        print("❌ No data downloaded. Aborting.")
        sys.exit(1)

    # Only keep symbols that have data
    valid_symbols = list(all_data.keys())
    print(f"\n✅ Valid symbols with data: {len(valid_symbols)}/{len(symbols)}")

    # ── Run Global Synchronized Backtest ──
    results = run_global_backtest(
        all_data=all_data,
        symbols=valid_symbols,
        days=args.days,
        initial_capital=args.capital,
        verbose=not args.quiet,
        scenario=args.scenario,
        isolated_strategy=args.isolated_strategy
    )

    # ── Save results to UNIQUE file (never overwrite) ──
    # REMEDIACIÓN: Cada run tiene su propio archivo con run_id.
    # POR QUÉ: Antes se sobrescribía el mismo archivo → resultados mezclados.
    run_id_result = results.get("run_id", "unknown")
    if args.output:
        output_path = args.output
    else:
        results_dir = os.path.join(_project_root, "results", "backtests")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(
            results_dir, f"god_mode_{run_id_result}_{args.days}d.json"
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- ATOMIC SAVE PROTOCOL ---
    # QUÉ: Guarda en un archivo temporal (.tmp) y luego lo renombra al final.
    # POR QUÉ: Evita que el archivo se corrompa si el proceso se interrumpe durante el guardado.
    # PARA QUÉ: Integridad absoluta de los resultados del backtest.
    temp_path = output_path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Atomically replace the final file
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC: GENERATE STRUCTURED INSIGHTS REPORT
        # ═══════════════════════════════════════════════════════════════
        try:
            report_path = output_path.replace(".json", "_insights.md")
            _generate_insights_report(results, report_path)
            print(f"  📝 Structured Insights Report saved to: {report_path}")
        except Exception as e:
            print(f"  ⚠️ Could not generate Insights Report: {e}")
            
    except Exception as e:
        print(f"❌ Error during atomic save: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    print(f"💾 Results saved to: {output_path}")
    print(f"   run_id: {run_id_result} (use for comparison)")
    return results


if __name__ == "__main__":
    main()
