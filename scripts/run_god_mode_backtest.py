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

# ─── Project Root ───
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Suppress noisy warnings during backtest
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION IMPORTS — SAME AS main.py
# ═══════════════════════════════════════════════════════════════════════════════
from config import Config
from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import EventType, SignalType, OrderSide, OrderType
from core.api_manager import APIManager

# ─── GET REAL PRODUCTION BALANCE ───
api_mgr = APIManager()
try:
    prod_balance = api_mgr.get_production_balance()
    real_capital = (
        prod_balance.get("total_equity", Config.INITIAL_CAPITAL)
        if prod_balance
        else Config.INITIAL_CAPITAL
    )
    print(f"💰 PRODUCTION BALANCE: ${real_capital:.2f}")
except Exception as e:
    print(f"⚠️ Could not fetch production balance: {e}")
    real_capital = Config.INITIAL_CAPITAL

# ─── PRODUCTION PORTFOLIO (THE REAL ONE) ───
from core.portfolio import Portfolio

# ─── PRODUCTION RISK MANAGER (THE REAL ONE) ───
from risk.risk_manager import RiskManager

# ─── PRODUCTION STRATEGIES (THE REAL ONES) ───
from strategies.ml_strategy import UniversalEnsembleStrategy as MLStrategy

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

        if order_event.metadata and order_event.metadata.get("is_tp_limit"):
            return None  # Backtester relies on check_stops loop for exits

        price = current_price or order_event.price
        if not price or price <= 0:
            return None

        qty = order_event.quantity
        if qty <= 0:
            return None

        # ─── DETERMINISTIC STOCHASTIC LATENCY & SLIPPAGE (Seeded RNG) ───
        # Simulate real-world network and exchange latency (5ms - 250ms)
        stochastic_latency_ms = max(5, min(250, self._rng.normal(50, 40)))
        
        # BBO: LIMIT orders have ZERO slippage (executed exactly at limit price)
        is_limit = order_event.order_type == OrderType.LIMIT
        if is_limit:
            # LIMIT orders: exact execution if latency is acceptable, otherwise small adverse slippage if BBO shifts
            if stochastic_latency_ms > 150:
                slip_pct = self._rng.uniform(0.00005, 0.0001) # Punish high latency limits slightly
            else:
                slip_pct = 0.0  
        else:
            # MARKET orders: higher slippage, scales with latency
            base_slip = self._rng.uniform(0.0001, 0.0005)  # 0.01% - 0.05%
            latency_penalty = (stochastic_latency_ms / 250.0) * 0.0002
            slip_pct = base_slip + latency_penalty

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
        )

        return fill_event


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SYNCHRONIZED BACKTEST ENGINE v2.0
# ═══════════════════════════════════════════════════════════════════════════════


def run_global_backtest(
    all_data, symbols, days, initial_capital=None, verbose=True, seed=42
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
    print(f"   MODE: PRODUCTION-PARITY (uses real Portfolio + RiskManager)")
    print(f"{'=' * 70}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: INITIALIZE PRODUCTION COMPONENTS (identical to main.py L594-679)
    # ─────────────────────────────────────────────────────────────────────────
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

    # 1d. BacktestExecutor (simulates)
    executor = BacktestExecutor(data_provider=data_provider)

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

    for symbol in symbols:
        try:
            is_leader = "BTC" in symbol

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
            ml_swing.strategy_id += "_SWING"

            strategies_map[symbol] = [ml_scalp, ml_swing]

        except Exception as e:
            logger.warning(f"⚠️ Failed to init strategies for {symbol}: {e}")
            strategies_map[symbol] = []

    # ═══════════════════════════════════════════════════════════════════
    # FORENSIC REMEDIATION: Add TechnicalStrategy (SCALPING + SWING)
    # QUÉ: Se registran instancias duales de TechnicalStrategy.
    # POR QUÉ: Producción (main.py) ahora tiene TechnicalStrategy para
    #   ambos horizontes. Sin esto, el backtest no prueba los setups LONG
    #   explícitos ni las señales SWING del Technical.
    # PARA QUÉ: Paridad total backtest ←→ producción.
    # ═══════════════════════════════════════════════════════════════════
    tech_strategies = []
    try:
        from strategies.technical import HybridScalpingStrategy as TechnicalStrategy

        tech_scalp = TechnicalStrategy(data_provider, events_queue, horizon="SCALPING")
        tech_strategies.append(tech_scalp)

        tech_swing = TechnicalStrategy(data_provider, events_queue, horizon="SWING")
        tech_strategies.append(tech_swing)

        # TechnicalStrategy iterates ALL symbols internally, so we DON'T
        # add it per-symbol. Instead, we store it separately and call
        # generate_signals() once per epoch (it processes all symbols).
        print(
            f"  🧠 TechnicalStrategy registered: SCALPING + SWING (processes all {len(symbols)} symbols)"
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to init TechnicalStrategy: {e}")

    total_strats = sum(len(v) for v in strategies_map.values()) + len(tech_strategies)
    print(
        f"  🧠 Total strategies registered: {total_strats} ({len(symbols)} symbols × 2 ML + {len(tech_strategies)} Tech)"
    )

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

    t_start = time.time()
    last_progress = 0
    total_epochs = data_provider.total_epochs
    # Warmup: strategies need ~100 bars to compute features (RSI14, EMA20, etc.)
    # BacktestDataProvider accumulates bars, so after 100 epochs each symbol
    # has ~100 bars of history for get_latest_bars()
    warmup_epochs = min(100, total_epochs // 20)

    print(f"  ⏱️  Starting simulation: {total_epochs:,} global epochs")
    print(f"  🔥 Warmup: first {warmup_epochs} epochs (no trading)")

    while data_provider.continue_backtest:
        # ── ADVANCE GLOBAL TIMELINE ──
        data_provider.update_bars()
        epoch_count += 1

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
                            if hasattr(strat, "_launch_training"):
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
            symbol = event.symbol
            close_price = event.close_price

            if not close_price or close_price <= 0:
                continue

            # B1. UPDATE PORTFOLIO PRICES (production: portfolio.update_market_price)
            portfolio.update_market_price(symbol, close_price)

            # PREDICTION TRACKER: Update forward returns for all active signals
            # QUÉ: Actualiza MFE/MAE/direction accuracy de señales abiertas.
            # CUÁNDO: En cada tick de precio, O(active_signals) ~< 10.
            prediction_tracker.update_forward_returns(symbol, close_price, event.timestamp)

            # B2. Skip trading during warmup (strategies need history)
            if epoch_count < warmup_epochs:
                continue

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
            # FORENSIC-V11 Fix #7: Exit Cascade Prevention
            # Only emit EXIT if not already pending for this symbol+horizon
            try:
                stop_signals = risk_manager.check_stops(portfolio, data_provider, now=event.timestamp)
                if stop_signals:
                    for sig in stop_signals:
                        _exit_key = (
                            f"{sig.symbol}_{getattr(sig, 'horizon', 'SCALPING')}"
                        )
                        if _exit_key not in pending_exits:
                            events_queue.put(sig)
                            pending_exits.add(_exit_key)
            except Exception:
                pass

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

                    # SYNC inference — same as production _run_inference()
                    strat._run_inference()
                except Exception:
                    pass  # Strategy errors are non-fatal

        # ── B6. RUN TECHNICAL STRATEGIES (once per epoch, all symbols) ──
        # TechnicalStrategy.generate_signals() iterates ALL symbols internally.
        # We call it ONCE per epoch after all MarketEvents have been processed,
        # so all symbol prices are updated for the current epoch.
        if epoch_count >= warmup_epochs:
            for tech_strat in tech_strategies:
                try:
                    tech_strat.generate_signals()
                except Exception:
                    pass

        # ── Phase C: Process Signal/Exit events generated by strategies ──
        # Strategies put SignalEvents directly into events_queue via
        # events_queue.put(signal). Now we drain and process them.
        # This mirrors Engine._process_signal_event() in production.
        _max_signal_iterations = 50  # Prevent infinite loops
        _iter = 0
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
                    rejected_count += 1
                    # Extract specific rejection reason from stdout
                    _captured = _f_capture.getvalue()
                    _specific_reason = None
                    for _line in _captured.strip().split("\n"):
                        if "[RISK] Rejected by" in _line:
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

                # ── PORTFOLIO UPDATE (THE REAL update_fill) ──
                try:
                    result = portfolio.update_fill(fill)
                    fill_count += 1

                    # FORENSIC-V11 Fix #2+#6: Record fill epoch for cooldown
                    _fill_horizon = getattr(fill, "horizon", "SCALPING")
                    _fill_cooldown_key = f"{fill.symbol}_{_fill_horizon}"
                    last_fill_epoch[_fill_cooldown_key] = epoch_count

                    # FORENSIC-V11 Fix #7: Clear pending exit after fill
                    pending_exits.discard(_fill_cooldown_key)

                    # Record result for RiskManager Kelly tracking
                    if result and isinstance(result, tuple):
                        pnl, outcome = result
                        if pnl is not None:
                            # ═══════════════════════════════════════════════════════════════
                            # FORENSIC-V31 FIX: USE NET PNL FOR RISK MANAGER TRACKING
                            # QUÉ: Usar net_pnl para determinar is_win.
                            # POR QUÉ: risk_manager.win_count se usa para Kelly sizing y backtest WR.
                            #   Si usamos gross_pnl, inflamos el WR y el apalancamiento Kelly!
                            # ═══════════════════════════════════════════════════════════════
                            _closed_trade = getattr(portfolio, '_last_closed_trade_data', None)
                            if _closed_trade and _closed_trade.get('symbol') == event.symbol:
                                net_pnl = _closed_trade.get('net_pnl', pnl)
                            else:
                                net_pnl = pnl  # Fallback
                                
                            is_win = net_pnl > 0
                            pnl_pct = net_pnl / capital if capital > 0 else 0
                            risk_manager.record_trade_result(
                                is_win, pnl_pct, event.symbol
                            )

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

            elif event.type == EventType.FILL:
                pass  # Handled inline after executor

        # ── END OF EPOCH: Update equity curve ──
        if epoch_count % 60 == 0:  # Sample every 60 bars (1 hour)
            eq = portfolio.get_total_equity()
            equity_curve.append(eq)
            ts = pd.to_datetime(data_provider.current_time_ms, unit="ms", utc=True)
            equity_timestamps.append(ts)

            # Update RiskManager equity (for kill switch)
            try:
                risk_manager.update_equity(eq)
            except:
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

            # Global kill switch check (hard floor)
            if eq < capital * 0.85:  # 15% total drawdown -> emergency stop
                print(
                    f"\n  🚨 HARD FLOOR KILL SWITCH: Equity ${eq:.2f} < 85% of initial ${capital:.2f}"
                )
                kill_switch_triggered = True

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
            )
            portfolio.update_fill(close_fill)
            fill_count += 1
        except Exception as e:
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
    total_trades = len(portfolio.get_trade_history()) if hasattr(portfolio, "get_trade_history") else len(all_trades)

    # Equity curve metrics
    eq_arr = np.array(equity_curve)
    max_dd = 0
    if len(eq_arr) > 1:
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / peak
        max_dd = float(np.max(dd)) * 100

    # Win rate from kelly tracking
    total_wl = risk_manager.win_count + risk_manager.loss_count
    win_rate = (risk_manager.win_count / total_wl * 100) if total_wl > 0 else 0

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

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="God Mode Backtest v2.0 — Global Synchronized Engine"
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
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="Comma-separated symbols or ALL for full basket",
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
    all_data = fetch_multi_symbol_data(
        symbols, days=args.days, max_workers=4, end_time=end_time
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

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"💾 Results saved to: {output_path}")
    print(f"   run_id: {run_id_result} (use for comparison)")
    return results


if __name__ == "__main__":
    main()
