"""
🎯 PREDICTION TRACKER v1.0 — Real-Time Predictive Accuracy System
===================================================================
Trader Gemini — Cierre del Feedback Loop Predictivo

QUÉ: Módulo que registra cada señal generada en producción/backtest,
  rastrea su precisión direccional bar-a-bar, y expone métricas
  agregadas (accuracy, confidence_factor, optimal_ttl) para que
  RiskManager y BinanceExecutor las consuman en tiempo real.

POR QUÉ: El sistema genera prediction_metrics.json offline pero
  NUNCA lo consume en producción. No hay retroalimentación desde
  record_trade_result() hacia las estrategias ni el motor de ejecución.
  El sistema aprende win rate (Kelly) pero NO aprende precisión
  direccional ni duración óptima del edge.

PARA QUÉ: Alimentar el feedback loop que:
  1. Rechaza señales de estrategias con accuracy < 55%
  2. Modula sizing por confidence_factor (0.5x-1.2x)
  3. Ajusta LIMIT pricing por precisión histórica
  4. Calcula duración óptima del edge (TTL)

CÓMO: Singleton en memoria con ring buffer por strategy_id.
  - record_signal(): Registra nueva señal
  - update_forward_returns(): Actualiza MFE/MAE en cada price tick
  - get_strategy_metrics(): Retorna accuracy/confidence para filtros
  - export_metrics(): Persiste a prediction_metrics.json

CUÁNDO: Instanciado en main.py y run_god_mode_backtest.py.
DÓNDE: core/prediction_tracker.py
QUIÉN: Engine, RiskManager, BinanceExecutor.

DEPENDENCIAS: Ninguna crítica (solo numpy, json, collections).
  NO toca engine.py, risk_manager.py, binance_executor.py.
  Esos módulos lo CONSUMEN, no lo modifican.
"""

import os
import json
import time
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

from utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SIGNALS_PER_STRATEGY = 500   # Ring buffer size per strategy_id
FORWARD_WINDOWS = [1, 3, 5, 10, 15, 30, 60, 120]  # Bars to measure accuracy
MIN_SIGNALS_FOR_METRICS = 20     # Minimum signals before exposing metrics (reduced from 30 for micro-accounts)
DECAY_CLEANUP_BARS = 2000        # Clean signals older than this
PERSIST_INTERVAL = 100           # Persist to JSON every N signals
DEFAULT_CONFIDENCE = 1.0         # Neutral confidence when no data


class PredictionSignal:
    """
    Represents a single recorded prediction signal with its forward tracking state.
    
    Lightweight struct (~200 bytes per signal) optimized for ring buffer storage.
    """
    __slots__ = [
        'strategy_id', 'symbol', 'direction', 'horizon',
        'entry_price', 'sl_pct', 'tp_pct', 'confidence',
        'timestamp', 'bar_count',
        'mfe', 'mae', 'mfe_bar', 'mae_bar',
        'direction_correct_at',  # dict: {window: bool}
        'is_resolved', 'trade_outcome',  # win/loss/None
        'pnl_pct',
    ]

    def __init__(self, strategy_id: str, symbol: str, direction: str,
                 horizon: str, entry_price: float, sl_pct: float,
                 tp_pct: float, confidence: float, timestamp):
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.direction = direction  # 'long' or 'short'
        self.horizon = horizon
        self.entry_price = entry_price
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.confidence = confidence
        self.timestamp = timestamp
        self.bar_count = 0

        # Forward tracking
        self.mfe = 0.0   # Maximum Favorable Excursion (%)
        self.mae = 0.0   # Maximum Adverse Excursion (%)
        self.mfe_bar = 0
        self.mae_bar = 0
        self.direction_correct_at = {}  # {window_bars: True/False}

        # Resolution
        self.is_resolved = False
        self.trade_outcome = None  # 'win', 'loss', None
        self.pnl_pct = 0.0


class PredictionTracker:
    """
    🎯 Real-Time Prediction Accuracy Tracker

    Closes the feedback loop between signal generation and execution optimization.

    Usage:
        tracker = PredictionTracker()

        # On signal generation:
        tracker.record_signal('ML_SCALPING_BTC', 'BTC/USDT', 'long', 'SCALPING',
                              42000.0, 0.0025, 0.004, 0.85, timestamp)

        # On every price update:
        tracker.update_forward_returns('BTC/USDT', 42050.0, timestamp)

        # In RiskManager.generate_order():
        metrics = tracker.get_strategy_metrics('ML_SCALPING_BTC', 'SCALPING')
        if metrics['direction_accuracy'] < 0.55:
            return None  # Reject signal
    """

    def __init__(self, persist_path: str = None):
        # Ring buffers per strategy_id
        self._signals: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=MAX_SIGNALS_PER_STRATEGY)
        )
        # Active (unresolved) signals indexed by symbol for fast price updates
        self._active_by_symbol: Dict[str, list] = defaultdict(list)

        # Aggregated metrics cache (refreshed on demand)
        self._metrics_cache: Dict[str, Dict] = {}
        self._cache_dirty = True
        self._total_signals_recorded = 0

        # Persistence
        self._persist_path = persist_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'prediction_metrics.json'
        )
        self._signals_since_persist = 0

        # Load existing metrics if available
        self._load_existing_metrics()

    def _load_existing_metrics(self):
        """Load previously persisted metrics as warm start."""
        try:
            if os.path.exists(self._persist_path):
                with open(self._persist_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._metrics_cache = data
                    logger.info(
                        f"🎯 [PredictionTracker] Loaded {len(data)} strategy metrics from {self._persist_path}"
                    )
        except Exception as e:
            logger.warning(f"⚠️ [PredictionTracker] Could not load existing metrics: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL RECORDING
    # ═══════════════════════════════════════════════════════════════════════════

    def record_signal(self, strategy_id: str, symbol: str, direction: str,
                      horizon: str, entry_price: float, sl_pct: float,
                      tp_pct: float, confidence: float = 0.5,
                      timestamp=None) -> None:
        """
        📝 Registra una nueva señal predictiva para tracking.

        Llamado desde: Engine (después de strategy.calculate_signals())
        o desde el backtest loop (después de generar SignalEvent).
        """
        if entry_price <= 0:
            return

        ts = timestamp or datetime.now(timezone.utc)

        sig = PredictionSignal(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction.lower(),
            horizon=horizon,
            entry_price=entry_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            confidence=confidence,
            timestamp=ts,
        )

        self._signals[strategy_id].append(sig)
        self._active_by_symbol[symbol].append(sig)
        self._total_signals_recorded += 1
        self._cache_dirty = True

        # Periodic persistence
        self._signals_since_persist += 1
        if self._signals_since_persist >= PERSIST_INTERVAL:
            self._persist_metrics()
            self._signals_since_persist = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # FORWARD RETURN TRACKING
    # ═══════════════════════════════════════════════════════════════════════════

    def update_forward_returns(self, symbol: str, current_price: float,
                               timestamp=None) -> None:
        """
        📊 Actualiza MFE/MAE/direction accuracy de todas las señales activas
        para un símbolo dado.

        Llamado desde: Engine.process_market_event() o backtest epoch loop.
        Complejidad: O(active_signals_for_symbol) — típicamente < 10.
        """
        if current_price <= 0 or symbol not in self._active_by_symbol:
            return

        active = self._active_by_symbol[symbol]
        to_remove = []

        for i, sig in enumerate(active):
            if sig.is_resolved:
                to_remove.append(i)
                continue

            sig.bar_count += 1
            entry = sig.entry_price

            # Calculate unrealized return
            if sig.direction == 'long':
                unrealized_pct = (current_price - entry) / entry
                favorable = (current_price - entry) / entry
            else:
                unrealized_pct = (entry - current_price) / entry
                favorable = (entry - current_price) / entry

            # Update MFE/MAE
            if favorable > sig.mfe:
                sig.mfe = favorable
                sig.mfe_bar = sig.bar_count
            if favorable < sig.mae:
                sig.mae = favorable
                sig.mae_bar = sig.bar_count

            # Direction accuracy at forward windows
            for window in FORWARD_WINDOWS:
                if sig.bar_count == window and window not in sig.direction_correct_at:
                    sig.direction_correct_at[window] = (unrealized_pct > 0)

            # Auto-resolve after DECAY_CLEANUP_BARS
            if sig.bar_count >= DECAY_CLEANUP_BARS:
                sig.is_resolved = True
                to_remove.append(i)

        # Cleanup resolved signals from active list
        for idx in reversed(to_remove):
            if idx < len(active):
                active.pop(idx)

        if to_remove:
            self._cache_dirty = True

    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE OUTCOME RECORDING
    # ═══════════════════════════════════════════════════════════════════════════

    def record_trade_outcome(self, symbol: str, is_win: bool,
                             pnl_pct: float = 0.0, strategy_id: str = None) -> None:
        """
        📈 Records the actual outcome of a trade for the most recent signal.

        Called from: RiskManager.record_trade_result()
        """
        # Find the most recent unresolved signal for this symbol
        if symbol in self._active_by_symbol:
            for sig in reversed(self._active_by_symbol[symbol]):
                if not sig.is_resolved:
                    if strategy_id and sig.strategy_id != strategy_id:
                        continue
                    sig.is_resolved = True
                    sig.trade_outcome = 'win' if is_win else 'loss'
                    sig.pnl_pct = pnl_pct
                    self._cache_dirty = True
                    break

    # ═══════════════════════════════════════════════════════════════════════════
    # METRICS AGGREGATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _refresh_metrics(self) -> None:
        """Recalculate aggregated metrics from raw signals."""
        if not self._cache_dirty:
            return

        new_cache = {}

        for strat_id, ring in self._signals.items():
            signals = list(ring)
            n = len(signals)
            if n < 10:
                continue

            # Direction accuracy by window
            accuracy_by_window = {}
            for w in FORWARD_WINDOWS:
                correct = sum(
                    1 for s in signals
                    if w in s.direction_correct_at and s.direction_correct_at[w]
                )
                total = sum(1 for s in signals if w in s.direction_correct_at)
                if total > 0:
                    accuracy_by_window[w] = correct / total

            # Overall direction accuracy (use 15-bar window as primary)
            primary_window = 15
            if primary_window in accuracy_by_window:
                direction_accuracy = accuracy_by_window[primary_window]
            elif accuracy_by_window:
                direction_accuracy = np.mean(list(accuracy_by_window.values()))
            else:
                direction_accuracy = 0.5

            # MFE / MAE stats (include ALL signals, even 1-bar scalps)
            mfes = [s.mfe for s in signals if s.bar_count >= 1]
            maes = [s.mae for s in signals if s.bar_count >= 1]
            avg_mfe = float(np.mean(mfes)) if mfes else 0.0
            avg_mae = float(np.mean(maes)) if maes else 0.0

            # Prediction decay (bar where MFE is reached)
            mfe_bars = [s.mfe_bar for s in signals if s.mfe_bar > 0]
            optimal_ttl = int(np.median(mfe_bars)) if mfe_bars else 60

            # Confidence factor: scales linearly from 0.5 (at 50% acc) to 1.2 (at 80%+)
            if direction_accuracy > 0.50:
                c_factor = min(1.2, max(0.5, (direction_accuracy - 0.5) * 2.5 + 0.5))
            else:
                c_factor = 0.5

            # Trade outcome stats
            resolved = [s for s in signals if s.is_resolved and s.trade_outcome]
            wins = sum(1 for s in resolved if s.trade_outcome == 'win')
            losses = sum(1 for s in resolved if s.trade_outcome == 'loss')
            trade_win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5

            # Horizon breakdown
            horizon = signals[-1].horizon if signals else 'SCALPING'

            new_cache[strat_id] = {
                'strategy_id': strat_id,
                'horizon': horizon,
                'total_signals': n,
                'direction_accuracy': round(direction_accuracy, 4),
                'accuracy_by_window': {
                    str(k): round(v, 4) for k, v in accuracy_by_window.items()
                },
                'avg_mfe_pct': round(avg_mfe * 100, 4),
                'avg_mae_pct': round(avg_mae * 100, 4),
                'optimal_ttl_bars': optimal_ttl,
                'confidence_factor': round(c_factor, 3),
                'trade_win_rate': round(trade_win_rate, 4),
                'trades_resolved': wins + losses,
                'trades_won': wins,
                'trades_lost': losses,
            }

        self._metrics_cache = new_cache
        self._cache_dirty = False

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    def get_strategy_metrics(self, strategy_id: str,
                             horizon: str = None) -> Optional[Dict]:
        """
        📊 Returns aggregated prediction metrics for a strategy.

        Returns None if insufficient data (< MIN_SIGNALS_FOR_METRICS).

        Used by: RiskManager.generate_order() for confidence gate.
        """
        self._refresh_metrics()

        metrics = self._metrics_cache.get(strategy_id)
        if not metrics:
            return None

        if metrics.get('total_signals', 0) < MIN_SIGNALS_FOR_METRICS:
            return None

        if horizon and metrics.get('horizon') != horizon:
            # Try to find strategy with matching horizon
            for sid, m in self._metrics_cache.items():
                if sid.startswith(strategy_id) and m.get('horizon') == horizon:
                    return m if m.get('total_signals', 0) >= MIN_SIGNALS_FOR_METRICS else None

        return metrics

    def get_execution_params(self, strategy_id: str,
                             horizon: str = None) -> Dict:
        """
        🎯 Returns optimal execution parameters based on prediction accuracy.

        Used by: BinanceExecutor for smart LIMIT pricing.

        Returns:
            {
                'confidence_factor': float,  # 0.5-1.2
                'avg_mfe_pct': float,        # Average favorable excursion
                'optimal_ttl_bars': int,     # Optimal hold duration
                'limit_offset_pct': float,   # Suggested LIMIT offset
            }
        """
        metrics = self.get_strategy_metrics(strategy_id, horizon)
        if not metrics:
            return {
                'confidence_factor': DEFAULT_CONFIDENCE,
                'avg_mfe_pct': 0.1,
                'optimal_ttl_bars': 60,
                'limit_offset_pct': 0.0003,  # Default 0.03% offset
            }

        c_factor = metrics['confidence_factor']

        # Higher confidence → tighter spread (more aggressive LIMIT)
        # c_factor 0.5 → 0.05% offset (passive)
        # c_factor 1.2 → 0.01% offset (aggressive)
        limit_offset = max(0.0001, 0.0005 * (1.5 - c_factor))

        return {
            'confidence_factor': c_factor,
            'avg_mfe_pct': metrics['avg_mfe_pct'],
            'optimal_ttl_bars': metrics['optimal_ttl_bars'],
            'limit_offset_pct': round(limit_offset, 6),
        }

    def should_reject_signal(self, strategy_id: str,
                             horizon: str = None) -> Tuple[bool, str]:
        """
        🛡️ Returns whether a signal should be rejected based on historical accuracy.

        Used by: RiskManager.generate_order() as an additional gate.

        Returns:
            (should_reject: bool, reason: str)
        """
        metrics = self.get_strategy_metrics(strategy_id, horizon)
        if not metrics:
            return False, ""  # Not enough data to judge

        acc = metrics['direction_accuracy']
        n = metrics['total_signals']

        if n >= MIN_SIGNALS_FOR_METRICS and acc < 0.55:
            return True, (
                f"accuracy {acc:.1%} < 55% threshold "
                f"(n={n}, horizon={metrics.get('horizon', '?')})"
            )

        return False, ""

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════

    def export_metrics(self) -> Dict:
        """
        📤 Exports current metrics as a dict (also persists to JSON).

        Called at: End of backtest, periodic in production.
        """
        self._refresh_metrics()
        self._persist_metrics()
        return dict(self._metrics_cache)

    def _persist_metrics(self) -> None:
        """Write metrics to prediction_metrics.json."""
        try:
            self._refresh_metrics()
            with open(self._persist_path, 'w') as f:
                json.dump(self._metrics_cache, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"⚠️ [PredictionTracker] Persist failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_summary(self) -> str:
        """Human-readable summary for logging/dashboard."""
        self._refresh_metrics()
        lines = [
            "🎯 PREDICTION TRACKER SUMMARY",
            f"   Total signals recorded: {self._total_signals_recorded}",
            f"   Strategies tracked: {len(self._metrics_cache)}",
            "",
        ]

        for sid, m in sorted(self._metrics_cache.items()):
            acc = m['direction_accuracy']
            emoji = "✅" if acc > 0.60 else "⚠️" if acc > 0.55 else "❌"
            lines.append(
                f"   {emoji} {sid}: acc={acc:.1%} cf={m['confidence_factor']:.2f} "
                f"mfe={m['avg_mfe_pct']:.3f}% ttl={m['optimal_ttl_bars']}bars "
                f"n={m['total_signals']}"
            )

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"PredictionTracker(strategies={len(self._signals)}, "
            f"total_signals={self._total_signals_recorded})"
        )
