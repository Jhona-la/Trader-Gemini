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

MAX_SIGNALS_PER_STRATEGY = 200   # [Phase 6 RAM OFF-LOAD] Strict sliding window
FORWARD_WINDOWS = [1, 3, 5, 10, 15, 30, 60, 120]  # Bars to measure accuracy
MIN_SIGNALS_FOR_METRICS = 50     # Reduced from 100 to allow faster learning with smaller RAM footprint
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
        # CTOS Phase 3: Extended prediction fields
        'predicted_magnitude', 'predicted_target_price', 'predicted_duration_bars',
        'trade_id', 'thought_id',  # Forensic traceability IDs
        'is_vetoed', 'veto_reason',  # Meta-Arbitrator Diagnostics
    ]

    def __init__(self, strategy_id: str, symbol: str, direction: str,
                 horizon: str, entry_price: float, sl_pct: float,
                 tp_pct: float, confidence: float, timestamp,
                 predicted_magnitude: float = None,
                 predicted_target_price: float = None,
                 predicted_duration_bars: int = None,
                 trade_id: str = None, thought_id: str = None):
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

        # CTOS Phase 3: Extended prediction fields
        self.predicted_magnitude = predicted_magnitude  # Expected % move
        self.predicted_target_price = predicted_target_price  # Expected target price
        self.predicted_duration_bars = predicted_duration_bars  # Expected bars to target
        self.trade_id = trade_id  # Link to portfolio trade_id
        self.thought_id = thought_id  # Link to thought_id in DB
        
        # Meta-Arbitrator Veto Diagnostics
        self.is_vetoed = False
        self.veto_reason = ""


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

    def _resolve_strategy_id(self, strategy_id: str, symbol: str) -> str:
        if not symbol:
            return strategy_id
        clean_sym = symbol.replace('/', '_').replace('-', '_')
        if clean_sym not in strategy_id:
            return f"{strategy_id}_{clean_sym}"
        return strategy_id

    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL RECORDING
    # ═══════════════════════════════════════════════════════════════════════════

    def record_signal(self, strategy_id: str, symbol: str, direction: str,
                      horizon: str, entry_price: float, sl_pct: float,
                      tp_pct: float, confidence: float = 0.5,
                      timestamp=None,
                      predicted_magnitude: float = None,
                      predicted_target_price: float = None,
                      predicted_duration_bars: int = None,
                      trade_id: str = None,
                      thought_id: str = None) -> None:
        """
        📝 Registra una nueva señal predictiva para tracking.

        Llamado desde: Engine (después de strategy.calculate_signals())
        o desde el backtest loop (después de generar SignalEvent).
        
        CTOS Phase 3: Now accepts predicted_magnitude, predicted_target_price,
        and predicted_duration_bars for full prediction audit trail.
        """
        if entry_price <= 0:
            return

        ts = timestamp or datetime.now(timezone.utc)
        strategy_id = self._resolve_strategy_id(strategy_id, symbol)

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
            predicted_magnitude=predicted_magnitude,
            predicted_target_price=predicted_target_price,
            predicted_duration_bars=predicted_duration_bars,
            trade_id=trade_id,
            thought_id=thought_id,
        )

        self._signals[strategy_id].append(sig)
        self._active_by_symbol[symbol].append(sig)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4 FIX: RAM MEMORY LEAK PREVENTION (SSD OFFLOADING)
        # QUÉ: Limitar el tamaño de _active_by_symbol a 50 señales vivas.
        # POR QUÉ: Para evitar llenar la RAM con señales huérfanas en
        # simulaciones o ejecuciones de varios días.
        # ═══════════════════════════════════════════════════════════════
        if len(self._active_by_symbol[symbol]) > 50:
            while len(self._active_by_symbol[symbol]) > 50:
                old_sig = self._active_by_symbol[symbol].pop(0)
                old_sig.is_resolved = True
                
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

            # Direction accuracy at forward windows (ANTI-MENTIRA FIX)
            # QUÉ: Una predicción no es correcta solo porque esté en >0.
            # POR QUÉ: Si predijo ganar 1% y va en +0.01%, está estancado, no es correcto.
            # PARA QUÉ: Penalizar predicciones falsas o extremadamente optimistas.
            for window in FORWARD_WINDOWS:
                if sig.bar_count == window and window not in sig.direction_correct_at:
                    target = sig.predicted_magnitude or sig.tp_pct or 0.001
                    # Se espera un progreso proporcional al tiempo transcurrido
                    expected_duration = sig.predicted_duration_bars or 60
                    expected_progress = target * (window / max(1, expected_duration))
                    # Requerimos al menos el 50% del progreso esperado (mitad de inercia)
                    min_required = min(expected_progress * 0.5, target * 0.8)
                    sig.direction_correct_at[window] = (unrealized_pct >= min_required)

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
                             pnl_pct: float = 0.0, strategy_id: str = None,
                             trade_id: str = None) -> Optional[Dict]:
        """
        📈 Records the actual outcome of a trade for the most recent signal.
        
        CTOS Phase 3: Returns forensic data dict with prediction vs reality
        for the prediction_audit table.

        Called from: RiskManager.record_trade_result() / Portfolio._record_closed_trade()
        
        Returns:
            dict with prediction audit data, or None if no matching signal found.
        """
        audit_data = None
        if strategy_id:
            strategy_id = self._resolve_strategy_id(strategy_id, symbol)
        
        # Find the most recent unresolved signal for this symbol
        if symbol in self._active_by_symbol:
            for sig in reversed(self._active_by_symbol[symbol]):
                if not sig.is_resolved:
                    if strategy_id and sig.strategy_id != strategy_id:
                        continue
                    if trade_id and sig.trade_id and sig.trade_id != trade_id:
                        continue
                    # CTOS Phase 3/5: Anti-Mentira (Magnitude Verification)
                    sig.is_resolved = True
                    sig.pnl_pct = pnl_pct
                    self._cache_dirty = True
                    
                    target = sig.predicted_magnitude or sig.tp_pct or 0.0
                    
                    if target > 0:
                        # FORENSIC-V91 FIX: Alpha Decay may close trades early for safety.
                        # If a trade made at least 40% of its target, or achieved a positive
                        # return greater than 0.1%, it's a valid directional prediction.
                        # It's only a FAILURE if it closed negative or barely moved (<0.05%).
                        magnitude_achieved = sig.mfe >= (target * 0.4) or sig.mfe >= 0.001
                    else:
                        magnitude_achieved = (pnl_pct > 0)
                        
                    was_direction_correct = (pnl_pct > 0) or (sig.mfe > 0.001 and pnl_pct > -0.0005)
                    
                    # La predicción fue correcta direccionalmente si subió y logramos al menos un MFE decente.
                    was_correct = was_direction_correct and magnitude_achieved
                    
                    # Fix: No destruir la reputación del modelo si el RiskManager cerró por seguridad.
                    # Si el PnL es positivo, SIEMPRE ES UN WIN. No penalizar.
                    if pnl_pct > 0:
                        sig.trade_outcome = 'win'
                    else:
                        # Si PnL es negativo pero MFE fue alto (hit SL after running), es loss.
                        sig.trade_outcome = 'loss'
                    
                    # Calculate missed profit: Target - actual exit PnL
                    missed_profit = max(0.0, target - max(0.0, pnl_pct))
                    
                    audit_data = {
                        'strategy_id': sig.strategy_id,
                        'symbol': sig.symbol,
                        'horizon': sig.horizon,
                        'direction': sig.direction,
                        'confidence': sig.confidence,
                        'predicted_magnitude': sig.predicted_magnitude,
                        'predicted_target_price': sig.predicted_target_price,
                        'predicted_duration_bars': sig.predicted_duration_bars,
                        'actual_magnitude_pct': pnl_pct,
                        'actual_duration_bars': sig.bar_count,
                        'was_correct': was_correct,
                        'optimal_exit_bar': sig.mfe_bar,
                        'mfe_pct': sig.mfe,
                        'mae_pct': sig.mae,
                        'missed_profit_pct': missed_profit,
                        'trade_id': sig.trade_id or trade_id,
                        'thought_id': sig.thought_id,
                    }
                    break
        
        return audit_data

    def record_signal_veto(self, strategy_id: str, symbol: str, reason: str, trade_id: str = None) -> None:
        """
        🛑 CTOS Meta-Arbitrator Diagnostic Hook
        Marks an active signal as VETOED (rejected by the Meta-Arbitrator).
        This allows the system to quantify signals generated vs executed and
        calibrate the ML to avoid proposing trades in contexts that always fail.
        """
        if symbol not in self._active_by_symbol:
            return
            
        strategy_id = self._resolve_strategy_id(strategy_id, symbol)
        
        # Search from most recent
        for sig in reversed(self._active_by_symbol[symbol]):
            if not sig.is_resolved:
                if strategy_id and sig.strategy_id != strategy_id:
                    continue
                if trade_id and sig.trade_id and sig.trade_id != trade_id:
                    continue
                
                sig.is_resolved = True
                sig.is_vetoed = True
                sig.veto_reason = reason
                sig.trade_outcome = 'vetoed'
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

            # Overall direction accuracy using actual trade outcomes
            # FORENSIC-V92: Previous logic used 15-bar window which ignored
            # trades that hit TP/SL or were closed before 15 bars.
            resolved = [s for s in signals if s.is_resolved and s.trade_outcome in ('win', 'loss')]
            vetoed = [s for s in signals if getattr(s, 'is_vetoed', False)]
            
            wins = sum(1 for s in resolved if s.trade_outcome == 'win')
            losses = sum(1 for s in resolved if s.trade_outcome == 'loss')
            vetoes = len(vetoed)
            trade_win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5

            direction_accuracy = trade_win_rate
            
            # Keep accuracy_by_window for analytics but don't use it for gate logic
            primary_window = 15

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

            # Horizon breakdown
            horizon = signals[-1].horizon if signals else 'SCALPING'

            # CTOS Phase 6: Hyper-Growth (Streak Tracking)
            current_win_streak = 0
            current_loss_streak = 0
            if resolved:
                for s in reversed(resolved):
                    if s.trade_outcome == 'win':
                        if current_loss_streak > 0:
                            break
                        current_win_streak += 1
                    elif s.trade_outcome == 'loss':
                        if current_win_streak > 0:
                            break
                        current_loss_streak += 1

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
                'trades_vetoed': vetoes,
                'veto_rate': round(vetoes / max(1, n), 4),
                'current_win_streak': current_win_streak,
                'current_loss_streak': current_loss_streak,
            }

        self._metrics_cache = new_cache
        self._cache_dirty = False

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    def get_strategy_metrics(self, strategy_id: str,
                             horizon: str = None,
                             symbol: str = None) -> Optional[Dict]:
        """
        📊 Returns aggregated prediction metrics for a strategy.

        Returns None if insufficient data (< MIN_SIGNALS_FOR_METRICS).

        Used by: RiskManager.generate_order() for confidence gate.
        """
        self._refresh_metrics()
        strategy_id = self._resolve_strategy_id(strategy_id, symbol)

        metrics = self._metrics_cache.get(strategy_id)
        if not metrics:
            return None

        if metrics['total_signals'] < MIN_SIGNALS_FOR_METRICS:
            return None

        if horizon and metrics['horizon'] != horizon:
            # Try to find strategy with matching horizon
            for sid, m in self._metrics_cache.items():
                if sid.startswith(strategy_id) and m['horizon'] == horizon:
                    return m if m['total_signals'] >= MIN_SIGNALS_FOR_METRICS else None

        return metrics

    def get_execution_params(self, strategy_id: str,
                             horizon: str = None,
                             symbol: str = None) -> Dict:
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
        metrics = self.get_strategy_metrics(strategy_id, horizon, symbol)
        if not metrics:
            return {
                'confidence_factor': DEFAULT_CONFIDENCE,
                'avg_mfe_pct': 0.1,
                'optimal_ttl_bars': 60,
                'limit_offset_pct': 0.0003,  # Default 0.03% offset
            }

        c_factor = metrics['confidence_factor']
        accuracy = metrics['direction_accuracy']

        # 🎯 OPTIMIZACIÓN DE EJECUCIÓN LIMIT BASADA EN DATA
        # Si precisión > 75% → Órdenes LIMIT agresivas (push spread, offset negativo)
        # Si precisión 60-75% → Órdenes LIMIT conservadoras (passive maker, offset positivo)
        # Si precisión < 60% → Reevaluación de estrategia (offset estándar, pero será rechazada)
        if accuracy >= 0.75:
            limit_offset = -0.0002  # Agresivo: push 0.02% into spread para forzar fill
        elif accuracy >= 0.60:
            limit_offset = 0.0001   # Conservador: wait 0.01% away from BBO
        else:
            limit_offset = 0.0003   # Muy pasivo

        return {
            'confidence_factor': c_factor,
            'avg_mfe_pct': metrics['avg_mfe_pct'],
            'optimal_ttl_bars': metrics['optimal_ttl_bars'],
            'limit_offset_pct': round(limit_offset, 6),
        }

    def should_reject_signal(self, strategy_id: str,
                             horizon: str = None,
                             symbol: str = None) -> Tuple[bool, str]:
        """
        🛡️ Returns whether a signal should be rejected based on historical accuracy.

        Used by: RiskManager.generate_order() as an additional gate.

        [FORENSIC-AUDIT-V1] COLD-START FIX:
        QUÉ: Requiere al menos 30 trades RESUELTOS antes de activar el gate.
        POR QUÉ: Con solo señales (no trades), la accuracy es ~50% (coin-flip)
          y el gate bloquea todo, creando un deadlock: para subir accuracy se
          necesitan trades, pero para hacer trades se necesita pasar el gate.
        PARA QUÉ: Permitir que el sistema acumule suficientes trades reales
          antes de juzgar la calidad predictiva.
        CÓMO: Verificar trades_resolved >= 30 además de total_signals >= 50.

        Returns:
            (should_reject: bool, reason: str)
        """
        metrics = self.get_strategy_metrics(strategy_id, horizon, symbol)
        if not metrics:
            return False, ""  # Not enough data to judge

        acc = metrics['direction_accuracy']
        n = metrics['total_signals']
        resolved = metrics['trades_resolved']

        # [FORENSIC-AUDIT-V1] Require 10 RESOLVED trades before activating gate
        # This prevents cold-start deadlock where accuracy is ~50% from 
        # unresolved signals and the gate blocks all new trades.
        MIN_RESOLVED_FOR_GATE = 10
        if resolved < MIN_RESOLVED_FOR_GATE:
            logger.debug(
                f"🎯 [PREDICTION_GATE] {strategy_id} warming up: "
                f"{resolved}/{MIN_RESOLVED_FOR_GATE} resolved trades. "
                f"Gate bypassed (current acc={acc:.1%}, n={n})."
            )
            return False, ""

        # CTOS Phase 5: Sniper Accuracy Minimum (55%)
        # Si la estrategia no tiene Edge Matemático positivo, queda vetada.
        if n >= MIN_SIGNALS_FOR_METRICS and acc < 0.55:
            return True, (
                f"accuracy {acc:.1%} < 55% threshold "
                f"(n={n}, resolved={resolved}, horizon={metrics['horizon']})"
            )

        return False, ""

    def get_prediction_for_trade(self, symbol: str, strategy_id: str = None,
                                 trade_id: str = None) -> Optional[Dict]:
        """
        🔍 CTOS Phase 3: Returns the active prediction for a symbol/strategy.
        
        Used by: ExitOracle to know what the opening strategy predicted.
        This allows exit strategies to evaluate if the position is on track
        to reach its predicted target or has deviated.
        
        Returns:
            dict with prediction data, or None if no matching signal found.
        """
        if symbol not in self._active_by_symbol:
            return None
        
        if strategy_id:
            strategy_id = self._resolve_strategy_id(strategy_id, symbol)
        
        for sig in reversed(self._active_by_symbol[symbol]):
            if sig.is_resolved:
                continue
            if strategy_id and sig.strategy_id != strategy_id:
                continue
            if trade_id and sig.trade_id and sig.trade_id != trade_id:
                continue
            
            return {
                'strategy_id': sig.strategy_id,
                'symbol': sig.symbol,
                'direction': sig.direction,
                'horizon': sig.horizon,
                'entry_price': sig.entry_price,
                'confidence': sig.confidence,
                'predicted_magnitude': sig.predicted_magnitude,
                'predicted_target_price': sig.predicted_target_price,
                'predicted_duration_bars': sig.predicted_duration_bars,
                'bar_count': sig.bar_count,
                'mfe': sig.mfe,
                'mae': sig.mae,
                'mfe_bar': sig.mfe_bar,
                'trade_id': sig.trade_id,
                'thought_id': sig.thought_id,
                'sl_pct': sig.sl_pct,
                'tp_pct': sig.tp_pct,
            }
        return None

    def calculate_realtime_edge(self, strategy_id: str, elapsed_bars: float, horizon: str = None, symbol: str = None) -> float:
        """
        📉 CTOS Phase 3: Dynamic Alpha Decay
        Calcula el 'Edge Probability' (0.0 a 1.0) usando la función matemática de decaimiento continuo
        en vez de un corte binario basado en un umbral de tiempo.
        
        Args:
            strategy_id: ID de la estrategia.
            elapsed_bars: Barras o minutos transcurridos desde que se abrió el trade.
            horizon: SCALPING o SWING.
            
        Returns:
            float: Edge probability actual. Si es menor a 0.45, el trade perdió su inercia matemática.
        """
        metrics = self.get_strategy_metrics(strategy_id, horizon, symbol)
        
        # Fallback values
        initial_accuracy = 0.5
        ttl_bars = 25.0
        
        if metrics:
            initial_accuracy = metrics['direction_accuracy']
            ttl_bars = float(metrics['optimal_ttl_bars'])
            
            # Si el TTL histórico es muy corto o absurdo, ponerle límites (min 5, max 120)
            ttl_bars = max(5.0, min(120.0, ttl_bars))
        
        # Traemos la funcion Numba de FastMath para el Alpha Decay continuo
        try:
            from utils.math_kernel import compute_alpha_decay_jit
            # compute_alpha_decay_jit(signal_strength, elapsed_seconds, ttl_seconds)
            # En nuestro caso, strength = accuracy inicial, y enviamos barras en lugar de segundos
            edge_prob = compute_alpha_decay_jit(initial_accuracy, elapsed_bars, ttl_bars)
        except ImportError:
            # Fallback en caso de que math_kernel no cargue
            import math
            if ttl_bars <= 0.0:
                return 0.0
            lam = 1.0 / ttl_bars
            edge_prob = initial_accuracy * math.exp(-lam * elapsed_bars)
            
        return edge_prob


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
