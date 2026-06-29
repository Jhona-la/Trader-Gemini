"""
FeedbackProcessor — Auto-Conciencia del Sistema (CTOS)
═══════════════════════════════════════════════════════════════
QUÉ: Módulo que cierra el loop de retroalimentación del sistema.
  Captura el resultado de cada trade (Fill) y propaga aprendizajes.
POR QUÉ: Sin feedback loops, el sistema es estático. Las estrategias
  generan señales pero nunca saben si esas señales fueron buenas.
PARA QUÉ: Convertir el sistema en ADAPTATIVO:
  - Si hay slippage recurrente → ajustar política de ejecución.
  - Si una estrategia pierde en un régimen → bajar su peso.
  - Si exits cortan ganancias prematuramente → ajustar umbrales.
CÓMO: Escucha FillEvents, compara resultados reales vs esperados,
  y publica ajustes a través del EventBus.
CUÁNDO: Cada vez que se recibe un FillEvent con PnL cerrado.
DÓNDE: core/feedback_processor.py (este archivo)
QUIÉN: Quant Developer + Risk Manager (análisis) + Engine (receptor)
═══════════════════════════════════════════════════════════════
"""

import time
import logging
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Record of a completed trade for feedback analysis."""
    symbol: str
    strategy_id: str
    horizon: str
    direction: str
    pnl: float
    duration_seconds: float
    entry_price: float
    exit_price: float
    slippage_pct: float
    regime_at_entry: str
    timestamp: float = field(default_factory=time.time)


class FeedbackProcessor:
    """
    Cierra el loop: EXECUTION → RESULT → LEARNING → ADJUSTMENT.
    
    Bucles de retroalimentación implementados:
    1. SLIPPAGE TRACKER: Si slippage > threshold → ajustar ejecución
    2. STRATEGY WEIGHT ADJUSTER: WR por régimen → ajustar confianza
    3. EXIT QUALITY ANALYZER: ¿Los exits capturan MFE o cortan early?
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeedbackProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Rolling window of recent outcomes
        self._outcomes: deque = deque(maxlen=200)
        
        # Strategy performance by (strategy_id, regime)
        self._strategy_regime_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Format: { strategy_id: { regime: { 'wins': N, 'losses': N, 'total_pnl': X } } }
        
        # Slippage tracker per symbol
        self._slippage_history: Dict[str, deque] = {}
        
        # Adjustment signals emitted
        self._pending_adjustments: List[Dict[str, Any]] = []
        
        logger.info("🔄 [FEEDBACK] FeedbackProcessor initialized — closing the learning loop.")
        
        # CTOS PHASE 2: Auto-subscribe to EventBus FILLS channel
        # QUÉ: Suscribe este processor al canal FILLS del EventBus.
        # POR QUÉ: Ruta de aprendizaje dual — la directa (engine calls) +
        #   la desacoplada (EventBus publish/subscribe). Si una falla, la otra persiste.
        # CUÁNDO: Al inicializar el singleton.
        try:
            from core.event_bus import event_bus, EventChannel
            event_bus.subscribe(EventChannel.FILLS, self._on_fill_event)
            logger.info("🔄 [FEEDBACK] Subscribed to EventBus FILLS channel.")
        except Exception as e:
            logger.debug(f"[FEEDBACK] EventBus subscription skipped: {e}")
    
    def _on_fill_event(self, payload: Dict[str, Any]):
        """
        EventBus callback for FILLS channel.
        Receives fill data asynchronously from the EventBus.
        NOTE: This is a SECONDARY path. The PRIMARY path is the direct
        call from engine._process_fill_event(). This handler only logs
        receipt for telemetry — the actual analysis is done by the direct call.
        """
        symbol = payload['symbol']
        pnl = payload['pnl']
        logger.debug(f"🔄 [FEEDBACK-BUS] Received fill via EventBus: {symbol} PnL={pnl:+.4f}")
    
    # ════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ════════════════════════════════════════════════════════════════
    
    def process_fill_outcome(self, fill_event, pnl: float, 
                              strategy_id: str = "Unknown",
                              entry_price: float = 0.0,
                              duration_seconds: float = 0.0):
        """
        Called after a trade is closed to record outcome and trigger analysis.
        
        Args:
            fill_event: The FillEvent that closed the position
            pnl: Realized PnL of the trade
            strategy_id: Which strategy generated the signal
            entry_price: Original entry price
            duration_seconds: Time in trade
        """
        symbol = getattr(fill_event, 'symbol', 'UNKNOWN')
        horizon = getattr(fill_event, 'horizon', 'SCALPING')
        
        # Calculate slippage
        fill_price = getattr(fill_event, 'fill_price', 0.0)
        slippage_pct = 0.0
        if entry_price > 0 and fill_price > 0:
            slippage_pct = abs(fill_price - entry_price) / entry_price
        
        # Get current regime from SSOT
        try:
            from core.global_state import global_state
            regime = global_state.market_regime
        except Exception:
            regime = "UNKNOWN"
        
        # Record outcome
        outcome = TradeOutcome(
            symbol=symbol,
            strategy_id=strategy_id,
            horizon=horizon,
            direction=getattr(fill_event, 'side', 'UNKNOWN'),
            pnl=pnl,
            duration_seconds=duration_seconds,
            entry_price=entry_price,
            exit_price=fill_price,
            slippage_pct=slippage_pct,
            regime_at_entry=regime,
        )
        self._outcomes.append(outcome)
        
        # Run feedback analysis
        self._analyze_slippage(outcome)
        self._analyze_strategy_performance(outcome)
        self._analyze_exit_quality(outcome)
        
        logger.debug(
            f"🔄 [FEEDBACK] Recorded {symbol} {horizon} | "
            f"PnL: {pnl:+.4f} | Strategy: {strategy_id} | Regime: {regime}"
        )
    
    # ════════════════════════════════════════════════════════════════
    # FEEDBACK LOOP 1: SLIPPAGE ANALYSIS
    # ════════════════════════════════════════════════════════════════
    
    def _analyze_slippage(self, outcome: TradeOutcome):
        """
        If slippage is consistently high for a symbol, emit adjustment.
        """
        sym = outcome.symbol
        if sym not in self._slippage_history:
            self._slippage_history[sym] = deque(maxlen=20)
        
        self._slippage_history[sym].append(outcome.slippage_pct)
        
        if len(self._slippage_history[sym]) >= 5:
            avg_slippage = sum(self._slippage_history[sym]) / len(self._slippage_history[sym])
            
            # Threshold: if average slippage > 0.15% (3x maker fee), escalate
            if avg_slippage > 0.0015:
                self._pending_adjustments.append({
                    'type': 'SLIPPAGE_ALERT',
                    'symbol': sym,
                    'avg_slippage_pct': avg_slippage,
                    'recommendation': 'INCREASE_LIMIT_TOLERANCE' if avg_slippage < 0.003 else 'REDUCE_POSITION_SIZE',
                    'timestamp': time.time()
                })
                logger.warning(
                    f"⚠️ [FEEDBACK] High slippage on {sym}: avg {avg_slippage*100:.3f}% "
                    f"→ Recommending adjustment"
                )

    # ════════════════════════════════════════════════════════════════
    # FEEDBACK LOOP 2: STRATEGY PERFORMANCE BY REGIME
    # ════════════════════════════════════════════════════════════════
    
    def _analyze_strategy_performance(self, outcome: TradeOutcome):
        """
        Tracks win rate per (strategy, regime) pair.
        If a strategy loses >60% in a specific regime, emit weight reduction.
        """
        sid = outcome.strategy_id
        regime = outcome.regime_at_entry
        
        if sid not in self._strategy_regime_stats:
            self._strategy_regime_stats[sid] = {}
        if regime not in self._strategy_regime_stats[sid]:
            self._strategy_regime_stats[sid][regime] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        
        stats = self._strategy_regime_stats[sid][regime]
        stats['total_pnl'] += outcome.pnl
        if outcome.pnl > 0:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        total = stats['wins'] + stats['losses']
        if total >= 10:  # Minimum sample
            wr = stats['wins'] / total
            if wr < 0.40:
                self._pending_adjustments.append({
                    'type': 'STRATEGY_WEIGHT_REDUCTION',
                    'strategy_id': sid,
                    'regime': regime,
                    'win_rate': wr,
                    'total_trades': total,
                    'recommendation': 'REDUCE_WEIGHT_50PCT',
                    'timestamp': time.time()
                })
                logger.warning(
                    f"📉 [FEEDBACK] Strategy {sid} underperforming in {regime}: "
                    f"WR={wr:.1%} over {total} trades → Reducing weight"
                )
    
    # ════════════════════════════════════════════════════════════════
    # FEEDBACK LOOP 3: EXIT QUALITY
    # ════════════════════════════════════════════════════════════════
    
    def _analyze_exit_quality(self, outcome: TradeOutcome):
        """
        Checks if exits are systematically leaving money on the table
        or cutting winners too early (duration-based heuristic).
        """
        # Scalping exit too fast with small PnL = potential exit problem
        if outcome.horizon == 'SCALPING' and outcome.pnl > 0:
            if outcome.duration_seconds < 5 and outcome.pnl < 0.001:
                self._pending_adjustments.append({
                    'type': 'EXIT_TOO_EARLY',
                    'symbol': outcome.symbol,
                    'strategy_id': outcome.strategy_id,
                    'pnl': outcome.pnl,
                    'duration': outcome.duration_seconds,
                    'recommendation': 'WIDEN_TRAILING_STOP',
                    'timestamp': time.time()
                })
    
    # ════════════════════════════════════════════════════════════════
    # OUTPUT API
    # ════════════════════════════════════════════════════════════════
    
    def get_pending_adjustments(self) -> List[Dict[str, Any]]:
        """Returns and clears pending adjustment signals."""
        adjustments = self._pending_adjustments.copy()
        self._pending_adjustments.clear()
        return adjustments
    
    def get_strategy_stats(self, strategy_id: str) -> Dict[str, Dict[str, float]]:
        """Returns performance stats for a strategy across all regimes."""
        return self._strategy_regime_stats[strategy_id]
    
    def get_recent_outcomes(self, n: int = 20) -> List[TradeOutcome]:
        """Returns the N most recent trade outcomes."""
        return list(self._outcomes)[-n:]


# ════════════════════════════════════════════════════════════════
# SINGLETON
# ════════════════════════════════════════════════════════════════
feedback_processor = FeedbackProcessor()
