"""
📝 SOPHIA-INTELLIGENCE §4.2: Post-Mortem Comparator

QUÉ: Compara la intención pre-trade con el resultado real y calcula el Brier Score.
POR QUÉ: La calibración es CRÍTICA. Si el bot dice "78% de ganar" y solo gana
     el 40% de las veces que dice eso, el modelo está MAL CALIBRADO.
     El Brier Score mide exactamente esto.
PARA QUÉ: Detectar si la confianza del bot es confiable. Si Brier > 0.25,
     necesita recalibración urgente.
CÓMO: Almacena TradeIntent al abrir → Al cerrar: Brier = (p - outcome)²
     donde outcome = 1 si WIN, 0 si LOSS.
CUÁNDO: Al abrir posición (store intent) y al cerrar posición (compute score).
DÓNDE: sophia/post_mortem.py
QUIÉN: PostMortemComparator, invocado por Portfolio.update_fill().
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from utils.logger import logger
from sophia.narrative import NarrativeGenerator


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TradeIntent:
    """
    📋 Pre-trade intention record.
    
    QUÉ: Snapshot de lo que el bot "pensaba" al abrir la posición.
    POR QUÉ: Para comparar contra el resultado real (Post-Mortem).
    """
    trade_id: str
    symbol: str
    direction: str
    timestamp: str
    
    # SOPHIA predictions
    win_probability: float
    expected_exit_mins: float
    signal_strength: float
    top_features: List[Dict]
    entropy: float
    entropy_label: str
    narrative: str
    
    # NÉMESIS: Price at signal generation for slippage forensics
    trigger_price: float = 0.0
    
    # Full sophia report dict for deep analysis
    sophia_report: Dict = field(default_factory=dict)


@dataclass 
class PostMortemResult:
    """
    📊 Post-trade comparison result.
    
    QUÉ: Resultado de comparar predicción vs realidad.
    POR QUÉ: Datos para calibración rolling del modelo.
    """
    trade_id: str
    symbol: str
    direction: str
    
    # Prediction
    predicted_prob: float
    predicted_exit_mins: float
    
    # Reality
    actual_outcome: str  # "WIN" or "LOSS"
    actual_pnl: float
    actual_duration_mins: float
    
    # Score
    brier_score: float
    time_error_mins: float
    
    # Narrative
    narrative: str


# ============================================================
# POST-MORTEM COMPARATOR
# ============================================================

class PostMortemComparator:
    """
    📊 SOPHIA §4.2: Brier Score Calibration Tracker.
    
    QUÉ: Almacena intenciones pre-trade y las compara contra resultados reales.
    POR QUÉ: Sin post-mortem, no sabemos si la confianza del bot es calibrada.
         Brier Score = 0 → perfecto. Brier = 0.25 → igual que azar.
    PARA QUÉ: Rolling calibration → si Brier crece, el modelo necesita ajuste.
    CÓMO:
         1. store_intent(trade_id, sophia_report) → al abrir
         2. compute_post_mortem(trade_id, pnl, duration) → al cerrar
         3. get_calibration_status() → rolling health check
    CUÁNDO: Integrado en Portfolio.update_fill().
    DÓNDE: sophia/post_mortem.py → PostMortemComparator
    QUIÉN: Portfolio llama store al abrir, compute al cerrar.
    """
    
    def __init__(self, rolling_window: int = 100):
        """
        Args:
            rolling_window: Number of recent trades for rolling Brier Score.
        """
        self.pending_intents: Dict[str, TradeIntent] = {}  # trade_id → intent
        self.rolling_scores: deque = deque(maxlen=rolling_window)
        self.rolling_window = rolling_window
        self.total_trades = 0
        self.total_brier_sum = 0.0
        
        logger.info(f"📊 [SOPHIA] PostMortem initialized (window={rolling_window})")
    
    def store_intent(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        sophia_report: Dict,
        narrative: str = "",
        trigger_price: float = 0.0,
    ):
        """
        Store trade intention BEFORE opening position.
        
        QUÉ: Guarda el "pensamiento" del bot para comparación futura.
        CUÁNDO: Inmediatamente después de generar SophiaReport, antes de la orden.
        """
        intent = TradeIntent(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            timestamp=datetime.now(timezone.utc).isoformat(),
            win_probability=sophia_report.get('win_probability', 0.5),
            expected_exit_mins=sophia_report.get('expected_exit_mins', 10.0),
            signal_strength=sophia_report.get('signal_strength', 0.5),
            top_features=sophia_report.get('top_features', []),
            entropy=sophia_report.get('decision_entropy', 0.0),
            entropy_label=sophia_report.get('entropy_label', 'N/A'),
            narrative=narrative,
            trigger_price=trigger_price,
            sophia_report=sophia_report,
        )
        
        self.pending_intents[trade_id] = intent
        
        # Cleanup: remove intents older than 1000 entries to prevent memory leak
        if len(self.pending_intents) > 1000:
            # Remove oldest 100
            keys = list(self.pending_intents.keys())[:100]
            for k in keys:
                del self.pending_intents[k]
    
    def compute_post_mortem(
        self,
        trade_id: str,
        actual_pnl: float,
        duration_seconds: float = 0.0,
    ) -> Optional[PostMortemResult]:
        """
        Compute post-mortem comparison when position closes.
        
        QUÉ: Calcula Brier Score comparando predicción con resultado.
        POR QUÉ: Brier Score = (forecast - outcome)²
             outcome = 1 si PnL > 0,[OMITTED] 0 si PnL ≤ 0
             forecast = win_probability predicho
        PARA QUÉ: Saber si el bot "sabía" que iba a ganar/perder.
        
        Args:
            trade_id: UUID of the closed trade
            actual_pnl: Realized PnL in USD
            duration_seconds: How long the trade was open
            
        Returns:
            PostMortemResult or None if no matching intent found.
        """
        intent = self.pending_intents.pop(trade_id, None)
        
        if intent is None:
            # Try matching by partial trade_id
            for tid, i in list(self.pending_intents.items()):
                if tid[:8] == trade_id[:8]:
                    intent = self.pending_intents.pop(tid)
                    break
        
        if intent is None:
            return None
        
        # Determine outcome
        actual_outcome = "WIN" if actual_pnl > 0 else "LOSS"
        outcome_binary = 1.0 if actual_pnl > 0 else 0.0
        
        # Brier Score = (predicted_prob - actual_outcome)²
        brier = (intent.win_probability - outcome_binary) ** 2
        
        # Duration comparison
        duration_mins = duration_seconds / 60.0
        time_error = abs(duration_mins - intent.expected_exit_mins)
        
        # Generate narrative
        pm_narrative = NarrativeGenerator.generate_post_mortem_narrative(
            symbol=intent.symbol,
            direction=intent.direction,
            predicted_prob=intent.win_probability,
            actual_outcome=actual_outcome,
            brier_score=brier,
            pnl=actual_pnl,
            duration_mins=duration_mins,
            predicted_exit_mins=intent.expected_exit_mins,
        )
        
        result = PostMortemResult(
            trade_id=trade_id,
            symbol=intent.symbol,
            direction=intent.direction,
            predicted_prob=intent.win_probability,
            predicted_exit_mins=intent.expected_exit_mins,
            actual_outcome=actual_outcome,
            actual_pnl=actual_pnl,
            actual_duration_mins=duration_mins,
            brier_score=brier,
            time_error_mins=time_error,
            narrative=pm_narrative,
        )
        
        # Update rolling stats
        self.rolling_scores.append(brier)
        self.total_trades += 1
        self.total_brier_sum += brier
        
        # Log
        logger.info(f"   {pm_narrative}")
        
        return result
    
    def get_rolling_brier(self) -> float:
        """Returns rolling Brier Score over last N trades."""
        if not self.rolling_scores:
            return 0.0
        return sum(self.rolling_scores) / len(self.rolling_scores)
    
    def get_lifetime_brier(self) -> float:
        """Returns lifetime average Brier Score."""
        if self.total_trades == 0:
            return 0.0
        return self.total_brier_sum / self.total_trades
    
    def get_calibration_status(self) -> Dict:
        """
        Returns calibration health status.
        
        QUÉ: Resumen de la calibración del modelo.
        POR QUÉ: Si rolling Brier > 0.25, el modelo necesita recalibración.
        
        Returns:
            Dict with rolling_brier, lifetime_brier, status, n_trades.
        """
        rolling = self.get_rolling_brier()
        lifetime = self.get_lifetime_brier()
        
        if rolling < 0.05:
            status = "EXCELENTE 🎯"
        elif rolling < 0.10:
            status = "BUENA ✅"
        elif rolling < 0.20:
            status = "ACEPTABLE ⚠️"
        elif rolling < 0.25:
            status = "DEGRADADA ⚠️⚠️"
        else:
            status = "CRÍTICA ❌ — RECALIBRAR"
        
        # Trend detection
        trend = "STABLE"
        if len(self.rolling_scores) >= 20:
            first_half = list(self.rolling_scores)[:len(self.rolling_scores)//2]
            second_half = list(self.rolling_scores)[len(self.rolling_scores)//2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first * 1.3:
                trend = "DETERIORATING 📉"
            elif avg_second < avg_first * 0.7:
                trend = "IMPROVING 📈"
        
        return {
            'rolling_brier': round(rolling, 4),
            'lifetime_brier': round(lifetime, 4),
            'status': status,
            'trend': trend,
            'n_trades_rolling': len(self.rolling_scores),
            'n_trades_total': self.total_trades,
        }
    
    def get_summary_log(self) -> str:
        """Returns one-line calibration summary for logs."""
        cal = self.get_calibration_status()
        return (
            f"[SOPHIA Calibration] Brier={cal['rolling_brier']:.4f} "
            f"({cal['status']}) | Trend: {cal['trend']} | "
            f"N={cal['n_trades_total']}"
        )
