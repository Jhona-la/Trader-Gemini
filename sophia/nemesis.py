"""
⚔️ PROTOCOLO NÉMESIS-RETROSPECCIÓN: Post-Mortem Estadístico y Re-Calibración

QUÉ: Motor de autopsia técnica y psicológica de cada posición cerrada.
POR QUÉ: Sin retrospección, el bot repite errores. Compara la 'Intención Inicial'
     (SOPHIA) contra la 'Realidad del Mercado' para detectar sesgos, recalibrar
     confianza y penalizar genes que producen trades pobres.
PARA QUÉ: Cierre del bucle de aprendizaje: Error → Diagnóstico → Ajuste → Mejora.
CÓMO: NemesisEngine.full_autopsy() orquesta 8 analizadores especializados:
     §I BrierAudit → §II TemporalForensics → §III BiasAudit → §IV FeedbackLoop.
CUÁNDO: Se invoca inmediatamente después de compute_post_mortem() en Portfolio.update_fill().
DÓNDE: sophia/nemesis.py
QUIÉN: NemesisEngine (facade), invocado por Portfolio._sophia_post_mortem_check().
"""

import math
import os
import time
import json
from collections import deque
from dataclasses import dataclass, field
from utils.metrics_exporter import metrics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from utils.logger import logger
from sophia.axioma import AxiomDiagnoser, FallaBase  # CRITERIO-AXIOMA Protocol


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class NemesisReport:
    """
    📋 Complete autopsy result for a single closed trade.

    QUÉ: Reporte completo de la retrospección post-mortem.
    POR QUÉ: Centraliza los diagnósticos de TODAS las dimensiones:
         probabilística, temporal, sesgos, eficiencia, slippage.
    PARA QUÉ: Trazabilidad completa + alimentación del feedback loop.
    """
    trade_id: str
    symbol: str
    direction: str

    # §I: Calibración Probabilística
    brier_score: float
    brier_bucket: str                   # "0-50%", "50-70%", "70-85%", "85-100%"
    overconfidence_active: bool         # True si penalty activo
    overconfidence_penalty_factor: float # 1.0 = sin penalización
    false_positive: bool                # True si P>85% y LOSS
    false_positive_reason: str          # TAIL_EVENT / VOLATILITY_SPIKE / SIGNAL_DECAY / UNKNOWN / N/A

    # §II: Forense Temporal
    time_deviation_ratio: float         # actual / predicted
    time_deviation_class: str           # PRECISE / ALPHA_LEAK / VOLATILITY_STALL / PREMATURE_EXIT
    efficiency_factor: float            # PnL / duration_mins
    efficiency_class: str               # EXCELLENT / GOOD / POOR / CAPITAL_TRAPPED

    # §III: Sesgos Psico-Digitales
    bias_detected: str                  # NONE / PREMATURE_PROFIT / LOSS_HOLDING
    disposition_score: float            # rolling avg(loss_hold) / avg(win_hold)
    shap_accuracy: float                # feature hit rate (0-1)
    shap_mismatches: List[str]          # features that missed
    slippage_pct: float                 # |fill - trigger| / trigger × 100
    slippage_alert: bool                # True si slippage > threshold

    # §IV: Feedback
    gene_penalty: float                 # amount deducted from fitness
    gene_flagged: bool                  # True if genotype flagged for replacement

    # CRITERIO-AXIOMA: Root Cause Diagnosis
    falla_base: str                     # CALCULO, PROFUNDIDAD, TESIS_DECAY, NO_FALLA
    residual_pct: float                 # Target vs Realized dev
    reco_accion: str                    # Accion dictada por el oráculo

    # Narrative
    manifest: str                       # Auto-critique full text
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'brier_score': round(self.brier_score, 4),
            'brier_bucket': self.brier_bucket,
            'overconfidence_active': self.overconfidence_active,
            'overconfidence_penalty_factor': round(self.overconfidence_penalty_factor, 3),
            'false_positive': self.false_positive,
            'false_positive_reason': self.false_positive_reason,
            'time_deviation_ratio': round(self.time_deviation_ratio, 3),
            'time_deviation_class': self.time_deviation_class,
            'efficiency_factor': round(self.efficiency_factor, 6),
            'efficiency_class': self.efficiency_class,
            'bias_detected': self.bias_detected,
            'disposition_score': round(self.disposition_score, 3),
            'shap_accuracy': round(self.shap_accuracy, 3),
            'shap_mismatches': self.shap_mismatches,
            'slippage_pct': round(self.slippage_pct, 4),
            'slippage_alert': self.slippage_alert,
            'gene_penalty': round(self.gene_penalty, 4),
            'gene_flagged': self.gene_flagged,
            'falla_base': self.falla_base,
            'residual_pct': round(self.residual_pct, 4),
            'reco_accion': self.reco_accion,
            'manifest': self.manifest,
            'timestamp': self.timestamp,
        }

    def to_log_line(self) -> str:
        bias_str = f"Sesgo: {self.bias_detected}" if self.bias_detected != "NONE" else "Sesgo: NINGUNO"
        return (
            f"[NÉMESIS] Brier={self.brier_score:.3f} ({self.brier_bucket}) | "
            f"T_dev={self.time_deviation_ratio:.2f}x ({self.time_deviation_class}) | "
            f"E={self.efficiency_factor:.4f}$/min ({self.efficiency_class}) | "
            f"{bias_str} | Slip={self.slippage_pct:.3f}%"
            f"{' ⚠️OC' if self.overconfidence_active else ''}"
            f"{' 🚨FP' if self.false_positive else ''}"
            f" | AXIOM:[Falla:{self.falla_base} Dev:{self.residual_pct:.4f} 👉 {self.reco_accion}]"
        )


# ============================================================
# §I.2: OVERCONFIDENCE PENALIZER
# ============================================================

class OverconfidencePenalizer:
    """
    ⚠️ NÉMESIS §I.2: Adaptive Confidence Threshold Penalizer.

    QUÉ: Si el Brier Score promedio sube, CASTIGA la confianza del bot
         incrementando los umbrales de entrada para las próximas 10 operaciones.
    POR QUÉ: Un bot con Brier > 0.20 está sobreestimando su capacidad predictiva.
         Necesita ser más conservador hasta que su calibración mejore.
    PARA QUÉ: Auto-regulación. El bot se "auto-frena" cuando está descalibrado.
    CÓMO: penalty_factor = 1.0 + max(0, (avg_brier - 0.15) × 3.0)
         P(Win)_adjusted = P(Win) / penalty_factor
    CUÁNDO: Evaluado después de cada trade. Si se activa, dura 10 trades.
    """

    def __init__(self, lookback: int = 10, brier_threshold: float = 0.20):
        self.lookback = lookback
        self.brier_threshold = brier_threshold
        self.recent_briers: deque = deque(maxlen=lookback)
        self.penalty_trades_remaining = 0
        self.current_penalty_factor = 1.0

    def record_brier(self, brier: float):
        """Record a new Brier score and check if penalty should activate."""
        self.recent_briers.append(brier)

        # Decrement penalty counter
        if self.penalty_trades_remaining > 0:
            self.penalty_trades_remaining -= 1
            if self.penalty_trades_remaining == 0:
                self.current_penalty_factor = 1.0
                logger.info("✅ [NÉMESIS] Penalización por exceso de confianza EXPIRADA. Umbrales restaurados.")

        # Check if we need to activate/renew penalty
        if len(self.recent_briers) >= 5:
            avg_brier = sum(self.recent_briers) / len(self.recent_briers)
            if avg_brier > self.brier_threshold:
                self.current_penalty_factor = 1.0 + max(0, (avg_brier - 0.15) * 3.0)
                self.penalty_trades_remaining = self.lookback
                logger.warning(
                    f"⚠️ [NÉMESIS] Exceso de confianza detectado! "
                    f"Brier_avg={avg_brier:.3f}. "
                    f"Penalty factor={self.current_penalty_factor:.2f}x "
                    f"para próximos {self.lookback} trades."
                )

    def get_penalty_factor(self) -> float:
        """Returns current penalty factor (1.0 = no penalty)."""
        return self.current_penalty_factor

    def is_active(self) -> bool:
        return self.penalty_trades_remaining > 0

    def adjust_probability(self, p_win: float) -> float:
        """Apply penalty: scale down confidence."""
        if self.current_penalty_factor <= 1.0:
            return p_win
        return max(0.01, p_win / self.current_penalty_factor)


# ============================================================
# §I.3: FALSE POSITIVE ANALYZER
# ============================================================

class FalsePositiveAnalyzer:
    """
    🚨 NÉMESIS §I.3: High-Confidence Failure Diagnostics.

    QUÉ: Identifica si entradas con >85% de probabilidad fallaron.
    POR QUÉ: Un fallo cuando el bot estaba "muy seguro" es más grave que
         uno con 55%. Puede indicar cisnes negros o mala lectura del mercado.
    PARA QUÉ: Clasificar la causa raíz:
         TAIL_EVENT → distribución fat-tailed ignorada
         VOLATILITY_SPIKE → GARCH vol duplicó durante el trade
         SIGNAL_DECAY → trade duró >2× la tesis de alfa
         UNKNOWN → causas no determinables
    CÓMO: Al cerrar, si P_pred > 0.85 AND outcome == LOSS, analiza el SOPHIA report.
    """

    HIGH_CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, fp_window: int = 20):
        self.fp_window = fp_window
        self.recent_high_conf: deque = deque(maxlen=fp_window)  # (was_loss, reason)

    def analyze(
        self,
        predicted_prob: float,
        actual_pnl: float,
        sophia_report: Dict,
        actual_duration_secs: float,
    ) -> Tuple[bool, str]:
        """
        Check if this trade is a false positive and classify the reason.

        Returns:
            (is_false_positive, reason)
        """
        is_loss = actual_pnl <= 0
        is_high_conf = predicted_prob >= self.HIGH_CONFIDENCE_THRESHOLD

        if not is_high_conf:
            return False, "N/A"

        if not is_loss:
            self.recent_high_conf.append((False, "N/A"))
            return False, "N/A"

        # High confidence + Loss = FALSE POSITIVE
        reason = self._classify_reason(sophia_report, actual_duration_secs)

        self.recent_high_conf.append((True, reason))
        logger.warning(
            f"🚨 [NÉMESIS] FALSE POSITIVE: P={predicted_prob:.0%} → LOSS. "
            f"Razón: {reason}"
        )

        return True, reason

    def _classify_reason(self, sophia: Dict, duration_secs: float) -> str:
        """Classify the root cause of a high-confidence failure."""
        # Check for tail event
        excess_kurt = sophia.get('excess_kurtosis', 0.0)
        if excess_kurt > 3.0:
            return "TAIL_EVENT"

        # Check for signal decay
        alpha_threshold_mins = sophia.get('alpha_decay_threshold_mins', 999)
        actual_mins = duration_secs / 60.0
        if actual_mins > 2.0 * alpha_threshold_mins and alpha_threshold_mins > 0:
            return "SIGNAL_DECAY"

        # Check GARCH volatility inconsistency (if pre-trade vol was low but
        # the trade moved violently, that's a volatility spike)
        tail_warning = sophia.get('tail_risk_warning', False)
        if tail_warning:
            return "VOLATILITY_SPIKE"

        return "UNKNOWN"

    def get_fp_rate(self) -> float:
        """Rolling false positive rate among high-confidence trades."""
        if not self.recent_high_conf:
            return 0.0
        fp_count = sum(1 for was_loss, _ in self.recent_high_conf if was_loss)
        return fp_count / len(self.recent_high_conf)

    def is_critical(self) -> bool:
        """True if FP rate exceeds 25%."""
        return self.get_fp_rate() > 0.25


# ============================================================
# §II.4: TIME DEVIATION ANALYZER
# ============================================================

class TimeDeviationAnalyzer:
    """
    ⏱️ NÉMESIS §II.4: Temporal Prediction Accuracy Auditor.

    QUÉ: Compara cuánto DURÓ el trade vs cuánto PREDIJO SOPHIA que duraría.
    POR QUÉ: Si el bot dice "10 min" y el trade dura 60 min, hay un problema:
         o el precio se estancó (volatility stall) o se filtró el alfa (alpha leak).
    PARA QUÉ: Detectar ineficiencias temporales y ajustar el SurvivalEstimator.
    CÓMO: ratio = actual / predicted → clasificar.
    """

    def __init__(self, rolling_window: int = 50):
        self.rolling_ratios: deque = deque(maxlen=rolling_window)

    def analyze(
        self,
        actual_duration_mins: float,
        predicted_duration_mins: float,
        actual_pnl: float,
    ) -> Tuple[float, str]:
        """
        Compute time deviation ratio and classify.

        Returns:
            (ratio, classification)
        """
        if predicted_duration_mins <= 0:
            predicted_duration_mins = 10.0  # Default fallback

        ratio = actual_duration_mins / predicted_duration_mins
        self.rolling_ratios.append(ratio)

        # Classify
        if 0.5 <= ratio <= 2.0:
            classification = "PRECISE"
        elif ratio > 2.0 and actual_pnl > 0:
            classification = "ALPHA_LEAK"
        elif ratio > 2.0 and actual_pnl <= 0:
            classification = "VOLATILITY_STALL"
        elif ratio < 0.5:
            classification = "PREMATURE_EXIT"
        else:
            classification = "PRECISE"

        return round(ratio, 3), classification

    def get_avg_ratio(self) -> float:
        """Rolling average time deviation ratio."""
        if not self.rolling_ratios:
            return 1.0
        return sum(self.rolling_ratios) / len(self.rolling_ratios)

    def generate_narrative(
        self,
        ratio: float,
        classification: str,
        actual_mins: float,
        predicted_mins: float,
    ) -> str:
        """Human-readable time deviation narrative."""
        narratives = {
            "PRECISE": f"Duración real {actual_mins:.1f}min ≈ estimado {predicted_mins:.1f}min. Predicción temporal PRECISA.",
            "ALPHA_LEAK": f"Duración real {actual_mins:.1f}min vs estimado {predicted_mins:.1f}min → FUGA DE ALFA: ganó pero tardó {ratio:.1f}x lo esperado.",
            "VOLATILITY_STALL": f"Duración real {actual_mins:.1f}min vs estimado {predicted_mins:.1f}min → ESTANCAMIENTO: perdió tras esperar {ratio:.1f}x lo previsto.",
            "PREMATURE_EXIT": f"Duración real {actual_mins:.1f}min vs estimado {predicted_mins:.1f}min → SALIDA PREMATURA: cerró en {ratio:.1f}x del horizonte.",
        }
        return narratives.get(classification, f"Ratio temporal: {ratio:.2f}x")


# ============================================================
# §II.5: EFFICIENCY CALCULATOR
# ============================================================

class EfficiencyCalculator:
    """
    📊 NÉMESIS §II.5: Capital Efficiency Factor.

    QUÉ: E = PnL / Tiempo de Exposición ($/min).
    POR QUÉ: Un trade que gana $0.05 en 60 minutos es INEFICIENTE.
         Ese capital de $13 USDT estuvo "atrapado" mientras otros trades
         podían haber generado más retorno.
    PARA QUÉ: Detectar "capital traps" y mejorar la selección de entradas.
    CÓMO: E = PnL / duration_mins. E_norm = E / avg_E. Clasificar.
    """

    def __init__(self, rolling_window: int = 50):
        self.rolling_efficiencies: deque = deque(maxlen=rolling_window)

    def compute(
        self,
        actual_pnl: float,
        duration_mins: float,
    ) -> Tuple[float, float, str]:
        """
        Compute efficiency factor.

        Returns:
            (efficiency, normalized_efficiency, classification)
        """
        if duration_mins <= 0:
            duration_mins = 1.0  # Minimum 1 minute

        efficiency = actual_pnl / duration_mins
        self.rolling_efficiencies.append(efficiency)

        # Normalize against rolling average
        avg_e = self._get_avg_efficiency()
        if abs(avg_e) > 1e-8:
            e_normalized = efficiency / abs(avg_e)
        else:
            e_normalized = 1.0 if efficiency >= 0 else -1.0

        # Classify
        if e_normalized > 1.5:
            classification = "EXCELLENT"
        elif e_normalized >= 0.8:
            classification = "GOOD"
        elif actual_pnl > 0 and e_normalized < 0.3:
            classification = "CAPITAL_TRAPPED"
        else:
            classification = "POOR"

        return round(efficiency, 6), round(e_normalized, 3), classification

    def _get_avg_efficiency(self) -> float:
        if not self.rolling_efficiencies:
            return 0.0
        return sum(self.rolling_efficiencies) / len(self.rolling_efficiencies)


# ============================================================
# §III.6: DISPOSITION BIAS DETECTOR
# ============================================================

class DispositionBiasDetector:
    """
    🧠 NÉMESIS §III.6: Disposition Effect Auditor.

    QUÉ: Detecta si el bot cierra prematuramente las ganancias (miedo) o
         mantiene las pérdidas más allá del tiempo proyectado (esperanza).
    POR QUÉ: El "sesgo de disposición" es el error cognitivo #1 en trading:
         vender ganadores rápido + mantener perdedores demasiado.
         Si un bot lo hace, su código tiene el equivalente digital de un sesgo humano.
    PARA QUÉ: Diagnosticar si la lógica de SL/TP y salidas necesita ajuste.
    CÓMO:
         WINs: win_hold_ratio = actual_duration / time_to_tp_estimate
               Si < 0.5 → PREMATURE_PROFIT_TAKING
         LOSSes: loss_hold_ratio = actual_duration / time_to_sl_estimate
               Si > 2.0 → LOSS_HOLDING
         disposition_score = avg(loss_hold) / avg(win_hold)
               Si > 1.5 → DISPOSITION EFFECT CONFIRMED
    """

    def __init__(self, rolling_window: int = 30):
        self.win_hold_ratios: deque = deque(maxlen=rolling_window)
        self.loss_hold_ratios: deque = deque(maxlen=rolling_window)

    def analyze(
        self,
        actual_pnl: float,
        actual_duration_mins: float,
        predicted_tp_mins: float,
        predicted_sl_mins: float,
    ) -> Tuple[str, float]:
        """
        Detect disposition bias for this trade.

        Returns:
            (bias_type, disposition_score)
        """
        is_win = actual_pnl > 0
        bias = "NONE"

        if is_win:
            if predicted_tp_mins > 0:
                ratio = actual_duration_mins / predicted_tp_mins
                self.win_hold_ratios.append(ratio)
                if ratio < 0.5:
                    bias = "PREMATURE_PROFIT"
        else:
            if predicted_sl_mins > 0:
                ratio = actual_duration_mins / predicted_sl_mins
                self.loss_hold_ratios.append(ratio)
                if ratio > 2.0:
                    bias = "LOSS_HOLDING"

        # Compute disposition score
        disposition_score = self._get_disposition_score()

        return bias, round(disposition_score, 3)

    def _get_disposition_score(self) -> float:
        """
        disposition_score = avg(loss_hold_ratios) / avg(win_hold_ratios).
        > 1.5 = confirmed disposition effect.
        """
        if not self.win_hold_ratios or not self.loss_hold_ratios:
            return 1.0  # Neutral

        avg_win = sum(self.win_hold_ratios) / len(self.win_hold_ratios)
        avg_loss = sum(self.loss_hold_ratios) / len(self.loss_hold_ratios)

        if avg_win < 0.01:
            return 1.0
        return avg_loss / avg_win

    def has_confirmed_bias(self) -> bool:
        return self._get_disposition_score() > 1.5


# ============================================================
# §III.7: POST-TRADE SHAP COMPARATOR
# ============================================================

class PostTradeSHAPComparator:
    """
    📊 NÉMESIS §III.7: Feature Attribution Accuracy Tracker.

    QUÉ: Compara qué indicadores fueron importantes AL ENTRAR vs cuáles
         realmente movieron el precio DURANTE el trade.
    POR QUÉ: Si el RSI fue el factor #1 en la entrada pero el mercado respondió
         al volumen, el bot sobreestimó al RSI y subestimó al volumen.
    PARA QUÉ: Ajustar pesos de features en la "Trinidad Omega".
    CÓMO:
         1. Entry SHAP: top_features from SOPHIA (before trade)
         2. Post-trade: did the price move in the direction the feature predicted?
            - RSI oversold → price went UP? → HIT
            - RSI oversold → price went DOWN? → MISS
         3. SHAP_accuracy = hits / (hits + misses)
    """

    def __init__(self, rolling_window: int = 50):
        # Per-feature tracking: {feature_name: deque of booleans (hit=True)}
        self.feature_hits: Dict[str, deque] = {}
        self.rolling_window = rolling_window

    # Feature-to-direction mapping: how each feature predicts direction
    FEATURE_DIRECTION = {
        'RSI': 'contrarian',        # RSI oversold → expect UP
        'BB Position': 'contrarian', # BB low → expect UP
        'ADX': 'neutral',           # Strength, not direction
        'Volume Ratio': 'confirming', # High vol confirms signal direction
        'MTF Confluence': 'confirming',
        'MACD Histogram': 'momentum', # MACD+ → UP
        'Trend Alignment': 'confirming',
        'ATR %': 'neutral',
    }

    def analyze(
        self,
        top_features: List[Dict],
        actual_pnl: float,
        direction: str,
    ) -> Tuple[float, List[str]]:
        """
        Compare pre-trade feature attributions against actual outcome.

        Returns:
            (shap_accuracy, list_of_mismatched_features)
        """
        if not top_features:
            return 1.0, []

        hits = 0
        misses = 0
        mismatches = []
        trade_succeeded = actual_pnl > 0

        for feat in top_features:
            name = feat.get('feature', '')
            contribution = feat.get('contribution', 0.0)

            # A feature "hit" if:
            # - Its contribution was positive AND the trade won, OR
            # - Its contribution was negative AND the trade lost
            # (i.e., the feature correctly predicted the outcome)
            feature_predicted_success = contribution > 0
            is_hit = feature_predicted_success == trade_succeeded

            if is_hit:
                hits += 1
            else:
                misses += 1
                mismatches.append(name)

            # Track per-feature accuracy
            if name not in self.feature_hits:
                self.feature_hits[name] = deque(maxlen=self.rolling_window)
            self.feature_hits[name].append(is_hit)

        total = hits + misses
        accuracy = hits / total if total > 0 else 1.0

        return round(accuracy, 3), mismatches

    def get_feature_accuracy(self, feature_name: str) -> float:
        """Rolling accuracy for a specific feature."""
        hits = self.feature_hits.get(feature_name, deque())
        if not hits:
            return 1.0
        return sum(1 for h in hits if h) / len(hits)

    def get_underperforming_features(self, threshold: float = 0.40) -> List[str]:
        """Features with accuracy below threshold → candidate for weight reduction."""
        weak = []
        for name, hits in self.feature_hits.items():
            if len(hits) >= 10:  # Need enough data
                acc = sum(1 for h in hits if h) / len(hits)
                if acc < threshold:
                    weak.append(name)
        return weak


# ============================================================
# §III.8: SLIPPAGE FORENSICS
# ============================================================

class SlippageForensics:
    """
    📉 NÉMESIS §III.8: Trigger vs Fill Price Forensics.

    QUÉ: Cuantifica la diferencia entre el precio al generar la señal
         y el precio real de ejecución.
    POR QUÉ: En scalping con $13 USDT, un slippage de 0.05% = $0.0065 que
         se come directamente el PnL. Con trades que buscan 0.5-1.5% de TP,
         un slippage de 0.1% es CATASTRÓFICO (comería 10-20% del profit).
    PARA QUÉ: Si el slippage es recurrente, recalibrar el motor de ejecución.
    CÓMO: slippage_pct = |fill_price - trigger_price| / trigger_price × 100
    """

    def __init__(self, rolling_window: int = 100, alert_threshold_pct: float = 0.05):
        self.rolling_slippages: deque = deque(maxlen=rolling_window)
        self.alert_threshold = alert_threshold_pct

    def compute(
        self,
        trigger_price: float,
        fill_price: float,
    ) -> Tuple[float, bool]:
        """
        Compute slippage percentage and check alert threshold.

        Returns:
            (slippage_pct, is_alert)
        """
        if trigger_price <= 0:
            return 0.0, False

        slippage = abs(fill_price - trigger_price) / trigger_price * 100.0
        self.rolling_slippages.append(slippage)

        is_alert = self.get_avg_slippage() > self.alert_threshold

        return round(slippage, 4), is_alert

    def get_avg_slippage(self) -> float:
        if not self.rolling_slippages:
            return 0.0
        return sum(self.rolling_slippages) / len(self.rolling_slippages)

    def get_p95_slippage(self) -> float:
        if len(self.rolling_slippages) < 5:
            return 0.0
        sorted_s = sorted(self.rolling_slippages)
        idx = int(len(sorted_s) * 0.95)
        return sorted_s[min(idx, len(sorted_s) - 1)]

    def get_max_slippage(self) -> float:
        if not self.rolling_slippages:
            return 0.0
        return max(self.rolling_slippages)


# ============================================================
# §IV.9: GENE PENALIZER
# ============================================================

class GenePenalizer:
    """
    🧬 NÉMESIS §IV.9: Genetic Algorithm Fitness Penalty.

    QUÉ: Reduce el fitness_score del Genotype que produjo un trade con Brier pobre.
    POR QUÉ: Los genes que repetidamente hacen predicciones incorrectas deben ser
         marcados como "recesivos" para evitar que se repliquen en la flota.
    PARA QUÉ: Presión evolutiva hacia genes mejor calibrados.
    CÓMO: brier_penalty = brier_score × 0.5. Si 3+ consecutivas pobres → flag replacement.
    """

    POOR_BRIER_THRESHOLD = 0.30

    def __init__(self):
        self.consecutive_poor: Dict[str, int] = {}  # genotype_id → count

    def evaluate(
        self,
        brier_score: float,
        axioma: AxiomDiagnoser, # CRITERIO-AXIOMA integration
        genotype_id: str = "default",
    ) -> Tuple[float, bool]:
        """
        Compute gene penalty and check for replacement flag.
        Returns: (penalty_amount, should_flag_for_replacement)
        """
        # CRITERIO-AXIOMA: Safe-Mode on Math Calculation Errors
        if axioma.tipo_falla == FallaBase.CALCULO:
            logger.error(f"☠️ [NÉMESIS] Error de Cálculo detectado (Slippage/Precisión). Posible Safe-Mode trigger.")
            # We don't penalize the genotype for engine bugs, but we alert loudly.
            return 0.0, False
            
        # CRITERIO-AXIOMA: Severe penalty on Thesis Decay
        if axioma.tipo_falla == FallaBase.TESIS_DECAY:
            self.consecutive_poor[genotype_id] = self.consecutive_poor.get(genotype_id, 0) + 1
            # Double the Brier penalty because the model fundamentally failed the macro regime
            penalty = brier_score * 1.0 
            should_flag = self.consecutive_poor[genotype_id] >= 2 # Stricter: 2 strikes out
            
            if should_flag:
                logger.warning(
                    f"🧬 [NÉMESIS] Genotype '{genotype_id}' flagged for REPLACEMENT: "
                    f"Failures on TESIS_DECAY."
                )
            return round(penalty, 4), should_flag
            
        # CRITERIO-AXIOMA: Premium on NO_FALLA
        if axioma.tipo_falla == FallaBase.NO_FALLA:
            # Reward context: we return a negative penalty (bonus)
            self.consecutive_poor[genotype_id] = 0
            return -0.5, False # -0.5 is a fitness bonus

        # Standard Brier Evaluation
        if brier_score <= self.POOR_BRIER_THRESHOLD:
            # Good trade — reset consecutive counter
            self.consecutive_poor[genotype_id] = 0
            return 0.0, False

        # Poor trade
        penalty = brier_score * 0.5
        self.consecutive_poor[genotype_id] = self.consecutive_poor.get(genotype_id, 0) + 1

        should_flag = self.consecutive_poor[genotype_id] >= 3

        if should_flag:
            logger.warning(
                f"🧬 [NÉMESIS] Genotype '{genotype_id}' flagged for REPLACEMENT: "
                f"{self.consecutive_poor[genotype_id]} consecutive poor trades."
            )

        return round(penalty, 4), should_flag


# ============================================================
# §IV.10: MANIFEST WRITER
# ============================================================

class ManifestWriter:
    """
    📝 NÉMESIS §IV.10: Auto-Critique Narrative Generator.

    QUÉ: Genera una conclusión en lenguaje humano de lo que falló (o acertó).
    POR QUÉ: "Pensé que ganaría por X, pero perdí por Y. Error de calibración
         detectado en el módulo de Volatilidad. Ajustando umbrales."
    PARA QUÉ: Trazabilidad humana + debugging + lecciones automáticas.
    CÓMO: Template engine con datos del NemesisReport.
    """

    LOG_DIR = os.path.join("sophia", "nemesis_logs")

    @staticmethod
    def generate_manifest(
        trade_id: str,
        symbol: str,
        direction: str,
        predicted_prob: float,
        actual_pnl: float,
        brier_score: float,
        time_deviation_class: str,
        efficiency_class: str,
        bias_detected: str,
        false_positive_reason: str,
        shap_mismatches: List[str],
        overconfidence_active: bool,
        penalty_factor: float,
        axioma: AxiomDiagnoser,
        gene_penalty: float,
    ) -> str:
        """Generate the full auto-critique manifest."""
        outcome = "gané" if actual_pnl > 0 else "perdí"
        outcome_emoji = "✅" if actual_pnl > 0 else "❌"

        # Build reason chain
        reasons = []

        if false_positive_reason == "TAIL_EVENT":
            reasons.append("evento de cola gruesa (cisne negro)")
        elif false_positive_reason == "VOLATILITY_SPIKE":
            reasons.append("spike de volatilidad no anticipado")
        elif false_positive_reason == "SIGNAL_DECAY":
            reasons.append("la señal expiró antes de alcanzar el objetivo")

        if time_deviation_class == "ALPHA_LEAK":
            reasons.append("fuga de alfa — el movimiento fue más lento que lo predicho")
        elif time_deviation_class == "VOLATILITY_STALL":
            reasons.append("estancamiento por baja volatilidad")
        elif time_deviation_class == "PREMATURE_EXIT":
            reasons.append("salida prematura — el precio no tuvo tiempo de moverse")

        if bias_detected == "PREMATURE_PROFIT":
            reasons.append("sesgo de disposición: cerré ganancias prematuramente")
        elif bias_detected == "LOSS_HOLDING":
            reasons.append("sesgo de disposición: mantuve pérdidas por esperanza")

        if efficiency_class == "CAPITAL_TRAPPED":
            reasons.append("capital atrapado — baja eficiencia temporal")

        if shap_mismatches:
            mm = ", ".join(shap_mismatches[:3])
            reasons.append(f"features que fallaron: {mm}")

        reason_str = "; ".join(reasons) if reasons else "ejecución limpia"

        # Build adjustments
        adjustments = []
        if overconfidence_active:
            pct = (penalty_factor - 1.0) * 100
            adjustments.append(f"Umbrales de entrada +{pct:.0f}% por penalización de confianza")
        if shap_mismatches:
            adjustments.append(f"Reducir peso de {', '.join(shap_mismatches[:2])} en próximas señales")
        if bias_detected != "NONE":
            adjustments.append("Revisar lógica de SL/TP para corregir sesgo de disposición")
            
        # CRITERIO-AXIOMA Additions
        if axioma.tipo_falla != FallaBase.NO_FALLA:
            adjustments.append(f"Oráculo Axioma: {axioma.razon} ({axioma.accion_recomendada})")
        if gene_penalty > 0:
            adjustments.append(f"Restados {gene_penalty:.2f} pts al Genoma")
        elif gene_penalty < 0:
            adjustments.append(f"Sumados {-gene_penalty:.2f} pts al Genoma (Premio Axioma)")

        adjust_str = ". ".join(adjustments) if adjustments else "Sin ajustes necesarios"

        expectation = "ganaría" if actual_pnl > 0 else "perdería"

        manifest = (
            f"{outcome_emoji} Pensé que {expectation} con {predicted_prob:.0%} de probabilidad "
            f"en {symbol} {direction}, y {outcome} (PnL=${actual_pnl:+.4f}). "
            f"Brier={brier_score:.4f}. "
            f"Diagnóstico: {reason_str}. "
            f"Ajuste: {adjust_str}."
        )

        return manifest

    @staticmethod
    def persist_to_disk(trade_id: str, nemesis_report: Dict):
        """Save manifest as JSON to sophia/nemesis_logs/."""
        try:
            os.makedirs(ManifestWriter.LOG_DIR, exist_ok=True)
            filepath = os.path.join(
                ManifestWriter.LOG_DIR,
                f"nemesis_{trade_id[:12]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            )
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(nemesis_report, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"[NÉMESIS] Failed to persist manifest: {e}")


# ============================================================
# §I.1: BRIER BUCKET ANALYZER
# ============================================================

class BrierBucketAnalyzer:
    """
    📊 NÉMESIS §I.1: Enhanced Brier Score by Probability Bucket.

    QUÉ: Segmenta el Brier Score por rango de probabilidad predicha.
    POR QUÉ: Un Brier promedio de 0.15 puede esconder que el bot es EXCELENTE
         en trades de 50-70% pero PÉSIMO en trades de 85-100%. El análisis
         por bucket revela dónde está descalibrado.
    PARA QUÉ: Saber en qué rango de confianza confiar.
    """

    BUCKETS = {
        "0-50%": (0.0, 0.50),
        "50-70%": (0.50, 0.70),
        "70-85%": (0.70, 0.85),
        "85-100%": (0.85, 1.01),
    }

    def __init__(self):
        self.bucket_data: Dict[str, List[float]] = {b: [] for b in self.BUCKETS}

    def record(self, predicted_prob: float, brier_score: float):
        """Record a Brier score to the appropriate bucket."""
        for bucket_name, (lo, hi) in self.BUCKETS.items():
            if lo <= predicted_prob < hi:
                self.bucket_data[bucket_name].append(brier_score)
                return

    def get_bucket(self, predicted_prob: float) -> str:
        """Get which bucket a probability falls into."""
        for bucket_name, (lo, hi) in self.BUCKETS.items():
            if lo <= predicted_prob < hi:
                return bucket_name
        return "85-100%"

    def get_bucket_analysis(self) -> Dict[str, Dict]:
        """Returns per-bucket Brier mean and count."""
        analysis = {}
        for bucket_name, scores in self.bucket_data.items():
            if scores:
                analysis[bucket_name] = {
                    'mean_brier': round(sum(scores) / len(scores), 4),
                    'count': len(scores),
                    'worst': round(max(scores), 4),
                }
            else:
                analysis[bucket_name] = {
                    'mean_brier': 0.0,
                    'count': 0,
                    'worst': 0.0,
                }
        return analysis


# ============================================================
# FACADE: NEMESIS ENGINE
# ============================================================

class NemesisEngine:
    """
    ⚔️ NÉMESIS-RETROSPECCIÓN: Full Autopsy Facade.

    QUÉ: Punto de entrada único para ejecutar todos los diagnósticos post-mortem.
    POR QUÉ: Portfolio solo llama `nemesis.full_autopsy(...)` y recibe un
         NemesisReport completo con diagnóstico, narrativa y acciones correctivas.
    PARA QUÉ: Cierre del bucle de aprendizaje: Error → Diagnóstico → Ajuste → Mejora.
    CÓMO: Orquesta: BrierBucket → OverconfidencePenalizer → FalsePositiveAnalyzer →
         TimeDeviationAnalyzer → EfficiencyCalculator → DispositionBiasDetector →
         PostTradeSHAPComparator → SlippageForensics → GenePenalizer → ManifestWriter.
    CUÁNDO: Después de cada trade cerrado, invocado por Portfolio.
    DÓNDE: sophia/nemesis.py → NemesisEngine
    QUIÉN: Portfolio._sophia_post_mortem_check()
    """

    def __init__(self):
        # §I: Calibración
        self.brier_buckets = BrierBucketAnalyzer()
        self.overconfidence = OverconfidencePenalizer(lookback=10, brier_threshold=0.20)
        self.false_positives = FalsePositiveAnalyzer(fp_window=20)

        # §II: Temporal
        self.time_deviation = TimeDeviationAnalyzer(rolling_window=50)
        self.efficiency = EfficiencyCalculator(rolling_window=50)

        # §III: Sesgos
        self.disposition = DispositionBiasDetector(rolling_window=30)
        self.shap_comparator = PostTradeSHAPComparator(rolling_window=50)
        self.slippage = SlippageForensics(rolling_window=100, alert_threshold_pct=0.05)

        # §IV: Feedback
        self.gene_penalizer = GenePenalizer()

        logger.info("⚔️ [NÉMESIS] Retrospección engine initialized")

    def full_autopsy(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        predicted_prob: float,
        predicted_exit_mins: float,
        predicted_tp_mins: float,
        predicted_sl_mins: float,
        actual_pnl: float,
        actual_duration_mins: float,
        brier_score: float,
        sophia_report: Dict,
        top_features: List[Dict],
        trigger_price: float = 0.0,
        fill_price: float = 0.0,
        genotype_id: str = "default",
        persist_manifest: bool = True,
    ) -> NemesisReport:
        """
        Execute complete post-mortem autopsy.

        QUÉ: Ejecuta TODOS los diagnósticos §I-§IV en secuencia.
        POR QUÉ: Un solo método para obtener el diagnóstico completo.
        PARA QUÉ: Portfolio lo invoca con una sola llamada.

        Returns:
            NemesisReport with all fields populated.
        """
        ts = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        # ── §I.1: Brier Bucket ──
        brier_bucket = self.brier_buckets.get_bucket(predicted_prob)
        self.brier_buckets.record(predicted_prob, brier_score)

        # ── §I.2: Overconfidence ──
        self.overconfidence.record_brier(brier_score)
        oc_active = self.overconfidence.is_active()
        oc_factor = self.overconfidence.get_penalty_factor()

        # ── §I.3: False Positive ──
        is_fp, fp_reason = self.false_positives.analyze(
            predicted_prob, actual_pnl, sophia_report, actual_duration_mins * 60
        )

        # ── §II.4: Time Deviation ──
        time_ratio, time_class = self.time_deviation.analyze(
            actual_duration_mins, predicted_exit_mins, actual_pnl
        )

        # ── §II.5: Efficiency ──
        eff, eff_norm, eff_class = self.efficiency.compute(
            actual_pnl, actual_duration_mins
        )

        # ── §III.6: Disposition Bias ──
        bias, disp_score = self.disposition.analyze(
            actual_pnl, actual_duration_mins, predicted_tp_mins, predicted_sl_mins
        )

        # ── §III.7: Post-Trade SHAP ──
        shap_acc, shap_misses = self.shap_comparator.analyze(
            top_features, actual_pnl, direction
        )

        # ── §III.8: Slippage ──
        slip_pct, slip_alert = self.slippage.compute(trigger_price, fill_price)

        # ── Criterio-Axioma: Root Cause Diagnosis ──
        axioma = AxiomDiagnoser.diagnose(
            pnl=actual_pnl,
            direction=direction,
            trigger_price=trigger_price,
            fill_price=fill_price,
            sophia_report=sophia_report or {},
            duration_mins=actual_duration_mins
        )

        # ── §IV.9: Gene Penalty ──
        gene_pen, gene_flag = self.gene_penalizer.evaluate(brier_score, axioma, genotype_id)

        # ── §IV.10: Manifest ──
        manifest = ManifestWriter.generate_manifest(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            predicted_prob=predicted_prob,
            actual_pnl=actual_pnl,
            brier_score=brier_score,
            time_deviation_class=time_class,
            efficiency_class=eff_class,
            bias_detected=bias,
            false_positive_reason=fp_reason,
            shap_mismatches=shap_misses,
            overconfidence_active=oc_active,
            penalty_factor=oc_factor,
            axioma=axioma,
            gene_penalty=gene_pen
        )

        # Build report
        report = NemesisReport(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            brier_score=brier_score,
            brier_bucket=brier_bucket,
            overconfidence_active=oc_active,
            overconfidence_penalty_factor=oc_factor,
            false_positive=is_fp,
            false_positive_reason=fp_reason,
            time_deviation_ratio=time_ratio,
            time_deviation_class=time_class,
            efficiency_factor=eff,
            efficiency_class=eff_class,
            bias_detected=bias,
            disposition_score=disp_score,
            shap_accuracy=shap_acc,
            shap_mismatches=shap_misses,
            slippage_pct=slip_pct,
            slippage_alert=slip_alert,
            gene_penalty=gene_pen,
            gene_flagged=gene_flag,
            falla_base=axioma.tipo_falla.value,
            residual_pct=axioma.residual_pct,
            reco_accion=axioma.accion_recomendada,
            manifest=manifest,
            timestamp=ts,
        )

        # Log
        logger.info(f"   ⚔️ {report.to_log_line()}")
        logger.info(f"   📝 {manifest}")

        # ── SOPHIA-VIEW: Emit Prometheus Metrics & Loki JSON Log ──
        try:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            metrics.record_nemesis_autopsy(
                symbol=symbol,
                actual_pnl=actual_pnl,
                brier_score=brier_score,
                brier_bucket=brier_bucket,
                predicted_mins=predicted_exit_mins,
                actual_mins=actual_duration_mins,
                efficiency=eff,
                shap_accuracy=shap_acc,
                mismatches=shap_misses,
                overconfidence_active=oc_active,
                penalty_factor=oc_factor,
                gene_flagged=gene_flag,
                genotype_id=genotype_id,
                latency_ms=latency_ms
            )
            
            loki_payload = {
                "type": "nemesis_event",
                "trade_id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "predicted_mins": round(predicted_exit_mins, 2),
                "actual_mins": round(actual_duration_mins, 2),
                "brier_score": round(brier_score, 4),
                "latency_ms": round(latency_ms, 2),
                "axioma_falla": axioma.tipo_falla.value,
                "axioma_residual": round(axioma.residual_pct, 4)
            }
            logger.info(json.dumps(loki_payload))
        except Exception as e:
            logger.debug(f"[SOPHIA-VIEW] Metrics emission skipped: {e}")

        # Persist to disk
        if persist_manifest:
            ManifestWriter.persist_to_disk(trade_id, report.to_dict())

        return report

    def get_calibration_health(self) -> Dict:
        """Returns comprehensive calibration health summary."""
        return {
            'brier_buckets': self.brier_buckets.get_bucket_analysis(),
            'overconfidence': {
                'active': self.overconfidence.is_active(),
                'penalty_factor': self.overconfidence.get_penalty_factor(),
                'remaining_trades': self.overconfidence.penalty_trades_remaining,
            },
            'false_positive_rate': round(self.false_positives.get_fp_rate(), 3),
            'fp_critical': self.false_positives.is_critical(),
            'avg_time_deviation': round(self.time_deviation.get_avg_ratio(), 3),
            'disposition_bias': self.disposition.has_confirmed_bias(),
            'weak_features': self.shap_comparator.get_underperforming_features(),
            'avg_slippage_pct': round(self.slippage.get_avg_slippage(), 4),
            'p95_slippage_pct': round(self.slippage.get_p95_slippage(), 4),
        }
