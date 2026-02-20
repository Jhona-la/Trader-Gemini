"""
🧠 SOPHIA-INTELLIGENCE §1-3: Core XAI Engine

QUÉ: Motor de Explainable AI que genera reportes de intención antes de cada trade.
POR QUÉ: Sin explicabilidad, el bot es una caja negra. Necesitamos saber POR QUÉ
     se toma cada decisión, con qué PROBABILIDAD de éxito, y CUÁNDO caduca la tesis.
PARA QUÉ: Calibración probabilística (Bayesian P(Win|Signal)), atribución de features
     (SHAP-like), horizonte temporal (Survival Analysis), y telemetría estadística
     (Shannon Entropy + Fat Tails).
CÓMO: BayesianCalibrator → FeatureAttributor → SurvivalEstimator → EntropyAnalyzer
     → TailRiskAnalyzer → SophiaReport.
CUÁNDO: Se invoca ANTES de emitir cada SignalEvent en technical.py:generate_signals().
DÓNDE: sophia/intelligence.py
QUIÉN: SophiaIntelligence (facade), invocado por HybridScalpingStrategy.
"""

import numpy as np
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from utils.logger import logger

try:
    from utils.math_kernel import bayesian_probability_jit, calculate_garch_jit
except ImportError:
    # Fallback if numba not available during testing
    def bayesian_probability_jit(s, t, v):
        prior = 0.5
        lr = 0.5 + s * 1.5
        post_odds = (prior / (1 - prior)) * lr
        return post_odds / (1 + post_odds)
    
    def calculate_garch_jit(returns, omega=1e-6, alpha=0.05, beta=0.90):
        n = len(returns)
        v = np.zeros(n)
        if n < 2:
            return v
        v[0] = np.var(returns)
        for t in range(1, n):
            v[t] = omega + alpha * returns[t-1]**2 + beta * v[t-1]
        return v


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class FeatureAttribution:
    """Single feature's contribution to the prediction."""
    feature: str
    value: float
    contribution: float  # δP: positive = helps, negative = hurts
    
    def to_dict(self) -> Dict:
        return {
            'feature': self.feature,
            'value': round(self.value, 4),
            'contribution': round(self.contribution, 4),
        }


@dataclass
class SurvivalEstimate:
    """Time-to-exit estimation for a position."""
    time_to_tp_mins: float
    time_to_sl_mins: float
    expected_exit_mins: float
    confidence_lower_mins: float  # -1σ
    confidence_upper_mins: float  # +1σ
    garch_volatility: float


@dataclass
class TailRiskMetrics:
    """Fat tail statistics for tick distribution."""
    excess_kurtosis: float
    skewness: float
    tail_ratio: float  # P(|ret| > 3σ) / P_normal
    has_fat_tails: bool
    sl_adjustment_factor: float  # 1.0 = no adjustment, >1.0 = widen SL


@dataclass
class SophiaReport:
    """
    📋 Complete XAI intention report for a single trade decision.
    
    QUÉ: Reporte completo de intención generado por SOPHIA antes de cada trade.
    POR QUÉ: Centraliza TODA la información explicativa en un solo objeto.
    PARA QUÉ: Inyectado en SignalEvent.metadata['sophia'] para trazabilidad completa.
    """
    # Block 1: Calibración Probabilística
    win_probability: float          # P(Win|Signal) ∈ [0,1]
    prior_win_rate: float           # Prior from historical data
    top_features: List[Dict]        # Top-5 SHAP-like attributions
    
    # Block 2: Horizonte Temporal
    expected_exit_mins: float       # E[T] in minutes
    time_to_tp_mins: float          # Estimated time to TP
    time_to_sl_mins: float          # Estimated time to SL
    alpha_decay_threshold_mins: float  # When signal expires
    
    # Block 3: Telemetría
    decision_entropy: float         # Shannon entropy H
    entropy_label: str              # "Alta Convicción" / "Moderada" / "Indeciso"
    excess_kurtosis: float          # Fat tail metric
    skewness: float                 # Distribution asymmetry
    tail_risk_warning: bool         # True if fat tails detected
    
    # Metadata
    timestamp: str = ""
    symbol: str = ""
    direction: str = ""
    signal_strength: float = 0.0
    
    def to_log_line(self) -> str:
        """Compact one-line format for logging."""
        top3 = ", ".join(f"{f['feature']}={f['contribution']:+.3f}" for f in self.top_features[:3])
        return (
            f"[SOPHIA] P(Win)={self.win_probability:.1%} | "
            f"E[T]={self.expected_exit_mins:.0f}min | "
            f"Top3=[{top3}] | "
            f"H={self.decision_entropy:.2f} ({self.entropy_label})"
            f"{' ⚠️FAT-TAILS' if self.tail_risk_warning else ''}"
        )
    
    def to_dict(self) -> Dict:
        """Full report as dictionary for SignalEvent.metadata."""
        return {
            'win_probability': round(self.win_probability, 4),
            'prior_win_rate': round(self.prior_win_rate, 4),
            'top_features': self.top_features,
            'expected_exit_mins': round(self.expected_exit_mins, 1),
            'time_to_tp_mins': round(self.time_to_tp_mins, 1),
            'time_to_sl_mins': round(self.time_to_sl_mins, 1),
            'alpha_decay_threshold_mins': round(self.alpha_decay_threshold_mins, 1),
            'decision_entropy': round(self.decision_entropy, 4),
            'entropy_label': self.entropy_label,
            'excess_kurtosis': round(self.excess_kurtosis, 4),
            'skewness': round(self.skewness, 4),
            'tail_risk_warning': self.tail_risk_warning,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'direction': self.direction,
            'signal_strength': round(self.signal_strength, 4),
        }


# ============================================================
# BLOCK 1: BAYESIAN CALIBRATOR
# ============================================================

class BayesianCalibrator:
    """
    🧠 SOPHIA §1.1: Bayesian Win Probability Estimator.
    
    QUÉ: Calcula P(Win|Signal) usando Teorema de Bayes con prior adaptativo.
    POR QUÉ: La probabilidad cruda de bayesian_probability_jit usa prior=0.5.
         Pero con historial de trades, podemos tener un prior más informado.
         Si el bot tiene 60% win rate histórico, el prior debería ser 0.6.
    PARA QUÉ: Probabilidad calibrada que refleja tanto el contexto actual del
         mercado (señal, tendencia, volatilidad) como el historial del bot.
    CÓMO: P(Win|Signal) = bayesian_probability_jit(signal, trend, vol_z)
         ajustado con prior adaptativo Beta(wins+α, losses+β).
    CUÁNDO: Antes de cada SignalEvent.
    DÓNDE: sophia/intelligence.py → BayesianCalibrator
    QUIÉN: Invocado por SophiaIntelligence.analyze().
    """
    
    def __init__(self, prior_alpha: int = 10, prior_beta: int = 10):
        """
        Args:
            prior_alpha: Beta distribution α (pseudo-wins). Higher = more confident prior.
            prior_beta: Beta distribution β (pseudo-losses).
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.observed_wins = 0
        self.observed_losses = 0
    
    def get_prior_win_rate(self) -> float:
        """
        Bayesian posterior win rate using Beta distribution.
        
        QUÉ: E[Beta(α + wins, β + losses)] = (α + wins) / (α + β + wins + losses)
        POR QUÉ: La media de la distribución Beta posterior ES la tasa de acierto
             bayesiana. Con pocos trades, domina el prior (0.5). Con muchos trades,
             domina la evidencia observada.
        """
        a = self.prior_alpha + self.observed_wins
        b = self.prior_beta + self.observed_losses
        return a / (a + b)
    
    def update_prior(self, won: bool):
        """Update posterior with new evidence (called after each trade closes)."""
        if won:
            self.observed_wins += 1
        else:
            self.observed_losses += 1
    
    def sync_from_risk_manager(self, wins: int, losses: int):
        """Bulk sync from RiskManager's historical counts."""
        self.observed_wins = wins
        self.observed_losses = losses
    
    def compute_posterior(
        self,
        signal_strength: float,
        trend_strength: float,
        volatility_z: float,
    ) -> float:
        """
        Compute calibrated P(Win|Signal).
        
        QUÉ: Combina el prior adaptativo con la evidencia del mercado actual.
        CÓMO:
            1. Calcula P(Win|context) via bayesian_probability_jit (likelihood-based)
            2. Ajusta con el prior histórico del bot:
               P_calibrated = w * P_jit + (1-w) * P_prior
               donde w = min(1.0, n_trades / 100) (más peso a JIT con más datos)
        """
        # 1. Context-based probability (from math_kernel)
        p_context = float(bayesian_probability_jit(
            signal_strength, trend_strength, volatility_z
        ))
        
        # 2. Historical prior
        p_prior = self.get_prior_win_rate()
        
        # 3. Weighted blend (more trades → trust context more)
        n_trades = self.observed_wins + self.observed_losses
        context_weight = min(1.0, n_trades / 100.0)
        
        # With few trades: lean on prior. With many: lean on context probability.
        # But always blend — pure context can be overconfident
        p_calibrated = 0.7 * p_context + 0.3 * p_prior
        
        # If we have enough history, adjust based on how well calibrated we've been
        if n_trades > 50:
            # Shrink toward prior if context predictions have been unreliable
            p_calibrated = context_weight * p_context + (1 - context_weight) * p_prior
        
        return np.clip(p_calibrated, 0.01, 0.99)


# ============================================================
# BLOCK 1.2: FEATURE ATTRIBUTION (SHAP-LIKE)
# ============================================================

class FeatureAttributor:
    """
    🧠 SOPHIA §1.2: Marginal Feature Attribution.
    
    QUÉ: Calcula la contribución marginal de cada feature a P(Win|Signal).
    POR QUÉ: SHAP requiere un modelo ML. Aquí usamos permutación marginal:
         zeroeamos cada feature y medimos cuánto cambia P(Win).
    PARA QUÉ: Saber EXACTAMENTE qué impulsó la decisión (RSI? BB? Volume?).
    CÓMO: Para cada feature_i:
         δP_i = P(Win|all_features) - P(Win|all_features con feature_i=neutral)
         Ranking por |δP_i| descendiente → top 5.
    CUÁNDO: Parte del analyze() pre-trade.
    """
    
    # Feature definitions with their neutral (baseline) values
    FEATURE_DEFS = {
        'rsi':            {'neutral': 50.0,  'desc': 'RSI'},
        'bb_position':    {'neutral': 0.5,   'desc': 'BB Position'},
        'adx':            {'neutral': 20.0,  'desc': 'ADX'},
        'volume_ratio':   {'neutral': 1.0,   'desc': 'Volume Ratio'},
        'confluence':     {'neutral': 0.5,   'desc': 'MTF Confluence'},
        'macd_hist':      {'neutral': 0.0,   'desc': 'MACD Histogram'},
        'trend_aligned':  {'neutral': 0.0,   'desc': 'Trend Alignment'},
        'atr_pct':        {'neutral': 0.01,  'desc': 'ATR %'},
    }
    
    def __init__(self, calibrator: BayesianCalibrator):
        self.calibrator = calibrator
    
    def _features_to_bayesian_inputs(
        self, features: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """
        Maps setup features to bayesian_probability_jit inputs.
        
        QUÉ: bayesian_probability_jit espera (signal_strength, trend_strength, volatility_z).
             Necesitamos mapear nuestras 8 features a estos 3 inputs.
        CÓMO:
            signal_strength = weighted combo of RSI extremeness + BB position + volume
            trend_strength = trend_aligned + ADX normalization
            volatility_z = ATR z-score approximation
        """
        rsi = features.get('rsi', 50.0)
        bb = features.get('bb_position', 0.5)
        adx = features.get('adx', 20.0)
        vol_ratio = features.get('volume_ratio', 1.0)
        confluence = features.get('confluence', 0.5)
        macd = features.get('macd_hist', 0.0)
        trend = features.get('trend_aligned', 0.0)
        atr_pct = features.get('atr_pct', 0.01)
        
        # Signal strength: how strong is the entry signal?
        rsi_extremeness = abs(rsi - 50.0) / 50.0  # 0 at 50, 1 at 0 or 100
        bb_extremeness = abs(bb - 0.5) * 2.0       # 0 at middle, 1 at edges
        vol_boost = min(1.0, (vol_ratio - 1.0) / 2.0) if vol_ratio > 1.0 else 0.0
        
        signal_strength = np.clip(
            0.3 * rsi_extremeness + 
            0.25 * bb_extremeness + 
            0.2 * vol_boost + 
            0.25 * confluence,
            0.0, 1.0
        )
        
        # Trend strength: how aligned is the trend?
        adx_norm = min(1.0, adx / 50.0)  # 0-1 normalization
        trend_strength = np.clip(trend * adx_norm, -1.0, 1.0)
        
        # Volatility Z: how extreme is current volatility?
        # ATR% > 2% is high vol, < 0.3% is low
        vol_z = (atr_pct - 0.01) / 0.005  # Centered around 1% ATR
        vol_z = np.clip(vol_z, -3.0, 3.0)
        
        return signal_strength, trend_strength, vol_z
    
    def compute_attributions(
        self,
        features: Dict[str, float],
    ) -> List[FeatureAttribution]:
        """
        Compute marginal contribution of each feature.
        
        QUÉ: Para cada feature, reemplázala por su valor neutral y mide el cambio en P.
        POR QUÉ: Esto nos dice cuánto "ayudó" o "perjudicó" esa feature.
        """
        # Baseline: P(Win|all features)
        ss, ts, vz = self._features_to_bayesian_inputs(features)
        p_full = float(self.calibrator.compute_posterior(ss, ts, vz))
        
        attributions = []
        
        for feat_name, feat_def in self.FEATURE_DEFS.items():
            if feat_name not in features:
                continue
            
            # Create features with this one zeroed to neutral
            perturbed = features.copy()
            perturbed[feat_name] = feat_def['neutral']
            
            # Compute P without this feature
            ss_p, ts_p, vz_p = self._features_to_bayesian_inputs(perturbed)
            p_without = float(self.calibrator.compute_posterior(ss_p, ts_p, vz_p))
            
            # δP = P(with) - P(without)
            delta_p = p_full - p_without
            
            attributions.append(FeatureAttribution(
                feature=feat_def['desc'],
                value=features[feat_name],
                contribution=delta_p,
            ))
        
        # Sort by absolute contribution (descending)
        attributions.sort(key=lambda a: abs(a.contribution), reverse=True)
        
        return attributions[:5]  # Top 5


# ============================================================
# BLOCK 2: SURVIVAL ESTIMATOR
# ============================================================

class SurvivalEstimator:
    """
    ⏱️ SOPHIA §2.1: Time-to-Exit Estimator via GARCH Volatility.
    
    QUÉ: Estima cuántos minutos tardará el precio en alcanzar TP o SL.
    POR QUÉ: Saber el horizonte temporal permite al trader (y al bot) planificar.
         Si E[T] > TTL, la señal caducará antes de llegar al objetivo.
    PARA QUÉ: Calcular time_to_tp, time_to_sl, expected_exit.
    CÓMO: E[T] = (distancia_a_target / σ_GARCH) × √(timeframe_minutes)
         Basado en la aproximación de random walk: E[T_first_passage] ≈ d²/σ²
         donde d es la distancia normalizada.
    CUÁNDO: Pre-trade.
    """
    
    def __init__(self, bar_minutes: float = 5.0):
        """
        Args:
            bar_minutes: Timeframe in minutes (5m = 5.0 for scalping).
        """
        self.bar_minutes = bar_minutes
    
    def estimate(
        self,
        current_price: float,
        tp_pct: float,
        sl_pct: float,
        returns: Optional[np.ndarray] = None,
        garch_vol: Optional[float] = None,
    ) -> SurvivalEstimate:
        """
        Estimate time-to-exit for a new position.
        
        Args:
            current_price: Current asset price
            tp_pct: Take profit as fraction (e.g. 0.01 = 1%)
            sl_pct: Stop loss as fraction (e.g. 0.005 = 0.5%)
            returns: Recent log returns for GARCH estimation
            garch_vol: Pre-calculated GARCH volatility (overrides returns calc)
        """
        # 1. Get GARCH volatility per bar
        if garch_vol is not None and garch_vol > 0:
            sigma_bar = garch_vol
        elif returns is not None and len(returns) >= 10:
            try:
                garch_vars = calculate_garch_jit(returns.astype(np.float64))
                sigma_bar = float(np.sqrt(garch_vars[-1]))
                sigma_bar = max(sigma_bar, 1e-8)
            except Exception:
                sigma_bar = float(np.std(returns)) if len(returns) > 1 else 0.001
        else:
            sigma_bar = 0.001  # Default ~0.1% per 5m bar
        
        # 2. Distances in price-units (as fraction of price)
        dist_tp = abs(tp_pct)
        dist_sl = abs(sl_pct)
        
        # 3. First passage time approximation
        # For a random walk with drift ≈ 0, E[T] ~ (d/σ)² bars
        # This is the expected number of bars to reach distance d
        bars_to_tp = (dist_tp / sigma_bar) ** 2 if sigma_bar > 0 else 999
        bars_to_sl = (dist_sl / sigma_bar) ** 2 if sigma_bar > 0 else 999
        
        # Cap at reasonable values
        bars_to_tp = min(bars_to_tp, 500)
        bars_to_sl = min(bars_to_sl, 500)
        
        # Convert bars to minutes
        time_to_tp = bars_to_tp * self.bar_minutes
        time_to_sl = bars_to_sl * self.bar_minutes
        
        # Expected exit: weighted by probability of hitting each first
        # P(TP first) ≈ sl / (tp + sl) for symmetric random walk
        p_tp_first = dist_sl / (dist_tp + dist_sl) if (dist_tp + dist_sl) > 0 else 0.5
        expected_exit = p_tp_first * time_to_tp + (1 - p_tp_first) * time_to_sl
        
        # 4. Confidence interval (±1σ)
        # Variance of first passage time ≈ 2 * d² / σ⁴ (rough approximation)
        std_estimate = expected_exit * 0.5  # ~50% uncertainty
        
        return SurvivalEstimate(
            time_to_tp_mins=round(time_to_tp, 1),
            time_to_sl_mins=round(time_to_sl, 1),
            expected_exit_mins=round(expected_exit, 1),
            confidence_lower_mins=round(max(1.0, expected_exit - std_estimate), 1),
            confidence_upper_mins=round(expected_exit + std_estimate, 1),
            garch_volatility=round(sigma_bar, 6),
        )


# ============================================================
# BLOCK 2.2: ALPHA DECAY FUNCTION
# ============================================================

class AlphaDecayFunction:
    """
    ⏱️ SOPHIA §2.2: Signal Thesis Expiration.
    
    QUÉ: Define cuándo la señal original pierde validez.
    POR QUÉ: Una señal de scalping a 5m tiene vida útil limitada. Si no se
         ejecuta en ~3-15 minutos, el contexto del mercado ha cambiado y la
         tesis ya no es válida.
    PARA QUÉ: El bot puede explicar "tesis caducada" cuando un trade se arrastra.
    CÓMO: α(t) = signal_strength × exp(-λ × t), donde λ = 1/TTL_seconds.
         Threshold por defecto: 0.30 (si α < 0.30, tesis expirada).
    """
    
    def __init__(self, min_threshold: float = 0.30):
        self.min_threshold = min_threshold
    
    def compute_decay(
        self,
        signal_strength: float,
        elapsed_seconds: float,
        ttl_seconds: float = 180.0,
    ) -> float:
        """Returns current alpha value after elapsed time."""
        if ttl_seconds <= 0:
            return 0.0
        lam = 1.0 / ttl_seconds
        return signal_strength * math.exp(-lam * elapsed_seconds)
    
    def get_expiration_time_mins(
        self,
        signal_strength: float,
        ttl_seconds: float = 180.0,
    ) -> float:
        """
        Returns minutes until alpha drops below threshold.
        
        Solve: strength * exp(-t/TTL) = threshold
        → t = -TTL * ln(threshold / strength)
        """
        if signal_strength <= self.min_threshold:
            return 0.0
        
        ratio = self.min_threshold / signal_strength
        if ratio >= 1.0:
            return 0.0
        
        t_seconds = -ttl_seconds * math.log(ratio)
        return t_seconds / 60.0
    
    def is_thesis_expired(
        self,
        signal_strength: float,
        elapsed_seconds: float,
        ttl_seconds: float = 180.0,
    ) -> bool:
        """Check if signal thesis has expired."""
        alpha = self.compute_decay(signal_strength, elapsed_seconds, ttl_seconds)
        return alpha < self.min_threshold


# ============================================================
# BLOCK 3: ENTROPY ANALYZER
# ============================================================

class EntropyAnalyzer:
    """
    📊 SOPHIA §3.1: Shannon Entropy of Decision Distribution.
    
    QUÉ: Mide la incertidumbre de la decisión.
    POR QUÉ: Si el modelo da probabilidades similares a LONG/SHORT/HOLD,
         la entropía es alta → la decisión es "dudosa". Si da 80% a LONG,
         la entropía es baja → la decisión es confiada.
    PARA QUÉ: Alertar cuando el bot está "indeciso" y la señal no es confiable.
    CÓMO: H = -Σ p_i × log2(p_i) para la distribución [P(LONG), P(SHORT), P(HOLD)].
    """
    
    @staticmethod
    def compute_entropy(action_probs: List[float]) -> float:
        """
        Shannon entropy of action probability distribution.
        
        Args:
            action_probs: [P(LONG), P(SHORT), P(HOLD)] — must sum to ~1.0
        
        Returns:
            H ≥ 0 (0 = certain, log2(N) = max uncertainty)
        """
        h = 0.0
        for p in action_probs:
            if p > 1e-10:
                h -= p * math.log2(p)
        return h
    
    @staticmethod
    def classify_entropy(h: float) -> str:
        """
        Classify entropy level.
        
        For 3 actions: max H = log2(3) ≈ 1.585
        """
        if h < 0.5:
            return "Alta Convicción"
        elif h < 1.0:
            return "Moderada"
        else:
            return "Indeciso"
    
    @staticmethod
    def from_signal(win_prob: float, signal_type: str) -> Tuple[float, str]:
        """
        Derive entropy from win probability and signal type.
        
        QUÉ: Dado P(Win) y la dirección, construye la distribución de acciones.
        CÓMO: Si direction=LONG:
              P(LONG) = win_prob
              P(SHORT) = (1-win_prob) * 0.3  (small chance of wrong direction)
              P(HOLD) = (1-win_prob) * 0.7   (mostly should hold if wrong)
        """
        p_win = np.clip(win_prob, 0.01, 0.99)
        p_lose = 1.0 - p_win
        
        if signal_type in ('LONG', 'SHORT'):
            p_action = p_win
            p_opposite = p_lose * 0.3
            p_hold = p_lose * 0.7
        else:
            p_action = 0.33
            p_opposite = 0.33
            p_hold = 0.34
        
        # Normalize
        total = p_action + p_opposite + p_hold
        probs = [p_action / total, p_opposite / total, p_hold / total]
        
        h = EntropyAnalyzer.compute_entropy(probs)
        label = EntropyAnalyzer.classify_entropy(h)
        
        return h, label


# ============================================================
# BLOCK 3.2: TAIL RISK ANALYZER
# ============================================================

class TailRiskAnalyzer:
    """
    📊 SOPHIA §3.2: Fat Tail Detection via Kurtosis/Skewness.
    
    QUÉ: Analiza la distribución de retornos recientes para detectar colas gruesas.
    POR QUÉ: En distribuciones normales, P(|ret| > 3σ) = 0.27%.
         En cripto, puede ser 2-5% (fat tails). Esto invalida el SL convencional
         porque los movimientos extremos son MUCHO más frecuentes de lo esperado.
    PARA QUÉ: Ajustar el SL dinámicamente cuando se detectan fat tails.
    CÓMO: Calcular excess kurtosis (>3 = leptokúrtica) y tail ratio.
    """
    
    NORMAL_TAIL_PROB = 0.0027  # P(|Z| > 3) for normal distribution
    
    @staticmethod
    def analyze(returns: np.ndarray, window: int = 1000) -> TailRiskMetrics:
        """
        Analyze recent returns for fat tail characteristics.
        
        Args:
            returns: Array of log returns
            window: Analysis window (default 1000 ticks)
        """
        r = returns[-window:] if len(returns) > window else returns
        n = len(r)
        
        if n < 30:
            return TailRiskMetrics(
                excess_kurtosis=0.0,
                skewness=0.0,
                tail_ratio=1.0,
                has_fat_tails=False,
                sl_adjustment_factor=1.0,
            )
        
        mean = np.mean(r)
        std = np.std(r)
        
        if std < 1e-10:
            return TailRiskMetrics(
                excess_kurtosis=0.0,
                skewness=0.0,
                tail_ratio=1.0,
                has_fat_tails=False,
                sl_adjustment_factor=1.0,
            )
        
        # Standardize
        z = (r - mean) / std
        
        # Excess Kurtosis (normal = 0, >0 = heavier tails)
        kurtosis = float(np.mean(z ** 4) - 3.0)
        
        # Skewness
        skewness = float(np.mean(z ** 3))
        
        # Empirical tail ratio
        extreme_count = np.sum(np.abs(z) > 3.0)
        empirical_tail_prob = extreme_count / n
        tail_ratio = empirical_tail_prob / TailRiskAnalyzer.NORMAL_TAIL_PROB if TailRiskAnalyzer.NORMAL_TAIL_PROB > 0 else 1.0
        
        # Fat tails if kurtosis > 3 OR tail_ratio > 2
        has_fat_tails = kurtosis > 3.0 or tail_ratio > 2.0
        
        # SL adjustment: widen SL proportionally to excess kurtosis
        # Mild fat tails (kurtosis 3-6): 1.1x-1.3x SL
        # Heavy fat tails (kurtosis >6): 1.3x-1.5x SL
        if has_fat_tails:
            sl_factor = 1.0 + min(0.5, kurtosis * 0.05)
        else:
            sl_factor = 1.0
        
        return TailRiskMetrics(
            excess_kurtosis=round(kurtosis, 4),
            skewness=round(skewness, 4),
            tail_ratio=round(tail_ratio, 2),
            has_fat_tails=has_fat_tails,
            sl_adjustment_factor=round(sl_factor, 3),
        )


# ============================================================
# FACADE: SOPHIA INTELLIGENCE
# ============================================================

class SophiaIntelligence:
    """
    🧠 SOPHIA-INTELLIGENCE: Facade for all XAI subsystems.
    
    QUÉ: Punto de entrada único para generar un SophiaReport completo.
    POR QUÉ: Simplifica la integración en technical.py. Un solo método analyze()
         que devuelve todo lo necesario.
    PARA QUÉ: Se invoca así:
         sophia = SophiaIntelligence()
         report = sophia.analyze(symbol, setups, returns, tp_pct, sl_pct, ...)
         signal.metadata['sophia'] = report.to_dict()
    CÓMO: Orquesta BayesianCalibrator → FeatureAttributor → SurvivalEstimator →
         EntropyAnalyzer → TailRiskAnalyzer → NarrativeGenerator.
    CUÁNDO: Antes de cada SignalEvent en generate_signals().
    DÓNDE: sophia/intelligence.py → SophiaIntelligence
    QUIÉN: HybridScalpingStrategy.generate_signals()
    """
    
    def __init__(self, bar_minutes: float = 5.0):
        self.calibrator = BayesianCalibrator(prior_alpha=10, prior_beta=10)
        self.attributor = FeatureAttributor(self.calibrator)
        self.survival = SurvivalEstimator(bar_minutes=bar_minutes)
        self.decay = AlphaDecayFunction(min_threshold=0.30)
        self.tail_analyzer = TailRiskAnalyzer()
        
        logger.info("🧠 [SOPHIA] Intelligence engine initialized")
    
    def sync_history(self, wins: int, losses: int):
        """Sync calibrator with historical win/loss data from RiskManager."""
        self.calibrator.sync_from_risk_manager(wins, losses)
        logger.info(
            f"🧠 [SOPHIA] Prior synced: {wins}W/{losses}L → "
            f"P(Win)_prior = {self.calibrator.get_prior_win_rate():.2%}"
        )
    
    def update_after_trade(self, won: bool):
        """Update Bayesian prior after a trade closes."""
        self.calibrator.update_prior(won)
    
    def analyze(
        self,
        symbol: str,
        direction: str,
        signal_strength: float,
        setups: Dict[str, Any],
        confluence_score: float,
        tp_pct: float,
        sl_pct: float,
        returns: Optional[np.ndarray] = None,
        ttl_seconds: float = 180.0,
    ) -> SophiaReport:
        """
        Generate complete XAI report for a trade decision.
        
        Args:
            symbol: Trading pair (e.g. "BTC/USDT")
            direction: "LONG" or "SHORT"
            signal_strength: Signal strength from strategy (0-1)
            setups: Dict with RSI, ADX, BB position, etc.
            confluence_score: Multi-timeframe confluence (0-1)
            tp_pct: Take profit fraction
            sl_pct: Stop loss fraction
            returns: Recent returns array for GARCH/tail analysis
            ttl_seconds: Signal TTL
            
        Returns:
            SophiaReport with all XAI fields populated.
        """
        start_ns = time.perf_counter_ns()
        
        # ── BLOCK 1: Bayesian Calibration ──
        features = {
            'rsi': setups.get('rsi', 50.0),
            'bb_position': setups.get('bb_position', 0.5),
            'adx': setups.get('adx', 20.0),
            'volume_ratio': setups.get('volume_ratio', 1.0),
            'confluence': confluence_score,
            'macd_hist': setups.get('macd_hist', 0.0),
            'trend_aligned': 1.0 if setups.get('in_uptrend') else (-1.0 if setups.get('in_downtrend') else 0.0),
            'atr_pct': (setups.get('atr', 0.0) / setups.get('close', 1.0)) if setups.get('close', 0) > 0 else 0.01,
        }
        
        # Compute trend strength for Bayesian input
        trend_val = features['trend_aligned']
        vol_z = (features['atr_pct'] - 0.01) / 0.005
        vol_z = np.clip(vol_z, -3.0, 3.0)
        
        win_prob = self.calibrator.compute_posterior(
            signal_strength, trend_val, vol_z
        )
        prior_wr = self.calibrator.get_prior_win_rate()
        
        # Feature attributions (top 5)
        attributions = self.attributor.compute_attributions(features)
        top_features = [a.to_dict() for a in attributions]
        
        # ── BLOCK 2: Temporal Horizon ──
        survival = self.survival.estimate(
            current_price=setups.get('close', 0.0),
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            returns=returns,
        )
        
        decay_mins = self.decay.get_expiration_time_mins(signal_strength, ttl_seconds)
        
        # ── BLOCK 3: Statistical Telemetry ──
        entropy, entropy_label = EntropyAnalyzer.from_signal(win_prob, direction)
        
        # Tail risk (if returns available)
        if returns is not None and len(returns) >= 30:
            tail = self.tail_analyzer.analyze(returns)
        else:
            tail = TailRiskMetrics(
                excess_kurtosis=0.0,
                skewness=0.0,
                tail_ratio=1.0,
                has_fat_tails=False,
                sl_adjustment_factor=1.0,
            )
        
        # ── Build Report ──
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000
        
        report = SophiaReport(
            win_probability=win_prob,
            prior_win_rate=prior_wr,
            top_features=top_features,
            expected_exit_mins=survival.expected_exit_mins,
            time_to_tp_mins=survival.time_to_tp_mins,
            time_to_sl_mins=survival.time_to_sl_mins,
            alpha_decay_threshold_mins=decay_mins,
            decision_entropy=entropy,
            entropy_label=entropy_label,
            excess_kurtosis=tail.excess_kurtosis,
            skewness=tail.skewness,
            tail_risk_warning=tail.has_fat_tails,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            direction=direction,
            signal_strength=signal_strength,
        )
        
        # Log compact line
        logger.info(f"   {report.to_log_line()} [{elapsed_us:.0f}μs]")
        
        return report
