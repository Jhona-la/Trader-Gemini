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
from collections import deque

from utils.logger import logger

try:
    from utils.math_kernel import bayesian_probability_jit, calculate_garch_jit, compute_shannon_entropy_jit, compute_alpha_decay_jit
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
        
    def compute_shannon_entropy_jit(probs):
        h = 0.0
        for p in probs:
            if p > 1e-10:
                h -= p * math.log2(p)
        return h
        
    def compute_alpha_decay_jit(signal_strength, elapsed_seconds, ttl_seconds):
        if ttl_seconds <= 0.0:
            return 0.0
        lam = 1.0 / ttl_seconds
        return signal_strength * math.exp(-lam * elapsed_seconds)


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
    
    # Block 1.5: Chronos Temporal Horizon (V5.15)
    expected_high_pct: float  # Exp. price +% in horizon
    expected_low_pct: float   # Exp. price -% in horizon
    drift_factor: float       # Coin-specific price drift
    
    # Block 1.6: Holographic Trajectory (V5.16)
    path_score: float         # 0-1 Intensity of the trajectory
    
    # Block 1.7: Quantum Sovereignty (V5.17)
    hurst_exponent: float          # >0.5 Trend, <0.5 Mean Rev
    quantum_leverage: float        # Adaptive leverage multiplier (up to 50x)
    
    # Block 1.8: Vortex Singularity (V5.18)
    vortex_pulse: float            # 0-N Volume standard deviation pulse
    is_vortex_regime: bool         # True if volume > 2.5 sigmas
    
    # Block 1.9: Apex Singularity (V5.19)
    whale_ratio: float             # Current vol / 4H mean vol
    is_breakout: bool              # Price at 50-bar high/low
    
    # Block 1.10: Noise Predator (V5.20)
    noise_level: float             # Spectral density (0-1)
    noise_sigma: float             # Noise standard deviation for SL adjustment
    
    # Block 1.24: Ocean of PnL (V5.24)
    noise_trend: str               # "STABLE", "DECAYING", "RISING"
    
    # Block 1.26: Omniscient Predator (V5.26)
    omniscient_score: float        # Single decision score (0-1)
    
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
    
    # Block 1.29: The Oracle (Chaos Prediction) (V5.29)
    entropy_velocity: float         # ΔH (Change in entropy)
    lyapunov_horizon: float         # Local predictability horizon (bars)
    
    # Block 1.60: Regime Awareness (Phase 6)
    market_regime: str              # Multi-scale consensus regime
    
    # Block 1.30: The Oracle Awakened (V5.30)
    entropy_acceleration: float     # Δ²H (Acceleration of chaos)
    entropy_forecast: float         # H_pred (Predicted future chaos)
    
    # Block 1.31: The Hurricane Hunter (V5.31)
    noise_color: str                # WHITE, PINK, BROWN
    hurricane_flow: float           # 0-1, Structured chaos energy
    
    # Block 1.32: The Singularity Pivot (V5.32)
    chaos_compactness: float        # Phase space density metric
    singularity_force: float        # Boost factor from hidden patterns
    
    # Block 1.33: The Butterfly Effect (V5.33)
    butterfly_sensitivity: float    # Sensitivity to initial conditions (χ)
    micro_entropy: float            # Entropy in 5-bar micro-window
    butterfly_force: float          # Boost factor from micro-order
    
    # Block 1.34: The Chaos Resonance (V5.34)
    resonance_index: float          # Takens Embedding cyclic resonance
    quantum_tunneling: float        # Energy bypass factor
    
    # Block 1.36: The Schrödinger Edge (V5.36)
    wave_amplitude: float           # Quantum probability amplitude (ψ)
    entanglement_factor: float      # Cross-symbol correlation energy
    heisenberg_shield: float        # Price/Time uncertainty buffer
    
    # Block 1.37: The Dirac Sea (V5.37)
    interference_pattern: float     # Constructive/Destructive energy (I)
    dirac_energy: float             # Vacuum excitation level (E)
    temporal_tunneling: float       # Wave auto-resonance
    
    # Block 1.38: The Quantum Feedback Loop (V5.38)
    quantum_coherence: float        # System adaptation factor (κ)
    feedback_bias: float            # Observer correction term
    
    # Block 1.39: Quantum Neural Fabric (V5.39)
    fabric_tension: float           # Physics/Technical resonance (T)
    liquid_modulation: float        # Adaptive neural weighting
    
    # Block 1.40: Quantum Singularity (V5.40)
    singularity_horizon: float      # Event horizon proximity (Rs)
    gravitational_boost: float      # Final force multiplier
    
    # Block 1.41: Fabric Perfection (V5.41)
    fabric_harmony: float           # Autotuned tension stability
    
    # Block 1.42: Quantum Superposition (V5.42)
    superposition_coherence: float  # Alignment between parallel paths (|φ⟩)
    collapsed_path: str             # Selected path (CONSERVATIVE/DYNAMIC/AGGRESSIVE)

    # Block 1.46: The Neuro-Evolutionary Fabric (V5.46) [NEW]
    meta_reasoning: Dict = field(default_factory=dict)
    parameter_drift: float = 0.0    # Suggested infinitesimal shift (±0.000001%)
    
    # Metadata
    timestamp: str = ""
    symbol: str = ""
    direction: str = ""
    signal_strength: float = 0.0
    metadata: dict = None # V5.14 catalyst
    
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
            # V5.15+ Blocks
            'expected_high_pct': round(getattr(self, 'expected_high_pct', 0.0), 5),
            'expected_low_pct': round(getattr(self, 'expected_low_pct', 0.0), 5),
            'drift_factor': round(getattr(self, 'drift_factor', 0.0), 6),
            'path_score': round(getattr(self, 'path_score', 0.0), 4),
            'hurst_exponent': round(getattr(self, 'hurst_exponent', 0.5), 4),
            'quantum_leverage': round(getattr(self, 'quantum_leverage', 1.0), 2),
            'vortex_pulse': round(getattr(self, 'vortex_pulse', 1.0), 2),
            'is_vortex_regime': getattr(self, 'is_vortex_regime', False),
            'whale_ratio': round(getattr(self, 'whale_ratio', 1.0), 2),
            'is_breakout': getattr(self, 'is_breakout', False),
            'noise_level': round(getattr(self, 'noise_level', 0.5), 4),
            'noise_sigma': round(getattr(self, 'noise_sigma', 0.001), 6),
            'noise_trend': getattr(self, 'noise_trend', "STABLE"),
            'omniscient_score': round(getattr(self, 'omniscient_score', 0.0), 4),
            'meta_reasoning': getattr(self, 'meta_reasoning', {}),
            'parameter_drift': round(getattr(self, 'parameter_drift', 0.0), 9)
        }


# ============================================================
# BLOCK 1: THE MULTI-HORIZON ORACLE (PHASE 3)
# ============================================================
class MultiHorizonOracle:
    """
    🔮 SOPHIA §1.0: Multi-Horizon Predictive Oracle.
    
    QUÉ: Analiza el contexto de múltiples marcos métimos (1m, 5m, 1h, 1d, 1w) y calcula un "Clash Vector".
    POR QUÉ: Evita tomar operaciones "perfectas" en 5 minutos que van directo contra una tendencia de 1 semana.
    PARA QUÉ: Filtrar operaciones con alto riesgo de golpear el Stop Loss (Hard Veto).
    CÓMO: Revisa la alineación de componentes direccionales en el dict de timeframe_data.
    CUÁNDO: Se invoca en Technical.generate_signals antes de autorizar el trade.
    """
    
    @staticmethod
    def evaluate_clash_vector(timeframe_data: Dict, direction: str, horizon: str = "SCALPING") -> Dict:
        """
        Retorna:
            - is_vetoed (bool): True si el macro prohíbe el trade.
            - clash_score (float): 0.0 (Perfectamente alineado) a 1.0 (Choque total).
            - macro_context (str): Descripción del mercado.
        """
        is_vetoed = False
        clash_score = 0.0
        details = []
        
        # 1. Extraer Macro y Estructural
        tf_1d = timeframe_data.get('1d')
        tf_1w = timeframe_data.get('1w')
        
        up_votes = 0
        down_votes = 0
        total_votes = 0
        
        # Ponderación severa para Macro:
        # Si 1W y 1D apuntan a la baja fuertemente, vetar compras.
        for tf, data_pkg in [('1d', tf_1d), ('1w', tf_1w)]:
            if not data_pkg: continue
            inds = data_pkg['inds']
            if len(inds['rsi']) == 0: continue
            
            # Condición alcista estructural:
            is_up = inds['in_uptrend'][-1]
            is_down = inds['in_downtrend'][-1]
            last_rsi = inds['rsi'][-1]
            
            # Peso mayor para 1w
            weight = 2 if tf == '1w' else 1
            total_votes += weight
            
            if is_up and last_rsi > 40:
                up_votes += weight
            elif is_down and last_rsi < 60:
                down_votes += weight
                
            details.append(f"{tf}:{'UP ' if is_up else 'DN ' if is_down else 'SIDE'} (RSI:{last_rsi:.1f})")

        if total_votes == 0:
             return {'is_vetoed': False, 'clash_score': 0.0, 'macro_context': "NO_MACRO_DATA"}
             
        # Cálculo del choque
        # Up-ratio = (up_votes / total_votes)
        up_ratio = up_votes / total_votes
        down_ratio = down_votes / total_votes
        
        if direction == 'LONG':
            clash_score = down_ratio # Si el mercado es 100% bajista, clash=1.0
            if clash_score > 0.6: # Si más del 60% del peso macro es opuesto
                 is_vetoed = True
        elif direction == 'SHORT':
            clash_score = up_ratio
            if clash_score > 0.6:
                 is_vetoed = True

        # ================================================================
        # IMPLEMENTACIÓN DE SHORTS SIMÉTRICOS (Sophia AI)
        # ================================================================
        if horizon == 'SCALPING' and direction == 'SHORT':
            # Evitar que el macro (1D/1W alcista) bloquee un scalp ultrarrápido bajista
            is_vetoed = False
            clash_score = clash_score * 0.5  # Relajar penalización al 50%
            details.append("SCALP_SHORT_RELAXED")
        # ================================================================
                 
        context = " | ".join(details)
        return {
            'is_vetoed': is_vetoed,
            'clash_score': clash_score,
            'macro_context': context
        }

# ============================================================
# BLOCK 1.1: BAYESIAN CALIBRATOR
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

    def calculate_exhaustion(self, macd_hist: np.ndarray, rsi: float) -> float:
        """
        🧠 SOPHIA EXTENSION V5.13: Momentum Exhaustion Predictor.
        
        QUÉ: Detecta si el impulso actual está perdiendo fuerza (curvatura del histograma).
        POR QUÉ: Si el histograma MACD se achica mientras el RSI está en extremo,
             la probabilidad de continuación cae drásticamente.
        PARA QUÉ: Activar salidas predictivas antes de que el mercado se de la vuelta.
        """
        if len(macd_hist) < 3:
            return 0.5
            
        curr_h = macd_hist[-1]
        prev_h = macd_hist[-2]
        prev2_h = macd_hist[-3]
        
        # Detectar pérdida de aceleración (curvatura)
        exhaustion = 0.5
        if curr_h > 0: # Bullish Momentum
            if curr_h < prev_h: exhaustion += 0.2
            if prev_h < prev2_h: exhaustion += 0.1
            if rsi > 70: exhaustion += 0.1
        elif curr_h < 0: # Bearish Momentum
            if curr_h > prev_h: exhaustion += 0.2
            if prev_h > prev2_h: exhaustion += 0.1
            if rsi < 30: exhaustion += 0.1
            
        return np.clip(exhaustion, 0.0, 1.0)


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
        # MEJORA 7: Dynamic RSI Percentile para Sophia
        # Ajustamos el punto neutral del RSI según la fuerza de la tendencia
        rsi_neutral = 50.0
        if trend > 0.3:
            rsi_neutral = 60.0 # Support en Bull market es más alto
        elif trend < -0.3:
            rsi_neutral = 40.0 # Resistance en Bear market es más bajo
            
        rsi_extremeness = abs(rsi - rsi_neutral) / 40.0  # Distancia máxima ~40
        rsi_extremeness = np.clip(rsi_extremeness, 0.0, 1.0)
        
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
                baseline_sigma = (math.sqrt(self.bar_minutes / 5.0) * 0.001) if self.bar_minutes > 0 else 0.001
                sigma_bar = float(np.std(returns)) if len(returns) > 1 else baseline_sigma
        else:
            baseline_sigma = (math.sqrt(self.bar_minutes / 5.0) * 0.001) if getattr(self, 'bar_minutes', 5.0) > 0 else 0.001
            sigma_bar = baseline_sigma  # Scaled by time instead of static 0.1%
        
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
        """Returns current alpha value after elapsed time using nano-speed JIT calculation."""
        return compute_alpha_decay_jit(float(signal_strength), float(elapsed_seconds), float(ttl_seconds))
    
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
        [NANO-SPEED] Shannon entropy using JIT kernel computation.
        
        Args:
            action_probs: [P(LONG), P(SHORT), P(HOLD)] — must sum to ~1.0
        
        Returns:
            H ≥ 0 (0 = certain, log2(N) = max uncertainty)
        """
        arr = np.array(action_probs, dtype=np.float64)
        return compute_shannon_entropy_jit(arr)
    
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

    @staticmethod
    def calculate_micro_entropy(returns: np.ndarray, window: int = 5) -> float:
        """
        V5.33: Multi-Scale Chaos. 
        Calculates entropy of returns in a micro-window.
        """
        if returns is None or len(returns) < window:
            return 1.585
            
        try:
            micro = returns[-window:]
            # Discretize into 3 buckets: Down, Flat, Up
            std = np.std(returns[-20:]) if len(returns) >= 20 else 0.001
            buckets = []
            for r in micro:
                if r < -0.5 * std: buckets.append(0) # Down
                elif r > 0.5 * std: buckets.append(1) # Up
                else: buckets.append(2) # Flat
            
            counts = np.bincount(buckets, minlength=3)
            probs = counts / len(buckets)
            return EntropyAnalyzer.compute_entropy(probs.tolist())
        except:
            return 1.585


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
        self.last_noise = {} # V5.24: State for noise trend {symbol: last_noise_level}
        self.entropy_history = {} # V5.30: State for chaos prediction {symbol: deque([h1, h2...], maxlen=5)}
        # Adaptive Evolution Protocol: Horizon-Aware Dampening
        self.chaos_dampening_factor = 1.0   # 1.0 = full penalty (scalping default)
        self.certainty_floor = 0.0          # 0.0 = no floor (scalping default)
        self.horizon_profile = 'SCALPING'
        
        logger.info("🧠 [SOPHIA] Intelligence engine initialized")
    
    def set_horizon_profile(self, horizon_days: int):
        """
        Adaptive Evolution Protocol: Modula los dampeners de Sophia
        según el horizonte temporal de operación.
        
        QUÉ: Ajusta chaos_dampening_factor y certainty_floor.
        POR QUÉ: En horizontes de 15D, los dampeners cuánticos (Heisenberg,
                  Lyapunov, Chaos Penalty) penalizan agresivamente porque
                  hay ruido Y tendencia simultáneamente. Esto crea el
                  "Valle de la Muerte" (Win Rate cae a 38.6%).
        PARA QUÉ: Reducir la penalización en horizontes más largos donde
                   el ruido es transitorio y las tendencias son reales.
        CÓMO: 1D→full penalty, 15D→50% penalty, 30D→30% penalty.
              Certainty floor: 1D→0, 15D→0.50, 30D→0.70.
        CUÁNDO: Al inicio de cada sesión junto con MarketRegimeDetector.
        DÓNDE: sophia/intelligence.py → SophiaIntelligence
        QUIÉN: Engine.py o run_backtest.py.
        """
        if horizon_days <= 1:
            self.chaos_dampening_factor = 1.0
            self.certainty_floor = 0.65  # [PHASE 3] DARWINIAN VETO: 65% min confidence
            self.horizon_profile = 'SCALPING'
        elif horizon_days <= 7:
            self.chaos_dampening_factor = 0.7
            self.certainty_floor = 0.65
            self.horizon_profile = 'SHORT_TERM'
        elif horizon_days <= 15:
            self.chaos_dampening_factor = 0.5
            self.certainty_floor = 0.65
            self.horizon_profile = 'MID_TERM'
        else:
            self.chaos_dampening_factor = 0.3
            self.certainty_floor = 0.75
            self.horizon_profile = 'MACRO'
        
        logger.info(
            f"🧠 [SOPHIA] Horizon Profile: {self.horizon_profile} → "
            f"Chaos Dampening={self.chaos_dampening_factor}, "
            f"Certainty Floor={self.certainty_floor}"
        )
    
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
    
    def apply_nemesis_feedback(self, fp_rate: float, avg_brier: float):
        """
        🔄 PHASE 4: Evolutionary Feedback Loop (Némesis → Sophia)
        
        QUÉ: Ajusta parámetros internos de Sophia basándose en el feedback de Némesis.
        POR QUÉ: Sin este feedback, Sophia opera en modo abierto permanentemente,
                 ignorando las lecciones de los trades anteriores.
        PARA QUÉ: Cerrar el bucle de aprendizaje para que el sistema sea verdaderamente
                  adaptativo y evolutivo.
        CÓMO: Si FP_rate > 25%, aumenta chaos_dampening (más conservador).
              Si Brier > 0.25, reduce certainty_floor (menos certeza forzada).
        CUÁNDO: Invocado por NemesisEngine después de cada autopsia completa.
        DÓNDE: sophia/intelligence.py → SophiaIntelligence
        QUIÉN: NemesisEngine.full_autopsy() → SophiaIntelligence.apply_nemesis_feedback()
        """
        adjusted = False
        
        if fp_rate > 0.25:
            # Too many false positives → increase chaos dampening (be more conservative)
            old_val = self.chaos_dampening_factor
            self.chaos_dampening_factor = min(1.5, self.chaos_dampening_factor * 1.2)
            adjusted = True
            logger.info(
                f"🔄 [NÉMESIS→SOPHIA] FP Rate High ({fp_rate:.2%}): "
                f"ChaosDamp {old_val:.2f} → {self.chaos_dampening_factor:.2f}"
            )
        
        if avg_brier > 0.25:
            # Poor calibration → reduce certainty floor (trust less)
            old_val = self.certainty_floor
            self.certainty_floor = max(0.0, self.certainty_floor * 0.7)
            adjusted = True
            logger.info(
                f"🔄 [NÉMESIS→SOPHIA] Brier High ({avg_brier:.3f}): "
                f"CertaintyFloor {old_val:.2f} → {self.certainty_floor:.2f}"
            )
        
        if adjusted:
            logger.info(
                f"🔄 [NÉMESIS→SOPHIA] Feedback Applied. "
                f"New State: ChaosDamp={self.chaos_dampening_factor:.2f}, "
                f"CertaintyFloor={self.certainty_floor:.2f}"
            )
    
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
        btc_returns: Optional[np.ndarray] = None,
        regime: str = "UNKNOWN",
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
        
        # ── V5.36/V5.37: QUANTUM STATE ANALYSIS ──
        psi_l_raw, psi_s_raw = self._calculate_quantum_amplitude_vectorial(returns)
        
        # ── V5.38: QUANTUM FEEDBACK LOOP (Observer Effect) ──
        # prior_wr acts as the 'Observation' that collapses the wave more or less.
        prior_wr = self.calibrator.get_prior_win_rate()
        psi_l, psi_s, kappa, q_bias = self._apply_quantum_feedback(psi_l_raw, psi_s_raw, prior_wr)
        
        psi_amplitude = (psi_l + psi_s) / 2.0
        
        interference = self._calculate_interference_pattern(psi_l, psi_s)
        dirac_e = self._calculate_dirac_energy(returns)
        t_tunnel = self._calculate_temporal_tunneling(returns)
        
        entanglement = self._calculate_entanglement_factor(symbol, returns, btc_returns)
        
        # Quantum Boost (V5.37/V5.38): Interference & Coherence
        # We search for Constructive Interference and high System Coherence (kappa)
        quantum_boost = 1.0
        if (interference > 1.2 or (dirac_e > 0.8 and t_tunnel > 0.7)) and kappa > 0.8:
            quantum_boost = 1.0 + (psi_amplitude * interference * kappa * 0.7)
            logger.info(f"⚛️ [OBSERVER/FEEDBACK] {symbol} Coherent Wave: I={interference:.2f}, κ={kappa:.2f}, ψ={psi_amplitude:.2f}")
        
        # Destructive Interference Penalty (V5.37 / V5.45 Bridge)
        # SUPREMO-V4: Desactivar penalización por interferencia en SCALPING
        # POR QUÉ: En 1m, las ondas son puro ruido y siempre se interfieren.
        # Matar el 90% de señales por esto es un error matemático.
        interference_penalty = 0.0
        if self.horizon_profile != 'SCALPING':
            if (interference < 0.3 and (psi_l > 0.4 and psi_s > 0.4)) or kappa < 0.75:
                bridge_protection = min(0.15, confluence_score * 0.2)
                raw_penalty = (0.12 if kappa < 0.75 else 0.10) - bridge_protection
                interference_penalty = max(0.0, raw_penalty)
            
            if kappa < 0.75:
                logger.warning(f"🛡️ [COHERENCE SHIELD] {symbol} Decoherece Detected (κ={kappa:.2f}). Penalty modulated by Bridge: {interference_penalty:.2f}")
            else:
                logger.debug(f"🔉 [DESTRUCTIVE-MILD] {symbol} Conflicting waves (normal for scalping). Penalty: {interference_penalty:.2f}")

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
        # ── BLOCK 6: Chronos Horizon Prediction (V5.15) ──
        expected_high, expected_low, drift = self._predict_range(returns, horizon_bars=4)
        
        # ── BLOCK 7: Holographic Path Intensity (V5.16) ──
        path_score = self._calculate_path_intensity(setups, direction)
        
        # ── BLOCK 8: Quantum Sovereignty (V5.17) ──
        hurst = self._calculate_hurst_exponent(returns)
        q_leverage = self._calculate_shannon_leverage(entropy, win_prob)
        
        # ── BLOCK 9: Vortex Singularity (V5.18) ──
        v_pulse, is_v_regime = self._detect_vortex_pulse(setups)
        
        # ── BLOCK 10: Apex Singularity (V5.19) ──
        whale_r = self._detect_whales(setups)
        is_break = self._detect_breakout(setups, direction)
        
        # ── BLOCK 11: Noise Predator (V5.20) ──
        n_level, n_sigma = self._calculate_noise_density(returns)
        
        # V5.24: Noise Trend Analysis
        prev_noise = self.last_noise.get(symbol, n_level)
        if n_level < prev_noise * 0.95:
            n_trend = "DECAYING"
        elif n_level > prev_noise * 1.05:
            n_trend = "RISING"
        else:
            n_trend = "STABLE"
        self.last_noise[symbol] = n_level
        
        # FIX A-1: Removed forced breakout anticipation.
        # Breakout must be determined ONLY by real price data (via _detect_breakout),
        # not forced by Sophia's internal probability estimate.
        
        # ── BLOCK 12: Omniscient Score (V5.29 THE ORACLE) ──
        # Collapse ALL filter dimensions into a single weighted score.
        edge_for_dir = abs(expected_high if direction == "LONG" else expected_low)
        risk_for_dir = abs(expected_low if direction == "LONG" else expected_high)
        
        edge_norm = min(1.0, edge_for_dir / 0.01)       # 1% edge = 1.0
        energy_norm = min(1.0, v_pulse / 5.0)            # 5 sigmas = 1.0
        noise_inv = max(0.0, 1.0 - n_level)              # low noise = high score
        momentum = path_score                             # 0-1
        
        # Chaos & Entropy Momentum (V5.30 THE ORACLE AWAKENED)
        if symbol not in self.entropy_history:
            self.entropy_history[symbol] = deque(maxlen=5)
        
        history = self.entropy_history[symbol]
        history.append(entropy)
        
        h_velocity = 0.0
        h_accel = 0.0
        h_forecast = entropy
        
        if len(history) >= 2:
            h_velocity = history[-1] - history[-2]
        if len(history) >= 3:
            prev_velocity = history[-2] - history[-3]
            h_accel = h_velocity - prev_velocity
            
            # Simple Linear Forecast for next bar
            x = np.arange(len(history))
            y = np.array(history)
            try:
                m, b = np.polyfit(x, y, 1)
                h_forecast = m * (len(history)) + b
            except:
                h_forecast = entropy + h_velocity
        
        # V5.31 THE HURRICANE HUNTER: Spectroscopic Chaos Analysis
        n_color, h_flow = self._classify_noise_color(returns)
        
        # V5.32 THE SINGULARITY PIVOT: Phase Space & Entropy Reorganization
        compactness = self._calculate_chaos_compactness(returns)
        
        # V5.33 THE BUTTERFLY EFFECT: Multi-Scale Chaos & Sensitivity
        h_micro = EntropyAnalyzer.calculate_micro_entropy(returns, window=5)
        chi_sensitivity = self._calculate_butterfly_sensitivity(returns)
        
        # V5.34 THE CHAOS RESONANCE: Takens Embedding & Cyclic Order
        r_index = self._calculate_resonance_state(returns)
        
        # Detect Resonance Alert (V5.35): Requires stronger correlation for gating
        is_resonant = r_index > 0.45
        
        # Detect Butterfly Trigger: Local order emerging in global chaos
        # Large H (>1.0) but small micro H (<0.6) + High sensitivity.
        is_butterfly = entropy > 1.0 and h_micro < 0.6 and chi_sensitivity > 0.7
        
        # Detect Singularity: High entropy equilibrium transitioning to order
        # If entropy is near max (>1.4) but acceleration is negative, we are at a pivot.
        is_pivot = entropy > 1.4 and h_accel < 0
        
        # Lyapunov Horizon (Predictability)
        divergence = 0.5
        if returns is not None and len(returns) > 10:
            diffs = np.abs(np.diff(returns[-10:]))
            sigma_local = np.std(returns[-10:])
            divergence = np.mean(diffs) / sigma_local if sigma_local > 0 else 0.5
        
        l_horizon = 1.0 / max(0.01, divergence)
        l_horizon = np.clip(l_horizon, 1.0, 20.0)
        
        # C-3 FIX: Heisenberg Shield calculation moved to L1364 (was duplicated here)
        
        # Resonance Force (V5.34/V5.35): Multiplier for cyclic chaos
        # V5.35: Boost only if resonance is truly confirmed
        res_boost = 1.0
        if is_resonant:
            res_boost = 1.0 + (r_index * 1.0) # Up to 2.0x boost
            logger.info(f"🌀 [RESONANCE] {symbol} Cyclic Order Found: Force={res_boost:.2f} (Index={r_index:.2f})")

        # Butterfly Force (V5.33/V5.35): Power injected by fractal order
        # V5.35 Fractal Alignment: Only if some minimal resonance is detected
        butterfly_boost = 1.0
        if is_butterfly and r_index > 0.35:
            butterfly_boost = 1.0 + (chi_sensitivity * 1.0) # Up to 2.0x boost
            logger.info(f"🦋 [BUTTERFLY] {symbol} Micro-Order Found: Force={butterfly_boost:.2f} (χ={chi_sensitivity:.2f})")

        # Singularity Force (V5.32/V5.35): Power injected by hidden patterns
        singularity_boost = 1.0
        if compactness > 0.7 or is_pivot:
            # We found an attractor or an entropy pivot
            singularity_boost = 1.0 + (compactness * 0.5) + (0.25 if is_pivot else 0.0)
            logger.info(f"🌌 [SINGULARITY] {symbol} Hidden Order Found: Force={singularity_boost:.2f} (Compact={compactness:.2f})")

        # Eye Boost (V5.31): If we are in Pink/Brown noise, chaos is structured.
        eye_boost = 1.0
        if n_color in ["PINK", "BROWN"] and h_flow > 0.4:
            eye_boost = 1.0 + (h_flow * 0.5) # Max 1.5x boost
            logger.info(f"🌀 [HURRICANE] {symbol} Flow State Active ({n_color}): Boost x{eye_boost:.2f}")

        # V5.39/V5.41: QUANTUM NEURAL FABRIC (Tejido)
        # Tension measures alignment between Technical and Quantum.
        # Liquidity modulates weights based on Coherence (kappa).
        _fabric_direction = "LONG" if (win_prob > 0.5) else "SHORT" # Use prior belief for early tension
        fabric_t, fabric_h = self._calculate_fabric_tension(setups, psi_l, psi_s, _fabric_direction)
        liquid_mod = self._liquid_neural_modulation(kappa, fabric_t, fabric_h)
        
        # V5.40: QUANTUM SINGULARITY (Horizonte de Eventos)
        # Event Horizon Rs: Point of inevitable momentum.
        rs_horizon, g_boost = self._calculate_quantum_singularity(psi_l, psi_s, h_accel)
        
        # V5.36: Heisenberg Shield for Superposition
        h_shield = 1.0
        if entropy > 1.2:
            h_shield = 1.0 + (entropy - 1.2) * 2.0
            
        # V5.42: QUANTUM SUPERPOSITION (|φ⟩)
        # Placeholder base_omni for simulation before actual calculation
        _sim_omni = (win_prob * 0.4 + confluence_score * 0.6)
        s_coherence, s_path = self._simulate_superposition_paths(_sim_omni, psi_l, psi_s, h_shield, g_boost, kappa, fabric_t)
        
        # V5.45: QUANTUM HARMONY (The Bridge)
        # Instead of forcing entry, we reward harmony between technical and quantum.
        # Harmony is high if fabric tension is high and superposition is coherent.
        harmony_boost = 1.0
        if fabric_t > 0.7 and s_coherence > 0.6:
            harmony_boost = 1.0 + (fabric_t - 0.7) * 0.5 + (s_coherence - 0.6) * 0.5
            logger.info(f"🌈 [HARMONY] Market & Quantum are aligned: Boost x{harmony_boost:.2f}")

        # Oracle Boost (V5.30):
        # Rewards setups where chaos is stabilizing or decaying.
        oracle_boost = 1.0
        if h_forecast < entropy: # Entropy predicted to drop
            oracle_boost += 0.10
        if h_accel < 0: # Chaos is decelerating
            oracle_boost += 0.05
        
        # Chaos Penalty (V5.29/V5.30): 
        # Penalize rising entropy unless acceleration is negative OR Hurricane/Singularity/Butterfly/Resonance is active.
        # V5.29 CHAOS PENALTY:
        chaos_penalty = 0.0
        is_protected = (eye_boost > 1.1) or (singularity_boost > 1.2) or (butterfly_boost > 1.3) or (res_boost > 1.4) or (quantum_boost > 1.2)
        
        # Phase 47.5: Altcoin Chaos Mitigation
        is_btc = 'BTC' in symbol
        chaos_mult = 1.0 if is_btc else 0.4 # Dampen chaos penalty for volatile alts
        
        if h_velocity > 0 and h_accel >= 0 and not is_protected:
            chaos_penalty += min(0.12 * chaos_mult, h_velocity * 0.6 * chaos_mult)
        if l_horizon < 4.0 and not is_protected:
            chaos_penalty += 0.12 * chaos_mult * (1.0 - (l_horizon / 4.0))
        
        # Adaptive Evolution Protocol: Horizon-Aware Chaos Dampening (V5.29/V5.30)
        chaos_penalty *= self.chaos_dampening_factor

        # Phase 6: Regime-Aware Chaos Modulation (Multiscale Consensus)
        if 'TRENDING' in regime:
            chaos_penalty *= 0.7 # Reduce penalty in trending markets
        elif 'CHOPPY' in regime or 'DIVERGENT' in regime:
            chaos_penalty *= 1.3 # Increase penalty in choppy or divergent markets

        # FIX A-2: Removed duplicate chaos modulation block (was applied 2x)

        # Tail Risk Penalty (V5.28): 
        risk_penalty = 0.0
        if edge_for_dir > 0 and (risk_for_dir / edge_for_dir) > 1.5:
            risk_penalty = min(0.15, (risk_for_dir / edge_for_dir - 1.5) * 0.1)
            
        base_omni = (
            win_prob * 0.25 + 
            edge_norm * 0.20 +
            energy_norm * 0.15 +
            momentum * 0.20 +
            noise_inv * 0.20
        ) - risk_penalty - chaos_penalty - interference_penalty
        
        # ================================================================
        # V5.51: MOMENTUM INERTIA BONUS FOR SCALPING SHORTS
        # QUÉ: Bonus de +0.15 a la confianza para shorts cuando la 
        #   volatilidad actual excede 0.5% (ATR/Price).
        # POR QUÉ: Los mejores trades rápidos bajistas ocurren cuando hay
        #   inercia de momentum: el mercado se mueve con fuerza y los shorts
        #   capturan esa energía direccional. Sin volatilidad, no hay edge.
        # PARA QUÉ: Premiar entradas SHORT que tienen "fuel" para llegar al TP.
        # CÓMO: Si direction == SHORT y ATR% > 0.005, inyectar +0.15.
        # CUÁNDO: Solo en evaluaciones pre-trade de Sophia.
        # DÓNDE: sophia/intelligence.py → analyze() → base_omni calculation
        # QUIÉN: SophiaIntelligence
        # ================================================================
        atr_pct = features.get('atr_pct', 0.01)
        if direction == "SHORT" and atr_pct > 0.005:
            inertia_bonus = 0.15
            base_omni += inertia_bonus
            logger.info(f"⚡ [MOMENTUM_INERTIA] {symbol} SHORT bonus +{inertia_bonus:.2f} (vol={atr_pct*100:.2f}%)")
        
        # V5.45: FREQUENTIST BRIDGE FLOOR (Restored for Scalping Parity)
        # We need a floor because small timeframes have mathematically high entropy/chaos,
        # which drives base_omni to zero. If technical confidence is good or win_prob is high, anchor it.
        if confluence_score > 0.50 or win_prob > 0.60:
            base_omni = max(base_omni, 0.38)
            logger.debug(f"🌉 [BRIDGE FLOOR] {symbol} base_omni anchored to 0.38 (Tech/Prob criteria).")
        
        # Uncertainty Penalty (V5.28): Multiplicative dampening based on entropy.
        normalized_entropy = min(1.0, entropy / 1.585)
        
        uncertainty_penalty = 1.0 - normalized_entropy
        # We increase the gate energy if Dirac Sea (E) is high, interference is Constructive (I > 1.2), 
        # OR if we are in QUANTUM HARMONY (V5.45)
        if is_resonant or (quantum_boost > 1.3 and interference > 1.2) or (harmony_boost > 1.2):
            energy_bypass = min(1.0, base_omni * 2.5) # Even deeper bypass for V5.37
            # V5.45: Harmony Tunneling allows more flow even in high entropy
            q_tunneling = max(0.45, (1.0 - normalized_entropy * 0.40) + (energy_bypass * 0.40))
            if harmony_boost > 1.2:
                logger.info(f"🌈 [HARMONY TUNNELING] {symbol} Bypassing entropy dampeners.")
        else:
            q_tunneling = (1.0 - normalized_entropy)
        
        # V5.35: LYAPUNOV SHIELD — Hard cap if horizon is blind (< 2.0 bars)
        # FIX FORENSIC-2: Raised floor from 0.1/0.4 to 0.50/0.65.
        # POR QUÉ: With 1-minute candles, Lyapunov divergence is ALWAYS high
        # (noise dominates), so l_horizon is almost always < 2.0. The old floor
        # of 0.1 multiplied certainty by 0.1 → killed ALL signals (0.40 × 0.1 = 0.04).
        # PARA QUÉ: Allow Sophia to generate trades on scalping timeframes.
        # A floor of 0.50 means "uncertain but not dead" — still penalizes but survivable.
        l_shield = 1.0
        if l_horizon < 2.0:
            shield_floor = 0.50 if is_btc else 0.65  # Was 0.1/0.4 — FORENSIC FIX
            l_shield = max(shield_floor, 0.5 * (l_horizon / 2.0))
            if is_btc:
                logger.debug(f"🛡️ [LYAPUNOV SHIELD] {symbol} Reduced blind penalty: Shield={l_shield:.2f}")
            else:
                logger.debug(f"🛡️ [ALT-SHIELD] {symbol} Mitigation: Shield={l_shield:.2f}")

        # FORENSIC FIX #3: SIMPLIFY QUANTUM CASCADE (20+ Multipliers -> 5 Core Factors)
        # Previous multiplicative chain: q_tunneling * l_shield * oracle * eye * singularity * butterfly * resonance * quantum * liquid * g_boost * harmony
        # This 11-factor chain collapsed certainty to ~0.01 in high-noise environments (Scalping), suppressing 99.9% of signals.
        # Now we only use the core 5 scientifically sound metrics:
        # 1. q_tunneling (Entropy/Shannon)
        # 2. l_shield (Lyapunov chaos horizon)
        # 3. quantum_boost (Interference/Coherence)
        # 4. liquid_mod (Neural alignment)
        # 5. harmony_boost (Frequentist/Quantum Bridge)
        
        certainty = q_tunneling * l_shield * quantum_boost * liquid_mod * harmony_boost
        
        # Superposition adjustment (V5.42): Signal is stronger if timelines are coherent
        certainty *= (0.8 + s_coherence * 0.4)
        
        # Coherence Injector (V5.43)
        if s_coherence > 0.7:
            injector_boost = 1.0 + (s_coherence - 0.7) * 0.5 # Up to 1.15x
            certainty *= injector_boost
            logger.debug(f"💉 [COHERENCE] Injecting x{injector_boost:.2f} boost to certainty")
            
        # Removed Divine Resonance (x1.4) and Force Collapse (x1.6) which bloated scores artificially

        certainty = min(3.0, certainty) # Probabilistic sanity limit
        
        # Adaptive Evolution Protocol: Horizon-Aware Certainty Floor
        # In longer horizons, prevent over-dampening that kills valid signals
        if self.certainty_floor > 0:
            certainty = max(certainty, self.certainty_floor)
        
        # FIX FORENSIC-3: MINIMUM CERTAINTY FLOOR FOR ALL TIMEFRAMES
        # POR QUÉ: The multiplicative chain of 20+ quantum filters (l_shield × q_tunneling ×
        # oracle_boost × eye_boost × singularity × butterfly × resonance × quantum × liquid ×
        # g_boost × harmony × superposition × coherence_injector) can drive certainty to 0.012.
        # With base_omni ≈ 0.40, omni_score = 0.40 × 0.012 = 0.005 → NO TRADES EVER.
        # PARA QUÉ: Ensure Sophia can still generate signals in high-noise environments.
        # A floor of 0.25 means "low confidence but actionable" — combined with base_omni 0.40,
        # omni_score = 0.40 × 0.25 = 0.10, which can pass the entry gate (0.18-0.25).
        certainty = max(certainty, 0.25)
        
        omni_score = base_omni * certainty
        
        # V5.45: QUANTUM OVERRIDE (Frequentist Bridge)
        # If technicals show any potential or harmony is even slightly present, we force entry.
        # This is the "Balanced" Frequentist bridge to ensure capital movement with precision.
        # Phase 50: Sovereign Alt-Floor (0.18 vs 0.25 for BTC) - Raised from 0.12
        entry_floor = 0.25 if is_btc else 0.18
        
        # FIX A-3: Removed Elite Bridge (redundant with entry_floor).

        # Phase 50: Quantum Override (Restored leniency for Scalping/High Conviction)
        # If technicals are extremely strong OR ML probability is high, ensure it doesn't fail the gate.
        if confluence_score > 0.75 or win_prob > 0.75 or (confluence_score > 0.65 and win_prob > 0.65):
            omni_score = max(omni_score, entry_floor) 
            logger.info(f"⚡ [QUANTUM OVERRIDE] {symbol} Score forced to {entry_floor} (Confluence={confluence_score:.2f}, WinProb={win_prob:.2f})")
        
        logger.info(f"🔮 [ORACLE] {symbol}: ψ={psi_amplitude:.2f}, Rs={rs_horizon:.2f}, |φ⟩={s_path} → Final={omni_score:.3f} (Coh={s_coherence:.2f})")
        
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000
        
        report = SophiaReport(
            win_probability=win_prob,
            prior_win_rate=prior_wr,
            top_features=top_features,
            expected_high_pct=expected_high,
            expected_low_pct=expected_low,
            drift_factor=drift,
            path_score=path_score,
            hurst_exponent=hurst,
            quantum_leverage=q_leverage,
            vortex_pulse=v_pulse,
            is_vortex_regime=is_v_regime,
            # Block 1.9
            whale_ratio=whale_r,
            is_breakout=is_break,
            # Block 1.10
            noise_level=n_level,
            noise_sigma=n_sigma,
            # Block 1.24
            noise_trend=n_trend,
            # Block 1.26
            omniscient_score=omni_score,
            # Block 1.29: The Oracle (V5.29)
            entropy_velocity=h_velocity,
            lyapunov_horizon=l_horizon,
            # Block 1.30: The Oracle Awakened (V5.30)
            entropy_acceleration=h_accel,
            entropy_forecast=h_forecast,
            # Block 1.31: The Hurricane Hunter (V5.31)
            noise_color=n_color,
            hurricane_flow=h_flow,
            # Block 1.32: The Singularity Pivot (V5.32)
            chaos_compactness=compactness,
            singularity_force=singularity_boost,
            # Block 1.33: The Butterfly Effect (V5.33)
            butterfly_sensitivity=chi_sensitivity,
            micro_entropy=h_micro,
            butterfly_force=butterfly_boost,
            # Block 1.34: The Chaos Resonance (V5.34)
            resonance_index=r_index,
            quantum_tunneling=q_tunneling,
            # Block 1.36: The Schrödinger Edge (V5.36)
            wave_amplitude=psi_amplitude,
            entanglement_factor=entanglement,
            heisenberg_shield=h_shield,
            # Block 1.37: The Dirac Sea (V5.37)
            interference_pattern=interference,
            dirac_energy=dirac_e,
            temporal_tunneling=t_tunnel,
            # Block 1.38: The Quantum Feedback Loop (V5.38)
            quantum_coherence=kappa,
            feedback_bias=q_bias,
            # Block 1.39: Quantum Neural Fabric (V5.39)
            fabric_tension=fabric_t,
            liquid_modulation=liquid_mod,
            # Block 1.40: Quantum Singularity (V5.40)
            singularity_horizon=rs_horizon,
            gravitational_boost=g_boost,
            # Block 1.41: Fabric Perfection (V5.41)
            fabric_harmony=fabric_h,
            # Block 1.42: Quantum Superposition (V5.42)
            superposition_coherence=s_coherence,
            collapsed_path=s_path,
            # Block 1.46: Meta-Reasoning (V5.46)
            meta_reasoning=self._generate_meta_reasoning(
                symbol, win_prob, entropy, fabric_t, s_coherence
            ),
            parameter_drift=1e-6 if s_coherence > 0.8 else -1e-6,
            # Block 2 (Survival)
            expected_exit_mins=survival.expected_exit_mins,
            time_to_tp_mins=survival.time_to_tp_mins,
            time_to_sl_mins=survival.time_to_sl_mins,
            alpha_decay_threshold_mins=decay_mins,
            # Block 3 (Telemetry)
            decision_entropy=entropy,
            entropy_label=entropy_label,
            excess_kurtosis=tail.excess_kurtosis,
            skewness=tail.skewness,
            tail_risk_warning=tail.has_fat_tails,
            # Metadata
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            direction=direction,
            signal_strength=signal_strength,
            market_regime=regime,
            metadata={
                'boost_factor': 1.5 if win_prob > 0.90 else (1.2 if win_prob > 0.85 else 1.0),
                'calibrated': win_prob > 0.6
            }
        )
        
        # Log compact line
        logger.info(f"   {report.to_log_line()} [{elapsed_us:.0f}μs]")
        
        return report

    def _predict_range(self, returns: np.ndarray, horizon_bars: int = 4) -> Tuple[float, float, float]:
        """
        V5.15 Chronos: Predicts the expected range for the next N bars.
        Uses drift + GARCH vol to estimate a 1-std confidence corridor.
        """
        if returns is None or len(returns) < 10:
            return 0.01, -0.01, 0.0
            
        # 1. Calculate Drift (Average return per bar)
        drift = float(np.mean(returns))
        
        # 2. Calculate Volatility (GARCH or Simple Std)
        # We'll use simple std for reliability in backtest
        sigma = float(np.std(returns))
        
        # 3. Project for N bars
        # Price process: P_t = P_0 * exp(drift*N + sigma*sqrt(N)*Z)
        exp_drift = drift * horizon_bars
        exp_vol = sigma * math.sqrt(horizon_bars)
        
        # Upper/Lower Bound (1-sigma)
        expected_high = exp_drift + exp_vol
        expected_low = exp_drift - exp_vol
        
        return expected_high, expected_low, drift

    def _calculate_path_intensity(self, setups: Dict[str, float], direction: str) -> float:
        """
        V5.16 Hologram: Measures how 'sharp' the price path is.
        Score 1.0 = Perfect explosive alignment. 
        Score 0.0 = Chaotic noise.
        """
        # Heuristic based on volume, RSI acceleration and ATR expansion
        vol_boost = min(1.0, setups.get('volume_ratio', 1.0) / 3.0)
        
        # RSI alignment with direction
        rsi = setups.get('rsi', 50)
        rsi_align = 0.0
        if direction == 'LONG':
            rsi_align = (rsi - 40) / 40 # 0 if 40, 1 if 80
        else:
            rsi_align = (60 - rsi) / 40 # 0 if 60, 1 if 20
        rsi_align = np.clip(rsi_align, 0.0, 1.0)
        
        # Final intensity score
        score = (vol_boost * 0.4) + (rsi_align * 0.6)
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_hurst_exponent(self, returns: np.ndarray) -> float:
        """
        V5.17: Hurst Exponent Axiom.
        H > 0.5: Trend (Memory)
        H < 0.5: Mean Reversion (Anti-persistent)
        """
        if returns is None or len(returns) < 50:
            return 0.5
            
        # Simplified R/S approximation for performance
        try:
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(np.clip(poly[0] * 2.0, 0.2, 0.8))
        except:
            return 0.5

    def _classify_noise_color(self, returns: np.ndarray) -> Tuple[str, float]:
        """
        V5.31: Classifies noise structure (Spectral Slope α).
        - White (α ≈ 0): Pure random chaos.
        - Pink (α ≈ 1): Fractal/Structured chaos (Standard for markets).
        - Brown (α ≈ 2): Strong persistence (Trending noise).
        Returns: (color_name, hurricane_flow_score)
        """
        if returns is None or len(returns) < 20:
            return "WHITE", 0.0
            
        try:
            # Calculate autocorrelation at various lags
            lags = [1, 2, 3, 5]
            coeffs = []
            for l in lags:
                c = np.corrcoef(returns[l:], returns[:-l])[0, 1]
                coeffs.append(abs(c))
            
            # Estimate spectral power slope (simplified)
            # alpha is the decay rate of autocorrelation
            alpha = np.mean(coeffs) * 2.5 # Heuristic mapping to alpha
            
            if alpha < 0.4:
                return "WHITE", 0.0
            elif alpha < 1.3:
                # Pink Noise: High energy fractal order
                flow = (alpha - 0.4) / 0.9
                return "PINK", float(np.clip(flow, 0.0, 1.0))
            else:
                # Brown Noise: High persistence
                flow = 0.8 + (alpha - 1.3) * 0.2
                return "BROWN", float(np.clip(flow, 0.8, 1.0))
        except:
            return "WHITE", 0.0

    def _calculate_chaos_compactness(self, returns: np.ndarray) -> float:
        """
        V5.32: Phase Space Reconstruction & Compactness Analysis.
        Measures if the chaos is orbiting a stable attractor (compact) 
        or if it is divergent and purely random.
        """
        if returns is None or len(returns) < 10:
            return 0.5
            
        try:
            # Simple 2D Phase Space Reconstruction (returns[t] vs returns[t-1])
            x = returns[1:]
            y = returns[:-1]
            
            # Distance from origin in phase space (Energy state)
            distances = np.sqrt(x**2 + y**2)
            
            # Density / Compactness: Inverse of standard deviation of orbit
            # If orbit is stable, std is low.
            orbit_stability = np.std(distances)
            mean_energy = np.mean(distances)
            
            if mean_energy == 0: return 0.5
            
            # Compactness metric: High if orbit is tight/predictable
            compactness = 1.0 / (1.0 + (orbit_stability / mean_energy))
            return float(np.clip(compactness, 0.0, 1.0))
        except:
            return 0.5

    def _calculate_butterfly_sensitivity(self, returns: np.ndarray) -> float:
        """
        V5.33: Sensitivity to Initial Conditions (χ).
        Measures the local divergence rate (Short-term Lyapunov).
        High χ = High sensitivity (Butterfly Effect), potential for explosive moves.
        """
        if returns is None or len(returns) < 8:
            return 0.5
            
        try:
            # Look at the last 8 returns and their cumulative sum (price path)
            window = returns[-8:]
            # Local divergence: sum of absolute changes vs absolute total change
            # If path is zigzagging (high divergence), sensitivity is high.
            abs_sum = np.sum(np.abs(window))
            total_abs_move = np.abs(np.sum(window))
            
            if total_abs_move == 0: return 0.9 # Pure noise / sensitive
            
            # Sensitivity: How much 'extra' activity exists vs the net result.
            chi = abs_sum / (total_abs_move + 1e-6)
            # Normalize to 0-1 range (heuristic)
            return float(np.clip(chi / 4.0, 0.0, 1.0))
        except:
            return 0.5

    def _calculate_resonance_state(self, returns: np.ndarray) -> float:
        """
        V5.34: Chaos Resonance (Simplified Takens Embedding).
        Reconstructs state space with delay tau=2 to find cyclic order.
        If the 'orbit' in phase space is resonant, index is high.
        """
        if returns is None or len(returns) < 12:
            return 0.0
            
        try:
            # Takens Embedding with delay tau=2
            tau = 2
            x = returns[tau:]
            y = returns[:-tau]
            
            # Resonance: Correlation between time-delayed states
            # In purely random noise, correlation with delayed state is 0.
            # In structured chaos (attractors), cyclic resonance appears.
            resonance = np.corrcoef(x, y)[0, 1]
            # We look for ANY structured correlation (positive or negative)
            index = abs(resonance)
            
            return float(np.clip(index * 2.5, 0.0, 1.0)) # Scaled to highlight resonance
        except:
            return 0.0

    def _simulate_superposition_paths(self, base_omni: float, psi_l: float, psi_s: float, h_shield: float, g_boost: float, kappa: float, fabric_t: float) -> Tuple[float, str]:
        """
        V5.42: Quantum Superposition of Strategies (|φ⟩).
        Simulates 3 parallel timelines: Conservative, Dynamic, Aggressive.
        Returns: (superposition_coherence, collapsed_path)
        """
        try:
            # Línea 1: Conservadora (|φ_c⟩)
            # Alta selectividad, busca pureza total del tejido.
            score_c = base_omni * 0.8 * fabric_t * kappa / h_shield
            
            # Línea 2: Dinámica (|φ_d⟩) 
            # Equilibrio fluido, adaptándose al ritmo.
            score_d = base_omni * g_boost * fabric_t * kappa
            
            # Línea 3: Agresiva (|φ_a⟩)
            # Caza singularidades con fuerza gravitacional.
            score_a = base_omni * g_boost * 1.5 * kappa
            
            # Coherencia de Superposición: ¿Qué tan alineadas están?
            # Si el std es bajo, hay alta coherencia.
            scores = np.array([score_c, score_d, score_a])
            coherence = 1.0 - np.std(scores) / (np.mean(scores) + 0.01)
            coherence = float(np.clip(coherence, 0.0, 1.0))
            
            # Colapso de Onda: Elegimos la línea dominante
            # Normalmente la dinámica, pero si Rs es muy alto, la Agresiva colapsa.
            if score_a > 1.2 * score_d:
                path = "AGGRESSIVE"
            elif score_c > 0.9 * score_d:
                path = "CONSERVATIVE"
            else:
                path = "DYNAMIC"
                
            return coherence, path
        except:
            return 0.5, "DYNAMIC"

    def _generate_meta_reasoning(self, symbol: str, win_prob: float, entropy: float, fabric_t: float, coherence: float) -> Dict[str, Any]:
        """
        V5.46 Meta-Cognition: Generates explainable reasoning for parameter drift.
        """
        reasoning = {
            'state': "STABLE" if coherence > 0.7 else "UNSTABLE",
            'fabric_tension': fabric_t,
            'suggestion': "No change required.",
            'logic': ""
        }
        
        if coherence < 0.5:
            reasoning['suggestion'] = "Increase SL Buffer"
            reasoning['logic'] = f"The Neuro-Evolutionary Fabric for {symbol} is unstable (Coh={coherence:.2f}). Increasing defensive parameters."
        elif win_prob > 0.85:
            reasoning['suggestion'] = "Tighten TP"
            reasoning['logic'] = f"High conviction detected. Optimization suggests locking profits faster in this micro-universe."
        elif fabric_t > 1.2:
            reasoning['suggestion'] = "Scale Weights"
            reasoning['logic'] = f"Physics-Technical resonance is high. Strengthening neural fabric connectivity."
            
        return reasoning

    def _calculate_quantum_singularity(self, psi_l: float, psi_s: float, h_accel: float) -> Tuple[float, float]:
        """
        V5.40: Quantum Singularity (Horizonte de Eventos).
        Detects the point where uncertainty collapses and momentum becomes inevitable.
        Returns: (horizon_proximity (Rs), gravitational_boost)
        """
        try:
            # Determinamos la fuerza del campo gravitacional (Amplitud + Aceleración)
            q_force = (psi_l + psi_s) / 2.0
            
            # Singularity condition: High wave amplitude AND negative entropy acceleration (collapsing uncertainty)
            # R_s (Event Horizon proximity): closer to 1.0 means inevitable collapse
            rs = q_force * (1.0 - np.clip(h_accel, -1.0, 1.0))
            rs = float(np.clip(rs, 0.0, 1.0))
            
            # Gravitational Boost: Force multiplier if we are near the horizon
            g_boost = 1.0
            if rs > 0.85:
                # We are at the Event Horizon
                g_boost = 1.0 + (rs * 1.5) # Massive boost for inevitable breakouts
                
            return rs, float(g_boost)
        except:
            return 0.0, 1.0

    def _calculate_fabric_tension(self, setups: Dict[str, float], psi_l: float, psi_s: float, direction: str) -> Tuple[float, float]:
        """
        V5.37/V5.39/V5.41: Fabric Tension (T) with Autotuning.
        Measures harmony with dynamic sensitivity based on market state.
        Returns: (tension, harmony_stability)
        """
        try:
            rsi = setups.get('rsi', 50.0)
            adx = setups.get('adx', 20.0)
            atr_pct = setups.get('atr_pct', 0.001)
            
            # Autotuning: Sensitivity (S) increases with lower volatility (search for micro-tension)
            # and decreases with high volatility (avoiding over-triggering)
            sensitivity = 1.0 / (1.0 + atr_pct * 100)
            sensitivity = np.clip(sensitivity, 0.5, 1.5)
            
            tech_dir = 0.5
            if direction == "LONG":
                tech_dir = (rsi / 100.0) * (adx / 50.0)
            else:
                tech_dir = ((100.0 - rsi) / 100.0) * (adx / 50.0)
            
            q_dir = psi_l if direction == "LONG" else psi_s
            
            # Tension = Difference weighted by sensitivity
            raw_tension = abs(tech_dir - q_dir)
            tension = 1.0 - (raw_tension * sensitivity)
            
            # Harmony Stability (V5.41): How reliable is this tension
            stability = 1.0 - (atr_pct * 10.0)
            
            return float(np.clip(tension, 0.0, 1.0)), float(np.clip(stability, 0.0, 1.0))
        except:
            return 0.5, 0.5

    def _liquid_neural_modulation(self, kappa: float, tension: float, stability: float) -> float:
        """
        V5.39/V5.41: Liquid Neural Modulation with Hysteresis.
        Refines weighting based on both tension and historical stability.
        """
        # Linear Modulation
        modulation = kappa * tension
        
        # Hysteresis (V5.41): Stability reinforces the modulation
        if stability > 0.7:
            modulation *= (1.0 + (stability - 0.7)) # Reward consistent fabric
            
        # Power scaling for harmony
        if tension > 0.85 and stability > 0.8:
            modulation *= 1.3 # Super-Resonancia de Seda
            
        return float(np.clip(modulation, 0.4, 2.0))

    def _apply_quantum_feedback(self, psi_l: float, psi_s: float, prior_wr: float) -> Tuple[float, float, float, float]:
        """
        V5.38: Quantum Feedback Loop (Observer Effect).
        Recent performance (prior_wr) acts as a Back-Action on the wave function.
        
        > [!WARNING]
        > **M-1: Directional Bias In Quantum Feedback**
        > The adjustment `psi_s_adj = psi_s * kappa - bias` means that if the prior win rate is
        > high (>60%), `bias` is positive and `psi_s` (bearish amplitude) is artificially reduced.
        > This creates a permanent bias towards LONG positions when the bot has a good win rate.
        
        Returns: Adjusted (psi_l, psi_s), coherence (kappa), bias.
        """
        # Target WR is 60% (0.60)
        target_wr = 0.60
        delta_p = prior_wr - target_wr
        
        # Coherence (kappa): 1.0 (Target) +/- sensitivity
        # If WR is high, kappa > 1.0 (Constructive Feedback)
        # If WR is low, kappa < 1.0 (Wave Diffusion / Protective)
        kappa = 1.0 + (delta_p * 0.5)
        kappa = float(np.clip(kappa, 0.7, 1.3))
        
        # Feedback Bias: Corrects the wave to favor successful orbits
        bias = delta_p * 0.2
        
        # Apply feedback to amplitudes (Renormalization)
        # FORENSIC FIX: Remove directional bias. High WR should boost overall coherence, not artificially force LONG signals.
        psi_l_adj = np.clip(psi_l * kappa, 0.0, 1.0)
        psi_s_adj = np.clip(psi_s * kappa, 0.0, 1.0)
        
        return float(psi_l_adj), float(psi_s_adj), float(kappa), float(bias)

    def _calculate_quantum_amplitude_vectorial(self, returns: np.ndarray) -> Tuple[float, float]:
        """
        V5.37: Vectorial Schrödinger Amplitude. 
        Separates energy into Bullish (ψL) and Bearish (ψS) components.
        """
        if returns is None or len(returns) < 20:
            return 0.5, 0.5
            
        try:
            # Separar retornos positivos y negativos
            pos_returns = returns[returns > 0]
            neg_returns = returns[returns < 0]
            
            # Potencial del sistema (V)
            potential = np.std(returns[-20:]) if len(returns) >= 20 else 0.001
            
            # ψL (Long Amplitude)
            e_l = np.mean(pos_returns[-10:]) if len(pos_returns) > 0 else 0
            psi_l = np.abs(e_l) / potential if potential > 0 else 0.5
            
            # ψS (Short Amplitude)
            e_s = np.mean(neg_returns[-10:]) if len(neg_returns) > 0 else 0
            psi_s = np.abs(e_s) / potential if potential > 0 else 0.5
            
            return float(np.clip(psi_l, 0.0, 1.0)), float(np.clip(psi_s, 0.0, 1.0))
        except:
            return 0.5, 0.5

    def _calculate_interference_pattern(self, psi_l: float, psi_s: float) -> float:
        """
        V5.37: Quantum Interference Matrix.
        I = (ψL + ψS)^2 - (ψL - ψS)^2 normalized.
        Detects if forces are amplifying or canceling each other.
        """
        # Simplificación: Diferencia de fase simulada basada en la dominancia
        diff = abs(psi_l - psi_s)
        total = psi_l + psi_s
        
        if total == 0: return 1.0
        
        # Coherence: High if one dominates, Low if both are balanced (Destructive)
        coherence = diff / total
        
        # Interference Factor: 0.0 (Destructive) to 2.0 (Constructive)
        return float(np.clip(coherence * 2.0, 0.0, 2.0))

    def _calculate_dirac_energy(self, returns: np.ndarray) -> float:
        """
        V5.37: Dirac Sea Energy Level (E).
        Measures the 'vacuum' excitation. High energy in the Dirac Sea 
        precedes a Quantum Jump (Breakout).
        """
        if returns is None or len(returns) < 50: return 0.5
        
        try:
            # Use rolling variance acceleration as excitation proxy
            recent_std = np.std(returns[-10:])
            baseline_std = np.std(returns[-50:])
            
            if baseline_std == 0: return 0.5
            
            excitation = recent_std / baseline_std
            return float(np.clip(excitation, 0.0, 1.0))
        except:
            return 0.5

    def _calculate_temporal_tunneling(self, returns: np.ndarray) -> float:
        """
        V5.37: Temporal Tunneling.
        Cross-correlates the wave with its own past to find hidden cycles.
        """
        if returns is None or len(returns) < 40: return 0.0
        
        try:
            # Correlación de la ventana actual con la ventana desplazada
            w1 = returns[-20:]
            w2 = returns[-40:-20]
            corr = np.corrcoef(w1, w2)[0, 1]
            return float(np.clip(abs(corr), 0.0, 1.0))
        except:
            return 0.0

    def _calculate_quantum_amplitude(self, returns: np.ndarray) -> float:
        """
        V5.36: Schrödinger Probability Amplitude (ψ).
        Calculates the wave function energy based on return density and volatility.
        Higher amplitude means the 'wave' is collapsing into a deterministic direction.
        """
        if returns is None or len(returns) < 20:
            return 0.5
            
        try:
            # Energy E = Mean returns normalized
            energy = np.mean(returns[-10:])
            # Potential V = Volatility (Standard Deviation)
            potential = np.std(returns[-20:])
            
            # Wave Function Amplitude (simplified)
            # ψ ~ exp(E/V) if V > 0
            if potential > 0:
                psi = np.abs(energy) / potential
            else:
                psi = 0.5
                
            return float(np.clip(psi, 0.0, 1.0))
        except:
            return 0.5

    def _calculate_entanglement_factor(self, symbol: str, returns: np.ndarray, btc_returns: np.ndarray) -> float:
        """
        V5.36: Quantum Entanglement.
        Measures instant correlation with the market leader (BTC).
        If Entanglement is high, the symbol's wave is coupled with BTC.
        """
        if returns is None or btc_returns is None:
            return 1.0
            
        try:
            # Sync lengths
            min_len = min(len(returns), len(btc_returns), 20)
            if min_len < 10: return 1.0
            
            # Cross-correlation (Entanglement)
            corr = np.corrcoef(returns[-min_len:], btc_returns[-min_len:])[0, 1]
            # We care about the coherence (absolute correlation)
            return float(np.clip(abs(corr), 0.0, 1.0))
        except:
            return 1.0

    def _calculate_shannon_leverage(self, entropy: float, win_prob: float) -> float:
        """
        V5.17: Adaptive leverage based on certainty.
        Low entropy + High WinProb = High Sovereign Force.
        """
        if entropy > 1.2 or win_prob < 0.6:
            return 1.0
            
        # Scaling factor: If entropy is 0 and win_prob is 1.0, mult = 5.0 (50x cap in engine)
        mult = (1.5 - entropy) * win_prob * 2.0
        return float(np.clip(mult, 1.0, 5.0))

    def _detect_vortex_pulse(self, setups: Dict[str, float]) -> Tuple[float, bool]:
        """
        V5.18: Detects Liquidity Vortexes.
        Measures volume deviation from standard activity.
        """
        v_ratio = setups.get('volume_ratio', 1.0)
        
        # High-Energy: Volume > 2.5 is rare and indicates explosive momentum
        is_regime = v_ratio > 2.5
        return float(v_ratio), bool(is_regime)

    def _detect_whales(self, setups: Dict[str, float]) -> float:
        """
        V5.19: Detects institutional activity (Whales).
        Compares current volume ratio vs a broader window.
        """
        # We assume 'volume_ratio_4h' is passed in setups from strategy
        return float(setups.get('volume_ratio_4h', 1.0))

    def _detect_breakout(self, setups: Dict[str, float], direction: str) -> bool:
        """
        V5.19: Detects price breakouts of the last 50 periods.
        """
        # We assume 'is_50_bar_high' / 'is_50_bar_low' passed in setups
        if direction == "LONG":
            return bool(setups.get('is_50_bar_high', False))
        else:
            return bool(setups.get('is_50_bar_low', False))

    def _calculate_noise_density(self, returns: np.ndarray) -> Tuple[float, float]:
        """
        V5.20: Noise Predator core logic.
        Measures 'Spectral Density' of noise vs signal.
        QUÉ: Calcula la desviación estándar de los residuos.
        CÓMO: Residuos = Retornos - Media Móvil de Retornos.
        """
        if returns is None or len(returns) < 15:
            return 0.5, 0.001
            
        # Window of 15 bars for local noise
        recent = returns[-15:]
        mean_ret = np.mean(recent)
        residuals = recent - mean_ret
        noise_sigma = float(np.std(residuals))
        
        # Noise Level normalization: Compare local std vs historical std
        hist_std = np.std(returns) if len(returns) > 0 else 0.001
        noise_level = float(np.clip(noise_sigma / hist_std, 0.0, 1.0)) if hist_std > 0 else 0.5
        
        return noise_level, noise_sigma
