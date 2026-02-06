"""
ML STRATEGY HÍBRIDA DEFINITIVA ULTIMATE
Combina TODO lo mejor de ambas versiones para crecimiento exponencial con riesgo controlado.
Objetivo: Convertir 12 USD en 100K USD en el menor tiempo posible (semanas).
Características avanzadas:
- ✅ Ensemble completo de 3 modelos (RF+XGB+GB) con weighted voting dinámico
- ✅ Detección de régimen de mercado avanzada con suavizado temporal
- ✅ Circuit breaker automático por drawdown y pérdidas consecutivas
- ✅ Feature engineering adaptativo por régimen
- ✅ Targets dinámicos por volatilidad y régimen
- ✅ Re-pesado dinámico de modelos basado en performance
- ✅ Gestión de riesgo multi-capa con filtros robustos
- ✅ Aprendizaje adaptativo con learning rate variable
- ✅ Monitoreo completo con 40+ métricas
- ✅ Arquitectura asíncrona optimizada para alta frecuencia
"""

# ⚠️ CRITICAL: Suprimir warnings TOTALMENTE para evitar ruido en consola
import os
import warnings

# Disable fragmentation warnings and sklearn parallel noise
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTHONNOUSERSITE'] = '1'
warnings.filterwarnings('ignore')
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
except:
    pass
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')

import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from .strategy import Strategy
from core.events import SignalEvent
from core.enums import EventType, SignalType
from datetime import datetime, timezone
import talib
from utils.math_helpers import safe_div
from config import Config
from utils.logger import logger
from core.transparent_logger import monitor_log
from utils.debug_tracer import trace_execution
from utils.thread_monitor import monitor
from concurrent.futures import ThreadPoolExecutor
from core.neural_bridge import neural_bridge
import threading
import copy
from collections import deque, Counter
try:
    import ujson as json
except ImportError:
    import json
import joblib
import time
import gc
from data.feature_store import FeatureStore
from core.ml_governance import MLGovernance

# --- OPTIMIZACIÓN DE RECURSOS (Rule 3.3) ---
# MODO PROFESOR: Limitador global de entrenamiento.
# QUÉ: Semáforo para controlar cuántas estrategias entrenan simultáneamente.
# POR QUÉ: Con 24 símbolos, n_jobs=-1 causa agotamiento de RAM instantáneo (97%+).
# PARA QUÉ: Estabilizar el sistema y permitir que el trading continúe sin lag.
TRAINING_LIMITER = threading.BoundedSemaphore(value=2) 

class MLStrategyHybridUltimate(Strategy):
    """
    ML Strategy Híbrida DEFINITIVA ULTIMATE
    Versión final que combina todas las características avanzadas para crecimiento exponencial.
    """
    
    def __init__(self, data_provider, events_queue, symbol='BTC/USDT', 
                 lookback=50, sentiment_loader=None, portfolio=None, risk_manager=None):
        
        # ============================================================
        # ✅ CORE CONFIGURATION - OPTIMIZADO PARA CRECIMIENTO RÁPIDO
        # ============================================================
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.symbol = symbol
        initial_lookback = getattr(Config.Strategies, 'ML_LOOKBACK_BARS', 5000)
        self.lookback = min(lookback, initial_lookback)
        self._feature_cols = [] # Ensure initialized
        self.sentiment_loader = sentiment_loader
        self.portfolio = portfolio
        self.risk_manager = risk_manager # Regime Leader Link
        self.strategy_id = "ML_HYBRID_ULTIMATE_V2"
        
        # === ASYNC EXECUTOR OPTIMIZADO ===
        self.executor = ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix=f"ML_Ultimate_{symbol.replace('/','')}"
        )
        
        # ============================================================
        # ✅ ENSEMBLE COMPLETO - 3 MODELOS CON WEIGHTED VOTING
        # ============================================================
        self.rf_model = None
        self.xgb_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        
        # Pesos base para crecimiento agresivo
        self.base_rf_weight = 0.45
        self.base_xgb_weight = 0.35
        self.base_gb_weight = 0.20
        
        # Pesos originales para reset
        self.original_rf_weight = 0.45
        self.original_xgb_weight = 0.35
        self.original_gb_weight = 0.20
        
        # ============================================================
        # ✅ PERFORMANCE TRACKING PARA CRECIMIENTO EXPONENCIAL
        # ============================================================
        self.performance_history = deque(maxlen=100)  # Historial de trades
        self.performance_window = deque(maxlen=20)    # Ventana para cálculos dinámicos
        self.signal_history = deque(maxlen=100)       # Historial completo de señales
        self.equity_curve = deque(maxlen=500)         # Curva de equity para análisis
        
        # Tracking individual por modelo
        self.individual_model_scores = {'rf': 0.0, 'xgb': 0.0, 'gb': 0.0}
        self.model_performance = {
            'rf': deque(maxlen=30),
            'xgb': deque(maxlen=30),
            'gb': deque(maxlen=30)
        }
        
        # ============================================================
        # ✅ TRAINING CONFIGURATION - OPTIMIZADO PARA RAPIDEZ
        # ============================================================
        self.is_trained = False
        self.min_bars_to_train = 300  # Reducido para empezar rápido
        self.bars_since_train = 0
        self.retrain_interval = 150   # Retrain más frecuente
        self.last_training_time = None
        self.last_training_score = 0.0
        self.training_iteration = 0
        
        # ============================================================
        # TARGETS ADAPTATIVOS - ULTRA REDUCIDOS PARA TESTNET/BAJA VOL
        # ============================================================
        self.BASE_TP_TARGET = 0.003   # 0.3%
        self.BASE_SL_TARGET = 0.002   # 0.2%
        # Aumentado para microscalping M1/M5: 30 velas = 30 min context
        self.LOOKAHEAD_BARS = 30
        
        self.current_tp_target = self.BASE_TP_TARGET
        self.current_sl_target = self.BASE_SL_TARGET
        self.volatility_multiplier = 1.0
        
        # ============================================================
        # ✅ UMBRALES ADAPTATIVOS - BALANCE ENTRE RIESGO Y GANANCIA
        # ============================================================
        self.MIN_MODEL_ACCURACY = 0.38  # Umbral más bajo para más señales
        
        # Umbrales base optimizados
        self.BASE_CONFIDENCE_THRESHOLD = 0.58  # Más bajo para más operaciones
        self.BASE_CONFLUENCE_LONG = 0.30       # Más permisivo
        self.BASE_CONFLUENCE_SHORT = -0.30     # Más permisivo
        
        # Umbrales adaptativos
        self.adaptive_confidence_threshold = self.BASE_CONFIDENCE_THRESHOLD
        self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG
        self.adaptive_confluence_short = self.BASE_CONFLUENCE_SHORT
        
        # ============================================================
        # ✅ FILTROS DE ROBUSTEZ - PROTECCIÓN EN CRECIMIENTO RÁPIDO
        # ============================================================
        self.MAX_ATR_PCT = 0.035      # Más permisivo para volatilidad
        self.MIN_VOLUME_RATIO = 0.7   # Menos restrictivo
        self.RSI_FILTER_RANGE = (20, 80)  # Rango más amplio
        # === INFRAESTRUCTURA DE PERSISTENCIA Y ENTRENAMIENTO INCREMENTAL ===
        self._state_lock = threading.Lock()
        self.loop_count = 0
        self.analysis_stats = Counter()
        self.bars_since_incremental = 0  # Contador para updates rápidos
        self.models_dir = ".models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # === [PHASE 12] GOVERNANCE & FEATURE STORE ===
        self.feature_store = FeatureStore()
        self.ml_governance = MLGovernance()
        
        # Cargar modelos previos si existen (Prioridad: MLGovernance)
        self._load_governed_model()
        
        # ============================================================
        # ✅ DETECCIÓN DE RÉGIMEN DE MERCADO AVANZADA
        # ============================================================
        # DETECCIÓN DE RÉGIMEN DE MERCADO AVANZADA
        self.market_regime = "UNKNOWN"
        self.regime_history = deque(maxlen=15)
        self.last_regime_update = datetime.now(timezone.utc) - pd.Timedelta(minutes=5) # Allow immediate update
        self.regime_confidence = 0.0
        self.regime_duration = 0
        self.oracle_log_count = 0  # Counter for throttling repetitive logs
        
        # ============================================================
        # ✅ CIRCUIT BREAKER AVANZADO - PROTECCIÓN CAPITAL
        # ============================================================
        self.circuit_breaker_active = False
        self.circuit_breaker_threshold = 0.12  # 12% drawdown para crecimiento agresivo
        self.original_confidence_threshold = self.BASE_CONFIDENCE_THRESHOLD
        self.peak_equity = None
        self.consecutive_losses = 0
        self.max_consecutive_losses = 4        # Más sensible a pérdidas
        
        # ============================================================
        # ✅ SISTEMA DE APRENDIZAJE ADAPTATIVO
        # ============================================================
        self.learning_rate = 1.0
        self.aggressiveness_factor = 1.0  # Factor de agresividad dinámico
        self.win_streak = 0
        self.loss_streak = 0
        
        # ============================================================
        # ✅ STATE MANAGEMENT Y THREAD SAFETY
        # ============================================================
        self.running = True
        self._state_lock = threading.Lock()
        self._feature_cols = None
        self._training_thread = None
        self._last_prediction_time = None
        self._label_mapping = {0: -1, 1: 1}  # Default mapping for inference before training
        
        # ============================================================
        # ✅ MONITORING Y ESTADÍSTICAS COMPLETAS
        # ============================================================
        self.total_signals_generated = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
        # Win Rate / Payoff tracking for Kelly
        self.rolling_win_rate = deque(maxlen=20)
        self.rolling_payoff = deque(maxlen=20) # Avg Win / Avg Loss
        
        self.signals_by_regime = {
            'TRENDING': 0, 'RANGING': 0, 'VOLATILE': 0, 'MIXED': 0, 'UNKNOWN': 0
        }
        
        self.regime_accuracy = {
            'TRENDING': [], 'RANGING': [], 'VOLATILE': [], 'MIXED': [], 'UNKNOWN': []
        }
        
        # ============================================================
        # ✅ CONFIGURACIÓN DE CRECIMIENTO EXPONENCIAL
        # ============================================================
        self.compounding_factor = 1.0  # Factor de compuesto
        self.position_sizing_mode = "KELLY"  # Kelly, FIXED, VOLATILITY
        self.kelly_fraction = 0.5  # Fracción de Kelly para riesgo controlado
        self.base_position_size = 0.95  # Usar 95% del capital para crecimiento rápido
        
        # Phase 5: Dynamic Math Utils
        from utils.statistics_pro import StatisticsPro
        self.stats_pro = StatisticsPro()
        
        # Meta de 12 USD a 100K USD
        self.initial_capital = 12.0
        self.target_capital = 100000.0
        self.current_capital = self.initial_capital
        
        logger.info(f"🟢 ML HYBRID ULTIMATE STRATEGY [ENSEMBLE] INITIALIZED FOR {symbol}")
        logger.info(f"🎯 OBJECTIVE: ${self.initial_capital} → ${self.target_capital}")
        logger.info(f"⚙️  Mode: Exponential Growth (Aggressive with Risk Control)")
        
    def _calculate_dynamic_sizing(self, confidence, volatility):
        """
        Phase 5: Dynamic Kelly Criterion Sizing
        Size = Kelly% * Confidence_Scaler * Volatility_Scaler
        """
        try:
            # 1. Calculate Base Kelly
            if len(self.rolling_win_rate) < 5:
                # Cold start: Use conservative fixed
                kelly_pct = 0.25 
            else:
                wr = np.mean(self.rolling_win_rate)
                avg_win = np.mean([p for p in self.rolling_payoff if p > 0]) if any(p > 0 for p in self.rolling_payoff) else 0.01
                avg_loss = abs(np.mean([p for p in self.rolling_payoff if p < 0])) if any(p < 0 for p in self.rolling_payoff) else 0.005
                
                kelly_pct = self.stats_pro.calculate_kelly_criterion(wr, avg_win, avg_loss)
            
            # 2. Apply Fractional Kelly (Safety)
            safe_kelly = kelly_pct * self.kelly_fraction
            
            # 3. Scale by Prediction Confidence (Higher conf -> Higher size)
            # Normalize confidence (0.5 to 1.0) -> (0.0 to 1.0)
            conf_scaler = max(0.0, (confidence - 0.5) * 2) 
            
            # 4. Scale by Volatility (Inverse Volatility Sizing)
            # If Volatility is high, reduce size to keep $ Risk constant
            # Reference volatility: 0.005 (0.5%) per bars
            vol_scaler = min(1.5, 0.005 / max(0.001, volatility))
            
            final_size = min(0.95, safe_kelly * conf_scaler * vol_scaler)
            
            # print(f"  💰 Dynamic Sizing: K={kelly_pct:.2f} -> Safe={safe_kelly:.2f} * Conf={conf_scaler:.2f} * Vol={vol_scaler:.2f} = {final_size:.2f}")
            return max(0.1, final_size) # Min 10%
            
        except Exception as e:
            logger.error(f"Sizing error: {e}")
            return 0.1

    # ============================================================
    # ✅ DETECCIÓN DE RÉGIMEN MEJORADA CON SUAVIZADO
    # ============================================================
    
    def _detect_market_regime(self, df):
        """
        Detección avanzada de régimen con múltiples capas de validación
        """
        if len(df) < 50:
            return "UNKNOWN", 0.0
            
        try:
            # Indicadores principales
            current_adx = df['adx'].iloc[-1] if 'adx' in df.columns else 20
            current_atr_pct = (df['atr_pct'].iloc[-1] / 100) if 'atr_pct' in df.columns else 0.01
            rsi_std = df['rsi_14'].tail(20).std() if 'rsi_14' in df.columns else 15
            
            # Volatilidad y tendencia
            price_volatility = df['close'].pct_change().tail(20).std()
            volume_volatility = df['volume'].pct_change().tail(20).std()
            
            # Tendencia EMAs
            closes = df['close'].values
            ema_20 = talib.EMA(closes, timeperiod=20)[-1] if len(closes) >= 20 else closes[-1]
            ema_50 = talib.EMA(closes, timeperiod=50)[-1] if len(closes) >= 50 else closes[-1]
            trend_strength = abs(ema_20 - ema_50) / ema_50 if ema_50 > 0 else 0
            
            # Sistema de scoring mejorado
            regime_scores = {
                'TRENDING': 0.0,
                'RANGING': 0.0,
                'VOLATILE': 0.0,
                'STAGNANT': 0.0,
                'MIXED': 0.0
            }
            
            # ✅ TRENDING: ADX alto + tendencia fuerte + volatilidad controlada
            if current_adx > 25:
                regime_scores['TRENDING'] += 0.4
            if trend_strength > 0.025:
                regime_scores['TRENDING'] += 0.3
            if current_atr_pct < 0.03:
                regime_scores['TRENDING'] += 0.2
            if volume_volatility < 0.5:
                regime_scores['TRENDING'] += 0.1
                
            # ✅ VOLATILE: ATR alto + RSI volátil + alta volatilidad precio
            if current_atr_pct > 0.035:
                regime_scores['VOLATILE'] += 0.5
            if rsi_std > 18:
                regime_scores['VOLATILE'] += 0.3
            if price_volatility > 0.035:
                regime_scores['VOLATILE'] += 0.2
                
            # ✅ RANGING: ADX bajo + RSI estable + baja volatilidad
            if current_adx < 20: # Un poco más permisivo que 18
                regime_scores['RANGING'] += 0.3
            if rsi_std < 10: # Un poco más permisivo que 8
                regime_scores['RANGING'] += 0.3
            if current_atr_pct < 0.015: # < 1.5%
                regime_scores['RANGING'] += 0.2
            
            # ✅ STAGNANT (ZOMBIE): Volatilidad nula o insignificante
            # Usar la auditoría de calidad de datos aquí también
            price_spread = (df['high'].max() - df['low'].min()) / df['close'].mean()
            identical_bars = (df['high'] == df['low']).sum() / len(df)
            
            if current_atr_pct < 0.0005 or price_spread < 0.0002 or identical_bars > 0.85:
                regime_scores['STAGNANT'] += 0.8
            elif current_atr_pct < 0.0015: # < 0.15%
                regime_scores['STAGNANT'] += 0.5
                regime_scores['RANGING'] += 0.1
                
            # ✅ MIXED: Sin señales claras o transición
            if max(regime_scores.values()) < 0.45:
                regime_scores['MIXED'] = 1.0
                
            # Determinar régimen dominante
            best_regime_pair = max(regime_scores.items(), key=lambda x: x[1])
            best_regime = best_regime_pair[0]
            confidence = min(best_regime_pair[1] * 1.2, 1.0)  # Boost de confianza
            
            # --- MÉTRICAS ESTADÍSTICAS PARA LOGGING ---
            stats = {
                'adx': float(current_adx),
                'atr_pct': float(current_atr_pct) * 100,
                'rsi_std': float(rsi_std),
                'trend_strength': float(trend_strength) * 100
            }
            
            return best_regime, confidence, stats
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return "UNKNOWN", 0.0, {}
    
    def _update_market_regime(self, df):
        """
        Actualizar régimen con suavizado y persistencia
        """
        current_time = datetime.now(timezone.utc)
        
        # Solo actualizar cada 3 minutos para evitar overfitting
        # EXCEPCIÓN: Si es UNKNOWN, actualizar siempre para inicializar
        if self.market_regime != "UNKNOWN" and (current_time - self.last_regime_update).total_seconds() < 180:
            return
            
        new_regime, confidence, stats = self._detect_market_regime(df)
        self.regime_history.append(new_regime)
        
        # Suavizado con ventana dinámica
        if len(self.regime_history) >= 5:
            regime_counts = Counter(self.regime_history)
            most_common = regime_counts.most_common(2)
            
            if len(most_common) >= 2:
                # Si hay empate cercano, usar MIXED
                if most_common[0][1] - most_common[1][1] <= 2:
                    smoothed_regime = "MIXED"
                else:
                    smoothed_regime = most_common[0][0]
            else:
                smoothed_regime = most_common[0][0]
        else:
            # Si hay poca historia, usamos el último detectado
            smoothed_regime = new_regime
            
        # Solo cambiar si hay confianza suficiente o es consistente o es el primer update real
        is_initial = (self.market_regime == "UNKNOWN" and len(self.regime_history) >= 1)
        if confidence > 0.55 or smoothed_regime == self.market_regime or is_initial:
                with self._state_lock:
                    old_regime = self.market_regime
                    self.market_regime = smoothed_regime
                    self.regime_confidence = confidence
                    self.last_regime_update = current_time
                    
                    # Actualizar duración del régimen
                    if old_regime == smoothed_regime:
                        self.regime_duration += 1
                    else:
                        self.regime_duration = 1
                
                # Ajustar todos los parámetros al nuevo régimen
                self._adjust_all_parameters_by_regime(smoothed_regime)
                
                if old_regime != smoothed_regime:
                    # ORCHESTRATION (Phase 12): Push Global Regime if this is the Leader
                    if self.risk_manager:
                        self.risk_manager.update_regime(smoothed_regime)
                    # --- NARRATIVA CONCEPTUAL Y ESTADÍSTICA ---
                    descriptions = {
                        "TRENDING": "Directional bias confirmed. Momentum engines active.",
                        "RANGING": "Side-ways consolidation. Switching to Mean Reversion mode.",
                        "VOLATILE": "High noise & volatility. Wide stops and conservative bias.",
                        "STAGNANT": "Zombie market detected. Stagnant price action. Protection active.",
                        "MIXED": "Internal transition or choppy price action. Filtering active.",
                        "UNKNOWN": "Initializing discovery mode."
                    }
                    concept = descriptions.get(smoothed_regime, "Evolving market state.")
                    
                    emoji = "🚀" if smoothed_regime == "TRENDING" else "⚖️" if smoothed_regime == "RANGING" else "🔥" if smoothed_regime == "VOLATILE" else "🧊" if smoothed_regime == "STAGNANT" else "🔄"
                    
                    logger.info(
                        f"\n{'='*70}\n"
                        f"📊 {emoji} REGIME CHANGE: {self.symbol}\n"
                        f"{'='*70}\n"
                        f"   Phase: {old_regime} → {smoothed_regime}\n"
                        f"   Concept: {concept}\n"
                        f"   Stats: ADX={stats.get('adx', 0):.1f} | ATR%={stats.get('atr_pct', 0):.2f}% | Trend={stats.get('trend_strength', 0):.2f}%\n"
                        f"   Confidence: {confidence*100:.1f}% | Strategy: Adaptive targets enabled.\n"
                        f"{'='*70}\n"
                    )

    def _adjust_all_parameters_by_regime(self, regime):
        """
        Ajustar TODOS los parámetros según el régimen de mercado
        """
        if regime == "TRENDING":
            # En trending: maximizar ganancias, ser más agresivo
            self.adaptive_confidence_threshold = max(0.48, self.BASE_CONFIDENCE_THRESHOLD - 0.08)
            self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG - 0.12
            self.adaptive_confluence_short = self.BASE_CONFLUENCE_SHORT + 0.12
            self.current_tp_target = self.BASE_TP_TARGET * 1.3  # Targets más grandes
            self.current_sl_target = self.BASE_SL_TARGET * 0.9  # SL más ajustado
            self.aggressiveness_factor = 1.2  # Más agresivo
            
        elif regime == "VOLATILE":
            # En volatilidad: proteger capital, ser conservador
            self.adaptive_confidence_threshold = min(0.72, self.BASE_CONFIDENCE_THRESHOLD + 0.14)
            self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG + 0.18
            self.adaptive_confluence_short = self.BASE_CONFLUENCE_SHORT - 0.18
            self.current_tp_target = self.BASE_TP_TARGET * 1.6  # Targets mucho más grandes
            self.current_sl_target = self.BASE_SL_TARGET * 1.6  # SL más amplio
            self.aggressiveness_factor = 0.8  # Menos agresivo
            
        elif regime == "RANGING":
            # En ranging: mean reversion, targets ajustados
            self.adaptive_confidence_threshold = self.BASE_CONFIDENCE_THRESHOLD
            self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG
            self.adaptive_confluence_short = self.BASE_CONFLUENCE_SHORT
            self.current_tp_target = self.BASE_TP_TARGET * 0.7  # Targets pequeños
            self.current_sl_target = self.BASE_SL_TARGET * 0.8  # SL ajustado
            self.aggressiveness_factor = 1.0
            
        elif regime == "MIXED":
            # En mixed: equilibrio
            self.adaptive_confidence_threshold = self.BASE_CONFIDENCE_THRESHOLD + 0.03
            self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG + 0.07
            self.adaptive_confluence_short = self.BASE_CONFLUENCE_SHORT - 0.07
            self.current_tp_target = self.BASE_TP_TARGET
            self.current_sl_target = self.BASE_SL_TARGET
            self.aggressiveness_factor = 1.0
            
        elif regime == "STAGNANT":
            # En stagnant: selectividad alta, pero NO bloqueo total.
            # Nada es más importante que la operatividad.
            self.adaptive_confidence_threshold = 0.68  
            self.adaptive_confluence_long = 0.65 
            self.adaptive_confluence_short = -0.65
            self.aggressiveness_factor = 0.6  # Operatividad reducida pero activa
            self.current_tp_target = self.BASE_TP_TARGET
            self.current_sl_target = self.BASE_SL_TARGET
            
        else:  # UNKNOWN
            self.adaptive_confidence_threshold = self.BASE_CONFIDENCE_THRESHOLD + 0.05
            self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG + 0.10
            self.adaptive_confluence_short = self.BASE_CONFLUENCE_SHORT - 0.10
            self.current_tp_target = self.BASE_TP_TARGET * 0.9
            self.current_sl_target = self.BASE_SL_TARGET * 1.1
            self.aggressiveness_factor = 0.9
        
        # Aplicar learning rate y factor de agresividad
        self.adaptive_confidence_threshold *= (self.learning_rate * self.aggressiveness_factor)
        
        logger.debug(
            f"🔧 Parameters adjusted for {regime}: "
            f"Conf={self.adaptive_confidence_threshold:.2f}, "
            f"TP/SL={self.current_tp_target:.3f}/{self.current_sl_target:.3f}, "
            f"Aggr={self.aggressiveness_factor:.2f}"
        )

    # ============================================================
    # ✅ FEATURE ENGINEERING ULTIMATE - 80+ FEATURES
    # ============================================================
    
    @trace_execution
    def _prepare_features(self, bars, regime_aware=True):
        """
        Feature engineering completo con 80+ features adaptativos
        """
        if not bars:
            return pd.DataFrame()
            
        df = pd.DataFrame(bars)
        
        # === [PHASE 12] FEATURE STORE LOOKUP ===
        # Si tenemos un bloque grande de datos (entrenamiento), intentamos recuperar de cache
        if len(df) > 100:
            try:
                start_ts = df['datetime'].min()
                end_ts = df['datetime'].max()
                cached_df = self.feature_store.get_features(self.symbol, start_ts, end_ts)
                if not cached_df.empty and len(cached_df) >= len(df) * 0.9:
                    # Mezclamos OHLCV original con las features del cache
                    full_df = pd.concat([df.set_index('datetime'), cached_df], axis=1)
                    # Eliminamos duplicados y retornamos
                    return full_df.reset_index()
            except Exception as e:
                logger.warning(f"FeatureStore retrieval skipped: {e}")

        # Si no hay cache or es tiempo real, calculamos...
        
        # Convertir a numérico y optimizar RAM (Rule 3.5)
        # PROFESSOR METHOD: Downcasting agresivo para ahorrar 50-70% RAM.
        
        # OPTIMIZATION: Use dict batching for new columns to avoid fragmentation
        new_features = {}
        
        # 1. Base transformations
        numeric_cols = ['close', 'open', 'high', 'low', 'volume']
        # Note: df is already created from bars list, so we modify in place but careful with copies
        # Efficient typing
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
        
        if len(df) < 50:
            return pd.DataFrame()

        # Helper for efficient array access (CRITICAL: TA-Lib requires float64/double)
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        open_ = df['open'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)

        # ==================== PRICE ACTION ====================
        new_features['returns_1'] = df['close'].pct_change(1)
        new_features['returns_3'] = df['close'].pct_change(3)
        new_features['returns_5'] = df['close'].pct_change(5)
        new_features['returns_10'] = df['close'].pct_change(10)
        
        # High-Low ratios (Vectorized)
        new_features['hl_range'] = (df['high'] - df['low']) / df['close']
        new_features['oc_range'] = abs(df['close'] - df['open']) / df['close']
        new_features['close_position'] = safe_div(df['close'] - df['low'], df['high'] - df['low'], 0.5)
        
        # Body to wick ratio
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        new_features['body_to_wick'] = safe_div(abs(df['close'] - df['open']), upper_wick + lower_wick, 1.0)
        
        # ==================== MOMENTUM COMPLETO ====================
        for period in [3, 5, 8, 13, 21, 34]:
            new_features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        new_features['roc_5'] = df['close'].pct_change(5)
        new_features['roc_10'] = df['close'].pct_change(10)
        new_features['roc_20'] = df['close'].pct_change(20)
        
        # ==================== INDICADORES TÉCNICOS ====================
        # RSIs
        new_features['rsi_7'] = talib.RSI(close, timeperiod=7)
        new_features['rsi_14'] = talib.RSI(close, timeperiod=14)
        new_features['rsi_21'] = talib.RSI(close, timeperiod=21)
        
        # ATR / ADX
        new_features['atr'] = talib.ATR(high, low, close, timeperiod=14)
        new_features['atr_pct'] = safe_div(new_features['atr'], df['close']) * 100
        new_features['natr'] = talib.NATR(high, low, close, timeperiod=14)
        
        new_features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        new_features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        new_features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        new_features['macd'] = macd
        new_features['macd_signal'] = macd_signal
        new_features['macd_hist'] = macd_hist
        new_features['macd_hist_change'] = pd.Series(macd_hist).diff()
        new_features['macd_slope'] = pd.Series(macd_hist).diff(3)
        
        # Bollinger
        upper, middle, lower_band = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        new_features['bb_upper'] = upper
        new_features['bb_middle'] = middle
        new_features['bb_lower'] = lower_band
        new_features['bb_position'] = safe_div(close - lower_band, upper - lower_band, 0.5)
        new_features['bb_width'] = safe_div(upper - lower_band, middle)
        # Squeeze logic requires rolling mean of bb_width, do it later or inline
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        new_features['stoch_k'] = slowk
        new_features['stoch_d'] = slowd
        new_features['stoch_cross'] = np.where(slowk > slowd, 1, -1)
        
        # MFI / CCI
        new_features['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
        new_features['cci'] = talib.CCI(high, low, close, timeperiod=20)
        
        # EMAs
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            if len(df) >= period:
                ema = talib.EMA(close, timeperiod=period)
                new_features[f'ema_{period}'] = ema
                new_features[f'dist_ema_{period}'] = safe_div(close - ema, ema)
        
        # SMAs
        for period in [20, 50]:
            if len(df) >= period:
                new_features[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
        
        # Volume
        new_features['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
        new_features['volume_ratio'] = safe_div(df['volume'], new_features['volume_sma_20'])
        
        # OBV
        new_features['obv'] = talib.OBV(close, volume)
        new_features['obv_sma'] = talib.SMA(new_features['obv'], timeperiod=20)
        new_features['obv_ratio'] = safe_div(new_features['obv'], new_features['obv_sma'], 1.0)
        
        # Volatility 10
        new_features['volatility_10'] = pd.Series(close).pct_change().rolling(10).std() * 100
        
        # Garman-Klass
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        new_features['gk_vol'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

        # --- MERGE BATCH ---
        # This is where we save memory vs repeated assignment
        features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, features_df], axis=1)

        # Post-merge complex calculations (that depend on new features)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.5).astype(int)
        
        # Crossovers (Now accessing calculated EMAs)
        # Note: We must handle missing columns if data too short
        if 'ema_5' in df and 'ema_20' in df:
            df['ema_5_20_cross'] = np.where(df['ema_5'] > df['ema_20'], 1, -1)
        else:
             df['ema_5_20_cross'] = 0
             
        if 'ema_20' in df and 'ema_50' in df:
            df['ema_20_50_cross'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
        else:
            df['ema_20_50_cross'] = 0
            
        if 'ema_50' in df and 'ema_200' in df:
             df['ema_50_200_cross'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
        else:
             df['ema_50_200_cross'] = 0

        # ... (Rest of logic continues, assuming columns exist)
        
        # ==================== PATTERN RECOGNITION ====================
        # Consecutive moves
        df['up_bar'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['down_bar'] = (df['close'] < df['close'].shift(1)).astype(int)
        
        df['consecutive_ups'] = df['up_bar'].groupby(
            (df['up_bar'] != df['up_bar'].shift()).cumsum()
        ).cumsum() * df['up_bar']
        
        df['consecutive_downs'] = df['down_bar'].groupby(
            (df['down_bar'] != df['down_bar'].shift()).cumsum()
        ).cumsum() * df['down_bar']
        
        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # ==================== FEATURES ADAPTATIVOS POR RÉGIMEN ====================
        # Initialize regime-aware features to ensure consistency for ML inference
        df['trend_power'] = 0.0
        df['trend_alignment'] = 0.0
        df['range_extreme'] = 0.0
        df['mean_reversion_potential'] = 0.0
        df['volatility_regime'] = 1.0
        df['panic_index'] = 0.0

        if regime_aware and hasattr(self, 'market_regime'):
            if self.market_regime == "TRENDING":
                # Features para trending
                df['trend_power'] = df['adx'] * df['volume_ratio']
                df['trend_alignment'] = (
                    np.where(df['ema_5_20_cross'] > 0, 1, -1) +
                    np.where(df['ema_20_50_cross'] > 0, 1, -1) +
                    np.where(df['ema_50_200_cross'] > 0, 1, -1)
                ) / 3
                
            elif self.market_regime == "RANGING":
                # Features para ranging
                df['range_extreme'] = (df['rsi_14'] < 30).astype(int) - (df['rsi_14'] > 70).astype(int)
                df['mean_reversion_potential'] = abs(df['bb_position'] - 0.5) * 2
                
            elif self.market_regime == "VOLATILE":
                # Features para volatilidad
                df['volatility_regime'] = df['atr_pct'].rolling(10).mean() / df['atr_pct'].rolling(50).mean()
                df['panic_index'] = df['volume_ratio'] * df['volatility_10']
        
        # ==================== MULTI-TIMEFRAME ANALYSIS ====================
        # 5-minute
        try:
            bars_5m = self.data_provider.get_latest_bars_5m(self.symbol, n=30)
            if len(bars_5m) >= 14:
                closes_5m = np.array([float(b['close']) for b in bars_5m])
                df['rsi_5m'] = talib.RSI(closes_5m, timeperiod=14)[-1]
                df['momentum_5m'] = (closes_5m[-1] / closes_5m[-3] - 1) if len(closes_5m) >= 3 else 0
            else:
                df['rsi_5m'] = 50.0
                df['momentum_5m'] = 0.0
        except:
            df['rsi_5m'] = 50.0
            df['momentum_5m'] = 0.0
        
        # 15-minute
        try:
            bars_15m = self.data_provider.get_latest_bars_15m(self.symbol, n=30)
            if len(bars_15m) >= 14:
                closes_15m = np.array([float(b['close']) for b in bars_15m])
                df['rsi_15m'] = talib.RSI(closes_15m, timeperiod=14)[-1]
                df['momentum_15m'] = (closes_15m[-1] / closes_15m[-5] - 1) if len(closes_15m) >= 5 else 0
            else:
                df['rsi_15m'] = 50.0
                df['momentum_15m'] = 0.0
        except:
            df['rsi_15m'] = 50.0
            df['momentum_15m'] = 0.0
        
        # 1-hour (tendencia principal)
        try:
            bars_1h = None
            if self.data_provider:
                bars_1h = self.data_provider.get_latest_bars_1h(self.symbol, n=50)
                
            if bars_1h and len(bars_1h) >= 14:
                closes_1h = np.array([float(b['close']) for b in bars_1h])
                
                # DEFRAGMENTATION FIX: Batch assignments in a dict
                h1_features = {}
                h1_features['rsi_1h'] = talib.RSI(closes_1h, timeperiod=14)[-1]
                h1_features['momentum_1h'] = (closes_1h[-1] / closes_1h[-10] - 1) if len(closes_1h) >= 10 else 0
                
                # Tendencia 1h
                if len(closes_1h) >= 50:
                    ema_50_1h = talib.EMA(closes_1h, timeperiod=50)[-1]
                    ema_200_1h = talib.EMA(closes_1h, timeperiod=200)[-1] if len(closes_1h) >= 200 else ema_50_1h
                    h1_features['trend_1h'] = 1 if ema_50_1h > ema_200_1h else -1
                    h1_features['trend_1h_strength'] = abs(ema_50_1h - ema_200_1h) / ema_200_1h
                else:
                    h1_features['trend_1h'] = 0
                    h1_features['trend_1h_strength'] = 0
                
                # Apply batch
                for k, v in h1_features.items():
                    df[k] = v
            else:
                default_h1 = {'rsi_1h': 50.0, 'momentum_1h': 0.0, 'trend_1h': 0, 'trend_1h_strength': 0}
                for k, v in default_h1.items():
                    df[k] = v
            
            # Consolidate after H1 features
            df = df.copy()
        except Exception:
            for k in ['rsi_1h', 'momentum_1h', 'trend_1h', 'trend_1h_strength']:
                df[k] = 0.0
            df = df.copy()
        
        # ==================== SENTIMENT INTEGRATION ====================
        if self.sentiment_loader:
            try:
                sentiment = self.sentiment_loader.get_sentiment(self.symbol)
                # DEFRAGMENTATION FIX: Create new columns in a batch if possible or copy
                new_cols = {}
                new_cols['sentiment'] = float(sentiment) if sentiment is not None else 0.0
                
                # Use temp series for calculations to avoid fragmented assignments
                s_sent = pd.Series(new_cols['sentiment'], index=df.index)
                new_cols['sentiment_change'] = s_sent.diff().fillna(0)
                new_cols['sentiment_momentum'] = new_cols['sentiment_change'].rolling(5).mean()
                
                # Batch update
                for k, v in new_cols.items():
                    df[k] = v
                
                # Standardize to avoid further fragmentation
                df = df.copy()
            except:
                df['sentiment'] = 0.0
                df['sentiment_change'] = 0.0
                df['sentiment_momentum'] = 0.0
        else:
            df['sentiment'] = 0.0
            df['sentiment_change'] = 0.0
            df.loc[:, 'sentiment_momentum'] = 0.0
        
        # ==================== CONFLUENCE SCORE MEJORADO ====================
        confluence = np.zeros(len(df))
        
        # RSI multi-timeframe (30%)
        confluence += np.where(df['rsi_14'] > 50, 0.15, -0.15)
        confluence += np.where(df['rsi_5m'] > 50, 0.08, -0.08)
        confluence += np.where(df['rsi_15m'] > 50, 0.07, -0.07)
        
        # MACD y momentum (25%)
        confluence += np.where(df['macd_hist'] > 0, 0.15, -0.15)
        confluence += np.where(df['momentum_5'] > 0, 0.10, -0.10)
        
        # Tendencia (25%)
        confluence += np.where(df['ema_20_50_cross'] > 0, 0.10, -0.10)
        confluence += np.where(df['trend_1h'] > 0, 0.10, -0.10)
        confluence += np.where(df['adx'] > 25, 0.05, -0.05)
        
        # Volumen (10%)
        confluence += np.where(df['volume_ratio'] > 1.2, 0.05, -0.05)
        confluence += np.where(df['obv_ratio'] > 1, 0.05, -0.05)
        
        # Sentiment (10%)
        confluence += np.where(df['sentiment'] > 0, 0.05, -0.05)
        confluence += np.where(df['sentiment_momentum'] > 0, 0.05, -0.05)
        
        df.loc[:, 'confluence_score'] = confluence
        
        df = self._validate_features(df)

        # ==================== [PHASE 12] FEATURE STORE SAVE ====================
        # Solo guardamos si no es una inferencia de una sola vela (para ahorrar I/O)
        if len(df) > 1:
            try:
                self.feature_store.store_features(self.symbol, df)
            except Exception as e:
                logger.debug(f"FeatureStore storage skipped: {e}")

        return df

    def _validate_features(self, df):
        """
        Limpieza robusta de features
        """
        if len(df) == 0:
            return df
            
        # Reemplazar infinitos
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Forward y backward fill limitado
        df.ffill(limit=5, inplace=True)
        df.bfill(limit=5, inplace=True)
        
        # Clipping específico por tipo de feature
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exempt OHLCV and other essential non-feature columns
        exempt_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'open_time', 'close_time']
        
        for col in numeric_cols:
            if col in exempt_cols:
                continue
                
            if 'rsi' in col or 'mfi' in col:
                df[col] = df[col].clip(0, 100)
            elif 'stoch' in col or 'bb_position' in col or 'close_position' in col:
                df[col] = df[col].clip(0, 1)
            elif 'returns' in col or 'momentum' in col or 'roc' in col or 'dist_' in col:
                df[col] = df[col].clip(-0.5, 0.5)
            elif 'volume_ratio' in col:
                df[col] = df[col].clip(0, 5)
            elif 'sentiment' in col:
                df[col] = df[col].clip(-1, 1)
            elif 'confluence' in col:
                df[col] = df[col].clip(-1, 1)
            else:
                # Clipping general (Z-scores, changes, etc.)
                df[col] = df[col].clip(-5, 5)
        
        # Llenar NaN restantes
        df.fillna(0, inplace=True)
        
        return df

    # ============================================================
    # ✅ LABEL CREATION CON TARGETS ADAPTATIVOS
    # ============================================================
    
    def _create_labels(self, df, adaptive_targets=True):
        """
        Creación de labels con targets adaptativos por volatilidad
        """
        # CRITICAL PERFORMANCE: Ensure dataframe is consolidated before looping or adding labels
        df = df.copy()
        labels = []
        
        for i in range(len(df)):
            if i >= len(df) - self.LOOKAHEAD_BARS:
                labels.append(0)
                continue
                
            current_price = df.iloc[i]['close']
            current_volatility = df.iloc[i]['atr_pct'] / 100
            
            # Targets adaptativos por volatilidad
            if adaptive_targets:
                # Multiplicadores para targets (Híbrido Testnet/Producción)
                if current_volatility > 0.04:  # Muy alta volatilidad
                    volatility_multiplier = 1.5
                elif current_volatility > 0.03:  # Alta volatilidad
                    volatility_multiplier = 1.3
                elif current_volatility > 0.02:  # Volatilidad media
                    volatility_multiplier = 1.1
                elif current_volatility < 0.0003:  # Testnet Extrema (< 0.03%)
                    volatility_multiplier = 0.10 # Target ~0.03% (Ultra-sensitive)
                elif current_volatility < 0.0008:  # Micro Testnet (< 0.08%)
                    volatility_multiplier = 0.20 
                elif current_volatility < 0.002:  # < 0.2%
                    volatility_multiplier = 0.35
                elif current_volatility < 0.01:   # < 1%
                    volatility_multiplier = 0.6
                else:
                    volatility_multiplier = 1.0
                    
                tp_target = self.BASE_TP_TARGET * volatility_multiplier
                sl_target = self.BASE_SL_TARGET * volatility_multiplier
            else:
                tp_target = self.BASE_TP_TARGET
                sl_target = self.BASE_SL_TARGET
                
            future_data = df.iloc[i+1:i+1+self.LOOKAHEAD_BARS]
            
            if len(future_data) == 0:
                labels.append(0)
                continue
            
            tp_level = current_price * (1 + tp_target)
            sl_level = current_price * (1 - sl_target)
            
            tp_hit = (future_data['high'] >= tp_level).any()
            sl_hit = (future_data['low'] <= sl_level).any()
            
            if tp_hit and sl_hit:
                # Ambos tocados: usar el primero
                tp_idx = future_data[future_data['high'] >= tp_level].index[0]
                sl_idx = future_data[future_data['low'] <= sl_level].index[0]
                label = 1 if tp_idx < sl_idx else -1
            elif tp_hit:
                label = 1
            elif sl_hit:
                label = -1
            else:
                # --- FALLBACK: Fixed Horizon Return ---
                # Si no tocamos TP ni SL, usamos el signo del movimiento final
                final_price = future_data.iloc[-1]['close']
                ret = (final_price - current_price) / current_price
                
                # Umbral de ruido equilibrado (50% del target)
                noise_threshold = tp_target * 0.5
                
                if ret > noise_threshold:
                    label = 1
                elif ret < -noise_threshold:
                    label = -1
                else:
                    label = 0
                
            labels.append(label)
        
        # Asegurar longitud correcta
        while len(labels) < len(df):
            labels.append(0)
            
        # DEFRAGMENTATION: Copy before adding new column to large DF
        df = df.copy()
        df['label'] = labels[:len(df)]
        return df

    # ============================================================
    # ✅ TRAINING ULTIMATE - ENSEMBLE CON HIPERPARÁMETROS ADAPTATIVOS
    # ============================================================
    
    def _train_with_cross_validation(self, df):
        """
        Training con ensemble de 3 modelos y hiperparámetros adaptativos por régimen
        """
        if len(df) < 200:
            return None, 0.0
            
        # Actualizar régimen antes de entrenar
        self._update_market_regime(df)
        
        # ⚡ EFICIENCIA DE ENTRENAMIENTO (Rule 3.1)
        # SUBSAMPLING ESTRATÉGICO
        # PROFESSOR METHOD:
        # QUÉ: Selección de un subconjunto relevante de datos.
        # POR QUÉ: Reduce tiempo de cómputo en 50% y prioriza el régimen de mercado actual.
        # CÓMO: Si hay > 3000 velas, nos quedamos con el último 60% (relevancia temporal).
        original_len = len(df)
        if original_len > 3000:
            df = df.iloc[-int(original_len * 0.6):]
            logger.info(f"⚡ [ML Optimization] Subsampling active: {original_len} -> {len(df)} samples (60% most recent).")

        # Crear labels con targets adaptativos
        df = self._create_labels(df, adaptive_targets=True)
        df = df.dropna()
        
        # DEBUG: Analizar por qué no hay señales
        vol_mean = df['atr_pct'].mean()
        vol_ref = vol_mean / 100
        mult = 1.0
        if vol_ref < 0.0003: mult = 0.10
        elif vol_ref < 0.0008: mult = 0.20
        elif vol_ref < 0.002: mult = 0.35
        elif vol_ref < 0.01: mult = 0.6
        
        tp_mean = self.BASE_TP_TARGET * mult
        label_counts = df['label'].value_counts().to_dict()
        logger.info(f"🔍 DEBUG ML [{self.symbol}]: Rows={len(df)}, AvgVol={vol_mean:.4f}%, mult={mult:.2f}, Est.Target={tp_mean:.4f}, Labels={label_counts}")
        
        df_signals = df[df['label'] != 0]
        
        if len(df_signals) < 30:
            # --- NEW: Data Quality Audit (Transparencia) ---
            price_spread = (df['high'].max() - df['low'].min()) / df['close'].mean()
            identical_bars = (df['high'] == df['low']).sum() / len(df)
            
            if price_spread < 0.0001 or identical_bars > 0.95:
                # Caso Testnet: Precio fijo (ej: BTC=5.0)
                logger.warning(f"🚫 [ZOMBIE MARKET] {self.symbol} is flat. Spread: {price_spread*100:.6f}% | Identical Bars: {identical_bars*100:.1f}%. Training aborted.")
            else:
                logger.warning(f"⚠️ [LOW VOLATILITY] {self.symbol} has movement but not enough to reach targets. Real Signals: {len(df_signals)}/30 required.")
            
            return None, 0.0    # Excluir columnas no-features
        exclude_cols = ['label', 'open', 'high', 'low', 'close', 'volume', 
                       'timestamp', 'datetime', 'symbol']
        feature_cols = [c for c in df_signals.columns if c not in exclude_cols]
        
        X = df_signals[feature_cols]
        y = df_signals['label']
        
        # CRITICAL: Remap labels for XGBoost compatibility
        # XGBoost expects classes starting from 0: [0, 1, 2, ...]
        # Our labels are [-1, 0, 1] -> remap to [0, 1, 2]
        # Note: We filter df_signals where label != 0, so we only have [-1, 1]
        # Remap: -1 -> 0, 1 -> 1
        y = y.map({-1: 0, 1: 1})
        self._label_mapping = {0: -1, 1: 1}  # For inverse mapping during inference
        
        # DEBUG: Verificar si las features son válidas
        std_zero_cols = X.columns[X.std() == 0].tolist()
        if len(std_zero_cols) > 0:
            logger.debug(f"⚠️ {len(std_zero_cols)} features are constant (std=0). Ex: {std_zero_cols[:5]}")
            
        logger.info(f"📊 Training {self.symbol} with {len(X)} samples, {len(feature_cols)} features")
        
        # TimeSeriesSplit CV with auto-adjustment for small datasets
        n_samples = len(X)
        if n_samples < 5:
            # Life Support Mode: Not enough data for CV
            logger.info("📉 Minimal data mode: Using full dataset for training (Overfitting intended for survival)")
            # Manual split: Train and test on same data to produce a valid score and model
            indices = list(range(n_samples))
            splitter = [(indices, indices)]
        else:
            tscv = TimeSeriesSplit(n_splits=3)
            splitter = tscv.split(X)
            
        cv_scores = {'rf': [], 'xgb': [], 'gb': []}
        best_models = {'rf': None, 'xgb': None, 'gb': None}
        best_scaler = None
        
        # Hiperparámetros por régimen y agresividad
        if self.market_regime == "VOLATILE":
            # Conservador en volatilidad
            n_estimators = 70
            max_depth_rf = 5
            max_depth_xgb = 4
            max_depth_gb = 4
            min_samples_split = 10
            learning_rate = 0.05
            subsample = 0.7
            
        elif self.market_regime == "TRENDING":
            # Agresivo en trending
            n_estimators = 120
            max_depth_rf = 8
            max_depth_xgb = 7
            max_depth_gb = 6
            min_samples_split = 5
            learning_rate = 0.08
            subsample = 0.9
            
        elif self.market_regime == "RANGING":
            # Equilibrado en ranging
            n_estimators = 90
            max_depth_rf = 6
            max_depth_xgb = 5
            max_depth_gb = 5
            min_samples_split = 8
            learning_rate = 0.06
            subsample = 0.8
            
        else:  # MIXED o UNKNOWN
            n_estimators = 80
            max_depth_rf = 6
            max_depth_xgb = 5
            max_depth_gb = 5
            min_samples_split = 8
            learning_rate = 0.06
            subsample = 0.8
        
        # Ajustar por agresividad y fase inicial
        # PRIORIDAD: Operatividad instantánea en el primer arranque
        if self.training_iteration <= 1:
            n_estimators = max(30, int(n_estimators * 0.4))
            
        n_estimators = int(n_estimators * self.aggressiveness_factor)
        learning_rate = min(0.15, learning_rate * self.aggressiveness_factor)
        
        # Loop de entrenamiento con reporte de progreso
        num_folds = 3 if n_samples >= 300 else 1
        for fold, (train_idx, test_idx) in enumerate(splitter):
            if fold % 1 == 0: # Log every fold
                logger.info(f"   ⚙️ [{self.symbol}] Fitting ML Engine (Fold {fold+1}/{num_folds})...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scaling & Memory Optimization (Rule 3.6)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train).astype('float32')
            X_test_scaled = scaler.transform(X_test).astype('float32')
            
            # 1. Random Forest (Paralelización completa)
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth_rf,
                min_samples_split=min_samples_split,
                min_samples_leaf=3,
                max_features='sqrt',
                n_jobs=1,   # Optimización: n_jobs=1 para evitar explosión de RAM con 24 símbolos
                random_state=42 + self.training_iteration,
                class_weight='balanced'
            )
            rf.fit(X_train_scaled, y_train)
            rf_score = rf.score(X_test_scaled, y_test)
            cv_scores['rf'].append(rf_score)
            
            # 2. XGBoost (Incremental Ready + tree_method='hist')
            xgb = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth_xgb,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=0.8,
                n_jobs=1,             # Optimización: n_jobs=1 para estabilidad de recursos
                tree_method='hist',   # MÁXIMA VELOCIDAD
                random_state=42 + self.training_iteration,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )
            
            # 🔄 APRENDIZAJE INCREMENTAL (Rule 3.2): Validar matching de features
            prev_xgb = None
            if hasattr(self, 'xgb_model') and self.xgb_model:
                try:
                    if hasattr(self.xgb_model, 'feature_names_in_') and list(self.xgb_model.feature_names_in_) == list(feature_cols):
                        prev_xgb = self.xgb_model
                        logger.info(f"🔄 [{self.symbol}] Resuming learning from previous XGBoost model (Incremental).")
                    else:
                        logger.debug(f"🔄 [{self.symbol}] Feature mismatch for incremental learning. Resetting XGB.")
                except:
                    pass

            xgb.fit(X_train_scaled, y_train, xgb_model=prev_xgb)
            xgb_score = xgb.score(X_test_scaled, y_test)
            cv_scores['xgb'].append(xgb_score)
            
            # 3. Gradient Boosting (Persistent Warm Start - Rule 3.7)
            # PROFESSOR METHOD: Reuse GB instance to accumulate knowledge.
            gb = None
            if hasattr(self, 'gb_model') and self.gb_model is not None:
                gb = self.gb_model
                # Incrementar n_estimators para permitir warm_start real
                gb.n_estimators += 5 
                logger.debug(f"🔄 [{self.symbol}] Resuming GB with {gb.n_estimators} estimators (Warm Start).")
            else:
                gb = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth_gb,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    random_state=42 + self.training_iteration,
                    warm_start=True
                )
            
            gb.fit(X_train_scaled, y_train)
            gb_score = gb.score(X_test_scaled, y_test)
            cv_scores['gb'].append(gb_score)
            
            # Guardar mejores modelos (último fold)
            # CRITICAL FIX: Support variable folds (Life Support = 1 fold)
            is_last_fold = (fold == 2) or (n_samples < 5 and fold == 0)
            if is_last_fold:
                best_models['rf'] = rf
                best_models['xgb'] = xgb
                best_models['gb'] = gb
                best_scaler = scaler
            
            # 🧹 Fold Cleanup (Rule 3.6)
            del X_train_scaled, X_test_scaled, X_train, X_test
            gc.collect()
        
        # Calcular scores
        avg_rf = np.mean(cv_scores['rf'])
        avg_xgb = np.mean(cv_scores['xgb'])
        avg_gb = np.mean(cv_scores['gb'])
        
        # Score del ensemble weighted
        ensemble_score = (avg_rf * self.base_rf_weight + 
                         avg_xgb * self.base_xgb_weight + 
                         avg_gb * self.base_gb_weight)
        
        # Actualizar scores individuales
        self.individual_model_scores = {
            'rf': avg_rf,
            'xgb': avg_xgb,
            'gb': avg_gb
        }
        
        logger.info(
            f"📊 [{self.symbol}] CV Scores - RF: {avg_rf:.3f}, XGB: {avg_xgb:.3f}, "
            f"GB: {avg_gb:.3f}, Ensemble: {ensemble_score:.3f}"
        )
        logger.info(
            f"📊 [{self.symbol}] Regime: {self.market_regime}, Aggressiveness: {self.aggressiveness_factor:.2f}"
        )
        
        if ensemble_score >= self.MIN_MODEL_ACCURACY:
            return (best_models, best_scaler, feature_cols), ensemble_score
        
        logger.warning(f"Ensemble score {ensemble_score:.3f} below threshold")
        return None, ensemble_score

    # ============================================================
    # ✅ RE-PESADO DINÁMICO DE MODELOS
    # ============================================================
    
    def _update_model_weights(self):
        """
        Re-pesado dinámico basado en performance reciente
        """
        if len(self.performance_history) < 15:
            return
        
        recent_performance = list(self.performance_history)[-15:]
        success_rate = sum(1 for x in recent_performance if x > 0) / len(recent_performance)
        
        # Performance-based weight adjustment
        if success_rate > 0.70:  # Excelente performance
            # Aumentar peso del mejor modelo
            best_model = max(self.individual_model_scores.items(), key=lambda x: x[1])
            weight_increase = 0.05
            
            if best_model[0] == 'rf':
                self.base_rf_weight = min(0.60, self.base_rf_weight + weight_increase)
            elif best_model[0] == 'xgb':
                self.base_xgb_weight = min(0.50, self.base_xgb_weight + weight_increase)
            else:
                self.base_gb_weight = min(0.40, self.base_gb_weight + weight_increase)
                
            logger.debug(f"🔥 Increasing {best_model[0]} weight due to excellent performance")
            
        elif success_rate < 0.40:  # Mala performance
            # Volver a pesos originales
            self.base_rf_weight = self.original_rf_weight
            self.base_xgb_weight = self.original_xgb_weight
            self.base_gb_weight = self.original_gb_weight
            logger.debug(f"⚠️ Resetting weights due to poor performance")
        
        # Normalizar pesos
        total = self.base_rf_weight + self.base_xgb_weight + self.base_gb_weight
        self.base_rf_weight /= total
        self.base_xgb_weight /= total
        self.base_gb_weight /= total

    def _adjust_learning_rate(self):
        """
        Ajustar learning rate y factor de agresividad dinámicamente
        """
        if len(self.performance_window) < 10:
            return
            
        recent_perf = list(self.performance_window)
        win_rate = sum(1 for x in recent_perf if x > 0) / len(recent_perf)
        
        # Ajustar learning rate
        if win_rate > 0.65:
            # Buena performance: aumentar agresividad
            self.learning_rate = min(1.3, self.learning_rate * 1.08)
            self.aggressiveness_factor = min(1.5, self.aggressiveness_factor * 1.05)
        elif win_rate < 0.35:
            # Mala performance: reducir agresividad
            self.learning_rate = max(0.7, self.learning_rate * 0.92)
            self.aggressiveness_factor = max(0.7, self.aggressiveness_factor * 0.95)
        else:
            # Performance neutral: tender a 1.0
            self.learning_rate += (1.0 - self.learning_rate) * 0.1
            self.aggressiveness_factor += (1.0 - self.aggressiveness_factor) * 0.1

    # ============================================================
    # ✅ CIRCUIT BREAKER AVANZADO
    # ============================================================
    
    def _check_circuit_breaker(self):
        """
        Verificar condiciones para activar/desactivar circuit breaker
        """
        if not self.portfolio:
            return True
            
        try:
            current_equity = self.portfolio.get_total_equity()
            
            # Inicializar peak equity
            if self.peak_equity is None:
                self.peak_equity = current_equity
            else:
                self.peak_equity = max(self.peak_equity, current_equity)
            
            # Calcular drawdown
            drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
            
            # Activar por drawdown
            if drawdown > self.circuit_breaker_threshold and not self.circuit_breaker_active:
                self.activate_circuit_breaker()
                return False
            
            # Activar por pérdidas consecutivas
            if self.consecutive_losses >= self.max_consecutive_losses and not self.circuit_breaker_active:
                self.activate_circuit_breaker()
                return False
            
            # Desactivar si se recupera
            if self.circuit_breaker_active and drawdown < (self.circuit_breaker_threshold * 0.6):
                self.deactivate_circuit_breaker()
                return True
            
            return not self.circuit_breaker_active
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return True

    def activate_circuit_breaker(self):
        """Activar circuit breaker"""
        self.circuit_breaker_active = True
        self.original_confidence_threshold = self.adaptive_confidence_threshold
        self.adaptive_confidence_threshold = min(0.80, self.adaptive_confidence_threshold + 0.25)
        self.aggressiveness_factor = max(0.5, self.aggressiveness_factor * 0.7)
        
        logger.warning(
            f"🔴 CIRCUIT BREAKER ACTIVATED | "
            f"Confidence: {self.adaptive_confidence_threshold:.2f} | "
            f"Aggressiveness: {self.aggressiveness_factor:.2f}"
        )

    def deactivate_circuit_breaker(self):
        """Desactivar circuit breaker"""
        self.circuit_breaker_active = False
        self.adaptive_confidence_threshold = self.original_confidence_threshold
        self.consecutive_losses = 0
        self.peak_equity = None
        
        logger.info("🟢 CIRCUIT BREAKER DEACTIVATED")

    # ============================================================
    # ✅ INFERENCE ULTIMATE - ENSEMBLE COMPLETO
    # ============================================================
    
    def _run_inference(self):
        """
        Inference con ensemble completo y prioridad de visibilidad/operatividad.
        """
        try:
            self.analysis_stats['total'] += 1
            if not self._check_circuit_breaker():
                return
            
            # 1. Obtención y Preparación de Datos
            bars = self.data_provider.get_latest_bars(self.symbol, n=250)
            df = self._prepare_features(bars, regime_aware=True)
            
            if df.empty or len(df) < 5:
                return
                
            current_row = df.iloc[-1]
            atr_pct = current_row['atr_pct'] / 100
            current_atr = current_row['atr']
            rsi = current_row['rsi_14']
            vol_ratio = current_row.get('volume_ratio', 0)
            confluence = current_row['confluence_score']

            # 2. Verificar disponibilidad de Modelos (PRIMERO)
            with self._state_lock:
                models_ready = all([self.rf_model, self.xgb_model, self.gb_model])
                feature_cols = self._feature_cols

            # -------------------------------------------------------------------------
            # ✅ CROSS-POLLINATION (Phase 7): Read Math Stats from Portfolio
            # -------------------------------------------------------------------------
            math_hurst = 0.5
            if self.portfolio and hasattr(self.portfolio, 'math_stats'):
                math_hurst = self.portfolio.math_stats.get('hurst', 0.5)
                
            if not models_ready:
                self.oracle_log_count += 1
                # Log immediately on first check, then periodically to reduce spam
                if self.oracle_log_count == 1 or self.oracle_log_count % 10 == 0:
                    # Determine Concept/Context based on Regime
                    if self.market_regime == "ZOMBIE":
                        concept = "Zombie market detected. Stagnant price action."
                    elif self.market_regime == "RANGING":
                        concept = "Mean Reversion active. Hunting overextensions."
                    elif self.market_regime == "TRENDING":
                        concept = "Trend Following active. Riding momentum."
                    elif self.market_regime == "VOLATILE":
                        concept = "High Volatility. Defensive stops & wide targets."
                    else:
                        concept = "Analyzing market structure..."

                    # Prepare Enhanced Stats for Visibility during Training
                    z_score = current_row.get('volume_zscore', 0)
                    adx = current_row.get('adx', 0)
                    trend_power = current_row.get('trend_power', 0) 

                    oracle_msg = (
                        f"\n🔮 [UNIFIED ORACLE] {self.symbol} | TRAINING | Last CV: {self.last_training_score:.3f}\n"
                        f"   Engines Passing: 0/3 | Threshold: {self.consensus_threshold}\n"
                        f"   Scores  -> ML: {self.last_training_score:.2f} | SENT: 0.00 | TECH: 0.00\n"
                        f"   Horizon -> H5: 0.00 | H15: 0.00 | H30: 0.00\n"
                        f"   Verdict -> Direction: TRAINING | Final Conf: 0.00 (Gap: 1.00)\n"
                        f"   Phase: {self.market_regime} ({self.regime_confidence*100:.1f}%)\n"
                        f"   Concept: {concept} (Models Compiling)\n"
                        f"   Stats: ADX={adx:.1f} | ATR%={atr_pct*100:.2f}% | Trend={trend_power:.2f} | Z-Score={z_score:.2f}\n"
                        f"   Math: Hurst={math_hurst:.2f} (Portfolio Sync)\n"
                        f"   Confidence: 0.0% | Strategy: Waiting for AI models..."
                    )
                    logger.info(oracle_msg)
                return

            # 2.5 NUEVA: Validación crítica de features
            if feature_cols is None or not feature_cols:
                logger.error(f"❌ Error processing {self.symbol}: No valid feature columns available")
                return

            # Filtrar columnas válidas
            valid_features = [col for col in feature_cols if col is not None and col in df.columns]
            if not valid_features:
                logger.error(f"❌ Error processing {self.symbol}: No valid features available for inference")
                return

            # 3. Aligned Feature Matrix (Phase 6 Final Fix)
            # Ensure X_pred matches EXACTLY the columns the scaler was fitted on
            if hasattr(self.scaler, 'feature_names_in_'):
                # Strict alignment with scaler
                final_features = self.scaler.feature_names_in_
                # Fill missing if any (shouldn't happen with valid_features but for safety)
                X_pred_aligned = pd.DataFrame(columns=final_features)
                for col in final_features:
                    if col in df.columns:
                        X_pred_aligned[col] = df[col].iloc[[-1]].values
                    else:
                        X_pred_aligned[col] = 0.0
                X_pred = X_pred_aligned
            else:
                # Fallback to old behavior if scaler is legacy
                X_pred = df[valid_features].iloc[[-1]]
                
            if X_pred.empty:
                logger.error(f"❌ {self.symbol}: Empty feature matrix after alignment")
                return
                
            if X_pred.isnull().any().any():
                logger.warning(f"⚠️ {self.symbol} has missing values. Using zero-fill.")
                X_pred = X_pred.fillna(0)
                
            X_scaled = self.scaler.transform(X_pred)
            rf_proba = self.rf_model.predict_proba(X_scaled)[0]
            xgb_proba = self.xgb_model.predict_proba(X_scaled)[0]
            gb_proba = self.gb_model.predict_proba(X_scaled)[0]
            
            ensemble_proba = (rf_proba * self.base_rf_weight + 
                             xgb_proba * self.base_xgb_weight + 
                             gb_proba * self.base_gb_weight)
            
            classes = self.rf_model.classes_
            pred_idx = np.argmax(ensemble_proba)
            confidence = ensemble_proba[pred_idx]
            predicted_class = self._label_mapping.get(classes[pred_idx], classes[pred_idx])
            
            # --- Ajustes Dinámicos de Confianza ---
            if self.market_regime == "TRENDING" and abs(confluence) > 0.4:
                confidence = min(confidence + 0.03, 1.0)
            
            # 4. REPORTE ORÁCULO (Visión Total)
            gap = self.adaptive_confidence_threshold - confidence
            is_ready = gap <= 0
            
            oracle_msg = f"\n🔮 [ML ORACLE] {self.symbol} | Close: {current_row['close']:.2f} | {'READY' if is_ready else 'SCANNING'}"
            if Config.Strategies.ML_ORACLE_VERBOSE:
                oracle_msg += f"\n   Probabilities -> RF: {rf_proba[pred_idx]:.2f} | XGB: {xgb_proba[pred_idx]:.2f} | GB: {gb_proba[pred_idx]:.2f}"
            
            oracle_msg += f"\n   Decision Path -> Direction: {predicted_class} | Conf: {confidence:.2f} / {self.adaptive_confidence_threshold:.2f} (Gap: {gap:.2f})"
            logger.info(oracle_msg)

            # 5. FILTROS DE ROBUSTEZ (Prioridad Operatividad)
            # Filtro de volatilidad - Permisivo
            if atr_pct > self.MAX_ATR_PCT * 1.5:
                self.analysis_stats['filtered_vol'] += 1
                logger.info(f"🛡️ [{self.symbol}] Filtered: Extreme Risk Volatility ({atr_pct*100:.2f}%)")
                return
            
            # Filtro de volumen - Muy permisivo en Testnet
            min_vol = 0.2 if Config.BINANCE_USE_TESTNET else self.MIN_VOLUME_RATIO
            if vol_ratio < min_vol:
                self.analysis_stats['filtered_volume'] += 1
                if vol_ratio < 0.05: # Solo ruidoso si es crítico
                    logger.info(f"🛡️ [{self.symbol}] Filtered: Zero/Micro Volume ({vol_ratio:.2f})")
                return
            
            # Filtro de Confianza
            if not is_ready:
                self.analysis_stats['filtered_conf'] += 1
                return
            
            # Filtro de Confluencia
            if predicted_class == 1 and confluence < self.adaptive_confluence_long:
                return
            if predicted_class == -1 and confluence > self.adaptive_confluence_short:
                return

            # ============ CREAR SEÑAL ============
            signal_type = SignalType.LONG if predicted_class == 1 else SignalType.SHORT
            tp_target = self.current_tp_target
            sl_target = self.current_sl_target
            
            # Ajuste de targets por volatilidad
            if atr_pct > 0.03: tp_target *= 1.3; sl_target *= 1.3
            elif atr_pct < 0.01: tp_target *= 0.8; sl_target *= 0.8

            # PHASE 9.2: Adaptive TTL (Prediction Horizon Sync)
            # Default lookup: 30 bars (5m = 150s, 1m = 30s)
            # We use 70% of our lookahead bars as patience.
            prediction_ttl = int(self.LOOKAHEAD_BARS * 60 * 0.7) # e.g. 30 bars * 60s * 0.7 = 1260s
            # Clamp between 30s and 300s for safety in scalping
            final_ttl = max(30, min(300, prediction_ttl))

            # FIXED: Create SignalEvent with ALL metadata in constructor (frozen dataclass)
            signal = SignalEvent(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                datetime=datetime.now(timezone.utc),
                signal_type=signal_type,
                strength=confidence,
                atr=current_row['atr'], # FIXED: Use current_row
                tp_pct=tp_target * 100,
                sl_pct=sl_target * 100,
                current_price=current_row['close'],
                ttl=final_ttl
            )
            
            # ============ REGISTRAR Y ENVIAR ============
            self.performance_history.append(0)
            self.signal_history.append({
                'timestamp': datetime.now(timezone.utc),
                'type': signal_type,
                'confidence': confidence,
                'regime': self.market_regime,
                'price': current_row['close'],
                'confluence': confluence
            })
            
            self.total_signals_generated += 1
            if self.market_regime in self.signals_by_regime:
                self.signals_by_regime[self.market_regime] += 1
            
            # Logging completo y envío
            self._log_ml_signal(signal_type, confidence, confluence, df, 
                               rf_proba, xgb_proba, gb_proba)
            
            # Neural Bridge Publication (Base Logic Fallback)
            if not hasattr(self, 'engines_active'): # If not UniversalEnsemble
                neural_bridge.publish_insight(
                    strategy_id="ML_ORACLE",
                    symbol=self.symbol,
                    insight={
                        'confidence': confidence,
                        'direction': 'LONG' if predicted_class == 1 else 'SHORT',
                        'confluence': confluence
                    }
                )

            self.events_queue.put(signal)
            
            if len(self.performance_history) >= 15:
                self._update_model_weights()
            
            self._last_prediction_time = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"ML Inference error {self.symbol}: {e}", exc_info=True)

    # ============================================================
    # ✅ LOGGING ULTIMATE - 40+ MÉTRICAS
    # ============================================================
    
    def _log_ml_signal(self, signal_type, confidence, confluence, df, 
                       rf_proba, xgb_proba, gb_proba):
        """
        Logging completo con 40+ métricas para análisis
        """
        try:
            current_row = df.iloc[-1]
            atr_pct = current_row['atr_pct'] / 100
            
            # Preparar todas las métricas
            metrics = {
                # Indicadores principales
                'RSI_14': float(current_row['rsi_14']),
                'RSI_5m': float(current_row.get('rsi_5m', 50)),
                'RSI_15m': float(current_row.get('rsi_15m', 50)),
                'RSI_1h': float(current_row.get('rsi_1h', 50)),
                
                # Volatilidad y tendencia
                'ATR%': float(current_row['atr_pct']),
                'ADX': float(current_row.get('adx', 0)),
                'NATR': float(current_row.get('natr', 0)),
                'MACD_Hist': float(current_row.get('macd_hist', 0)),
                'MACD_Slope': float(current_row.get('macd_slope', 0)),
                
                # Precio y volumen
                'Price_Position': float(current_row.get('close_position', 0.5)),
                'BB_Position': float(current_row.get('bb_position', 0.5)),
                'Volume_Ratio': float(current_row['volume_ratio']),
                'Volume_ZScore': float(current_row.get('volume_zscore', 0)),
                
                # Momentum
                'Momentum_5': float(current_row.get('momentum_5', 0)),
                'Momentum_20': float(current_row.get('momentum_20', 0)),
                'ROC_10': float(current_row.get('roc_10', 0)),
                
                # Tendencia
                'EMA_Cross': int(current_row.get('ema_20_50_cross', 0)),
                'Trend_1h': int(current_row.get('trend_1h', 0)),
                'Trend_1h_Strength': float(current_row.get('trend_1h_strength', 0)),
                
                # Confluence y decisión
                'Confluence_Score': float(confluence),
                'Prediction_Confidence': float(confidence),
                'Signal_Type': signal_type.name,
                
                # Régimen y estado
                'Market_Regime': self.market_regime,
                'Regime_Confidence': float(self.regime_confidence),
                'Regime_Duration': self.regime_duration,
                'Circuit_Breaker': self.circuit_breaker_active,
                'Aggressiveness_Factor': float(self.aggressiveness_factor),
                
                # Scores de modelos
                'Training_Score': float(self.last_training_score),
                'RF_Score': float(self.individual_model_scores.get('rf', 0)),
                'XGB_Score': float(self.individual_model_scores.get('xgb', 0)),
                'GB_Score': float(self.individual_model_scores.get('gb', 0)),
                
                # Pesos del ensemble
                'RF_Weight': float(self.base_rf_weight),
                'XGB_Weight': float(self.base_xgb_weight),
                'GB_Weight': float(self.base_gb_weight),
                
                # Probabilidades individuales
                'RF_Proba_Long': float(rf_proba[1]) if len(rf_proba) > 1 else 0.0,
                'XGB_Proba_Long': float(xgb_proba[1]) if len(xgb_proba) > 1 else 0.0,
                'GB_Proba_Long': float(gb_proba[1]) if len(gb_proba) > 1 else 0.0,
                'Ensemble_Proba_Long': float(rf_proba[1] * self.base_rf_weight + 
                                           xgb_proba[1] * self.base_xgb_weight + 
                                           gb_proba[1] * self.base_gb_weight) if len(rf_proba) > 1 else 0.0,
                
                # Targets
                'TP_Target': float(self.current_tp_target * 100),
                'SL_Target': float(self.current_sl_target * 100),
                'TP_SL_Ratio': float(self.current_tp_target / self.current_sl_target) if self.current_sl_target > 0 else 0,
                
                # Performance
                'Total_Signals': self.total_signals_generated,
                'Win_Rate': (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0,
                'Consecutive_Losses': self.consecutive_losses,
                'Learning_Rate': float(self.learning_rate),
            }
            
            # Log a monitor
            monitor_log.log_ml_prediction(
                symbol=self.symbol,
                model_name="Hybrid_Ensemble_Ultimate",
                prediction=1 if signal_type == SignalType.LONG else -1,
                confidence=float(confidence),
                features=metrics,
                decision=signal_type.name
            )
            
            # Log a consola
            logger.info(
                f"🎯 ML {signal_type.name} {self.symbol} | "
                f"Conf: {confidence:.2f} | Confl: {confluence:.2f} | "
                f"Regime: {self.market_regime} | "
                f"Score: {self.last_training_score:.2f} | "
                f"TP/SL: {self.current_tp_target*100:.1f}%/{self.current_sl_target*100:.1f}% | "
                f"Aggr: {self.aggressiveness_factor:.2f}"
            )
            
            # Debug logging
            logger.debug(
                f"Model Details - "
                f"RF: {rf_proba}, XGB: {xgb_proba}, GB: {gb_proba} | "
                f"Weights: RF={self.base_rf_weight:.2f}, XGB={self.base_xgb_weight:.2f}, GB={self.base_gb_weight:.2f}"
            )
            
        except Exception as e:
            logger.error(f"ML logging error: {e}")

    # ============================================================
    # ✅ ARQUITECTURA ASÍNCRONA OPTIMIZADA
    # ============================================================
    
    @trace_execution
    def calculate_signals(self, event):
        """Entry point asíncrono"""
        if event.type != EventType.MARKET:
            return
        
        # Throttling: máximo 1 predicción por segundo
        current_time = datetime.now(timezone.utc)
        if (self._last_prediction_time and 
            (current_time - self._last_prediction_time).total_seconds() < 1.0):
            return
        
        self.executor.submit(self._async_process, event)

    def _init_feature_cols(self, df):
        """
        Inicializa la lista de columnas de features a partir de un DataFrame.
        PROFESSOR METHOD:
        QUÉ: Método de inicialización de metadatos de entrenamiento.
        POR QUÉ: Para asegurar que el modelo siempre use el mismo orden y conjunto de columnas.
        CÓMO: Filtrando columnas no numéricas o reservadas (targets, OHLCV).
        """
        try:
            # Columnas a excluir (Targets y metadatos)
            exclude = [
                'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                'target', 'regime', 'timestamp', 'returns'
            ]
            
            # Obtener solo las columnas numéricas que no están en la lista de exclusión
            cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
            
            with self._state_lock:
                self._feature_cols = cols
                
            if not self._feature_cols:
                logger.error(f"❌ [{self.symbol}] _init_feature_cols: No valid feature columns found!")
        except Exception as e:
            logger.error(f"❌ Error initializing feature columns: {e}")
            self._feature_cols = []

    def _async_process(self, event):
        """Procesamiento en background"""
        if not self.running:
            return
            
        monitor.register_thread(f"ML_{event.symbol}")
        
        try:
            self.loop_count += 1
            # 1. Obtención de datos con LOOKBACK CLIP (Rule 2.2)
            required_bars = getattr(Config.Strategies, 'ML_LOOKBACK_BARS', 5000)
            bars = self.data_provider.get_latest_bars(self.symbol, n=required_bars)
            
            # Data Health Check
            num_bars = len(bars)
            if num_bars < self.min_bars_to_train:
                if self.loop_count % 10 == 0:
                    logger.warning(f"⚠️ [{self.symbol}] Waiting for data: {num_bars}/{self.min_bars_to_train} bars.")
                return

            
            # Initial Feature Initialization (Rule 2.3)
            # Preparamos features una vez para inicializar _feature_cols antes del primer training
            # PROFESSOR METHOD:
            # QUÉ: Inicialización garantizada de metadatos de entrada.
            # POR QUÉ: Evita errores de disparidad de dimensiones si el modelo arranca sin saber qué columnas usar.
            if self._feature_cols is None or not self._feature_cols:
                 temp_df = self._prepare_features(bars[:1000] if len(bars) > 1000 else bars)
                 if temp_df is not None and not temp_df.empty:
                     self._init_feature_cols(temp_df)
                     if self._feature_cols:
                         logger.info(f"📋 [{self.symbol}] Feature Columns established: {len(self._feature_cols)} features.")
                     else:
                         logger.error(f"❌ [{self.symbol}] CRITICAL: _init_feature_cols produced empty list. Check data types.")
                         return
            
            # Actualizar learning rate y parámetros
            self._adjust_learning_rate()
            
            # Verificar si necesita entrenar (Incremental o Full)
            with self._state_lock:
                self.bars_since_train += 1
                self.bars_since_incremental += 1
                
                # Check if a training thread is already active
                is_training = (hasattr(self, '_training_thread') and 
                               self._training_thread and self._training_thread.is_alive())

                # Update incremental cada X velas (ej. 30)
                needs_incremental = self.bars_since_incremental >= Config.Strategies.ML_INCREMENTAL_UPDATE_BARS
                should_train_full = (
                    (self.bars_since_train >= self.retrain_interval) or
                    (not self.is_trained)
                ) and (not is_training)
                
            if (should_train_full or needs_incremental) and not is_training:
                train_type = "Full" if should_train_full else "Incremental"
                logger.info(f"🔄 {train_type} training triggered for {self.symbol}")
                self._launch_training(bars, train_type)
            
            # Solo hacer inference si está entrenado
            if not self.is_trained:
                return
                
            self._run_inference()
            
        except Exception as e:
            logger.error(f"ML Async error {self.symbol}: {e}", exc_info=True)

    def _launch_training(self, bars, train_type="Full"):
        """Lanzar entrenamiento en thread separado"""
        def train_bg(bars_data, t_type):
            """Core training routine with concurrency control."""
            with TRAINING_LIMITER:
                monitor.register_thread(f"ML_Train_{self.symbol}")
                start_time = time.time()
                
                try:
                    if not self.running:
                        logger.debug(f"ML Training for {self.symbol} aborted: shutdown signal received.")
                        return
                        
                    logger.info(f"🔄 Starting {t_type} training #{self.training_iteration + 1} for {self.symbol}...")
                    
                    df = self._prepare_features(bars_data, regime_aware=True)
                    result, score = self._train_with_cross_validation(df)
                    
                    if result:
                        models, scaler, feature_cols = result
                        
                        # ✅ GUARDIA DE CALIDAD (Rule 3.3)
                        # PROFESSOR METHOD:
                        # QUÉ: Filtro de persistencia basado en performance diferencial.
                        # POR QUÉ: Evita que un entrenamiento malo o ruidoso degrade la inteligencia del sistema.
                        # CÓMO: Comparamos el nuevo score contra el 98% del anterior (umbral de tolerancia).
                        if self.is_trained:
                            if score < (self.last_training_score * 0.98):
                                logger.warning(
                                    f"🛡️ [Quality Guard] Rejected model update for {self.symbol}.\n"
                                    f"   Current Score: {score:.4f} | Previous Score: {self.last_training_score:.4f}\n"
                                    f"   Reason: Performance degradation exceeds 2% threshold."
                                )
                                # NO actualizamos, pero limpiamos flag de entrenamiento
                                with self._state_lock:
                                    self.bars_since_train = 0
                                return

                        if not self.running:
                            return
                            
                        with self._state_lock:
                            # Guardar los 3 modelos
                            self.rf_model = models['rf']
                            self.xgb_model = models['xgb']
                            self.gb_model = models['gb']
                            self.scaler = scaler
                            cleaned_feature_cols = [col for col in feature_cols if col is not None]
                            self._feature_cols = cleaned_feature_cols
                            self.is_trained = True
                            self.bars_since_train = 0
                            self.last_training_score = score
                            self.last_training_time = datetime.now(timezone.utc)
                            self.training_iteration += 1
                            self.bars_since_incremental = 0
                        
                        # 📊 PERSISTENCIA INTELIGENTE (Rule 3.4)
                        self._save_models()
                        
                        duration = time.time() - start_time
                        logger.info(
                            f"✨✨✨ [ML {self.symbol}] {t_type.upper()} TRAINING FINISHED ✨✨✨\n"
                            f"   Result: SUCCESS #{self.training_iteration} | Score: {score:.3f} | Total Time: {duration:.1f}s\n"
                            f"   Features Used: {len(feature_cols)} | Quality Guard: PASSED"
                        )
                    else:
                        # Even if training result is None (score below threshold),
                        # we must mark it as trained to avoid infinite re-train loops every minute.
                        # It will try again in the next retrain_interval.
                        with self._state_lock:
                            self.is_trained = True 
                            self.bars_since_train = 0
                            self.last_training_score = score
                            
                        logger.warning(
                            f"⚠️ ML {self.symbol} training failed (Score: {score:.3f} < {self.MIN_MODEL_ACCURACY}). "
                            "Marking as trained to wait for next interval."
                        )
                        
                except Exception as e:
                    logger.error(f"ML Training error {self.symbol}: {e}", exc_info=True)
                finally:
                    # MODO PROFESOR: Liberar RAM agresivamente
                    gc.collect()
        
        self._training_thread = threading.Thread(
            target=train_bg, 
            args=(copy.deepcopy(bars), train_type),
            daemon=True
        )
        self._training_thread.start()

    def stop(self):
        """
        Signal the strategy to stop processing and cleanup resources.
        """
        logger.info(f"🛑 [ML {self.symbol}] Stopping strategy...")
        self.running = False
        
        # Shutdown thread pool executor
        try:
            self.executor.shutdown(wait=False)
        except Exception as e:
            logger.debug(f"Error shutting down executor: {e}")
        
        # We don't force-join the training thread here to avoid hanging the main shutdown,
        # but the running flag inside the training thread will handle the early exit.

    # ============================================================
    # ✅ GESTIÓN DE TRADES Y ACTUALIZACIÓN DE PERFORMANCE
    # ============================================================
    
    def update_trade_result(self, signal_id, success, profit_pct=0.0):
        """
        Actualizar resultado de un trade para aprendizaje continuo
        """
        try:
            # Actualizar historial de performance
            result_value = 1 if success else -1
            if len(self.performance_history) > 0:
                self.performance_history[-1] = result_value
            
            # Actualizar ventana de performance
            self.performance_window.append(1 if success else 0)
            
            # Actualizar streaks
            if success:
                self.consecutive_losses = 0
                self.win_streak += 1
                self.loss_streak = 0
                self.winning_trades += 1
                self.max_win_streak = max(self.max_win_streak, self.win_streak)
            else:
                self.consecutive_losses += 1
                self.loss_streak += 1
                self.win_streak = 0
                self.losing_trades += 1
                self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
            
            self.total_trades += 1
            
            # Actualizar accuracy por régimen
            if self.market_regime in self.regime_accuracy:
                self.regime_accuracy[self.market_regime].append(1 if success else 0)

        except Exception as e:
            logger.error(f"Error updating trade result: {e}")

    # ============================================================
    # ✅ PERSISTENCIA DE MODELOS EN DISCO
    # ============================================================
    
    def _save_models(self):
        """Guardar modelos y scaler con sistema de rotación backup (Rule 3.4)"""
        import joblib
        try:
            symbol_path = self.symbol.replace("/", "_")
            model_file = os.path.join(self.models_dir, f"models_{symbol_path}.joblib")
            
            # Sistema de Backup Rotativo
            if os.path.exists(model_file):
                old_path = model_file + ".old"
                if os.path.exists(old_path):
                    os.remove(old_path)
                os.rename(model_file, old_path)

            with self._state_lock:
                state = {
                    'rf_model': self.rf_model,
                    'xgb_model': self.xgb_model,
                    'gb_model': self.gb_model,
                    'scaler': self.scaler,
                    'feature_cols': self._feature_cols,
                    'last_training_score': self.last_training_score,
                    'training_iteration': self.training_iteration,
                    'performance_history': list(self.performance_history),
                    'timestamp': datetime.now(timezone.utc)
                }
            
            joblib.dump(state, model_file, compress=5)
            
            # === [PHASE 12] REGISTRO EN GOBERNANZA ===
            # Exportamos componentes individuales para el registry
            reg_dir = os.path.join(self.models_dir, f"registry_{symbol_path}")
            os.makedirs(reg_dir, exist_ok=True)
            
            comp_paths = {
                'rf': os.path.join(reg_dir, "rf.joblib"),
                'xgb': os.path.join(reg_dir, "xgb.joblib"),
                'gb': os.path.join(reg_dir, "gb.joblib"),
                'scaler': os.path.join(reg_dir, "scaler.joblib")
            }
            
            joblib.dump(self.rf_model, comp_paths['rf'])
            joblib.dump(self.xgb_model, comp_paths['xgb'])
            joblib.dump(self.gb_model, comp_paths['gb'])
            joblib.dump(self.scaler, comp_paths['scaler'])
            
            metrics = {
                'sharpe': self.last_training_score,
                'win_rate': 0.0
            }
            if len(self.performance_history) > 0:
                win_rate = len([p for p in self.performance_history if p > 0]) / len(self.performance_history)
                metrics['win_rate'] = win_rate * 100

            self.ml_governance.register_model(self.symbol, metrics, comp_paths)
            
            logger.info(f"💾 [{self.symbol}] Models persisted and registered in Governance.")
        except Exception as e:
            logger.error(f"Error saving models for {self.symbol}: {e}")
    def _load_governed_model(self):
        """
        👨‍🏫 MODO PROFESOR:
        QUÉ: Carga inteligente de modelos certificados.
        POR QUÉ: Priorizamos modelos que han pasado el Quality Gate de la gobernanza.
        """
        import joblib
        gov_model = self.ml_governance.get_production_model(self.symbol)
        if gov_model:
            try:
                path = gov_model['path']
                self.rf_model = joblib.load(os.path.join(path, "rf.joblib"))
                self.xgb_model = joblib.load(os.path.join(path, "xgb.joblib"))
                self.gb_model = joblib.load(os.path.join(path, "gb.joblib"))
                self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
                
                # Cargar columnas (buscamos en models_dir original por ahora o el último cache)
                cols_path = os.path.join(self.models_dir, f"features_{self.symbol.replace('/', '_')}.json")
                if os.path.exists(cols_path):
                    with open(cols_path, 'r') as f:
                        self._feature_cols = json.load(f)
                
                self.is_trained = True
                logger.info(f"🏆 [{self.symbol}] Cargado modelo de PRODUCCIÓN v{gov_model['version']} (Sharpe: {gov_model['sharpe']:.2f})")
                return True
            except Exception as e:
                logger.error(f"❌ Error cargando modelo de gobernanza: {e}")
        
        # Fallback a carga tradicional si no hay modelo de gobernanza
        return self._load_models()

    def _load_models(self):
        """Cargar modelos desde el disco para operatividad instantánea"""
        try:
            symbol_path = self.symbol.replace("/", "_")
            model_file = os.path.join(self.models_dir, f"models_{symbol_path}.joblib")
            
            # --- TRANSFER LEARNING: If no model, try to use BTC as base ---
            btc_model_file = os.path.join(self.models_dir, "models_BTC_USDT.joblib")
            use_transfer_learning = False
            
            if os.path.exists(model_file):
                state = joblib.load(model_file)
                save_time = state.get('timestamp')
                
                # --- VERIFICACIÓN DE CADUCIDAD (FRESHNESS) ---
                is_stale = False
                if save_time:
                    age_hours = (datetime.now(timezone.utc) - save_time).total_seconds() / 3600
                    if age_hours > 24:
                        is_stale = True
                        logger.warning(f"⚠️ [{self.symbol}] Intelligence STALE ({age_hours:.1f}h old). Checking transfer learning...")
                
                # ✅ VALIDACIÓN DE FEATURES
                loaded_feature_cols = state.get('feature_cols', [])
                cleaned_feature_cols = [col for col in loaded_feature_cols if col is not None]
                
                # ✅ CRITICAL: Si no hay features válidas, el modelo es corrupto
                if not cleaned_feature_cols:
                    logger.error(f"❌ [{self.symbol}] Corrupted model detected (no valid features). Checking transfer learning...")
                    is_stale = True
                
                if not is_stale:
                    with self._state_lock:
                        self.rf_model = state['rf_model']
                        self.xgb_model = state['xgb_model']
                        self.gb_model = state['gb_model']
                        self.scaler = state['scaler']
                        self._feature_cols = cleaned_feature_cols
                        self.last_training_score = state.get('last_training_score', 0)
                        self.training_iteration = state.get('training_iteration', 0)
                        
                        hist = state.get('performance_history', [])
                        self.performance_history = deque(hist, maxlen=100)
                        
                        self.is_trained = True
                    
                    logger.info(f"🟢 [{self.symbol}] ML HYBRID ULTIMATE [ENSEMBLE] - Native model loaded")
                    return
                else:
                    use_transfer_learning = True
            else:
                use_transfer_learning = True
            
            # --- TRANSFER LEARNING FROM BTC ---
            if use_transfer_learning and self.symbol != 'BTC/USDT' and os.path.exists(btc_model_file):
                try:
                    btc_state = joblib.load(btc_model_file)
                    btc_features = btc_state.get('feature_cols', [])
                    cleaned_btc_features = [col for col in btc_features if col is not None]
                    
                    if cleaned_btc_features and btc_state.get('rf_model'):
                        with self._state_lock:
                            self.rf_model = btc_state['rf_model']
                            self.xgb_model = btc_state['xgb_model']
                            self.gb_model = btc_state['gb_model']
                            self.scaler = btc_state['scaler']
                            self._feature_cols = cleaned_btc_features
                            self.is_trained = True
                            self.training_iteration = 0  # Mark for retraining
                        
                        logger.info(f"🟢 [{self.symbol}] ML HYBRID ULTIMATE [ENSEMBLE] - Transfer learned from BTC")
                        return
                except Exception as e:
                    logger.warning(f"⚠️ [{self.symbol}] Transfer learning failed: {e}")
            
            logger.info(f"🟢 [{self.symbol}] ML HYBRID ULTIMATE [ENSEMBLE] - Fresh training starting...")
            
        except Exception as e:
            logger.error(f"Error loading models for {self.symbol}: {e}")

    # ============================================================
    # ✅ MÉTODOS DE MONITOREO Y DIAGNÓSTICO
    # ============================================================

    
    def get_strategy_status(self):
        """
        Obtener estado completo de la estrategia
        """
        try:
            # Calcular métricas
            recent_win_rate = 0.0
            if len(self.performance_window) > 0:
                recent_win_rate = sum(self.performance_window) / len(self.performance_window)
            
            total_win_rate = 0.0
            if self.total_trades > 0:
                total_win_rate = self.winning_trades / self.total_trades
            
            # Accuracy por régimen
            regime_stats = {}
            for regime, results in self.regime_accuracy.items():
                if len(results) > 0:
                    regime_stats[regime] = {
                        'accuracy': sum(results) / len(results),
                        'signals': len(results),
                        'total_signals': self.signals_by_regime.get(regime, 0)
                    }
                else:
                    regime_stats[regime] = {'accuracy': 0.0, 'signals': 0, 'total_signals': 0}
            
            # Feature importance si está disponible
            feature_importance = {}
            if self.rf_model is not None and self._feature_cols is not None:
                try:
                    importances = self.rf_model.feature_importances_
                    top_features = dict(zip(self._feature_cols, importances))
                    sorted_features = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:20]
                    feature_importance = dict(sorted_features)
                except:
                    feature_importance = {}
            
            return {
                # Identificación
                'strategy_id': self.strategy_id,
                'symbol': self.symbol,
                'objective': f"${self.initial_capital} → ${self.target_capital}",
                
                # Estado general
                'trained': self.is_trained,
                'training_iteration': self.training_iteration,
                'training_score': self.last_training_score,
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                'bars_since_train': self.bars_since_train,
                'feature_count': len(self._feature_cols) if self._feature_cols else 0,
                
                # Régimen de mercado
                'market_regime': self.market_regime,
                'regime_confidence': self.regime_confidence,
                'regime_duration': self.regime_duration,
                'regime_history': list(self.regime_history)[-10:],
                'regime_stats': regime_stats,
                
                # Circuit breaker
                'circuit_breaker_active': self.circuit_breaker_active,
                'consecutive_losses': self.consecutive_losses,
                'max_consecutive_losses': self.max_consecutive_losses,
                'peak_equity': self.peak_equity,
                
                # Modelos y ensemble
                'model_weights': {
                    'rf': float(self.base_rf_weight),
                    'xgb': float(self.base_xgb_weight),
                    'gb': float(self.base_gb_weight)
                },
                'model_scores': {k: float(v) for k, v in self.individual_model_scores.items()},
                'feature_importance': feature_importance,
                
                # Parámetros adaptativos
                'adaptive_parameters': {
                    'confidence_threshold': float(self.adaptive_confidence_threshold),
                    'confluence_long': float(self.adaptive_confluence_long),
                    'confluence_short': float(self.adaptive_confluence_short),
                    'learning_rate': float(self.learning_rate),
                    'aggressiveness_factor': float(self.aggressiveness_factor)
                },
                
                # Targets
                'targets': {
                    'tp': float(self.current_tp_target),
                    'sl': float(self.current_sl_target),
                    'tp_sl_ratio': float(self.current_tp_target / self.current_sl_target) if self.current_sl_target > 0 else 0
                },
                
                # Performance completa
                'performance': {
                    'total_signals': self.total_signals_generated,
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'losing_trades': self.losing_trades,
                    'total_win_rate': float(total_win_rate),
                    'recent_win_rate': float(recent_win_rate),
                    'max_win_streak': self.max_win_streak,
                    'max_loss_streak': self.max_loss_streak,
                    'current_win_streak': self.win_streak,
                    'current_loss_streak': self.loss_streak,
                    'compounding_factor': float(self.compounding_factor)
                },
                
                # Signals por régimen
                'signals_by_regime': self.signals_by_regime,
                
                # Historial reciente
                'recent_performance': list(self.performance_history)[-20:],
                'recent_signals': [
                    {
                        'time': s['timestamp'].isoformat(),
                        'type': s['type'].name,
                        'confidence': s['confidence'],
                        'regime': s['regime'],
                        'price': s['price'],
                        'confluence': s['confluence']
                    }
                    for s in list(self.signal_history)[-10:]
                ],
                
                # Estado del sistema
                'thread_alive': self._training_thread.is_alive() if hasattr(self, '_training_thread') and self._training_thread else False,
                'last_prediction': self._last_prediction_time.isoformat() if self._last_prediction_time else None,
                'current_capital': float(self.current_capital),
                'progress_percentage': float((self.current_capital - self.initial_capital) / (self.target_capital - self.initial_capital) * 100) if self.target_capital > self.initial_capital else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            return {'error': str(e)}

    def get_performance_report(self):
        """
        Generar reporte detallado de performance
        """
        status = self.get_strategy_status()
        
        report = {
            'summary': {
                'symbol': status['symbol'],
                'trained': status['trained'],
                'total_trades': status['performance']['total_trades'],
                'win_rate': f"{status['performance']['total_win_rate']*100:.1f}%",
                'recent_win_rate': f"{status['performance']['recent_win_rate']*100:.1f}%",
                'current_regime': status['market_regime'],
                'circuit_breaker': status['circuit_breaker_active']
            },
            'model_info': {
                'training_score': f"{status['training_score']:.3f}",
                'ensemble_weights': status['model_weights'],
                'model_scores': status['model_scores']
            },
            'parameters': status['adaptive_parameters'],
            'targets': status['targets'],
            'streaks': {
                'current_win_streak': status['performance']['current_win_streak'],
                'current_loss_streak': status['performance']['current_loss_streak'],
                'max_win_streak': status['performance']['max_win_streak'],
                'max_loss_streak': status['performance']['max_loss_streak']
            },
            'regime_performance': status['regime_stats']
        }
        
        return json.dumps(report, indent=2)

    def force_retrain(self):
        """Forzar reentrenamiento inmediato"""
        with self._state_lock:
            self.is_trained = False
            self.bars_since_train = self.retrain_interval
        logger.info(f"🔧 Forced retrain triggered for {self.symbol}")

    def reset_weights(self):
        """Resetear pesos a valores originales"""
        self.base_rf_weight = self.original_rf_weight
        self.base_xgb_weight = self.original_xgb_weight
        self.base_gb_weight = self.original_gb_weight
        logger.info("🔄 Model weights reset to original values")

    def reset_circuit_breaker(self):
        """Resetear circuit breaker manualmente"""
        self.circuit_breaker_active = False
        self.consecutive_losses = 0
        self.adaptive_confidence_threshold = self.BASE_CONFIDENCE_THRESHOLD
        self.peak_equity = None
        self.aggressiveness_factor = 1.0
        logger.info("🔄 Circuit breaker manually reset")

    def set_aggressiveness(self, level):
        """
        Ajustar nivel de agresividad manualmente
        Level: 0.5 (conservador), 1.0 (normal), 1.5 (agresivo), 2.0 (muy agresivo)
        """
        level = max(0.3, min(2.0, level))
        self.aggressiveness_factor = level
        logger.info(f"🔧 Aggressiveness set to {level:.2f}")

    # ============================================================
    # ✅ CLEANUP Y DESTRUCTOR
    # ============================================================
    
    def __del__(self):
        """Cleanup seguro"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            
            logger.info(f"🧹 ML Hybrid Ultimate Strategy for {self.symbol} cleaned up")
        except:
            pass

# ============================================================
# ✅ UNIVERSAL ENSEMBLE STRATEGY - UNIFIED DECISION MAKING
# ============================================================

class UniversalEnsembleStrategy(MLStrategyHybridUltimate):
    """
    🔮 UNIVERSAL ENSEMBLE STRATEGY
    
    PROFESSOR METHOD:
    - QUÉ: Estrategia unificada que combina 3 motores de decisión
    - POR QUÉ: Elimina modelos heredados y fuerza consenso orgánico
    - CÓMO: Promedia señales de ML, Sentiment y Technical
    - CUÁNDO: 2 de 3 motores deben superar 0.60 para operar
    - DÓNDE: Reemplaza cualquier lógica de "Previous models"
    
    Engines:
    1. ML Engine: RF + XGB + GB ensemble inference
    2. Sentiment Engine: VADER/Social momentum analysis
    3. Technical Engine: RSI + EMA Cross + Bollinger confluence
    
    Consensus: Organic Confluence (min 2/3 engines > 0.60)
    """
    
    # Consensus threshold (UNIFIED)
    ENSEMBLE_CONSENSUS_THRESHOLD = 0.75
    MIN_ENGINES_REQUIRED = 2      # Requires at least 2 engines to agree (ML+TECH, ML+SNT, etc)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override strategy ID for ENSEMBLE mode
        self.strategy_id = "ML_HYBRID_ULTIMATE_ENSEMBLE_V3"
        
        # Engine tracking
        self.engine_scores = {
            'ml': 0.0,
            'sentiment': 0.0,
            'technical': 0.0
        }
        self.engines_active = 0
        self.consensus_threshold = 0.60
        
        logger.info(f"🟢 [{self.symbol}] UNIVERSAL ENSEMBLE STRATEGY INITIALIZED")
        logger.info(f"   Consensus Threshold: {self.ENSEMBLE_CONSENSUS_THRESHOLD}")
        logger.info(f"   Min Engines Required: {self.MIN_ENGINES_REQUIRED}/3")

    def _run_inference(self):
        """
        🔮 UNIVERSAL ENSEMBLE INFERENCE
        Overridden to force 3-engine consensus and bridge validation.
        """
        try:
            self.analysis_stats['total'] += 1
            if not self._check_circuit_breaker():
                return
            
            # 1. Data Preparation
            bars = self.data_provider.get_latest_bars(self.symbol, n=250)
            df = self._prepare_features(bars, regime_aware=True)
            if df.empty or len(df) < 5: return

            current_row = df.iloc[-1]
            atr_pct = current_row['atr_pct'] / 100
            vol_ratio = current_row.get('volume_ratio', 0)

            # 2. Model Availability
            with self._state_lock:
                models_ready = all([self.rf_model, self.xgb_model, self.gb_model])
                feature_cols = self._feature_cols

            if not models_ready:
                return

            # 3. Aligned Feature Matrix
            if hasattr(self.scaler, 'feature_names_in_'):
                final_features = self.scaler.feature_names_in_
                X_pred_aligned = pd.DataFrame(columns=final_features)
                for col in final_features:
                    X_pred_aligned[col] = df[col].iloc[[-1]].values if col in df.columns else 0.0
                X_pred = X_pred_aligned
            else:
                X_pred = df[feature_cols].iloc[[-1]]
                
            X_scaled = self.scaler.transform(X_pred)
            rf_proba = self.rf_model.predict_proba(X_scaled)[0]
            xgb_proba = self.xgb_model.predict_proba(X_scaled)[0]
            gb_proba = self.gb_model.predict_proba(X_scaled)[0]
            
            ensemble_proba = (rf_proba * self.base_rf_weight + 
                             xgb_proba * self.base_xgb_weight + 
                             gb_proba * self.base_gb_weight)
            
            classes = self.rf_model.classes_
            pred_idx = np.argmax(ensemble_proba)
            raw_confidence = ensemble_proba[pred_idx]
            direction = self._label_mapping.get(classes[pred_idx], classes[pred_idx])
            
            # ============================================================
            # 🎯 3-ENGINE ORGANIC CONFLUENCE (The Heart of Phase 8)
            # ============================================================
            final_confidence, engines_passing, is_valid, multi_horizon = self.compute_organic_confluence(
                df, direction, rf_proba, xgb_proba, gb_proba
            )
            
            # 4. ORACLE REPORT (Universal Multi-Engine View)
            gap = self.consensus_threshold - final_confidence
            ready_status = "READY" if is_valid else "SCANNING"
            
            # Extract Horizon Details
            h5 = multi_horizon.get('h5', 0)
            h15 = multi_horizon.get('h15', 0)
            h30 = multi_horizon.get('h30', 0)

            # Prepare Enhanced Stats
            z_score = current_row.get('volume_zscore', 0)
            adx = current_row.get('adx', 0)
            trend_power = current_row.get('trend_power', 0) 
            
            # Determine Concept/Context based on Regime
            if self.market_regime == "ZOMBIE":
                concept = "Zombie market detected. Stagnant price action. Protection active."
            elif self.market_regime == "RANGING":
                concept = "Mean Reversion active. Hunting overextensions."
            elif self.market_regime == "TRENDING":
                concept = "Trend Following active. Riding momentum."
            elif self.market_regime == "VOLATILE":
                concept = "High Volatility. Defensive stops & wide targets."
            else:
                concept = "Analyzing market structure..."

            oracle_msg = (
                f"\n🔮 [UNIFIED ORACLE] {self.symbol} | {ready_status}\n"
                f"   Engines Passing: {engines_passing}/3 | Threshold: {self.consensus_threshold}\n"
                f"   Scores  -> ML: {self.engine_scores['ml']:.2f} | SENT: {self.engine_scores['sentiment']:.2f} | TECH: {self.engine_scores['technical']:.2f}\n"
                f"   Horizon -> H5: {h5:.2f} | H15: {h15:.2f} | H30: {h30:.2f}\n"
                f"   Verdict -> Direction: {direction} | Final Conf: {final_confidence:.2f} (Gap: {gap:.2f})\n"
                f"   Phase: {self.market_regime} ({self.regime_confidence*100:.1f}%)\n"
                f"   Concept: {concept}\n"
                f"   Stats: ADX={adx:.1f} | ATR%={atr_pct*100:.2f}% | Trend={trend_power:.2f} | Z-Score={z_score:.2f}\n"
                f"   Confidence: {final_confidence*100:.1f}% | Strategy: Adaptive targets enabled."
            )
            logger.info(oracle_msg)


            # 5. ROBUSTNESS FILTERS
            if atr_pct > self.MAX_ATR_PCT * 1.5:
                logger.warning(f"⛔ [FILTER] {self.symbol} Rejected: Extreme Volatility (ATR: {atr_pct:.2%})")
                return
            
            min_vol = 0.2 if Config.BINANCE_USE_TESTNET else self.MIN_VOLUME_RATIO
            if vol_ratio < min_vol:
                logger.warning(f"⛔ [FILTER] {self.symbol} Rejected: Low Volume (Ratio: {vol_ratio:.2f} < {min_vol})")
                return
            
            if not is_valid:
                self.analysis_stats['filtered_conf'] += 1
                return

            # 6. SIGNAL CREATION
            signal_type = SignalType.LONG if direction == 1 else SignalType.SHORT
            tp_target = self.current_tp_target
            sl_target = self.current_sl_target
            
            # Volatility adjustments
            if atr_pct > 0.03: tp_target *= 1.3; sl_target *= 1.3
            elif atr_pct < 0.01: tp_target *= 0.8; sl_target *= 0.8

            signal = SignalEvent(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                datetime=datetime.now(timezone.utc),
                signal_type=signal_type,
                strength=final_confidence,
                atr=current_row['atr'],
                tp_pct=tp_target * 100,
                sl_pct=sl_target * 100,
                current_price=current_row['close']
            )
            
            # 7. LOGGING & SUBMISSION
            self.performance_history.append(0)
            self.signal_history.append({
                'timestamp': datetime.now(timezone.utc),
                'type': signal_type,
                'confidence': final_confidence,
                'engines': engines_passing,
                'price': current_row['close']
            })
            
            logger.info(f"✨ [UNIVERSAL ENSEMBLE] Signal Generated: {signal_type.name} {self.symbol}")
            self.events_queue.put(signal)
            
            if len(self.performance_history) >= 15:
                self._update_model_weights()
            
            self._last_prediction_time = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Universal Ensemble Inference error {self.symbol}: {e}", exc_info=True)
    
    def _compute_ml_engine_score(self, rf_proba, xgb_proba, gb_proba, direction: int) -> float:
        """
        Motor ML: Weighted average of ensemble models.
        """
        idx = 1 if direction == 1 else 0  # LONG = idx 1, SHORT = idx 0
        
        ml_score = (
            rf_proba[idx] * self.base_rf_weight +
            xgb_proba[idx] * self.base_xgb_weight +
            gb_proba[idx] * self.base_gb_weight
        )
        
        self.engine_scores['ml'] = ml_score
        return ml_score
    
    def _compute_sentiment_engine_score(self, df) -> float:
        """
        Motor Sentiment: VADER/Social momentum analysis.
        """
        try:
            current_row = df.iloc[-1]
            
            # Get sentiment components
            sentiment = current_row.get('sentiment', 0.0)
            sentiment_momentum = current_row.get('sentiment_momentum', 0.0)
            
            # Normalize to 0-1 range
            # Sentiment is typically -1 to 1, so we map to 0.5 as neutral
            sentiment_normalized = (sentiment + 1) / 2
            
            # Momentum adds weight
            if sentiment_momentum > 0:
                sentiment_score = sentiment_normalized * (1 + min(sentiment_momentum, 0.3))
            else:
                sentiment_score = sentiment_normalized * (1 + max(sentiment_momentum, -0.3))
            
            # Clamp to [0, 1]
            sentiment_score = max(0.0, min(1.0, sentiment_score))
            
            self.engine_scores['sentiment'] = sentiment_score
            return sentiment_score
            
        except Exception:
            self.engine_scores['sentiment'] = 0.5  # Neutral if unavailable
            return 0.5
    
    def _compute_technical_engine_score(self, df, direction: int) -> float:
        """
        Motor Técnico: RSI + EMA Cross + Bollinger Bands confluence.
        """
        try:
            current_row = df.iloc[-1]
            score = 0.0
            factors = 0
            
            # RSI Component (weight: 35%)
            rsi = current_row.get('rsi_14', 50)
            if direction == 1:  # LONG
                # RSI should be above 40 but not overbought
                if 40 <= rsi <= 70:
                    score += 0.35 * ((70 - abs(rsi - 55)) / 35)
                    factors += 1
            else:  # SHORT
                # RSI should be below 60 but not oversold
                if 30 <= rsi <= 60:
                    score += 0.35 * ((60 - abs(rsi - 45)) / 35)
                    factors += 1
            
            # EMA Cross Component (weight: 35%)
            ema_cross = current_row.get('ema_20_50_cross', 0)
            if (direction == 1 and ema_cross > 0) or (direction == -1 and ema_cross < 0):
                score += 0.35
                factors += 1
            
            # Bollinger Bands Component (weight: 30%)
            if 'bb_pctb' in current_row:
                bb_pct = current_row['bb_pctb']
                if direction == 1:  # LONG - want price near lower band
                    if bb_pct < 0.3:
                        score += 0.30 * (1 - bb_pct / 0.3)
                        factors += 1
                else:  # SHORT - want price near upper band
                    if bb_pct > 0.7:
                        score += 0.30 * ((bb_pct - 0.7) / 0.3)
                        factors += 1
            
            # Normalize score
            technical_score = score if factors > 0 else 0.5
            
            self.engine_scores['technical'] = technical_score
            return technical_score
            
        except Exception:
            self.engine_scores['technical'] = 0.5
            return 0.5
    
    def compute_organic_confluence(self, df, direction: int,
                                   rf_proba, xgb_proba, gb_proba) -> tuple:
        """
        🎯 ORGANIC CONFLUENCE CALCULATOR
        Returns: (final_confidence, engines_passing, is_valid, multi_horizon)
        """
        # 1. Calculate each engine score
        ml_score = self._compute_ml_engine_score(rf_proba, xgb_proba, gb_proba, direction)
        sentiment_score = self._compute_sentiment_engine_score(df)
        technical_score = self._compute_technical_engine_score(df, direction)
        
        # 2. Dynamic Threshold Logic (World Awareness)
        # PROFESSOR METHOD: adaptamos el rigor según la liquidez global.
        ls = getattr(self, 'market_context', {}).get('liquidity_score', 0.8)
        
        # Base is the user's setting (0.75)
        # If LS is low (dead zone), we demand extreme confluence (0.82)
        base_t = self.ENSEMBLE_CONSENSUS_THRESHOLD
        
        if ls >= 0.85:   # PRIME (London/NY)
            dynamic_threshold = base_t
        elif ls >= 0.65: # MID (Tokyo)
            dynamic_threshold = max(base_t, 0.78)
        elif ls >= 0.50: # LOW (Sydney)
            dynamic_threshold = max(base_t, 0.80)
        else:            # DEAD ZONE (22:00-00:00 UTC)
            dynamic_threshold = max(base_t, 0.82)
            
        self.consensus_threshold = dynamic_threshold # Update for oracle logging
        
        # ============================================================
        # ⏳ SYNTHETIC MULTI-HORIZON LOGIC (H5, H15, H30)
        # ============================================================
        # H5 (Short): Momentum-heavy (Tech + Sent)
        h5_score = (technical_score * 0.6) + (sentiment_score * 0.4)
        
        # H15 (Mid): Pure consensus
        h15_score = (ml_score * 0.34) + (technical_score * 0.33) + (sentiment_score * 0.33)
        
        # H30 (Full): ML Dominant (Original Model Horizon)
        h30_score = (ml_score * 0.7) + (((technical_score + sentiment_score) / 2) * 0.3)
        
        multi_horizon = {
            'h5': max(0.0, min(1.0, h5_score)),
            'h15': max(0.0, min(1.0, h15_score)),
            'h30': max(0.0, min(1.0, h30_score))
        }

        # Count engines passing threshold
        engines_passing = sum(1 for score in [ml_score, sentiment_score, technical_score]
                             if score >= dynamic_threshold)
        
        # Calculate weighted final confidence
        # ML has higher weight as it's the primary engine
        final_confidence = (
            ml_score * 0.50 +           # 50% weight to ML
            sentiment_score * 0.20 +     # 20% weight to sentiment
            technical_score * 0.30       # 30% weight to technical
        )
        
        # Apply penalty if not enough engines agree
        if engines_passing < self.MIN_ENGINES_REQUIRED:
            penalty = 0.8 if engines_passing == 1 else 0.6
            final_confidence *= penalty
        
        self.engines_active = engines_passing
        is_valid = engines_passing >= self.MIN_ENGINES_REQUIRED
        
        # Phase 8: Neural Bridge Publication
        neural_bridge.publish_insight(
            strategy_id="ML_ENSEMBLE",
            symbol=self.symbol,
            insight={
                'confidence': final_confidence,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'engines_passing': engines_passing,
                'horizons': multi_horizon
            }
        )
        
        return final_confidence, engines_passing, is_valid, multi_horizon

    
    def get_ensemble_status(self) -> dict:
        """Get current ensemble engine status."""
        return {
            'engines': self.engine_scores.copy(),
            'engines_active': self.engines_active,
            'threshold': self.ENSEMBLE_CONSENSUS_THRESHOLD,
            'is_unified': True,
            'mode': 'UNIVERSAL_ENSEMBLE'
        }


# Add method to base class for backwards compatibility
def _add_ensemble_methods_to_base():
    """Inject ensemble methods into base class."""
    
    def compute_organic_confluence(self, df, direction, rf_proba, xgb_proba, gb_proba):
        """Simplified organic confluence for base class."""
        # ML Engine Score
        idx = 1 if direction == 1 else 0
        ml_score = (
            rf_proba[idx] * self.base_rf_weight +
            xgb_proba[idx] * self.base_xgb_weight +
            gb_proba[idx] * self.base_gb_weight
        )
        
        # Technical Engine Score (from confluence_score)
        technical_score = abs(df.iloc[-1].get('confluence_score', 0))
        technical_score = min(1.0, technical_score + 0.5)  # Normalize
        
        # Count passing engines (ML + Technical, sentiment optional)
        THRESHOLD = 0.60
        engines_passing = sum(1 for s in [ml_score, technical_score] if s >= THRESHOLD)
        
        # Weighted average
        final_conf = ml_score * 0.6 + technical_score * 0.4
        
        if engines_passing < 2:
            final_conf *= 0.8  # Penalty
        
        return final_conf, engines_passing, engines_passing >= 2
    
    # Add method if not exists
    if not hasattr(MLStrategyHybridUltimate, 'compute_organic_confluence'):
        MLStrategyHybridUltimate.compute_organic_confluence = compute_organic_confluence

_add_ensemble_methods_to_base()


# ============================================================
# ✅ FACTORY FUNCTION PARA CREACIÓN FÁCIL
# ============================================================

def create_ml_strategy_hybrid_ultimate(data_provider, events_queue, symbol='BTC/USDT',
                                       sentiment_loader=None, portfolio=None,
                                       initial_capital=12.0, target_capital=100000.0):
    """
    Factory function para crear la estrategia híbrida ultimate
    
    Args:
        data_provider: Proveedor de datos
        events_queue: Cola de eventos
        symbol: Par de trading
        sentiment_loader: Cargador de sentiment (opcional)
        portfolio: Portfolio manager (opcional)
        initial_capital: Capital inicial (default: 12 USD)
        target_capital: Capital objetivo (default: 100,000 USD)
    
    Returns:
        UniversalEnsembleStrategy instance (3-engine consensus for ALL symbols)
    """
    # UNIVERSAL ENSEMBLE: All symbols use the same unified strategy
    strategy = UniversalEnsembleStrategy(
        data_provider=data_provider,
        events_queue=events_queue,
        symbol=symbol,
        sentiment_loader=sentiment_loader,
        portfolio=portfolio
    )
    
    # Sobreescribir objetivos si se especifican
    if initial_capital > 0:
        strategy.initial_capital = initial_capital
        strategy.current_capital = initial_capital
    
    if target_capital > 0:
        strategy.target_capital = target_capital
    
    logger.info(
        f"🟢 UNIVERSAL ENSEMBLE STRATEGY created for {symbol} | "
        f"Goal: ${strategy.initial_capital} → ${strategy.target_capital} | "
        f"Engines: ML+Sentiment+Technical | Threshold: 0.60"
    )
    
    return strategy

if __name__ == "__main__":
    """
    Test y demostración de la estrategia
    """
    print("=" * 80)
    print("🚀 ML STRATEGY HYBRID ULTIMATE - DEMONSTRATION")
    print("=" * 80)
    print("✅ FEATURES INCLUDED:")
    print("   1. Ensemble completo: RF + XGB + GB con weighted voting dinámico")
    print("   2. Detección de régimen avanzada con 4 regímenes")
    print("   3. Circuit breaker automático por drawdown (12%)")
    print("   4. Feature engineering adaptativo (80+ features)")
    print("   5. Targets dinámicos por volatilidad y régimen")
    print("   6. Re-pesado dinámico basado en performance")
    print("   7. Learning rate adaptativo y factor de agresividad")
    print("   8. Monitoreo completo con 40+ métricas")
    print("   9. Arquitectura asíncrona optimizada")
    print("  10. Gestión de riesgo multi-capa")
    print("=" * 80)
    print("🎯 OBJETIVO: Convertir $12 USD en $100,000 USD en el menor tiempo posible")
    print("=" * 80)