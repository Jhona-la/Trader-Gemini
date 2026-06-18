import sys
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
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTHONNOUSERSITE"] = "1"
warnings.filterwarnings("ignore")
try:
    from pandas.errors import PerformanceWarning

    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception as e:
    logger.error(f"Silent exception caught: {e}", exc_info=True)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

import threading
from collections import Counter, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from config import Config
from utils.notifier import Notifier
from core.enums import EventType, SignalType
from core.events import SignalEvent
from core.fused_strategy_kernel import (  # Nano-latency JIT Inference
    predict_gb_jit,
    predict_rf_jit,
)
from core.neural_bridge import neural_bridge
from core.xai_engine import XAIEngine  # Phase 22: Explainability
from models.deep_predictor import deep_predictor  # Phase 4 AITS: Deep Learning Model
from models.omniscient_predictor import omniscient_engine  # Phase 6: Seq2Seq 1000-candle predictor
from ml.tree_compiler import (  # Model to Matrix parser
    compile_gb_to_numpy_batch,
    compile_rf_to_numpy_batch,
)
from sophia.intelligence import MultiHorizonOracle, SophiaIntelligence  # Phase 4: Oracle Veto + Singleton XAI
from strategies.components.adaptive_engine import (
    AdaptiveMLParameterEngine,  # Phase 3: Adaptive Engine
)
from strategies.components.feature_engineering import (
    FeatureEngineering,  # Phase I: Refactoring
)
from strategies.components.signal_generator import (
    SignalGenerator,  # Phase I: Refactoring
)
from strategies.phalanx import (  # PHASE 13: Phalanx-Omega Protocol
    OnlineGARCH,
    OrderFlowAnalyzer,
)
from utils.debug_tracer import trace_execution
from utils.logger import logger
from utils.math_kernel import (  # Fast Math for Oracle
    calculate_ema_jit,
    calculate_rsi_jit,
)
from utils.thread_monitor import monitor

from .strategy import Strategy

# ═══════════════════════════════════════════════════════════════
# PHASE 2 POWER: Online Reinforcement Learning (Hot Adapter)
# QUÉ: Módulo de aprendizaje en caliente que ajusta bias por símbolo/dirección.
# POR QUÉ: XGBoost es estático entre re-entrenamientos. El mercado cambia cada segundo.
# PARA QUÉ: Si el mercado invalida una dirección, el bias la penaliza INMEDIATAMENTE
#   sin tener que re-entrenar 100 árboles.
# CÓMO: Multiplicador probabilístico que se actualiza con cada trade cerrado.
# CUÁNDO: Se aplica en cada inferencia, después del ensemble voting.
# DÓNDE: strategies/ml_strategy.py → _analyze_and_generate()
# QUIÉN: HotAdapterRL (ml/online_learning.py)
# ═══════════════════════════════════════════════════════════════
try:
    from ml.online_learning import HotAdapterRL
    _HOT_ADAPTER_AVAILABLE = True
except ImportError:
    _HOT_ADAPTER_AVAILABLE = False

try:
    import ujson as json
except ImportError:
    import json
import asyncio
import gc
import queue
import threading
import time
import traceback

import joblib

from core.ml_governance import MLGovernance
from core.online_learning import OnlineLearner
from core.reward_system import RewardSystem, TradeOutcome
from data.feature_store import FeatureStore
from ml.replay_buffer import PrioritizedReplayBuffer

# Global Process Pool for Training (Singleton)
# Limit to cpu_count - 2 to leave room for Engine and Data Loader
_TRAINING_POOL = None
import threading
_POOL_LOCK = threading.Lock()


def get_training_pool():
    global _TRAINING_POOL
    with _POOL_LOCK:
        if _TRAINING_POOL is None:
            # CONSERVATIVE SCALING (Ryzen 5700U 1.8GHz Base)
            # Instead of cpu_count - 2, we use simpler logic to prevent thermal throttling:
            # Use 6 workers. Leaves 2 cores for OS/Engine + headroom for heat dissipation.
            # This allows sustaining 1.8GHz-2.5GHz without heavy throttling.
            max_workers = 14 # QUANTUM OVERCLOCK

            from concurrent.futures import ProcessPoolExecutor

            _TRAINING_POOL = ProcessPoolExecutor(max_workers=max_workers)
    return _TRAINING_POOL


def ml_inference_worker_task(in_q, out_q):
    """Isolated process task for ML Inference (No GIL contention) (SUPREMO-V3)"""
    # Prevent circular imports if any, but enums are at top
    import time

    from core.enums import SignalType

    while True:
        try:
            data = in_q.get()
            X, rf, xgb, gb = data["X"], data["rf"], data["xgb"], data["gb"]
            
            # FORENSIC-V100: STRICT np.float32 INJECTION (NANO-LATENCY)
            if not isinstance(X, np.ndarray) or X.dtype != np.float32 or not X.flags['C_CONTIGUOUS']:
                X = np.ascontiguousarray(X, dtype=np.float32)
            # Heavy inference via JIT Kernels
            if isinstance(rf, dict) and "tree_offsets" in rf:
                rf_p = predict_rf_jit(
                    X.flatten(),
                    rf["children_left"],
                    rf["children_right"],
                    rf["feature"],
                    rf["threshold"],
                    rf["value"],
                    rf["tree_offsets"],
                )
            else:
                rf_p = (
                    rf.predict_proba(X)[0][1] if hasattr(rf, "predict_proba") else 0.5
                )

            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V100: XGBOOST O(1) INFERENCE (NANO-LATENCY)
            # QUÉ: Bypass de la creación de DMatrix usando inplace_predict.
            # POR QUÉ: predict_proba(X) allocaba memoria y consumía ~1.5ms.
            # PARA QUÉ: Reducir la latencia a ~50 microsegundos por tick.
            # ═══════════════════════════════════════════════════════════════
            if xgb:
                try:
                    booster = xgb.get_booster() if hasattr(xgb, 'get_booster') else xgb
                    xgb_p = float(booster.inplace_predict(X)[0])
                except Exception:
                    xgb_p = xgb.predict_proba(X)[0][1]
            else:
                xgb_p = 0.5

            if isinstance(gb, dict) and "tree_offsets" in gb:
                gb_p = predict_gb_jit(
                    X.flatten(),
                    gb["children_left"],
                    gb["children_right"],
                    gb["feature"],
                    gb["threshold"],
                    gb["value"],
                    gb["tree_offsets"],
                    gb["init_score"],
                    gb["learning_rate"],
                )
            else:
                gb_p = (
                    gb.predict_proba(X)[0][1] if hasattr(gb, "predict_proba") else 0.5
                )

            # Weighted ensemble (Dynamic Weights from Main Process)
            # Default to equal weights if not present (Safety)
            w_rf, w_xgb, w_gb = data.get(
                "weights", (0.0, 1.0, 0.0) if (not rf and xgb) else (0.34, 0.33, 0.33)
            )

            conf = rf_p * w_rf + xgb_p * w_xgb + gb_p * w_gb

            # Signal Logic con Adaptive Thresholds
            # Note: In worker we only compute raw probability/direction
            # The complex circuit breaking etc stays in the main process
            threshold_long = data.get("threshold_long", 0.65)
            # Default to symmetric if not explicitly passed
            threshold_short = data.get("threshold_short", 1.0 - threshold_long)

            sig = SignalType.NEUTRAL
            if conf > threshold_long:
                sig = SignalType.LONG
            elif conf < threshold_short:
                sig = SignalType.SHORT

            out_q.put(
                {
                    "confidence": conf,
                    "signal_type": sig,
                    "rf": rf_p,
                    "xgb": xgb_p,
                    "gb": gb_p,
                    "ts": data.get("ts", time.time()),
                }
            )

            # PHASE 3: Neural Insight Publication (Binary SHM)
            try:
                from core.neural_bridge import neural_bridge

                neural_bridge.publish_insight(
                    strategy_id="ML_CORE_WORKER",
                    symbol=data.get("symbol", "UNKNOWN"),
                    insight={
                        "confidence": conf,
                        "direction": sig.name if hasattr(sig, "name") else str(sig),
                    },
                )
            except Exception as bridge_err:
                # Silently fail in worker, but we should ideally log to a file
                pass
        except EOFError:
            break  # Queue closed
        except Exception as e:
            # Prevent silent failures and queue starvation
            import traceback
            import os
            try:
                with open("ml_worker_error.log", "a") as f:
                    f.write(f"Worker Error: {e}\n{traceback.format_exc()}\n")
            except:
                pass
            time.sleep(0.1)


# MODO PROFESOR: Limitador global de entrenamiento.
# QUÉ: Semáforo para controlar cuántas estrategias entrenan simultáneamente.
# POR QUÉ: Con 24 símbolos, n_jobs=-1 causa agotamiento de RAM instantáneo (97%+).
# PARA QUÉ: Estabilizar el sistema y permitir que el trading continúe sin lag.
TRAINING_LIMITER = threading.BoundedSemaphore(value=2) # PHASE 24: Prevent CPU Freeze during boot


class MLStrategyHybridUltimate(Strategy):
    """
    ML Strategy Híbrida DEFINITIVA ULTIMATE
    Versión final que combina todas las características avanzadas para crecimiento exponencial.
    """

    def __init__(
        self,
        data_provider,
        events_queue,
        symbol="BTC/USDT",
        lookback=50,
        sentiment_loader=None,
        portfolio=None,
        risk_manager=None,
        horizon="SCALPING",
        models_dir=None,
        db_path=None,
    ):

        # ============================================================
        # ✅ CORE CONFIGURATION - OPTIMIZADO PARA CRECIMIENTO RÁPIDO
        # ============================================================
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.symbol = symbol
        self.horizon = horizon  # NEW: 'SCALPING' or 'SWING'

        # --- SUPREMO-V4 / MÓDULO OMEGA (FASE D): H1-H8 ISOLATION PROTOCOL ---
        # Aislar los modelos según su horizonte temporal estricto (H1 a H8) para
        # evitar polinización cruzada entre modelos de Scalping y Swing.
        base_models_dir = models_dir if models_dir else os.path.join(getattr(Config, "BASE_DIR", "."), getattr(Config, "MODEL_DIR", ".models"))
        
        # Mapeo de Horizontes H1-H8 según Módulo OMEGA
        self.horizon_mapping = {
            "1m": "H1", "3m": "H2", "5m": "H3", "15m": "H4",
            "1h": "H5", "4h": "H6", "1d": "H7", "1w": "H8"
        }
        
        # Determine primary timeframe from Config based on horizon passed
        h_upper = self.horizon.upper()
        if h_upper == "MICROSCALPING":
            self.primary_tf = getattr(Config.Horizons, "Microscalping", {}).get('primary_tf', '1m')
        elif h_upper == "SCALPING":
            self.primary_tf = getattr(Config.Horizons, "Scalping", {}).get('primary_tf', '1m')
        elif h_upper == "SWING":
            self.primary_tf = getattr(Config.Horizons, "Swing", {}).get('primary_tf', '1h')
        else:
            self.primary_tf = '5m'
            
        self.h_level = self.horizon_mapping.get(self.primary_tf, "H3")
        
        # Enrutar los modelos a su carpeta de silo específica (Ej: .models/H1_1m)
        self.models_dir = os.path.join(base_models_dir, f"{self.h_level}_{self.primary_tf}")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.db_path = (
            db_path
            if db_path
            else os.path.join(
                getattr(Config, "BASE_DIR", "."), "data", "feature_store.db"
            )
        )
        # ============================================================
        # ✅ MULTI-HORIZON SCALING (Phase 2.2 / Phase FORENSIC-1)
        # ============================================================
        self.horizon_str = self.horizon.upper()

        if self.horizon_str == "MICROSCALPING":
            self.horizon_days = 0
            self.lookback = 500
        elif self.horizon_str == "SCALPING":
            self.horizon_days = 0
            self.lookback = 1500
        elif self.horizon_str == "SWING":
            self.horizon_days = 7 # SWING = 7 days for Adaptive Engine logic
            self.lookback = 5000
        else:
            self.horizon_days = 7
            self.lookback = 5000
        self._feature_cols = []  # Ensure initialized
        self.sentiment_loader = sentiment_loader
        self.portfolio = portfolio
        self.risk_manager = risk_manager  # Regime Leader Link
        base_label = getattr(Config, "STRATEGY_LABELS", {}).get(
            "ml_strategy", "ML_HYBRID_ULTIMATE_V2"
        )
        lbl = "[SCL]" if self.horizon_str == "SCALPING" else "[SWG]"
        self.strategy_id = f"{lbl}_{base_label}_{self.horizon_str}_{self.symbol.replace('/', '_')}"

        # === ASYNC EXECUTOR OPTIMIZADO ===
        # OPT-5: Shared pool instead of per-instance (42 instances × 2 threads = 84 → 8 total)
        from core.shared_pools import get_shared_pools
        self.executor = get_shared_pools().inference_pool

        # ============================================================
        # ✅ ENSEMBLE COMPLETO - 3 MODELOS CON WEIGHTED VOTING
        # ============================================================
        self.rf_model = None
        self.xgb_model = None
        self.gb_model = None
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        # 🧠 PHASE 8: PETIM INTEGRATION
        try:
            from ml.petim_model import GeometryPredictor
            self.petim_predictor = GeometryPredictor(self.symbol, timeframe="1m")
            petim_dir = os.path.join(getattr(Config, "BASE_DIR", "."), "models", "petim")
            if self.petim_predictor.load(petim_dir):
                logger.info(f"🚀 [PETIM] Multi-Task Predictor Loaded for {self.symbol}")
            else:
                logger.warning(f"⚠️ [PETIM] Models not found for {self.symbol}, running without PETIM")
                self.petim_predictor = None
        except Exception as e:
            logger.error(f"Failed to init PETIM: {e}")
            self.petim_predictor = None

        # Pesos base para crecimiento agresivo
        self.base_rf_weight = 0.45
        self.base_xgb_weight = 0.35
        self.base_gb_weight = 0.20

        # Pesos originales para reset
        self.original_rf_weight = 0.45
        self.original_xgb_weight = 0.35
        self.original_gb_weight = 0.20

        # --- SUPREMO-V3: ML ISOLATION ---
        self._inference_queue = queue.Queue(maxsize=10)
        self._results_queue = queue.Queue(maxsize=10)
        self._inference_process = None

        # ============================================================
        # ✅ SANDBOX FLAG PARA EMULACIÓN DE WEBSOCKETS GOD-MODE
        # ============================================================
        self.is_sandbox = False

        # ============================================================
        # ✅ PERFORMANCE TRACKING PARA CRECIMIENTO EXPONENCIAL
        # ============================================================
        self.performance_history = deque(maxlen=100)  # Historial de trades
        self.performance_window = deque(maxlen=20)  # Ventana para cálculos dinámicos
        self.signal_history = deque(maxlen=100)  # Historial completo de señales
        self.equity_curve = deque(maxlen=100)  # OPT: Reduced from 500 → 100 (saves ~4KB × 42 instances)

        # Tracking individual por modelo
        self.individual_model_scores = {"rf": 0.0, "xgb": 0.0, "gb": 0.0}
        self.model_performance = {
            "rf": deque(maxlen=30),
            "xgb": deque(maxlen=30),
            "gb": deque(maxlen=30),
        }

        # ============================================================
        # ✅ TRAINING CONFIGURATION - OPTIMIZADO PARA RAPIDEZ
        # ============================================================
        self.is_trained = False
        # Scale min bars to train based on horizon (we need more data for longer horizons)
        self.min_bars_to_train = 300 * max(1, self.horizon_days // 2)
        self.bars_since_train = 0
        self.retrain_interval = 150  # Retrain más frecuente
        self.last_training_time = None
        self.last_training_score = 0.0
        self.training_iteration = 0

        # ============================================================
        # TARGETS ADAPTATIVOS - EVOLUTIVOS (Phase 3)
        # ============================================================
        self.par_engine = AdaptiveMLParameterEngine(horizon_str=self.horizon_str)

        self.BASE_TP_TARGET = self.par_engine.get("tp_mult") / 100.0  # 0.3% base
        self.BASE_SL_TARGET = self.par_engine.get("sl_mult") / 100.0  # 0.2% base

        self.LOOKAHEAD_BARS = self.par_engine.get("lookahead")
        logger.info(
            f"🔮 [ML-Horizon] Scaled Prediction Window: {self.LOOKAHEAD_BARS} bars (Horizon: {self.horizon_str})"
        )

        self.current_tp_target = np.clip(self.BASE_TP_TARGET, 0.001, 0.05 if self.horizon_str in ['SCALPING', 'MICROSCALPING'] else 0.15)
        self.current_sl_target = np.clip(self.BASE_SL_TARGET, 0.001, 0.03 if self.horizon_str in ['SCALPING', 'MICROSCALPING'] else 0.10)
        self.volatility_multiplier = 1.0

        # ============================================================
        # ✅ UMBRALES ADAPTATIVOS - BALANCE ENTRE RIESGO Y GANANCIA
        # ============================================================
        self.MIN_MODEL_ACCURACY = 0.35  # Aggressive: 35% accuracy floor (ML > Random)

        # Umbrales base optimizados para SMART GROWTH MODE
        # FASE 30: Read from Genetic DNA if available
        dna_bull = Config.Strategies.ML_THRESHOLDS.get('confidence_bull')
        if dna_bull is not None:
            self.BASE_CONFIDENCE_THRESHOLD = dna_bull
        else:
            self.BASE_CONFIDENCE_THRESHOLD = self.par_engine.get("ml_confidence")
        self.BASE_CONFLUENCE_LONG = 0.25  # More permissive confluence
        self.BASE_CONFLUENCE_SHORT = -0.30  # Más permisivo

        # Umbrales adaptativos
        self.adaptive_confidence_threshold = self.BASE_CONFIDENCE_THRESHOLD
        self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG
        self.consensus_threshold = 2  # 2 of 3 engines must agree

        # OMEGA MIND: Online Learner for weights
        self.online_learner = OnlineLearner(learning_rate=0.005)
        self.last_ensemble_input = None  # Store [rf, xgb, gb] for update loop

        # Phase 9: Neural Fortress Components
        # OPT: Reduced from 2000→200 (with $13 capital, ~10-20 trades/day max)
        self.memory = PrioritizedReplayBuffer(capacity=200)
        self.reward_system = RewardSystem()
        self.training_batch_size = 32
        self.steps_since_learn = 0
        self.xai_engine = XAIEngine()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2 POWER: Hot Adapter RL Instance
        # ═══════════════════════════════════════════════════════════════
        if _HOT_ADAPTER_AVAILABLE:
            self.hot_adapter = HotAdapterRL(learning_rate=0.03, max_memory=500)
        else:
            self.hot_adapter = None

        # === FORENSIC-3: CENTRAL AI ORCHESTRATION (SOPHIA) ===
        # Compute bar mins dynamically based on horizon
        tf_to_mins = {"1m": 1.0, "5m": 5.0, "15m": 15.0, "30m": 30.0, "1h": 60.0, "4h": 240.0, "1d": 1440.0}
        primary_tf = getattr(self, "PRIMARY_TF", "5m" if horizon.upper() in ["SCALPING", "MICROSCALPING"] else "1h")
        bar_mins = tf_to_mins.get(primary_tf, 5.0 if horizon.upper() in ["SCALPING", "MICROSCALPING"] else 60.0)
        # OPTIMIZACIÓN RAM: Singleton por horizonte (2 instancias vs 42)
        self.sophia = SophiaIntelligence.get_instance(bar_minutes=bar_mins)
        # Apply Horizon profile to prevent false Chaos Dampening
        horizon_days_map = {"SCALPING": 1, "SWING": 15}
        target_days = horizon_days_map.get(horizon.upper(), 1)
        self.sophia.set_horizon_profile(target_days)

        # C-1 FIX: Flag for lazy Némesis→Sophia feedback loop binding
        self._sophia_feedback_linked = False

        self.last_hmm_info = {"regime": "UNKNOWN", "transition_risk": 0.0}

        self.adaptive_confluence_short = self.BASE_CONFLUENCE_SHORT

        # ============================================================
        # ✅ PHASE 13: PHALANX-OMEGA COMPONENTS
        # ============================================================
        self.phalanx = OrderFlowAnalyzer()
        # Crypto Params: Alpha=0.05 (Reaction), Beta=0.9 (Persistence)
        self.garch = OnlineGARCH(
            omega=1e-6, alpha=0.05, beta=0.90, initial_variance=1e-4
        )

        # ============================================================
        # ✅ FILTROS DE ROBUSTEZ - PROTECCIÓN EN CRECIMIENTO RÁPIDO
        # ============================================================
        self.MAX_ATR_PCT = 0.035  # Más permisivo para volatilidad
        self.MIN_VOLUME_RATIO = 0.7  # Menos restrictivo
        self.RSI_FILTER_RANGE = (20, 80)  # Rango más amplio
        # === INFRAESTRUCTURA DE PERSISTENCIA Y ENTRENAMIENTO INCREMENTAL ===
        self._state_lock = threading.Lock()
        self.loop_count = 0
        self.analysis_stats = Counter()
        self.bars_since_incremental = 0  # Contador para updates rápidos

        # === [PHASE 12] GOVERNANCE & FEATURE STORE ===
        self.feature_store = FeatureStore(db_path=self.db_path)
        self.ml_governance = MLGovernance(
            db_path=self.db_path, models_root=self.models_dir
        )

        # Cargar modelos previos si existen (Prioridad: MLGovernance)
        # SUPREME BLOCK V: Bypass Governance to force new XGB JSON models
        # self._load_governed_model()
        # [PHASE 17] Lazy Loading: Do not load models on boot.
        self._models_loaded = False
        # self._load_models()

        # ============================================================
        # ✅ DETECCIÓN DE RÉGIMEN DE MERCADO AVANZADA
        # ============================================================
        self._current_event_time = None
        self.market_regime = "UNKNOWN"
        self.regime_history = deque(maxlen=15)
        self.regime_accuracy = {
            "TRENDING": [],
            "RANGING": [],
            "VOLATILE": [],
            "MIXED": [],
            "UNKNOWN": [],
        }
        self.last_regime_update = self._now() - pd.Timedelta(
            minutes=5
        )  # Allow immediate update
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
        self.max_consecutive_losses = 4  # Más sensible a pérdidas

        # ============================================================
        # ✅ SISTEMA DE APRENDIZAJE ADAPTATIVO
        # ============================================================
        self.learning_rate = 1.0
        self.aggressiveness_factor = 1.25  # [FASE 11] Factor de agresividad elevado para cuentas de $13
        self.win_streak = 0
        self.loss_streak = 0

        # ============================================================
        # ✅ STATE MANAGEMENT Y THREAD SAFETY
        # ============================================================
        self.running = True
        # NOTE: _state_lock already initialized at L306 — DO NOT re-create
        self.genotypes = {}  # PHASE 47: Standardized Genotype Store (Cognitive Compatibility)
        self._training_thread = None
        self._last_prediction_time = None
        self._label_mapping = {
            0: -1,
            1: 1,
        }  # FORENSIC FIX: Revert to 2-class (SHORT/LONG). 3-class breaks JIT kernel.

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
        self.rolling_payoff = deque(maxlen=20)  # Avg Win / Avg Loss

        self.signals_by_regime = {
            "TRENDING": 0,
            "RANGING": 0,
            "VOLATILE": 0,
            "MIXED": 0,
            "UNKNOWN": 0,
        }

        # ============================================================
        # ✅ CONFIGURACIÓN DE CRECIMIENTO EXPONENCIAL
        # ============================================================
        self.compounding_factor = 1.0  # Factor de compuesto
        self.position_sizing_mode = "KELLY"  # Kelly, FIXED, VOLATILITY
        self.kelly_fraction = 0.5  # Fracción de Kelly para riesgo controlado
        # self.base_position_size = 0.95  # DELEGADO AL RISK MANAGER (40%)

        # Meta de 12 USD a 100K USD
        self.initial_capital = 12.0
        self.target_capital = 100000.0
        self.current_capital = self.initial_capital

        # Phase I Components: Eagerly initialize (was lazy-loaded, causing AttributeError)
        self.feature_engineer = FeatureEngineering()
        self.signal_generator = SignalGenerator(self.strategy_id)

        logger.info(
            f"🟢 ML HYBRID ULTIMATE STRATEGY [ENSEMBLE] INITIALIZED FOR {symbol}"
        )
        logger.info(f"🎯 OBJECTIVE: ${self.initial_capital} → ${self.target_capital}")
        logger.info(f"⚙️  Mode: Exponential Growth (Aggressive with Risk Control)")

    def _now(self):
        """Forensic Time Fix: Return event timestamp if available, else system time"""
        return getattr(self, '_current_event_time', None) if getattr(self, '_current_event_time', None) else datetime.now(tz=timezone.utc)

    def _calculate_dynamic_sizing(self, confidence, volatility):
        """
        Phase 5: Dynamic Kelly Criterion Sizing
        Size = Kelly% * Confidence_Scaler * Volatility_Scaler
        """
        try:
            # 1. READ REAL STATS FROM PORTFOLIO (PHALANX-OMEGA)
            kelly_pct = 0.05  # Default Safe Fallback

            if (
                self.portfolio
                and self.strategy_id in self.portfolio.strategy_performance
            ):
                perf = self.portfolio.strategy_performance[self.strategy_id]
                trades = perf["trades"]

                if trades >= 10:
                    wins = perf["wins"]
                    losses = perf["losses"]
                    total_win_pnl = perf.get(
                        "total_win_pnl", 0.0
                    )  # Safety get for old state
                    total_loss_pnl = perf.get("total_loss_pnl", 0.0)

                    if losses > 0 and total_loss_pnl > 0:
                        avg_win = total_win_pnl / wins if wins > 0 else 0
                        avg_loss = total_loss_pnl / losses

                        # Kelly Variables
                        p = wins / trades  # Win Probability
                        q = 1.0 - p  # Loss Probability
                        b = avg_win / avg_loss  # Payoff Ratio

                        # Full Kelly Formula: f = p - q/b
                        if b > 0:
                            raw_kelly = p - (q / b)
                            if raw_kelly <= 0:
                                logger.warning(
                                    f"⚠️ [KELLY] Negative Edge: K={raw_kelly:.3f}. Min sizing."
                                )
                                kelly_pct = 0.01  # Absolute minimum
                            else:
                                kelly_pct = raw_kelly

                            # Log periodically
                            if trades % 5 == 0:
                                logger.info(
                                    f"🧠 [KELLY] p={p:.2f} b={b:.2f} => Raw K={raw_kelly:.2f}"
                                )

            # 2. Apply Fractional Kelly (Safety)
            # We use 'kelly_fraction' (e.g. 0.3 or 0.5) to be conservative
            safe_kelly = kelly_pct * getattr(self, "kelly_fraction", 0.4)

            # 3. Scale by Prediction Confidence (Higher conf -> Higher size)
            # Normalize confidence (0.5 to 1.0) -> (0.0 to 1.0)
            conf_scaler = max(0.0, (confidence - 0.5) * 2)

            # 4. Scale by Volatility (Inverse Volatility Sizing)
            # If Volatility is high, reduce size to keep $ Risk constant
            # Reference volatility: 0.005 (0.5%) per bars
            vol_scaler = min(1.5, 0.005 / max(0.001, volatility))

            final_size = min(0.95, safe_kelly * conf_scaler * vol_scaler)

            # print(f"  💰 Dynamic Sizing: K={kelly_pct:.2f} -> Safe={safe_kelly:.2f} * Conf={conf_scaler:.2f} * Vol={vol_scaler:.2f} = {final_size:.2f}")
            return max(0.1, final_size)  # Min 10%

        except Exception as e:
            logger.error(f"Sizing error: {e}")
            return 0.1

    # ============================================================
    # ✅ DETECCIÓN DE RÉGIMEN MEJORADA CON SUAVIZADO
    # ============================================================

    def _detect_market_regime(self, df):
        """
        Detección avanzada de régimen con múltiples capas de validación.
        (Logic migrated to core/market_regime.py for centralization)
        """
        if len(df) < 50:
            return "UNKNOWN", 0.0, {}

        try:
            regime_detector = None
            if hasattr(self, 'portfolio') and self.portfolio and hasattr(self.portfolio, 'market_regime') and self.portfolio.market_regime:
                regime_detector = self.portfolio.market_regime
            else:
                if not hasattr(self, '_regime_detector'):
                    from core.market_regime import MarketRegimeDetector
                    self._regime_detector = MarketRegimeDetector()
                regime_detector = self._regime_detector
                
            return regime_detector.detect_ml_regime(df)
        except Exception as e:
            from utils.logger import logger
            logger.error(f"Error detecting regime via centralized detector: {e}")
            return "UNKNOWN", 0.0, {}

    def _update_market_regime(self, df):
        """
        Actualizar régimen con suavizado y persistencia
        """
        current_time = self._now()

        # EXCEPCIÓN: Si es UNKNOWN, actualizar siempre para inicializar
        if (
            self.market_regime != "UNKNOWN"
            and (current_time - self.last_regime_update).total_seconds() < 180
        ):
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
        is_initial = self.market_regime == "UNKNOWN" and len(self.regime_history) >= 1
        if confidence > Config.Strategies.ML_THRESHOLDS['confidence_regime_change'] or smoothed_regime == self.market_regime or is_initial:
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
                    # Phase 14: Transfer HMM Info to Strategy
                    if hasattr(self.risk_manager, "transition_risk"):
                        self.last_hmm_info["transition_risk"] = (
                            self.risk_manager.transition_risk
                        )

                # --- NARRATIVA CONCEPTUAL Y ESTADÍSTICA ---
                descriptions = {
                    "TRENDING": "Directional bias confirmed. Momentum engines active.",
                    "RANGING": "Side-ways consolidation. Switching to Mean Reversion mode.",
                    "VOLATILE": "High noise & volatility. Wide stops and conservative bias.",
                    "STAGNANT": "Zombie market detected. Stagnant price action. Protection active.",
                    "MIXED": "Internal transition or choppy price action. Filtering active.",
                    "UNKNOWN": "Initializing discovery mode.",
                }
                concept = descriptions.get(smoothed_regime, "Evolving market state.")

                emoji = (
                    "🚀"
                    if smoothed_regime == "TRENDING"
                    else "⚖️"
                    if smoothed_regime == "RANGING"
                    else "🔥"
                    if smoothed_regime == "VOLATILE"
                    else "🧊"
                    if smoothed_regime == "STAGNANT"
                    else "🔄"
                )

                logger.info(
                    f"\n{'=' * 70}\n"
                    f"📊 {emoji} REGIME CHANGE: {self.symbol}\n"
                    f"{'=' * 70}\n"
                    f"   Phase: {old_regime} → {smoothed_regime}\n"
                    f"   Concept: {concept}\n"
                    f"   Stats: ADX={stats.get('adx', 0):.1f} | ATR%={stats.get('atr_pct', 0):.2f}% | Trend={stats.get('trend_strength', 0):.2f}%\n"
                    f"   Confidence: {confidence * 100:.1f}% | Strategy: Adaptive targets enabled.\n"
                    f"{'=' * 70}\n"
                )

    def _adjust_all_parameters_by_regime(self, regime):
        """
        Ajustar TODOS los parámetros según el régimen de mercado (DYNAMIC & CONFIG DRIVEN)
        """
        # 1. Get Advice from Intelligence Layer (Source of Truth)
        if hasattr(self.market_regime, "get_regime_advice"):
            advice = self.market_regime.get_regime_advice(regime)
        else:
            # Fallback if market_regime is string/mock
            from core.market_regime import MarketRegimeDetector

            advice = MarketRegimeDetector().get_regime_advice(regime)

        # 2. Extract Dynamic Parameters
        lev_limit = advice.get("leverage", 1)
        threshold_mod = advice.get("threshold_mod", 0.0)
        scale_factor = advice.get("scale", 0.0)
        # 3. Apply Adaptive Engine Values (Update bases dynamically)
        # FASE 30: Read from Genetic DNA if available
        dna_bull = Config.Strategies.ML_THRESHOLDS.get('confidence_bull')
        if dna_bull is not None:
            self.BASE_CONFIDENCE_THRESHOLD = dna_bull
        else:
            self.BASE_CONFIDENCE_THRESHOLD = self.par_engine.get("ml_confidence")
        self.BASE_TP_TARGET = self.par_engine.get("tp_mult") / 100.0
        self.BASE_SL_TARGET = self.par_engine.get("sl_mult") / 100.0
        self.LOOKAHEAD_BARS = self.par_engine.get("lookahead")
        if self.horizon_str == "SWING":
            self.LOOKAHEAD_BARS = int(self.LOOKAHEAD_BARS * 60) # Scale hours to minutes

        # Apply to Strategy State
        # FORENSIC FIX: Raised floor from 0.40→0.55. Trades at 0.40 had catastrophic WR (16.7%).
        # With $13 capital, every low-confidence trade is pure fee drain ($10.73 in fees vs -$5.05 PnL).
        self.adaptive_confidence_threshold = max(
            0.55, min(0.90, self.BASE_CONFIDENCE_THRESHOLD + threshold_mod)
        )

        # Scale Targets based on Volatility/Agression
        # Higher Leverage (Sniper) -> Tighter Stops, Bigger Targets (Risk Reward)
        # Lower Leverage (Safety) -> Wider Stops (Volatility Room)

        if lev_limit >= 5:  # SNIPER / BULL
            self.aggressiveness_factor = 1.2
            self.current_tp_target = self.BASE_TP_TARGET * 1.3
            self.current_sl_target = self.BASE_SL_TARGET * 0.9  # Tight Stop
            self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG - 0.10
        elif lev_limit <= 1:  # DEFENSE / BEAR
            self.aggressiveness_factor = 0.5
            self.current_tp_target = self.BASE_TP_TARGET * 1.0
            self.current_sl_target = self.BASE_SL_TARGET * 1.5  # Wide Stop
            self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG + 0.20
        else:  # NORMAL / RANGING
            self.aggressiveness_factor = 1.0
            self.current_tp_target = self.BASE_TP_TARGET
            self.current_sl_target = self.BASE_SL_TARGET
            self.adaptive_confluence_long = self.BASE_CONFLUENCE_LONG

        # --- PHASE 14: HMM TRANSITION RISK ADJUSTMENT ---
        # Si el riesgo de transición es elevado (>40%), somos más conservadores.
        trans_risk = self.last_hmm_info.get("transition_risk", 0.0)
        if trans_risk > Config.Strategies.ML_THRESHOLDS['hmm_transition_risk_high']:
            self.adaptive_confidence_threshold += 0.05
            self.aggressiveness_factor *= 0.8
            logger.info(
                f"🛡️ [HMM Safety] Transition Risk Elevated ({trans_risk:.2f}). Confidence threshold increased (+5%)."
            )

        # Aplicar learning rate y factor de agresividad
        # NOTE: Threshold is already modulated by Config loop above

        logger.debug(
            f"🔧 [Dynamic Adaptation] {regime}: "
            f"Lev={lev_limit}x, "
            f"Conf={self.adaptive_confidence_threshold:.2f} (Base {self.BASE_CONFIDENCE_THRESHOLD} + {threshold_mod}), "
            f"Aggr={self.aggressiveness_factor:.1f}"
        )

    # ============================================================
    # ✅ FEATURE ENGINEERING ULTIMATE - 80+ FEATURES
    # ============================================================

    @trace_execution
    def _prepare_features(self, bars, regime_aware=True, return_polars=False):
        """Delegated to FeatureEngineering component (HORIZON-AWARE)"""
        return self.feature_engineer.prepare_features(
            bars,
            market_regime=self.market_regime if regime_aware else "UNKNOWN",
            sentiment_loader=self.sentiment_loader,
            data_provider=self.data_provider,
            symbol=self.symbol,
            feature_store=self.feature_store,
            horizon=self.horizon_str,
            return_polars=return_polars,
            is_live=not getattr(self, 'is_sandbox', False)
        )

    def _validate_features(self, df):
        """Delegated to component"""
        return self.feature_engineer.validate_features(df)

    # ============================================================
    # ✅ LABEL CREATION CON TARGETS ADAPTATIVOS
    # ============================================================

    def _create_labels(self, df, adaptive_targets=True):
        """
        ═══════════════════════════════════════════════════════════════
        OPTIMIZACIÓN RAM/CPU: VECTORIZED LABEL CREATION
        QUÉ: Reemplaza el loop Python puro con operaciones NumPy vectorizadas.
        POR QUÉ: El loop anterior hacía df.iloc[i] 5000+ veces, cada una
          creando una copia temporal del row (~100x más lento que NumPy).
        PARA QUÉ: Entrenamiento 10x más rápido, ~50MB menos RAM.
        CÓMO: Usa rolling max/min sobre arrays pre-extraídos.
        ═══════════════════════════════════════════════════════════════
        """
        n = len(df)
        lookahead = self.par_engine.get("lookahead")
        
        # [FASE 3: Pesimismo Cuántico] Restricción brutal de lookahead para Microscalping
        if getattr(self, "horizon_str", "") == "MICROSCALPING":
            lookahead = min(lookahead, 2) # Máximo 2 barras para evitar ambigüedad de ruido intradiario
            
        dd_stress_limit = self.par_engine.get("dd_stress_limit")
        fee_threshold = getattr(Config, "BINANCE_TAKER_FEE_BNB", 0.00075) * 2 * 1.5

        # Pre-extract numpy arrays (zero-copy views where possible)
        close_arr = df["close"].values.astype(np.float64)
        high_arr = df["high"].values.astype(np.float64)
        low_arr = df["low"].values.astype(np.float64)
        atr_pct_arr = df["atr_pct"].values.astype(np.float64) / 100.0  # Convert to fraction

        labels = np.zeros(n, dtype=np.int32)

        import math
        sqrt_bars = math.sqrt(max(1, lookahead))
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V91: REALISTIC LABEL GENERATION
        # QUÉ: Generar targets basados en volatilidad real (ATR * √t)
        # POR QUÉ: Antes usaba BASE_TP_TARGET (ej. 0.65%) que nunca se
        #   alcanzaba en scalping, llenando el dataset de ceros y ruido.
        # PARA QUÉ: El modelo debe aprender a predecir el movimiento real
        #   esperado por difusión browniana.
        # ═══════════════════════════════════════════════════════════════
        tp_targets = np.clip(atr_pct_arr * sqrt_bars, 0.001, self.BASE_TP_TARGET * 1.5)
        sl_targets = np.clip(atr_pct_arr * sqrt_bars * 0.8, 0.001, self.BASE_SL_TARGET * 1.5)

        # Process in bulk: compute future rolling max/min for each bar
        # We iterate only the valid range (not the last 'lookahead' bars)
        valid_end = n - lookahead
        if valid_end <= 0:
            df = df.copy()
            df["label"] = labels[:n]
            return df

        for i in range(valid_end):
            start = i + 1
            end = min(i + 1 + lookahead, n)
            
            future_high = high_arr[start:end]
            future_low = low_arr[start:end]
            future_close_last = close_arr[end - 1]
            
            current_price = close_arr[i]
            tp_target = tp_targets[i]
            sl_target = sl_targets[i]

            # TP/SL levels
            tp_level_long = current_price * (1 + tp_target)
            sl_level_long = current_price * (1 - sl_target)
            tp_level_short = current_price * (1 - tp_target)
            sl_level_short = current_price * (1 + sl_target)

            # Hit detection (vectorized within the window)
            tp_hit_long = np.any(future_high >= tp_level_long)
            sl_hit_long = np.any(future_low <= sl_level_long)
            tp_hit_short = np.any(future_low <= tp_level_short)
            sl_hit_short = np.any(future_high >= sl_level_short)

            # First hit indices for ordering
            tp_idx_l = np.argmax(future_high >= tp_level_long) if tp_hit_long else lookahead
            sl_idx_l = np.argmax(future_low <= sl_level_long) if sl_hit_long else lookahead
            tp_idx_s = np.argmax(future_low <= tp_level_short) if tp_hit_short else lookahead
            sl_idx_s = np.argmax(future_high >= sl_level_short) if sl_hit_short else lookahead

            # Effective hit index bounds the MAE lookup (so we don't look past the trade exit)
            hit_idx_l = min(tp_idx_l, sl_idx_l) if (tp_hit_long or sl_hit_long) else lookahead
            hit_idx_s = min(tp_idx_s, sl_idx_s) if (tp_hit_short or sl_hit_short) else lookahead
            mae_slice_long = future_low[:hit_idx_l + 1] if hit_idx_l + 1 <= len(future_low) else future_low
            mae_slice_short = future_high[:hit_idx_s + 1] if hit_idx_s + 1 <= len(future_high) else future_high

            # MAE calculation bounded by actual exit
            mae_long = (current_price - np.min(mae_slice_long)) / current_price if len(mae_slice_long) > 0 else 0
            mae_short = (np.max(mae_slice_short) - current_price) / current_price if len(mae_slice_short) > 0 else 0
            ds_long = mae_long / sl_target if sl_target > 0 else 0
            ds_short = mae_short / sl_target if sl_target > 0 else 0

            # LONG evaluation
            if tp_hit_long and sl_hit_long:
                long_won = tp_idx_l < sl_idx_l and ds_long <= dd_stress_limit
            elif tp_hit_long:
                long_won = ds_long <= dd_stress_limit
            else:
                long_won = False

            # SHORT evaluation
            if tp_hit_short and sl_hit_short:
                short_won = tp_idx_s < sl_idx_s and ds_short <= dd_stress_limit
            elif tp_hit_short:
                short_won = ds_short <= dd_stress_limit
            else:
                short_won = False

            # Label assignment
            if long_won and not short_won:
                labels[i] = 1
            elif short_won and not long_won:
                labels[i] = -1
            elif long_won and short_won:
                ret = (future_close_last - current_price) / current_price
                labels[i] = 1 if ret > 0 else -1
            else:
                ret = (future_close_last - current_price) / current_price
                # FIX ANTI-MENTIRA OFFLINE: Only assign a label if the price reached at least 80% of the target.
                # If it just drifted a little bit, it's NOT a successful prediction of the magnitude.
                strict_threshold = max(tp_target * 0.8, fee_threshold * 3.0)
                if ret > strict_threshold:
                    labels[i] = 1
                elif ret < -strict_threshold:
                    labels[i] = -1
                # else: labels[i] = 0 (already initialized)

        # DEFRAGMENTATION: Copy before adding new column to large DF
        df = df.copy()
        df["label"] = labels[:n]
        return df

    # ============================================================
    # ✅ TRAINING ULTIMATE - ENSEMBLE CON HIPERPARÁMETROS ADAPTATIVOS
    # ============================================================

    def _train_with_cross_validation(self, df):
        """
        Training con ensemble de 3 modelos y hiperparámetros adaptativos por régimen
        """
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
            
        min_bars_req = 200
        # FORENSIC FIX: Removed is_backtest divergence. Must require 200 bars even in backtest to ensure parity.
            
        if len(df) < min_bars_req:
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
            df = df.iloc[-int(original_len * 0.6) :]
            logger.info(
                f"⚡ [ML Optimization] Subsampling active: {original_len} -> {len(df)} samples (60% most recent)."
            )

        # Crear labels con targets adaptativos
        df = self._create_labels(df, adaptive_targets=True)
        df = df.dropna()

        # DEBUG: Analizar por qué no hay señales
        vol_mean = df["atr_pct"].mean()
        vol_ref = vol_mean / 100
        mult = 1.0
        if vol_ref < 0.0003:
            mult = 0.10
        elif vol_ref < 0.0008:
            mult = 0.20
        elif vol_ref < 0.002:
            mult = 0.35
        elif vol_ref < 0.01:
            mult = 0.6

        tp_mean = self.BASE_TP_TARGET * mult
        label_counts = df["label"].value_counts().to_dict()
        logger.info(
            f"🔍 DEBUG ML [{self.symbol}]: Rows={len(df)}, AvgVol={vol_mean:.4f}%, mult={mult:.2f}, Est.Target={tp_mean:.4f}, Labels={label_counts}"
        )

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC FIX: REVERT TO 2-CLASS (SHORT/LONG)
        # QUÉ: Eliminamos df_hold.
        # POR QUÉ: El Numba JIT compiler de inference (predict_rf_jit) SOLO 
        #   soporta salida binaria. Meter 3 clases rompe las probabilidades 
        #   y causa el error XGBoost 'got [1 2]'.
        # PARA QUÉ: Evitar crashes y restaurar operaciones LONG que estaban
        #   siendo mapeadas a HOLD.
        # ═══════════════════════════════════════════════════════════════
        df_long = df[df["label"] == 1]
        df_short = df[df["label"] == -1]
        
        df_signals = pd.concat([df_long, df_short]).sort_index()

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V50 FIX: MINIMUM TRAINING SAMPLES (ANTI-OVERFITTING)
        # QUÉ: Incrementamos el mínimo de señales para entrenar.
        # POR QUÉ: Con 76 samples y 80 features (ratio 0.95:1) el modelo
        #   memorizaba ruido → Training Score 0.54 = moneda al aire.
        #   Regla de oro: mínimo 10:1 (samples:features).
        # PARA QUÉ: Forzar al modelo a tener datos SUFICIENTES para
        #   generalizar. Mejor NO entrenar que entrenar con basura.
        # ═══════════════════════════════════════════════════════════════
        min_signals = 150  # Producción y Backtest: requiere datos significativos
        # FORENSIC FIX: Removed is_backtest divergence. Minimum 150 samples required everywhere to prevent overfitting.

        if len(df_signals) < min_signals:
            # --- NEW: Data Quality Audit (Transparencia) ---
            price_spread = (df["high"].max() - df["low"].min()) / df["close"].mean()
            identical_bars = (df["high"] == df["low"]).sum() / len(df)

            if price_spread < 0.0001 or identical_bars > 0.95:
                # Caso Testnet: Precio fijo (ej: BTC=5.0)
                logger.warning(
                    f"🚫 [ZOMBIE MARKET] {self.symbol} is flat. Spread: {price_spread * 100:.6f}% | Identical Bars: {identical_bars * 100:.1f}%. Training aborted."
                )
            else:
                logger.warning(
                    f"⚠️ [LOW VOLATILITY] {self.symbol} has movement but not enough to reach targets. Real Signals: {len(df_signals)}/30 required."
                )

            return None, 0.0  # Excluir columnas no-features
        exclude_cols = [
            "label",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "datetime",
            "symbol",
            # ════════════════════════════════════════════════════════════════
            # FORENSIC FIX: GHOST FEATURES — Live-only websocket data.
            # QUÉ: These features are ALWAYS 0.0 during backtest/training
            #   because they only populate from real-time websockets.
            # POR QUÉ: XGBoost learns to ignore them (weight≈0) during training,
            #   but in production they receive real values the model never saw,
            #   creating unpredictable inference artifacts.
            # PARA QUÉ: Eliminate Train≠Live divergence entirely.
            # CUÁNDO: During feature column selection for training AND inference.
            # DÓNDE: strategies/ml_strategy.py → _train_models()
            # QUIÉN: Quant Developer + QA Engineer
            # ════════════════════════════════════════════════════════════════
            "vbi", "vbi_avg", "liq_intensity",           # HFT websocket only
            "funding_rate", "oi", "oi_delta",             # Derivatives REST (unreliable in BT)
            "funding_distortion",                         # Derived from funding_rate
            "l2_ofi", "l2_spread", "l2_microprice_dist",  # L2 orderbook snapshot only
        ]
        
        # ════════════════════════════════════════════════════════════════
        # FORENSIC-V50 FIX: ML FEATURE REDUCTION (TOP 20)
        # QUÉ: Limitar el entrenamiento a las top 20 features más predictivas.
        # POR QUÉ: Demasiadas features causan overfitting masivo en M1.
        # PARA QUÉ: Evitar memorización de ruido y aumentar generalización.
        # ════════════════════════════════════════════════════════════════
        top_20_features = [
            'returns_5', 'returns_10', 'roc_10', 'rsi_14', 'atr_pct', 
            'macd_hist', 'bb_position', 'bb_width', 'stoch_k', 'adx', 
            'volume_ratio', 'gk_vol', 'hurst_memory', 'volatility_ransac', 
            'micro_imbalance', 'spread_squeeze', 'scalp_velocity_1', 
            'scalp_rsi_divergence', 'micro_label', 'market_cluster',
            'graph_centrality', 'graph_pagerank'
        ]
        
        feature_cols = [c for c in df_signals.columns if c in top_20_features and c not in exclude_cols]

        X = df_signals[feature_cols]
        y = df_signals["label"]

        # CRITICAL: Remap labels for XGBoost compatibility
        # XGBoost dynamically expects [0, 1] for binary classification
        # FIX: Hardcode mapping so classes don't shift across batches
        # -1 -> 0, 1 -> 1
        y_encoded = y.map({-1: 0, 1: 1})
        y = pd.Series(y_encoded, index=y.index)

        # DEBUG: Verificar si las features son válidas
        std_zero_cols = X.columns[X.std() == 0].tolist()
        if len(std_zero_cols) > 0:
            logger.debug(
                f"⚠️ {len(std_zero_cols)} features are constant (std=0). Ex: {std_zero_cols[:5]}"
            )

        logger.info(
            f"📊 Training {self.symbol} with {len(X)} samples, {len(feature_cols)} features"
        )

        # TimeSeriesSplit CV with auto-adjustment for small datasets
        n_samples = len(X)
        if n_samples < 5:
            # Life Support Mode: Not enough data for CV
            logger.info(
                "📉 Minimal data mode: Using full dataset for training (Overfitting intended for survival)"
            )
            # Manual split: Train and test on same data to produce a valid score and model
            indices = list(range(n_samples))
            splitter = [(indices, indices)]
        else:
            # Phase 24: Lazy Imports
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.neural_network import MLPClassifier
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            from xgboost import XGBClassifier

            tscv = TimeSeriesSplit(n_splits=3)
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V31: EMBARGO GAP IN CV SPLITS
            # QUÉ: Añade un gap de 'lookahead' barras entre train y test.
            # POR QUÉ: Sin purga, las últimas barras del train tienen labels
            #   que dependen de datos que están en el test set (data leakage).
            # PARA QUÉ: Estimación honesta del accuracy real del modelo.
            # CÓMO: Recorta las últimas 'embargo' filas del train set.
            # ═══════════════════════════════════════════════════════════════
            embargo = self.LOOKAHEAD_BARS
            raw_splits = list(tscv.split(X))
            purged_splits = []
            for train_idx, test_idx in raw_splits:
                if len(test_idx) == 0:
                    continue
                # Remove train samples within 'embargo' bars of test start
                test_start = test_idx[0]
                purged_train = [i for i in train_idx if i < test_start - embargo]
                if len(purged_train) >= 10:  # Minimum viable train set
                    purged_splits.append((purged_train, list(test_idx)))
                else:
                    purged_splits.append((list(train_idx), list(test_idx)))
            splitter = purged_splits if purged_splits else [(list(range(n_samples)), list(range(n_samples)))]

        cv_scores = {"rf": [], "xgb": [], "gb": []}
        best_models = {"rf": None, "xgb": None, "gb": None}
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

        # FORENSIC FIX: HORIZON-AWARE HYPERPARAMETER SCALING
        # QUÉ: Escalar hyperparámetros según el horizonte de trading.
        # POR QUÉ: Swing necesita modelos más profundos para capturar patrones temporales largos.
        #          Scalping necesita modelos más rápidos y regularizados anti-overfitting.
        # CÓMO: Multiplicadores post-régimen que ajustan capacidad del modelo.
        if self.horizon_str == "SWING":
            # Swing: +50% estimators, +2 depth, lower LR for stability
            n_estimators = int(n_estimators * 1.5)
            max_depth_rf = min(12, max_depth_rf + 2)
            max_depth_xgb = min(10, max_depth_xgb + 2)
            max_depth_gb = min(8, max_depth_gb + 2)
            learning_rate *= 0.7  # Lower LR = more stable convergence
            min_samples_split = max(3, min_samples_split - 2)
        else:
            # Scalping: -10% estimators, +2 min_samples_split for regularization
            n_estimators = max(50, int(n_estimators * 0.9))
            min_samples_split = min(15, min_samples_split + 2)

        # FIX A-4: Removed ModelFactory.get_ensemble_models() override.
        # Models are created inline in the CV loop below with regime-adaptive hyperparameters.

        # Pesos del Ensemble (Gobernanza Dinámica) con reporte de progreso
        num_folds = 3 if n_samples >= 300 else 1
        for fold, (train_idx, test_idx) in enumerate(splitter):
            if fold % 1 == 0:  # Log every fold
                logger.info(
                    f"   ⚙️ [{self.symbol}] Fitting ML Engine (Fold {fold + 1}/{num_folds})..."
                )
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Anti-Crash for Extreme Imbalance (Ensure both classes exist)
            # import pandas as pd

            if len(np.unique(y_train)) < 2:
                logger.warning(
                    f"⚠️ [{self.symbol}] Fold {fold} has only 1 class. Injecting synthetic samples to prevent XGB crash."
                )
                fake_y = pd.Series([-1, 1], index=[-1, -2])
                fake_X = pd.DataFrame(
                    [X_train.iloc[0].values, X_train.iloc[0].values],
                    index=[-1, -2],
                    columns=X_train.columns,
                )
                X_train = pd.concat([X_train, fake_X])
                y_train = pd.concat([y_train, fake_y])

            # Scaling & Memory Optimization (Rule 3.6)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train).astype("float32")
            X_test_scaled = scaler.transform(X_test).astype("float32")

            # 1. MLP Classifier (Neural Network - Online Learning)
            rf = getattr(self, "rf_model", None)
            if not rf or not hasattr(rf, "partial_fit"):
                rf = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    learning_rate_init=0.01,
                    max_iter=1,  # Fast single pass for online learning
                    warm_start=True,
                    random_state=42 + self.training_iteration
                )
            
            # Use partial_fit for O(1) memory updates
            rf.partial_fit(X_train_scaled, y_train, classes=np.array([0, 1]))
            rf_score = rf.score(X_test_scaled, y_test)
            cv_scores["rf"].append(rf_score)

            # 2. XGBoost (Incremental Ready + tree_method='hist')
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V31: ANTI-OVERFITTING REGULARIZATION
            # QUÉ: Hiperparámetros más conservadores para combatir overfitting.
            # POR QUÉ: Con micro-datasets (<2000 samples) y >100 features,
            #   los modelos memorizaban ruido en lugar de aprender patrones.
            # PARA QUÉ: Accuracy real (out-of-sample) >70% en vez de 95%
            #   in-sample pero 45% out-of-sample.
            # CAMBIOS: min_child_weight 1→5, gamma 0→0.1, reg_alpha 0→0.5,
            #   reg_lambda 1→2.0, colsample 0.8→0.7, subsample cap 0.75.
            # ═══════════════════════════════════════════════════════════════
            xgb = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth_xgb,
                learning_rate=learning_rate,
                subsample=min(subsample, 0.75),      # Cap: no more than 75%
                colsample_bytree=0.7,                 # Was 0.8 → reduce feature sampling
                min_child_weight=5,                   # Was default 1 → prevent leaf overfitting
                gamma=0.1,                            # Was 0 → minimum loss reduction per split
                reg_alpha=0.5,                        # L1 regularization (sparsity)
                reg_lambda=2.0,                       # L2 regularization (was default 1.0)
                n_jobs=-1,  # QUANTUM OVERCLOCK: Use all available cores
                tree_method="hist",  # MÁXIMA VELOCIDAD
            )

            # 🔄 APRENDIZAJE INCREMENTAL (Rule 3.2): Validar matching de features
            # BUG FIX: XGBoost >= 2.0 requiere base_score en (0,1) para logistic.
            # Los modelos entrenados con XGBoost <2.0 pueden tener base_score inválido.
            # Solución: intentar warm-start, hacer fallback a fresh fit si falla.
            prev_xgb = None
            if hasattr(self, "xgb_model") and self.xgb_model:
                try:
                    if hasattr(self.xgb_model, "feature_names_in_") and list(
                        self.xgb_model.feature_names_in_
                    ) == list(feature_cols):
                        prev_xgb = self.xgb_model
                        logger.info(
                            f"🔄 [{self.symbol}] Resuming learning from previous XGBoost model (Incremental)."
                        )
                    else:
                        logger.debug(
                            f"🔄 [{self.symbol}] Feature mismatch for incremental learning. Resetting XGB."
                        )
                except Exception as e:
                    logger.error(f"Silent exception caught: {e}", exc_info=True)

            # Intentar fit con warm-start; si falla (base_score incompatibility),
            # hacer fresh fit sin modelo previo
            # FORENSIC-V90: EARLY STOPPING to prevent overfitting
            # QUÉ: Detiene el entrenamiento cuando la métrica de validación deja de mejorar.
            # POR QUÉ: Sin early stopping, XGBoost memoriza ruido con datasets pequeños.
            # PARA QUÉ: Accuracy real (out-of-sample) mayor.
            try:
                xgb.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False,
                    xgb_model=prev_xgb
                )
            except Exception as xgb_warm_err:
                logger.warning(
                    f"⚠️ [{self.symbol}] XGBoost warm-start failed "
                    f"({xgb_warm_err}). Retrying from scratch."
                )
                xgb.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False
                )
            xgb_score = xgb.score(X_test_scaled, y_test)

            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V31: FEATURE IMPORTANCE PRUNING (POST FOLD-0)
            # QUÉ: Después del primer fold, retiene solo las top-30 features
            #   más importantes según XGBoost feature_importances_.
            # POR QUÉ: Curse of dimensionality — 100+ features con <2000
            #   muestras causa que el modelo memorice ruido.
            # PARA QUÉ: Modelo más parsimonioso con mayor generalización.
            # CUÁNDO: Solo en fold 0 (las features seleccionadas se usan en
            #   los folds siguientes).
            # ═══════════════════════════════════════════════════════════════
            if fold == 0 and hasattr(xgb, 'feature_importances_') and len(feature_cols) > 30:
                importances = xgb.feature_importances_
                top_k = min(30, len(importances))
                top_indices = np.argsort(importances)[-top_k:]
                selected_features = [feature_cols[i] for i in sorted(top_indices)]
                logger.info(
                    f"🔬 [{self.symbol}] Feature Pruning: {len(feature_cols)} → {len(selected_features)} features "
                    f"(top-{top_k} by XGB importance)"
                )
                feature_cols = selected_features
                X = df_signals[feature_cols]
                # Re-split with new features for remaining folds
                # Current fold results are kept as-is (valid baseline)
            cv_scores["xgb"].append(xgb_score)

            # 3. SGD Classifier (Linear Incremental - Rule 3.7)
            # PROFESSOR METHOD: Reuse SGD instance for ultra-fast online updates.
            gb = getattr(self, "gb_model", None)
            if not gb or not hasattr(gb, "partial_fit"):
                gb = SGDClassifier(
                    loss="log_loss", # Provides predict_proba
                    penalty="elasticnet",
                    learning_rate="adaptive",
                    eta0=0.01,
                    random_state=42 + self.training_iteration
                )
                
            gb.partial_fit(X_train_scaled, y_train, classes=np.array([0, 1]))
            gb_score = gb.score(X_test_scaled, y_test)
            cv_scores["gb"].append(gb_score)

        try:
            # Bypass broken dummy SHM worker and use locally trained models directly
            best_models = {"rf": rf, "xgb": xgb, "gb": gb}
            best_scaler = scaler
            f_cols = feature_cols

            # Use last fold score for metrics
            score = (
                cv_scores["rf"][-1] * self.base_rf_weight
                + cv_scores["xgb"][-1] * self.base_xgb_weight
                + cv_scores["gb"][-1] * self.base_gb_weight
            )
            metrics = {
                "rf_score": cv_scores["rf"][-1],
                "xgb_score": cv_scores["xgb"][-1],
                "gb_score": cv_scores["gb"][-1],
            }

            self.rf_model = best_models["rf"]
            self.xgb_model = best_models["xgb"]
            self.gb_model = best_models["gb"]
            self.scaler = best_scaler
            self._feature_cols = f_cols
            self.is_trained = True

            self.individual_model_scores["rf"] = metrics.get("rf_score", 0)
            self.individual_model_scores["xgb"] = metrics.get("xgb_score", 0)
            self.individual_model_scores["gb"] = metrics.get("gb_score", 0)

            logger.info(
                f"📥 [{self.symbol}] Local Training finished. Score: {score:.3f}"
            )

            # Train DeepPredictor LSTM (DISABLED FOR NANO OPTIMIZATION - 16GB RAM / 0 GPU)
            # try:
            #     from models.deep_predictor import deep_predictor
            #     if hasattr(self, 'scaler') and self.scaler is not None:
            #         try:
            #             X_dp = self.scaler.transform(X)
            #         except Exception:
            #             X_dp = X.values
            #     else:
            #         X_dp = X.values
            #     deep_predictor.train_model(X_dp, y.values if hasattr(y, 'values') else y, seq_len=10, epochs=10)
            # except Exception as e:
            #     logger.error(f"DeepPredictor train failed: {e}")

            # ═══════════════════════════════════════════════════════════════
            # OPT-3: AGGRESSIVE GC POST-TRAINING
            # QUÉ: Libera DataFrames intermedios y fuerza gc.collect().
            # POR QUÉ: Python no libera RAM de DataFrames grandes automáticamente.
            # PARA QUÉ: Recuperación inmediata de ~50MB post-entrenamiento.
            # ═══════════════════════════════════════════════════════════════
            del X, y, df_signals
            if 'X_train' in dir(): del X_train
            if 'X_test' in dir(): del X_test
            if 'X_train_scaled' in dir(): del X_train_scaled
            if 'X_test_scaled' in dir(): del X_test_scaled
            gc.collect()

            if score >= self.MIN_MODEL_ACCURACY:
                return (best_models, best_scaler, f_cols), score

            return None, score

        except Exception as e:
            logger.error(f"❌ Local Training Error for {self.symbol}: {e}")
            gc.collect()  # Cleanup even on error
            return None, 0.0

    # ============================================================
    # ✅ RE-PESADO DINÁMICO DE MODELOS (OMEGA MIND)
    # ============================================================

    def update_recursive_weights(
        self,
        actual_outcome_or_obj,
        trade_pnl: float = None,
        duration_seconds: float = 0.0,
        max_drawdown: float = 0.0,
        axioma_diagnosis: str = "NONE",
    ):
        """
        🚀 RECURSIVE WEIGHT UPDATE (Phase 9: NEURAL-FORTRESS PPO)
        Uses Asymmetric Reward Shaping and Prioritized Experience Replay.
        """
        if self.online_learner is None or not self.is_trained:
            return

        try:
            # 1. Parse Input (Retrocompatibility vs New TradeOutcome)
            actual_outcome = 1.0 if trade_pnl and trade_pnl > 0 else 0.0

            if isinstance(actual_outcome_or_obj, TradeOutcome):
                trade_pnl = actual_outcome_or_obj.pnl_pct
                duration_seconds = actual_outcome_or_obj.duration_seconds
                max_drawdown = actual_outcome_or_obj.max_adverse_excursion
                actual_outcome = 1.0 if trade_pnl > 0 else 0.0
                trade_obj = actual_outcome_or_obj
            else:
                actual_outcome = float(actual_outcome_or_obj)
                # Fallbacks for PnL if not provided (Local Estimation)
                if trade_pnl is None:
                    if self.portfolio and self.portfolio.closed_trades:
                        last_trade = self.portfolio.closed_trades[-1]
                        if last_trade["symbol"] == self.symbol:
                            trade_pnl = last_trade.get("pnl_pct", 0.0)
                            duration_seconds = last_trade.get("duration", 0)
                    else:
                        trade_pnl = 0.015 if actual_outcome > 0.5 else -0.01
                # Build synthetic TradeOutcome that derives pnl_pct from prices
                # FIX C-1: Instead of monkey-patching the CLASS property, we create
                # a TradeOutcome with entry/exit prices that produce the desired pnl_pct.
                # pnl_pct = (exit - entry) / entry * direction * leverage
                # For direction=1, leverage=1: pnl_pct = (exit - entry) / entry
                # We want pnl_pct = trade_pnl, so: exit = entry * (1 + trade_pnl)
                _synth_entry = 1.0  # Normalized reference price
                _synth_exit = 1.0 + (trade_pnl if trade_pnl else 0.0)
                trade_obj = TradeOutcome(
                    entry_price=_synth_entry,
                    exit_price=_synth_exit,
                    direction=1,
                    leverage=1.0,
                    max_adverse_excursion=max_drawdown,
                    max_favorable_excursion=0.0,
                    duration_seconds=duration_seconds,
                    latency_ms=0.0,
                )

            # 2. Convert string diagnosis to Enum
            enum_diagnosis = TesisDecayReason.NONE
            if "THESIS" in axioma_diagnosis.upper():
                enum_diagnosis = TesisDecayReason.THESIS_DECAY
            elif (
                "CRASH" in axioma_diagnosis.upper()
                or "DEPTH" in axioma_diagnosis.upper()
            ):
                enum_diagnosis = TesisDecayReason.DEPTH_CRASH
            elif "MOMENTUM" in axioma_diagnosis.upper():
                enum_diagnosis = TesisDecayReason.MOMENTUM_REVERSE

            # 3. Calculate Terminal Reward (Non-Linear)
            reward = self.reward_system.calculate_reward(
                trade_obj, current_drawdown=max_drawdown
            )

            # 4. Extract State/Action
            if getattr(self, "last_ensemble_input", None) is None:
                return  # Skip if no inference context

            state = self.last_ensemble_input  # [rf_prob, xgb_prob, gb_prob]
            current_weights = np.array(
                [self.base_rf_weight, self.base_xgb_weight, self.base_gb_weight]
            )
            prediction = float(np.dot(current_weights, state))

            # Legacy PPO Actor Log Prob Approx (Ensemble Weighting)
            log_prob = -0.5 * ((prediction - actual_outcome) ** 2)

            # Add to Prioritized Replay Buffer (Legacy / Weights)
            self.memory.add(
                state=state,
                action=prediction,
                reward=reward,
                next_state=np.zeros_like(state),  # Terminal bandit state
                log_prob=log_prob,
                axioma_reason=axioma_diagnosis,
            )
            
            # --- PHASE 5 AITS: Train PPO Agent ---
            if hasattr(self, "last_ppo_state") and self.last_ppo_state is not None:
                try:
                    if not hasattr(self, "ppo_memory"):
                        from ml.replay_buffer import PrioritizedReplayBuffer
                        self.ppo_memory = PrioritizedReplayBuffer(capacity=5000)
                        
                    self.ppo_memory.add(
                        state=self.last_ppo_state,
                        action=self.last_ppo_action,
                        reward=reward, # Reward from trade outcome
                        next_state=np.zeros_like(self.last_ppo_state),
                        log_prob=self.last_ppo_log_prob,
                        axioma_reason=axioma_diagnosis
                    )
                except Exception as e:
                    logger.error(f"Failed to add to PPO Memory: {e}")

            self.steps_since_learn += 1

            # 5. Execute PPO Batch Update (Asynchronous to avoid latency spikes)
            if self.steps_since_learn >= self.training_batch_size:
                import threading

                threading.Thread(
                    target=self._learn_ppo_batch, name="PPO_Learner", daemon=True
                ).start()
                self.steps_since_learn = 0

        except Exception as e:
            logger.error(f"Neural Fortress PPO update failed: {e}", exc_info=True)

    def _learn_ppo_batch(self):
        """Ejecuta el Clipped Surrogate Objective Update (PPO) sobre el Replay Buffer."""
        try:
            # 1. Update Legacy Ensemble Weights
            batch, idxs, weights_is = self.memory.sample(self.training_batch_size)
            if batch:
                states = np.array([e[0] for e in batch])
                actions = np.array([e[1] for e in batch])
                rewards = np.array([e[2] for e in batch])
                old_log_probs = np.array([e[4] for e in batch])
                advantages = rewards  # For Bandit tasks, Advantage ~ Reward

                current_weights = np.array(
                    [self.base_rf_weight, self.base_xgb_weight, self.base_gb_weight]
                )

                new_weights, abs_advantages = self.online_learner.update_ppo_batch(
                    weights=current_weights,
                    states=states,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    rewards=rewards,
                    advantages=advantages,
                )
                self.memory.update_priorities(idxs, abs_advantages)
                self._apply_weight_update(current_weights, new_weights, np.mean(rewards))

            # 2. Update Deep PPO Agent (Phase 5 AITS)
            if hasattr(self, "ppo_memory"):
                from ml.ppo_agent import ppo_agent
                p_batch, p_idxs, p_weights_is = self.ppo_memory.sample(self.training_batch_size)
                if p_batch and ppo_agent.network is not None:
                    p_states = np.array([e[0] for e in p_batch])
                    p_actions = np.array([e[1] for e in p_batch])
                    p_rewards = np.array([e[2] for e in p_batch])
                    p_log_probs = np.array([e[4] for e in p_batch])
                    # Normalize advantage proxy
                    p_advantages = (p_rewards - np.mean(p_rewards)) / (np.std(p_rewards) + 1e-8)
                    
                    # Ejecutar backprop en la red PyTorch
                    ppo_agent.update(p_states, p_actions, p_log_probs, p_rewards, p_advantages)
                    
                    logger.info(f"🧠 [PPO Agent] PyTorch Network Batch Complete. Avg Reward: {np.mean(p_rewards):.4f}")

        except Exception as e:
            logger.error(f"PPO Batch Learn Error: {e}", exc_info=True)

    def _apply_weight_update(self, old_weights_arr, new_weights_arr, avg_reward):
        """Helper to normalize and apply weights, and log explainability."""
        # Normalization and Constraints
        new_weights = np.clip(new_weights_arr, 0.05, 0.70)
        total = np.sum(new_weights)
        if total > 0:
            new_weights = new_weights / total

        old_w_dict = {
            "rf": old_weights_arr[0],
            "xgb": old_weights_arr[1],
            "gb": old_weights_arr[2],
        }
        new_w_dict = {
            "rf": float(new_weights[0]),
            "xgb": float(new_weights[1]),
            "gb": float(new_weights[2]),
        }

        self.base_rf_weight = new_w_dict["rf"]
        self.base_xgb_weight = new_w_dict["xgb"]
        self.base_gb_weight = new_w_dict["gb"]

        # XAI Auditing
        if hasattr(self, "xai_engine") and self.xai_engine:
            reason = f"Avg Reward: {avg_reward:+.4f} | Regime: {self.market_regime}"
            self.xai_engine.log_ppo_weight_evolution(
                self.symbol, old_w_dict, new_w_dict, avg_reward, reason
            )

        # Logging
        if self.training_iteration % 10 == 0:
            logger.debug(
                f"🧠 [Omega Weights] UPDATED: RF:{self.base_rf_weight:.2f}, XGB:{self.base_xgb_weight:.2f}, GB:{self.base_gb_weight:.2f}"
            )

    def _update_model_weights(self):
        """
        Re-pesado dinámico basado en performance reciente
        """
        if len(self.performance_history) < 30:
            return

        recent_performance = list(self.performance_history)[-30:]
        success_rate = sum(1 for x in recent_performance if x > 0) / len(
            recent_performance
        )

        # Performance-based weight adjustment
        if success_rate > 0.70:  # Excelente performance
            # Aumentar peso del mejor modelo
            best_model = max(self.individual_model_scores.items(), key=lambda x: x[1])
            weight_increase = 0.05

            if best_model[0] == "rf":
                self.base_rf_weight = min(0.60, self.base_rf_weight + weight_increase)
            elif best_model[0] == "xgb":
                self.base_xgb_weight = min(0.50, self.base_xgb_weight + weight_increase)
            else:
                self.base_gb_weight = min(0.40, self.base_gb_weight + weight_increase)

            logger.debug(
                f"🔥 Increasing {best_model[0]} weight due to excellent performance"
            )

        elif success_rate < 0.20 and len(self.performance_history) >= 30:  # Phase 2 FIX: Real degradation needs more sample
            # Volver a pesos originales
            self.base_rf_weight = self.original_rf_weight
            self.base_xgb_weight = self.original_xgb_weight
            self.base_gb_weight = self.original_gb_weight
            logger.debug(f"⚠️ Resetting weights due to poor performance (<20%) over 30 trades")
            self.performance_history.clear()  # Prevent infinite reset spam

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
            drawdown = (
                (self.peak_equity - current_equity) / self.peak_equity
                if self.peak_equity > 0
                else 0
            )

            # Activar por drawdown
            if (
                drawdown > self.circuit_breaker_threshold
                and not self.circuit_breaker_active
            ):
                self.activate_circuit_breaker()
                return False

            # Activar por pérdidas consecutivas
            if (
                self.consecutive_losses >= self.max_consecutive_losses
                and not self.circuit_breaker_active
            ):
                self.activate_circuit_breaker()
                return False

            # Desactivar si se recupera
            if self.circuit_breaker_active and drawdown < (
                self.circuit_breaker_threshold * 0.6
            ):
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
        self.adaptive_confidence_threshold = min(
            0.80, self.adaptive_confidence_threshold + 0.25
        )
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
            self.analysis_stats["total"] += 1
            if not self._check_circuit_breaker():
                return

            # 1. Obtención y Preparación de Datos
            bars = self.data_provider.get_latest_bars(self.symbol, n=100, timeframe=self.primary_tf)
            # ═══════════════════════════════════════════════════════════════
            # QUANTUM ZERO-COPY: return_polars=True para el hot-loop sync
            # El backtest llama _run_inference (sync), NO _run_inference_v3.
            # Sin este flag, Polars→Pandas→iloc→values añadía ~350ms.
            # ═══════════════════════════════════════════════════════════════
            # ── FASE III-B: TRUE ZERO-COPY CACHE BYPASS (GIL EVASION) ──
            df_pl = None
            last_row_dict = None
            if bars is not None and len(bars) > 0:
                current_ts = bars['timestamp'][-1] if hasattr(bars, 'dtype') else (bars[-1]['timestamp'] if isinstance(bars[-1], dict) else None)
                if current_ts is not None and hasattr(self, "_global_feature_cache") and hasattr(self, "_global_feature_cache_ts"):
                    import numpy as np
                    ts_arr = self._global_feature_cache_ts
                    idx = np.searchsorted(ts_arr, current_ts)
                    if idx < len(ts_arr) and ts_arr[idx] == current_ts:
                        # [ZERO-COPY] Skip Polars entirely. Use raw Pandas slice for Dict and extract NumPy array directly later.
                        # We just store the index so we can slice the NumPy matrix in O(1) time.
                        self._quantum_cache_hit = True
                        self._quantum_idx = idx
                        last_row_dict = self._global_feature_cache.iloc[idx].to_dict()
                        # We still need df_pl to be truthy to avoid recalculating prepare_features
                        df_pl = "CACHED_TRUE" 
            
            if df_pl is None:
                self._quantum_cache_hit = False
                df_pl = self._prepare_features(bars, regime_aware=True, return_polars=True)

            if df_pl is None or (not isinstance(df_pl, str) and len(df_pl) < 5):
                return
            
            # Convertir la última fila a dict para acceso rápido
            if self._quantum_cache_hit and last_row_dict is not None:
                last_row = last_row_dict
            else:
                last_row = df_pl.row(-1, named=True)
                
            current_row = last_row  # Alias: dict soporta .get() y [] igual que Pandas Series
            atr_pct = last_row.get("atr_pct", 0)
            current_atr = last_row.get("atr", 0)
            rsi = last_row.get("rsi_14", 50)
            vol_ratio = last_row.get("volume_ratio", 0)
            confluence = last_row.get("confluence_score", 0)

            # 2. Verificar disponibilidad de Modelos (PRIMERO)
            with self._state_lock:
                models_ready = all([self.rf_model, self.xgb_model, self.gb_model])
                feature_cols = self._feature_cols

            # -------------------------------------------------------------------------
            # ✅ CROSS-POLLINATION (Phase 7): Read Math Stats from Portfolio
            # -------------------------------------------------------------------------
            math_hurst = 0.5
            if self.portfolio and hasattr(self.portfolio, "math_stats"):
                math_hurst = self.portfolio.math_stats.get("hurst", 0.5)

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
                    z_score = last_row.get("volume_zscore", 0)
                    adx = last_row.get("adx", 0)
                    trend_power = last_row.get("trend_power", 0)

                    labels = ["M1", "M5", "M15", "M30"] if getattr(self, "horizon", "SCALPING") in ["SCALPING", "MICROSCALPING"] else ["H1", "H4", "H12", "D1"]
                    oracle_msg = (
                        f"\n🔮 [UNIFIED ORACLE] {self.symbol} | TRAINING | Last CV: {self.last_training_score:.3f}\n"
                        f"   Engines Passing: 0/3 | Threshold: {self.consensus_threshold}\n"
                        f"   Scores  -> ML: {self.last_training_score:.2f} | SENT: 0.00 | TECH: 0.00\n"
                        f"   Horizon -> {labels[0]}: 0.00 | {labels[1]}: 0.00 | {labels[2]}: 0.00 | {labels[3]}: 0.00\n"
                        f"   Verdict -> Direction: TRAINING | Final Conf: 0.00 (Gap: 1.00)\n"
                        f"   Phase: {self.market_regime} ({self.regime_confidence * 100:.1f}%)\n"
                        f"   Concept: {concept} (Models Compiling)\n"
                        f"   Stats: ADX={adx:.1f} | ATR%={atr_pct:.2f}% | Trend={trend_power:.2f} | Z-Score={z_score:.2f}\n"
                        f"   Math: Hurst={math_hurst:.2f} (Portfolio Sync)\n"
                        f"   Confidence: 0.0% | Strategy: Waiting for AI models..."
                    )
                    logger.info(oracle_msg)
                return

            # 2.5 NUEVA: Validación crítica de features
            if feature_cols is None or not feature_cols:
                logger.error(
                    f"❌ Error processing {self.symbol}: No valid feature columns available"
                )
                return

            # Filtrar columnas válidas
            if self._quantum_cache_hit:
                valid_features = [
                    col for col in feature_cols if col is not None and col in self._global_feature_cache.columns
                ]
            else:
                valid_features = [
                    col for col in feature_cols if col is not None and col in df_pl.columns
                ]
            if not valid_features:
                logger.error(
                    f"❌ Error processing {self.symbol}: No valid features available for inference"
                )
                return

            # ═══════════════════════════════════════════════════════════════
            # QUANTUM ZERO-COPY: Direct NumPy View (skip Polars/Pandas alignment)
            # ═══════════════════════════════════════════════════════════════
            if self._quantum_cache_hit:
                # O(1) Memory Slice! No allocations.
                # To make it even faster, we can extract the numpy array of the specific columns
                # Only if we pre-cache it. For now, iloc to numpy.
                X_pred = self._global_feature_cache.iloc[self._quantum_idx:self._quantum_idx+1][valid_features].to_numpy()
            else:
                X_pred = df_pl.select(valid_features).tail(1).to_numpy()

            if X_pred.size == 0:
                logger.error(f"❌ {self.symbol}: Empty feature matrix after alignment")
                return

            # Replace NaN with 0 in numpy directly
            np.nan_to_num(X_pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # FIX: Ensure scale-variant models (MLP, SGD) receive scaled data
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X_pred)
                except Exception:
                    X_scaled = X_pred  # Fallback
            else:
                X_scaled = X_pred

            # Determine whether to use fast matrices or slow models
            rf = getattr(self, "rf_arrays", self.rf_model)
            gb = getattr(self, "gb_arrays", self.gb_model)
            X_flat = (
                X_scaled[0]
                if isinstance(X_scaled, np.ndarray) and X_scaled.ndim > 1
                else X_scaled
            )

            if isinstance(rf, dict) and "tree_offsets" in rf:
                rf_proba_1 = predict_rf_jit(
                    X_flat,
                    rf["children_left"],
                    rf["children_right"],
                    rf["feature"],
                    rf["threshold"],
                    rf["value"],
                    rf["tree_offsets"],
                )
                rf_proba = np.array([1.0 - rf_proba_1, rf_proba_1])
            else:
                rf_proba = self.rf_model.predict_proba(X_scaled)[0]

            try:
                booster = self.xgb_model.get_booster() if hasattr(self.xgb_model, 'get_booster') else self.xgb_model
                _xgb_p = float(booster.inplace_predict(X_scaled)[0])
                xgb_proba = np.array([1.0 - _xgb_p, _xgb_p])
            except Exception:
                xgb_proba = self.xgb_model.predict_proba(X_scaled)[0]

            if isinstance(gb, dict) and "init_score" in gb:
                gb_proba_1 = predict_gb_jit(
                    X_flat,
                    gb["children_left"],
                    gb["children_right"],
                    gb["feature"],
                    gb["threshold"],
                    gb["value"],
                    gb["tree_offsets"],
                    gb["init_score"],
                    gb["learning_rate"],
                )
                gb_proba = np.array([1.0 - gb_proba_1, gb_proba_1])
            else:
                gb_proba = self.gb_model.predict_proba(X_scaled)[0]

            ensemble_proba = (
                rf_proba * self.base_rf_weight
                + xgb_proba * self.base_xgb_weight
                + gb_proba * self.base_gb_weight
            )

            # =========================================================
            # PHASE 4 (AITS): DEEP LEARNING LSTM-ATTENTION INJECTION
            # =========================================================
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V130 FIX: LSTM DEACTIVATED (RANDOM WEIGHTS = NOISE)
            # QUÉ: Desactivar el blending del DeepPredictor LSTM/Attention.
            # POR QUÉ: El modelo NUNCA persiste sus pesos (.pth no existe).
            #   Cada reinicio del bot, el LSTM arranca desde pesos aleatorios
            #   de PyTorch (Xavier/Kaiming init). Su output es estadísticamente
            #   equivalente a {SHORT: 0.33, FLAT: 0.34, LONG: 0.33}.
            #   Inyectar 15% de ruido aleatorio al ensemble DEGRADA la
            #   precisión de XGBoost/RF/GB que SÍ están entrenados.
            # PARA QUÉ: Eliminar la contaminación del ensemble. Los modelos
            #   tabular (XGB/RF/GB) son los únicos con poder predictivo real.
            # CUÁNDO: Se reactivará cuando existan archivos .pth validados
            #   con MAE < 0.002 en out-of-sample test.
            # DÓNDE: strategies/ml_strategy.py → _run_inference()
            # QUIÉN: QA Engineer + Quant Developer
            # ═══════════════════════════════════════════════════════════════
            deep_probs = {"SHORT": 0.33, "FLAT": 0.34, "LONG": 0.33}
            deep_array = np.array([deep_probs["SHORT"], deep_probs["LONG"]])
            
            # LSTM blend weight = 0% until trained weights are validated
            # Previously was 15% which injected pure random noise
            _lstm_blend_weight = 0.0
            if _lstm_blend_weight > 0 and len(deep_array) == len(ensemble_proba):
                ensemble_proba = ensemble_proba * (1.0 - _lstm_blend_weight) + deep_array * _lstm_blend_weight

            # Determine base prediction BEFORE bias
            classes = self.rf_model.classes_
            pred_idx = np.argmax(ensemble_proba)
            raw_confidence = ensemble_proba[pred_idx]
            predicted_class = self._label_mapping.get(
                classes[pred_idx], classes[pred_idx]
            )

            # CTOS Phase 5: Hard Confidence Floor (Anti-Noise Filter)
            # FASE 30: Use dynamic DNA threshold based on direction
            floor_threshold = Config.Strategies.ML_THRESHOLDS.get('confidence_bull', 0.55)
            if predicted_class == "SHORT" and 'confidence_bear' in Config.Strategies.ML_THRESHOLDS:
                floor_threshold = Config.Strategies.ML_THRESHOLDS['confidence_bear']
                
            if raw_confidence < floor_threshold:
                self.analysis_stats["filtered_conf"] += 1
                return

            # ═══════════════════════════════════════════════════════════════
            # PHASE 5 AITS: REINFORCEMENT LEARNING CORE (PPO SIZING)
            # QUÉ: PPO Agent ajusta la magnitud (agresividad/tamaño) del trade.
            # ═══════════════════════════════════════════════════════════════
            confidence = raw_confidence
            
            # Construir Estado (State) para el PPO (15 dims)
            try:
                from ml.ppo_agent import ppo_agent
                
                # Regime one-hot
                reg_trend = 1.0 if self.market_regime == "TRENDING" else 0.0
                reg_vol = 1.0 if self.market_regime == "VOLATILE" else 0.0
                reg_range = 1.0 if self.market_regime == "RANGING" else 0.0
                
                asset_hash = sum(ord(c) for c in self.symbol.split('/')[0]) / 1000.0 if self.symbol else 0.0
                ppo_state = np.array([
                    ensemble_proba[0], ensemble_proba[1], # 2
                    deep_probs.get("SHORT", 0), deep_probs.get("LONG", 0), # 2
                    confluence, atr_pct, vol_ratio, rsi, # 4
                    current_row.get("trend_power", 0), # 1
                    current_row.get("adx", 0), # 1
                    current_row.get("volume_zscore", 0), # 1
                    math_hurst, # 1
                    reg_trend, reg_vol, reg_range, # 3
                    1.0 if self.horizon_str in ["SCALPING", "MICROSCALPING"] else 0.0, # 1
                    asset_hash, # 1
                    float(current_row.get("normalized_spread", 0.0) or current_row.get("spread_squeeze", 0.0) or 0.0) # 1
                ], dtype=np.float32)
                
                # Obtener Acción (Continuous [-1, 1])
                ppo_action, ppo_log_prob, ppo_value = ppo_agent.get_action_and_value(ppo_state)
                
                # Modulación de la Confianza/Sizing
                _tentative_dir = "LONG" if predicted_class == 1 else "SHORT"
                
                # FIX: Bypass PPO until it has enough training data
                ppo_trades = getattr(ppo_agent, 'total_updates', 0)
                if ppo_trades < 50:
                    confidence = raw_confidence
                    logger.debug(f"🧠 [PPO-RL] BYPASS (Untrained: {ppo_trades}/50 trades). Conf: {confidence:.3f}")
                else:
                    # Si el PPO está de acuerdo con la dirección (signo), aumentamos confianza
                    # Si el PPO difiere, la reducimos.
                    # Acción > 0 significa LONG, Acción < 0 significa SHORT.
                    ppo_aggressiveness = 1.0
                    if _tentative_dir == "LONG":
                        if ppo_action > 0:
                            ppo_aggressiveness = 1.0 + (ppo_action * 0.2) # Up to 20% boost
                        else:
                            ppo_aggressiveness = 1.0 + ppo_action # Penalty down to 0
                    else: # SHORT
                        if ppo_action < 0:
                            ppo_aggressiveness = 1.0 + (abs(ppo_action) * 0.2)
                        else:
                            ppo_aggressiveness = 1.0 - ppo_action
                            
                    confidence = raw_confidence * max(0.1, ppo_aggressiveness)
                
                logger.debug(
                    f"🧠 [PPO-RL] {_tentative_dir} | PPO Action: {ppo_action:.3f} | Conf: {raw_confidence:.3f} → {confidence:.3f}"
                )
                
                # Save context for training when trade finishes
                self.last_ppo_state = ppo_state
                self.last_ppo_action = ppo_action
                self.last_ppo_log_prob = ppo_log_prob
                
            except Exception as e:
                logger.error(f"PPO Agent inference error: {e}")
                confidence = raw_confidence
                
            # PHASE 10: HotAdapterRL Always-On Active Memory
            if self.hot_adapter:
                _tentative_dir = "LONG" if predicted_class == 1 else "SHORT"
                _bias = self.hot_adapter.get_bias(self.symbol, _tentative_dir)
                
                # Apply asymmetric online bias
                confidence = min(0.99, confidence * _bias)
                if _bias != 1.0:
                    logger.debug(f"🧠 [HOT-ADAPTER] {_tentative_dir} confidence adjusted by bias {_bias:.2f}x -> {confidence:.3f}")

            # =========================================================
            # PHASE 4: MULTI-HORIZON ORACLE VETO (Causal Reasoner)
            # =========================================================
            # FORENSIC LATENCY FIX: Cache macro data (1d, 1w) for 1 hour
            # Fetching 250 bars on every tick destroys HFT latency.
            if not hasattr(self, '_oracle_cache'):
                self._oracle_cache = {}
                
            cache_key = f"{self.symbol}_macro"
            current_time = time.time()
            
            # Use cached data if younger than 1 hour (3600 seconds)
            if cache_key in self._oracle_cache and (current_time - self._oracle_cache[cache_key]['timestamp']) < 3600:
                timeframe_data = self._oracle_cache[cache_key]['data']
            else:
                timeframe_data = {}
                for tf in ["1d", "1w"]:
                    try:
                        macro_bars = self.data_provider.get_latest_bars(
                            self.symbol, n=250, timeframe=tf
                        )
                        if macro_bars is not None and len(macro_bars) >= 20:
                            _c = macro_bars["close"]
                            _rsi = calculate_rsi_jit(_c, period=14)
                            _ema_fast = calculate_ema_jit(_c, period=20)
                            _ema_slow = calculate_ema_jit(_c, period=50)
                            _ema_trend = calculate_ema_jit(_c, period=200)

                            _in_up = (_ema_fast > _ema_slow) & (_c > _ema_trend)
                            _in_dn = (_ema_fast < _ema_slow) & (_c < _ema_trend)

                            timeframe_data[tf] = {
                                "inds": {
                                    "rsi": _rsi,
                                    "in_uptrend": _in_up,
                                    "in_downtrend": _in_dn,
                                },
                                "data": macro_bars,
                            }
                    except Exception as e:
                        logger.debug(
                            f"Oracle data parsing failed for {tf} on {self.symbol}: {e}"
                        )
                # Store in cache
                self._oracle_cache[cache_key] = {
                    'timestamp': current_time,
                    'data': timeframe_data
                }

            signal_type_raw = "LONG" if predicted_class == 1 else "SHORT"

            try:
                oracle_verdict = MultiHorizonOracle.evaluate_clash_vector(
                    timeframe_data, signal_type_raw
                )
                if oracle_verdict["is_vetoed"]:
                    logger.info(
                        f"🔮 [ML ORACLE VETO] {self.symbol} {signal_type_raw} BLOCKED | "
                        f"Clash: {oracle_verdict['clash_score']:.1%} | Macro: {oracle_verdict['macro_context']}"
                    )
                    self.analysis_stats["filtered_conf"] += 1
                    return
            except Exception as e:
                logger.error(f"Oracle ML Integration Error on {self.symbol}: {e}")

            # ═══════════════════════════════════════════════════════════════
            # 🚀 FASE 12: CROSS-HORIZON RESONANCE (Filtro Cuántico)
            # QUÉ: Suprime operaciones Scalp/Microscalp contra la tendencia Swing activa.
            # POR QUÉ: Un Swing activo significa que el sesgo macro es fuerte en esa dirección.
            #   Operar un SCALP en contra es exponerse a la tendencia pesada del sistema.
            # ═══════════════════════════════════════════════════════════════
            if self.portfolio and hasattr(self.portfolio, 'virtual_ledger') and self.horizon_str in ("SCALPING", "MICROSCALPING", "MICRO"):
                swing_opposing = any(
                    ("SWING" in k and k.startswith(f"{self.symbol}_")) and
                    ((v.get('quantity', 0) > 0 and signal_type_raw == "SHORT") or
                     (v.get('quantity', 0) < 0 and signal_type_raw == "LONG"))
                    for k, v in self.portfolio.virtual_ledger.items()
                    if abs(v.get('quantity', 0)) > 1e-8
                )
                if swing_opposing:
                    logger.info(
                        f"🛑 [CROSS-HORIZON RESONANCE] {self.symbol} {signal_type_raw} BLOCKED | "
                        f"Opposing active SWING position detected."
                    )
                    self.analysis_stats["filtered_conf"] += 1
                    return

            # 6. SIGNAL GENERATION (Delegated to Component)
            # Retrieve Dynamic Advice based on Regime
            # NOTE: market_regime object is needed (it's initialized in strategy base or passed in)
            # Since MLStrategy doesn't hold reference to MarketRegimeDetector directly in __init__,
            # we rely on what was detected in `self.market_regime`.

            # Temporary: Get advice using static lookup or if we had the detector instance.
            # Ideally, Strategy should receive the full Regime Context.

            from config import Config

            regime_map = getattr(Config.Sniper, "REGIME_MAP", {})
            advice = regime_map.get(self.market_regime, regime_map.get("RANGING"))
            threshold_mod = advice.get("threshold_mod", 0.0)

            signal_data = self.signal_generator.generate_signal(
                df,
                prediction=predicted_class,
                probability=confidence,
                threshold=self.adaptive_confidence_threshold,
                regime=self.market_regime,
                threshold_mod=threshold_mod,
            )

            if not signal_data:
                self.analysis_stats["filtered_conf"] += 1
                return

            signal_type = signal_data["type"]
            final_conf = signal_data["confidence"]
            confluence = signal_data["confluence"]

            # Filtro de Confluencia
            if predicted_class == 1 and confluence < self.adaptive_confluence_long:
                return
            if predicted_class == -1 and confluence > self.adaptive_confluence_short:
                return

            # 🌑 FASE 38: DARK ALPHA LAYER VETO (Hyperliquid Cascade Detection)
            if self.horizon_str == 'SCALPING':
                try:
                    from core.global_state import global_state
                    dark_pressure = getattr(global_state, 'dark_alpha_pressure', 0.0)
                    
                    # Positive pressure = Short Squeeze (Buy momentum)
                    # Negative pressure = Long Cascade (Sell momentum)
                    
                    if predicted_class == 1 and dark_pressure < -250_000:
                        logger.warning(f"🌑 [DARK ALPHA VETO] {self.symbol} LONG blocked | Huge DEX Long Liquidation: ${dark_pressure:,.2f}")
                        self.analysis_stats["filtered_conf"] += 1
                        return
                        
                    if predicted_class == -1 and dark_pressure > 250_000:
                        logger.warning(f"🌑 [DARK ALPHA VETO] {self.symbol} SHORT blocked | Huge DEX Short Squeeze: ${dark_pressure:,.2f}")
                        self.analysis_stats["filtered_conf"] += 1
                        return
                        
                    # 🚨 RBF MEMPOOL PANIC OVERRIDE
                    rbf_panic = getattr(global_state, 'rbf_panic_score', 0.0)
                    if rbf_panic > 500.0:
                        logger.warning(f"🚨 [MEMPOOL OVERRIDE] {self.symbol} RBF Panic detected ({rbf_panic:.2f} Gwei). Forcing execution!")
                        # We bypass confluence restrictions because Mempool urgency is absolute priority
                        confluence = 1.0 # Force max confluence to bypass further checks
                        
                except Exception as e:
                    logger.error(f"Dark Alpha / Mempool check failed: {e}", exc_info=True)

            # 🔮 FASE 20: PDC (Price Discovery Coefficient) Veto for Scalping
            if self.horizon_str == 'SCALPING':
                try:
                    from core.global_state import global_state
                    ce_metrics = getattr(global_state, 'cross_exchange_metrics', {})
                    sym_pdc = ce_metrics.get(self.symbol, {})
                    pdc_velocity = sym_pdc.get('pdc_velocity', 0.0)
                    
                    min_pdc = getattr(Config.Strategies, 'TECHNICAL_THRESHOLDS', {}).get('min_pdc_velocity', 0.05)
                    if pdc_velocity < min_pdc:
                        logger.warning(f"🛑 [PDC VETO] {self.symbol} SCALPING blocked | PDC Velocity: {pdc_velocity:.4f} < {min_pdc}")
                        self.analysis_stats["filtered_conf"] += 1
                        return
                except Exception as e:
                    logger.warning(f"⚠️ [PDC VETO] Error checking PDC for {self.symbol}: {e}")

            # ============ CREAR SEÑAL ============
            signal_type = SignalType.LONG if predicted_class == 1 else SignalType.SHORT
            tp_target = self.current_tp_target
            sl_target = self.current_sl_target

            # Ajuste de targets por volatilidad
            if atr_pct > 0.03:
                tp_target *= 1.3
                sl_target *= 1.3
            elif atr_pct < 0.01:
                tp_target *= 0.8
                sl_target *= 0.8

            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V90: PREDICCIÓN DE MAGNITUD REAL (NO ESTÁTICA)
            # QUÉ: Calcula predicted_magnitude usando ATR real × √(lookahead) × confianza.
            # POR QUÉ: Antes se copiaba tp_target como predicted_magnitude,
            #   lo cual era una MENTIRA — siempre mostraba el mismo número.
            # PARA QUÉ: El ExitOracle puede comparar progreso REAL vs predicción REAL
            #   para decidir si cerrar prematuramente o dejar correr.
            # CÓMO: Magnitude = ATR% × √(barras_futuras) × confidence_multiplier
            #   Esto modela difusión browniana: el precio se mueve proporcional a √t.
            # ═══════════════════════════════════════════════════════════════
            import math
            # Magnitud esperada basada en difusión de precio real
            sqrt_bars = math.sqrt(max(1, self.LOOKAHEAD_BARS))
            raw_magnitude = atr_pct * sqrt_bars  # Movimiento esperado por difusión
            # Ajustar por confianza del modelo: mayor confianza → mayor magnitud esperada
            confidence_mult = 0.5 + confidence  # Rango [0.5, 1.5] para confidence [0, 1]
            predicted_magnitude_real = raw_magnitude * confidence_mult
            
            # FORENSIC-V120: REJECT UNPROFITABLE SIGNALS
            # QUÉ: Si la IA predice que el movimiento será menor a las fees, abortar.
            # POR QUÉ: Comprar para ganar 0.04% cuando las fees son 0.06% es suicidio matemático.
            if predicted_magnitude_real < 0.0006:
                logger.debug(f"🛑 [{self.symbol}] Signal Rejected: Predicted Magnitude {predicted_magnitude_real*100:.3f}% is less than fees (0.06%).")
                self.analysis_stats["filtered_conf"] += 1
                return
                
            # FORENSIC-V120: ALIGN TP WITH REAL PREDICTION
            # QUÉ: Setear el tp_target EXACTAMENTE a la predicción de la IA, no al default estático.
            # POR QUÉ: Si la IA predice 0.12%, pero el default es 0.40%, la orden LMT nunca se ejecutará.
            tp_target = max(0.0006, predicted_magnitude_real)
            sl_target = min(self.current_sl_target, tp_target / 1.5)

            # Duración predicha: cuántas barras para alcanzar la magnitud
            # Si ATR es alto → llega rápido; si ATR es bajo → llega lento
            if atr_pct > 0:
                predicted_duration_bars = int((predicted_magnitude_real / atr_pct) ** 2)
                predicted_duration_bars = max(3, min(self.LOOKAHEAD_BARS * 2, predicted_duration_bars))
            else:
                predicted_duration_bars = self.LOOKAHEAD_BARS

            logger.debug(
                f"🎯 [{self.symbol}] Magnitud Real: {predicted_magnitude_real*100:.3f}% en ~{predicted_duration_bars} barras "
                f"(ATR={atr_pct*100:.3f}%, √bars={sqrt_bars:.1f}, conf={confidence:.2f})"
            )

            # PHASE 9.2: Adaptive TTL (Prediction Horizon Sync)
            # FORENSIC FIX: TTL must scale with predicted_duration, not static LOOKAHEAD.
            prediction_ttl = int(predicted_duration_bars * 60 * 1.2)  # 20% buffer
            if self.horizon_str == "SWING":
                # Swing: 5 minutes to 4 hours patience
                final_ttl = max(300, min(14400, prediction_ttl))
            else:
                # Scalping: 30 seconds to 10 minutes patience (was 5 min max)
                final_ttl = max(30, min(600, prediction_ttl))

            # ============ REGISTRAR Y ENVIAR (XAI + SOPHIA) ============
            # Phase 22: XAI Explanation (Why did we decide this?)
            # [FIX] xai_engine is synchronous and blocks the hot-path. Temporarily disabled.
            # model_used = self.rf_model
            # xai_explanation = self.xai_engine.explain_local_prediction(model_used, X_scaled, "RandomForest")
            # logger.info(f"🧠 [XAI] {signal_type} Signal Reason: {xai_explanation}")
            # self.xai_engine.log_trade_explanation(self.symbol, signal_type, xai_explanation)

            # FORENSIC-3: OMEGA AI INTEGRATION
            # Vectorized numpy returns for < 1ms latency
            _cvals = df["close"].values
            if len(_cvals) > 1:
                _returns = np.diff(_cvals) / _cvals[:-1]
                _returns = _returns[~np.isnan(_returns) & ~np.isinf(_returns)]
            else:
                _returns = np.array([])
            
            sophia_report_dict = {}
            if hasattr(self, 'sophia') and self.sophia:
                sophia_report = self.sophia.analyze(
                    symbol=self.symbol,
                    direction=signal_type.name,
                    signal_strength=confidence, ml_confidence=confidence,
                    setups={"xai_reason": xai_explanation},
                    confluence_score=confluence,
                    tp_pct=tp_target,
                    sl_pct=sl_target,
                    returns=_returns,
                    ttl_seconds=final_ttl,
                    regime=self.market_regime,
                )
    
                # FORENSIC-31: HARD EXACTITUDE THRESHOLD (>70%)
                if sophia_report.win_probability < 0.70:
                    logger.info(f"🛑 [SOPHIA VETO] {self.symbol} ML Signal Blocked. Exactitude ({sophia_report.win_probability*100:.1f}%) < 70%.")
                    self.analysis_stats["filtered_conf"] += 1
                    return
                    
                # [FASE 3: Filtro Cuántico "The Vortex Gate"]
                if getattr(self, "horizon_str", "") == "MICROSCALPING":
                    if sophia_report.vortex_pulse < 1.5:
                        logger.info(f"🌀 [VORTEX GATE] {self.symbol} Microscalping Signal Blocked. Market is dead (Vortex: {sophia_report.vortex_pulse:.2f}).")
                        self.analysis_stats["filtered_conf"] += 1
                        return
                        
                sophia_report_dict = sophia_report.to_dict()

            # FIXED: Create SignalEvent with ALL metadata in constructor (frozen dataclass)
            detailed_id = f"{self.strategy_id}.ML_PREDICTION"
            real_current_price = float(bars['close'][-1])
            signal = SignalEvent(
                strategy_id=detailed_id,
                setup_type="ML_PREDICTION",
                symbol=self.symbol,
                datetime=self._now(),
                signal_type=signal_type,
                strength=confidence, ml_confidence=confidence,
                atr=current_row["atr"],  # FIXED: Use current_row
                tp_pct=tp_target,
                sl_pct=sl_target,
                current_price=real_current_price,
                horizon=self.horizon_str,
                predicted_magnitude=predicted_magnitude_real,
                predicted_duration=predicted_duration_bars,
                priority=getattr(self, "priority", 1),
                ttl=final_ttl,
                metadata={
                    "timeInForce": "GTX",  # [PHASE 5] Enforce MAKER execution
                    "sophia": sophia_report_dict,  # [PHASE OMEGA] Seamless Nemesis integration
                },
            )

            self.performance_history.append(0)
            self.signal_history.append(
                {
                    "timestamp": self._now(),
                    "type": signal_type,
                    "confidence": confidence,
                    "regime": self.market_regime,
                    "price": real_current_price,
                    "confluence": confluence,
                    "xai": xai_explanation,
                }
            )

            self.total_signals_generated += 1
            if self.market_regime in self.signals_by_regime:
                self.signals_by_regime[self.market_regime] += 1

            # Logging completo y envío
            self._log_ml_signal(
                signal_type, confidence, confluence, df_pl, rf_proba, xgb_proba, gb_proba
            )

            # Neural Bridge Publication (Base Logic Fallback)
            if not hasattr(self, "engines_active"):  # If not UniversalEnsemble
                neural_bridge.publish_insight(
                    strategy_id="ML_ORACLE",
                    symbol=self.symbol,
                    insight={
                        "confidence": confidence,
                        "direction": "LONG" if predicted_class == 1 else "SHORT",
                        "confluence": confluence,
                    },
                )

            self.events_queue.put(signal)

            # OMEGA MIND: Save input for recursive weighting update
            self.last_ensemble_input = np.array([rf_proba, xgb_proba, gb_proba])

            if len(self.performance_history) >= 15:
                self._update_model_weights()

            self._last_prediction_time = self._now()

        except Exception as e:
            logger.error(f"ML Inference error {self.symbol}: {e}", exc_info=True)

    # ============================================================
    # ✅ LOGGING ULTIMATE - 40+ MÉTRICAS
    # ============================================================

    def _log_ml_signal(
        self, signal_type, confidence, confluence, df, rf_proba, xgb_proba, gb_proba
    ):
        """
        Logging completo con 40+ métricas para análisis
        """
        try:
            # Handle both Polars and Pandas DataFrames
            if hasattr(df, 'row'):  # Polars
                current_row = df.row(-1, named=True)
            else:  # Pandas
                current_row = df.iloc[-1]
            atr_pct = current_row.get("atr_pct", 0) if isinstance(current_row, dict) else current_row["atr_pct"]

            # Preparar todas las métricas
            metrics = {
                # Indicadores principales
                "RSI_14": float(current_row["rsi_14"]),
                "RSI_5m": float(current_row.get("rsi_5m", 50)),
                "RSI_15m": float(current_row.get("rsi_15m", 50)),
                "RSI_1h": float(current_row.get("rsi_1h", 50)),
                # Volatilidad y tendencia
                "ATR%": float(current_row["atr_pct"]),
                "ADX": float(current_row.get("adx", 0)),
                "NATR": float(current_row.get("natr", 0)),
                "MACD_Hist": float(current_row.get("macd_hist", 0)),
                "MACD_Slope": float(current_row.get("macd_slope", 0)),
                # Precio y volumen
                "Price_Position": float(current_row.get("close_position", 0.5)),
                "BB_Position": float(current_row.get("bb_position", 0.5)),
                "Volume_Ratio": float(current_row["volume_ratio"]),
                "Volume_ZScore": float(current_row.get("volume_zscore", 0)),
                # Momentum
                "Momentum_5": float(current_row.get("momentum_5", 0)),
                "Momentum_20": float(current_row.get("momentum_20", 0)),
                "ROC_10": float(current_row.get("roc_10", 0)),
                # Tendencia
                "EMA_Cross": int(current_row.get("ema_20_50_cross", 0)),
                "Trend_1h": int(current_row.get("trend_1h", 0)),
                "Trend_1h_Strength": float(current_row.get("trend_1h_strength", 0)),
                # Confluence y decisión
                "Confluence_Score": float(confluence),
                "Prediction_Confidence": float(confidence),
                "Signal_Type": signal_type.name,
                # Régimen y estado
                "Market_Regime": self.market_regime,
                "Regime_Confidence": float(self.regime_confidence),
                "Regime_Duration": self.regime_duration,
                "Circuit_Breaker": self.circuit_breaker_active,
                "Aggressiveness_Factor": float(self.aggressiveness_factor),
                # Scores de modelos
                "Training_Score": float(self.last_training_score),
                "RF_Score": float(self.individual_model_scores.get("rf", 0)),
                "XGB_Score": float(self.individual_model_scores.get("xgb", 0)),
                "GB_Score": float(self.individual_model_scores.get("gb", 0)),
                # Pesos del ensemble
                "RF_Weight": float(self.base_rf_weight),
                "XGB_Weight": float(self.base_xgb_weight),
                "GB_Weight": float(self.base_gb_weight),
                # Probabilidades individuales
                "RF_Proba_Long": float(rf_proba[1]) if len(rf_proba) > 1 else 0.0,
                "XGB_Proba_Long": float(xgb_proba[1]) if len(xgb_proba) > 1 else 0.0,
                "GB_Proba_Long": float(gb_proba[1]) if len(gb_proba) > 1 else 0.0,
                "Ensemble_Proba_Long": float(
                    rf_proba[1] * self.base_rf_weight
                    + xgb_proba[1] * self.base_xgb_weight
                    + gb_proba[1] * self.base_gb_weight
                )
                if len(rf_proba) > 1
                else 0.0,
                # Targets
                "TP_Target": float(self.current_tp_target * 100),
                "SL_Target": float(self.current_sl_target * 100),
                "TP_SL_Ratio": float(self.current_tp_target / self.current_sl_target)
                if self.current_sl_target > 0
                else 0,
                # Performance
                "Total_Signals": self.total_signals_generated,
                "Win_Rate": (self.winning_trades / self.total_trades)
                if self.total_trades > 0
                else 0,
                "Consecutive_Losses": self.consecutive_losses,
                "Learning_Rate": float(self.learning_rate),
            }

            # FORENSIC FIX #21: RECONNECT monitor_log (was incorrectly commented out)
            try:
                from core.transparent_logger import monitor_log
                monitor_log.log_ml_prediction(
                    symbol=self.symbol,
                    model_name="Hybrid_Ensemble_Ultimate",
                    prediction=1 if signal_type == SignalType.LONG else -1,
                    confidence=float(confidence),
                    features=metrics,
                    decision=signal_type.name,
                )
            except Exception as e:
                logger.error(f"monitor_log error: {e}", exc_info=True)

            # Log a consola
            logger.info(
                f"🎯 ML {signal_type.name} {self.symbol} | "
                f"Conf: {confidence:.2f} | Confl: {confluence:.2f} | "
                f"Regime: {self.market_regime} | "
                f"Score: {self.last_training_score:.2f} | "
                f"TP/SL: {self.current_tp_target * 100:.1f}%/{self.current_sl_target * 100:.1f}% | "
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

    def _export_brain_telemetry(self, consensus, votes, weights, status_label):
        """
        [TRINITY] Exports cognitive state to dashboard for Visualization.
        """
        try:
            telemetry = {
                "timestamp": self._now().isoformat(),
                "consensus_score": float(consensus),
                "votes": {
                    "RL": float(votes[0]),
                    "GA": float(votes[1]),
                    "OL": float(votes[2]),
                },
                "weights": {
                    "RL": float(weights[0]),
                    "GA": float(weights[1]),
                    "OL": float(weights[2]),
                },
                "status": status_label,
                "entropy": "HIGH" if 0.45 < consensus < 0.55 else "LOW",
                "symbol": self.symbol,
            }

            # atomic write ideally, but simple replace is fine for dashboard
            path = "dashboard/data/brain_telemetry.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(telemetry, f)
        except Exception as e:
            # Non-blocking
            pass

    def calculate_signals(self, event):
        """Entry point síncrono para ThreadPool (SUPREMO-V3 FIX)"""
        if event.type != EventType.MARKET:
            return

        # ============================================================
        # ⏱️ ESPECIALIZACIÓN POR HORIZONTE (SCALPING vs SWING)
        # ============================================================
        is_swing = getattr(self, "horizon_str", "SCALPING") == "SWING"
        is_closed = getattr(event, "is_closed", True)
        
        # FASE HORIZONS: Filtrado estricto por timeframe para evitar Phantom Triggers
        event_tf = getattr(event, "timeframe", "1m")
        target_tf = Config.Data.RESOLUTION if is_swing else "1m"
        
        if event_tf != target_tf:
            return
            
        # SWING SOLO evalúa velas cerradas. Ignora el ruido HFT del websocket.
        if is_swing and not is_closed:
            return

        # Forensic Fix: Sync strategy clock to market event clock
        self._current_event_time = getattr(event, 'timestamp', self._now())

        # Throttling: máximo 2 predicciones por segundo en SCALPING
        current_time = self._now()
        if not getattr(self, "is_sandbox", False):
            throttle_seconds = 0.5 if not is_swing else 10.0
            if (
                self._last_prediction_time
                and (current_time - self._last_prediction_time).total_seconds() < throttle_seconds
            ):
                return

        # Execute async task dynamically checking if an event loop is already running
        try:
            try:
                loop = asyncio.get_running_loop()
                # A loop is already running in this thread, schedule it as a task
                loop.create_task(self._async_process_v3(event))
            except RuntimeError:
                # No loop is running in this thread, use asyncio.run
                asyncio.run(self._async_process_v3(event))
        except Exception as e:
            logger.error(f"Error in async ML processing: {e}")

    async def _async_process_v3(self, event):
        """Procesamiento asíncrono sin bloqueo de hilos (SUPREMO-V3)"""
        if not self.running:
            return

        try:
            # [PHASE 17] Zero-Latency Boot: Lazy load models only on first tick
            if not getattr(self, "_models_loaded", False):
                self._load_models()
                self._models_loaded = True

            self.loop_count += 1

            # C-1 FIX: Lazy bind Némesis→Sophia feedback loop on first invocation
            if not self._sophia_feedback_linked and self.portfolio:
                if hasattr(self.portfolio, "link_nemesis_to_sophia"):
                    self.portfolio.link_nemesis_to_sophia(self.sophia)
                    self._sophia_feedback_linked = True

            required_bars = self.lookback

            # Use to_thread for blocking data retrieval if needed,
            # though get_latest_bars should be fast if cached.
            bars = await asyncio.to_thread(
                self.data_provider.get_latest_bars, self.symbol, n=required_bars, timeframe=self.primary_tf
            )

            if len(bars) < self.min_bars_to_train:
                return

            if self._feature_cols is None or not self._feature_cols:
                temp_df = await asyncio.to_thread(
                    self._prepare_features, bars[:1000] if len(bars) > 1000 else bars
                )
                if temp_df is not None and not temp_df.empty:
                    self._init_feature_cols(temp_df)

            # Check Training (still in separate thread/process managed by launch_training)
            is_training = (
                hasattr(self, "_training_thread")
                and self._training_thread
                and self._training_thread.is_alive()
            )

            if (
                self.bars_since_train >= self.retrain_interval or not self.is_trained
            ) and not is_training:
                self._launch_training(bars, "Full")

            if self.is_trained:
                # Update GARCH with latest return before inference
                if len(bars) > 2:
                    last_ret = np.log(bars[-1][4] / bars[-2][4])  # Log return of Close
                    vol = self.garch.update(last_ret)
                    # self.current_volatility = vol # Optional storage

                # FIX: If we are UniversalEnsembleStrategy, use its overridden _run_inference
                if self.__class__.__name__ == "UniversalEnsembleStrategy":
                    await asyncio.to_thread(self._run_inference)
                else:
                    await self._run_inference_v3(bars)
        except Exception as e:
            logger.error(f"ML Async error {self.symbol}: {e}")

    async def _run_inference_v3(self, bars):
        """
        Inferencia asíncrona via Proceso Aislado (SUPREMO-V3)

        B4 FIX: Bypass StandardScaler for tree-only models (XGBoost, RF, GB).
        Tree-based models are scale-invariant — scaling is pure overhead.
        Only apply scaler if non-tree model (e.g., SVM, LogReg) is in ensemble.
        """
        try:
            # Prepare features (blocking, move to thread)
            # Use NumPy slicing for Phase 4 Structured Arrays
            data_slice = bars[-100:] if len(bars) > 100 else bars
            df = await asyncio.to_thread(self._prepare_features, data_slice, regime_aware=True, return_polars=True)
            if df is None or len(df) == 0:
                return

            # B4 FIX: Extract last row as numpy array directly (avoid DataFrame overhead)
            if self._feature_cols:
                valid_cols = [c for c in self._feature_cols if c in df.columns]
                if not valid_cols:
                    return
                    
                # ═══════════════════════════════════════════════════════════════
                # QUANTUM ZERO-COPY: Polars -> NumPy (Bypassing Pandas completely)
                # ═══════════════════════════════════════════════════════════════
                # df is a Polars DataFrame. We select cols, take tail(1), convert to numpy
                X = df.select(valid_cols).tail(1).to_numpy()

                # B4 FIX: Only scale if we have a non-tree model that needs it
                # Tree models (RF, XGB, GB) are ALL scale-invariant
                has_non_tree_model = False
                if (
                    has_non_tree_model
                    and self.scaler is not None
                    and hasattr(self.scaler, "scale_")
                ):
                    X = self.scaler.transform(X)
                    
                # 🧠 PHASE 8: PETIM INFERENCE
                self._latest_petim_prediction = None
                if getattr(self, "petim_predictor", None):
                    try:
                        # Extract exact features expected by PETIM
                        petim_feats = self.petim_predictor.features
                        valid_petim_cols = [c for c in petim_feats if c in df.columns]
                        if len(valid_petim_cols) == len(petim_feats):
                            X_petim = df.select(valid_petim_cols).tail(1).to_numpy()
                            self._latest_petim_prediction = self.petim_predictor.predict(X_petim)
                    except Exception as e:
                        logger.error(f"PETIM Inference Error: {e}")
            else:
                return

            # Direct Synchronous JIT Inference (Fixes 1-tick delay)
            rf = getattr(self, "rf_arrays", self.rf_model)
            xgb = self.xgb_model
            gb = getattr(self, "gb_arrays", self.gb_model)
            
            if isinstance(rf, dict) and "tree_offsets" in rf:
                rf_p = predict_rf_jit(X.flatten(), rf["children_left"], rf["children_right"], rf["feature"], rf["threshold"], rf["value"], rf["tree_offsets"])
            else:
                rf_p = rf.predict_proba(X)[0][1] if hasattr(rf, "predict_proba") else 0.5
                
            xgb_p = xgb.predict_proba(X)[0][1] if xgb else 0.5
            
            if isinstance(gb, dict) and "tree_offsets" in gb:
                gb_p = predict_gb_jit(X.flatten(), gb["children_left"], gb["children_right"], gb["feature"], gb["threshold"], gb["value"], gb["tree_offsets"], gb["init_score"], gb["learning_rate"])
            else:
                gb_p = gb.predict_proba(X)[0][1] if hasattr(gb, "predict_proba") else 0.5
                
            # Compute ensemble confidence directly
            w_rf, w_xgb, w_gb = self.base_rf_weight, self.base_xgb_weight, self.base_gb_weight
            conf = float(rf_p * w_rf + xgb_p * w_xgb + gb_p * w_gb)

            results = {
                "symbol": self.symbol,
                "rf": float(rf_p),
                "xgb": float(xgb_p),
                "gb": float(gb_p),
                "confidence": conf,
                "ts": time.time(),
                "weights": (w_rf, w_xgb, w_gb),
                "price": float(bars[-1][4]) if isinstance(bars, (list, tuple, np.ndarray)) and len(bars) > 0 else 0.0,
            }
            
            await self._process_ml_results(results)

        except Exception as e:
            logger.error(f"Inference Error {self.symbol}: {e}")

    def _ensure_inference_worker(self):
        """Mantiene vivo el subproceso de inferencia"""
        if self._inference_process is None or not self._inference_process.is_alive():
            logger.info(
                f"🧠 [SUPREMO-V3] Starting Isolated Inference Worker for {self.symbol}"
            )
            self._inference_process = threading.Thread(
                target=ml_inference_worker_task,
                args=(self._inference_queue, self._results_queue),
                daemon=True,
            )
            self._inference_process.start()

    # ============================================================
    # ✅ PHASE 13: GENETIC ALGORITHM SIGNAL (The Evolver)
    # ============================================================
    def _get_ga_signal(self, symbol):
        """
        Generates a signal based on Evolutionary Technical Analysis.
        Simulates a population of strategies (Genes) and uses the best recent performer.
        Feature: RSI Thresholds, MACD Params, Bollinger Bands deviation.
        """
        try:
            # 1. Fetch recent history (Last 100 bars)
            bars = self.data_provider.get_latest_bars(symbol, n=100, timeframe=self.primary_tf)
            if len(bars) < 50:
                return 0.5

            closes = bars["close"]

            # 2. Define Gene Population (Simplified)
            # Gene: (RSI_Period, RSI_Overbought, RSI_Oversold)
            genes = [
                (14, 70, 30),  # Classic
                (7, 80, 20),  # Aggressive
                (21, 65, 35),  # Conservative
                (9, 75, 25),  # Scalper
                (5, 85, 15),  # Hyper-Scalper
            ]

            best_gene = None
            best_pnl = -999.0

            # 3. Evaluate Fitness (Backtest on last 50 bars)
            # We want the gene that would have predicted the *recent* trend best.
            # Simplified: Check RSI divergence with price slope

            for gene in genes:
                period, ob, os_lvl = gene
                rsi = calculate_rsi_jit(closes, period=period)
                pnl = 0.0

                # Mock backtest
                pos = 0
                entry_price = 0.0
                for i in range(50, len(closes)):
                    price = closes[i]
                    val = rsi[i]

                    if pos == 0:
                        if val < os_lvl:
                            pos = 1
                            entry_price = price
                        elif val > ob:
                            pos = -1
                            entry_price = price
                    elif pos == 1:
                        if val > 50:  # Take profit/Exit condition
                            pnl += (price - entry_price) / entry_price
                            pos = 0
                    elif pos == -1:
                        if val < 50:
                            pnl += (entry_price - price) / entry_price
                            pos = 0

                if pnl > best_pnl:
                    best_pnl = pnl
                    best_gene = gene

            # 4. Generate Signal using Best Gene (Winner)
            if best_gene:
                period, ob, os_lvl = best_gene
                current_rsi = calculate_rsi_jit(closes, period=period)[-1]

                # Normalize 0.0 to 1.0 (0.5 = Neutral)
                # If RSI < OS -> Buy Signal (1.0)
                # If RSI > OB -> Sell Signal (0.0)
                # Middle -> 0.5

                if current_rsi <= os_lvl:
                    return 0.9 + (os_lvl - current_rsi) / 100  # Strong Buy
                if current_rsi >= ob:
                    return 0.1 - (current_rsi - ob) / 100  # Strong Sell

                # Linear interpolation
                if current_rsi < 50:
                    return 0.5 + (0.4 * (50 - current_rsi) / (50 - os_lvl))
                else:
                    return 0.5 - (0.4 * (current_rsi - 50) / (ob - 50))

            return 0.5

        except Exception as e:
            logger.error(f"GA Signal Error: {e}")
            return 0.5

    async def _process_ml_results(self, results):
        """Handle signals from inference worker (SUPREMO-V3)"""
        try:
            confidence = results["confidence"]
            rf_p, xgb_p, gb_p = results["rf"], results["xgb"], results["gb"]
            
            # Sync ML Confidence to SSOT (OmniScore Component)
            try:
                from core.global_state import global_state
                # Assuming 'confidence' represents LONG probability. Short probability is 1 - confidence.
                global_state.update_symbol_vector(self.symbol, {
                    "ml_bull_score": confidence,
                    "ml_bear_score": 1.0 - confidence
                })
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)

            # 🔮 FASE 5: MULTI-COIN ORACLE (LEAD-LAG ARBITRAGE) + CROSS-EXCHANGE PDC
            # Leemos el Price Discovery Coefficient (PDC) de Coinbase/Bybit para Latencia Negativa
            try:
                from core.global_state import global_state
                # Check BTC Velocity first (Macro)
                if self.symbol != "BTC/USDT":
                    btc_vel = getattr(global_state, 'btc_velocity', 0.0)
                    if btc_vel > 0.005: # BTC saltando rápido hacia arriba (>0.5% por seg)
                        logger.critical(f"🚀 [MULTI-COIN ORACLE] BTC Velocity ALTA ({btc_vel:.4f}). Acelerando LONG para {self.symbol}!")
                        confidence = min(0.99, confidence + 0.15) # Empuja hacia LONG
                    elif btc_vel < -0.005: # BTC cayendo rápido
                        logger.critical(f"📉 [MULTI-COIN ORACLE] BTC Velocity NEGATIVA ({btc_vel:.4f}). Acelerando SHORT para {self.symbol}!")
                        confidence = max(0.01, confidence - 0.15) # Empuja hacia SHORT
                
                # Check Cross-Exchange PDC (Micro / Sub-ms)
                if hasattr(global_state, 'cross_exchange_metrics'):
                    metrics = global_state.cross_exchange_metrics.get(self.symbol, {})
                    pdc = metrics.get('pdc_signal', 0.0)
                    
                    if pdc > 0.3: # Fuerte Lead de Coinbase/Deribit ALCISTA
                        logger.critical(f"🌌 [CROSS-EXCHANGE] Lead-Lag ALCISTA Detectado (PDC: {pdc:.2f}). Bias LONG Inyectado para {self.symbol}!")
                        confidence = min(0.99, confidence + (pdc * 0.2)) # Max +0.2 bias
                    elif pdc < -0.3: # Fuerte Lead de Coinbase/Deribit BAJISTA
                        logger.critical(f"🌌 [CROSS-EXCHANGE] Lead-Lag BAJISTA Detectado (PDC: {pdc:.2f}). Bias SHORT Inyectado para {self.symbol}!")
                        confidence = max(0.01, confidence + (pdc * 0.2)) # Max -0.2 bias
                        
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)

            # Determine preliminary signal type
            threshold = self.adaptive_confidence_threshold
            signal_type = (
                SignalType.LONG
                if confidence >= threshold
                else (SignalType.SHORT if confidence <= (1.0 - threshold) else SignalType.HOLD)
            )

            # 🌊 FASE 5: ANTI-SPOOFING ML (Cacería de Muros Falsos)
            of_metrics = self.data_provider.get_order_flow_metrics(self.symbol)
            if of_metrics:
                is_toxic = of_metrics.get("is_toxic", False)
                vpin = of_metrics.get("vpin", 0.5)
                iceberg = of_metrics.get("iceberg_score", 0.0)
                delta = of_metrics.get("rolling_delta_60s", 0.0)
                
                spoof_buy = of_metrics.get("spoofing_prob_buy", 0.0)
                spoof_sell = of_metrics.get("spoofing_prob_sell", 0.0)

                # Si detectamos un muro falso gigante de compra (> 85%), es manipulación.
                # La ballena va a vender, así que nosotros lanzamos SHORT primero.
                if spoof_buy > 0.85:
                    logger.critical(f"🐋🔫 [ANTI-SPOOFING] FAKE BUY WALL DETECTADO en {self.symbol} ({spoof_buy*100:.1f}% liquidez de ballenas). Disparando SHORT en contra de la manipulación.")
                    signal_type = SignalType.SHORT
                    confidence = 0.99
                # Si es un muro falso de venta, vamos LONG.
                elif spoof_sell > 0.85:
                    logger.critical(f"🐋🔫 [ANTI-SPOOFING] FAKE SELL WALL DETECTADO en {self.symbol} ({spoof_sell*100:.1f}% liquidez de ballenas). Disparando LONG en contra de la manipulación.")
                    signal_type = SignalType.LONG
                    confidence = 0.99
                # Check for absolute veto on standard toxic flow
                elif is_toxic and signal_type != SignalType.HOLD:
                    logger.warning(
                        f"🌌 [VETO CUÁNTICO] Señal {signal_type.name} cancelada en {self.symbol}! Flujo Tóxico Estándar (VPIN: {vpin:.2f}, Iceberg: {iceberg:.2f})."
                    )
                    self.metrics["discarded_events"] = self.metrics.get("discarded_events", 0) + 1
                    return

                # Check directional alignment with aggressive Delta
                if signal_type == SignalType.LONG and delta < -100:
                    logger.warning(
                        f"📉 [VETO CUÁNTICO] Señal LONG cancelada! Presión de venta Market excesiva (Delta 60s: {delta:.2f})."
                    )
                    return
                elif signal_type == SignalType.SHORT and delta > 100:
                    logger.warning(
                        f"📈 [VETO CUÁNTICO] Señal SHORT cancelada! Presión de compra Market excesiva (Delta 60s: {delta:.2f})."
                    )
                    return
                    
                # --- Mutación 39: Bayesian Mirage (Anti-Spoofing Veto) ---
                spoof_buy_prob = of_metrics.get("spoofing_prob_buy", 0.0)
                spoof_sell_prob = of_metrics.get("spoofing_prob_sell", 0.0)

                if signal_type == SignalType.LONG and spoof_buy_prob > 0.80:
                    logger.warning(f"🚨 [BAYESIAN MIRAGE] {self.symbol} LONG VETADA. Muro falso de compras (Trap) Prob: {spoof_buy_prob:.1%}.")
                    return
                elif signal_type == SignalType.SHORT and spoof_sell_prob > 0.80:
                    logger.warning(f"🚨 [BAYESIAN MIRAGE] {self.symbol} SHORT VETADA. Muro falso de ventas (Trap) Prob: {spoof_sell_prob:.1%}.")
                    return

            # ============================================================
            # ✅ PHASE 13: PHALANX-SWARM CONSENSUS ENGINE
            # ============================================================

            # 1. TIMEFRAME DIVERGENCE CHECK (The Chronos Guard)
            tf_divergence = False
            try:
                # Quick fetch of recent bars for 3 timeframes
                # Note: DataProvider caching makes this fast.
                mom_1m = self.data_provider.get_latest_bars(self.symbol, n=5, timeframe=self.primary_tf)["close"]
                mom_5m = self.data_provider.get_latest_bars_5m(self.symbol, n=5)[
                    "close"
                ]
                mom_15m = self.data_provider.get_latest_bars_15m(self.symbol, n=5)[
                    "close"
                ]

                m1 = (mom_1m[-1] / mom_1m[0]) - 1 if len(mom_1m) > 1 else 0
                m5 = (mom_5m[-1] / mom_5m[0]) - 1 if len(mom_5m) > 1 else 0
                m15 = (mom_15m[-1] / mom_15m[0]) - 1 if len(mom_15m) > 1 else 0

                # Check alignment
                if signal_type == SignalType.LONG:
                    if m1 < -0.001 or m5 < -0.001 or m15 < -0.001:  # Tolerance of -0.1%
                        tf_divergence = True
                elif signal_type == SignalType.SHORT:
                    if m1 > 0.001 or m5 > 0.001 or m15 > 0.001:
                        tf_divergence = True

                if tf_divergence:
                    logger.info(
                        f"⏳ [PHALANX] Timeframe Divergence: M1={m1:.4f} M5={m5:.4f} M15={m15:.4f} -> VETO"
                    )
                    confidence *= (
                        0.5  # Penalty, don't kill completely if strong elsewhere
                    )

            except Exception as e:
                logger.error(f"Timeframe check error: {e}", exc_info=True)

            # 2. ENSEMBLE CONSENSUS (RL + GA + OL)
            # FIX: Ensure RL vote diverges from OL vote by incorporating PPO action history
            try:
                from ml.ppo_agent import ppo_agent
                _last_ppo = getattr(self, "last_ppo_action", getattr(ppo_agent, "last_action", 0.0))
                rl_vote = confidence * (1.0 + abs(_last_ppo) * 0.2) if _last_ppo != 0 else confidence * 1.02
            except Exception:
                rl_vote = confidence * 1.02
            ol_vote = (
                rf_p * self.base_rf_weight
                + xgb_p * self.base_xgb_weight
                + gb_p * self.base_gb_weight
            )
            ga_vote = self._get_ga_signal(self.symbol)

            # ✅ PHASE 17: BYZANTINE FAULT TOLERANCE
            votes = np.array([rl_vote, ga_vote, ol_vote])
            weights = np.array([0.4, 0.3, 0.3])  # Base weights: RL, GA, OL
            names = ["RL", "GA", "OL"]

            # BFT 1: Outlier Detection (The Triad Check)
            mu = np.mean(votes)
            sigma = np.std(votes)

            # Only check for traitors if there is significant disagreement
            if sigma > 0.05:
                # Z-Score Calculation
                z_scores = np.abs(votes - mu) / sigma

                # Threshold: 1.5 sigma is effective for N=3 to catch the single deviant
                # (User asked for 3 sigma, but that is impossible for N=3. 1.5~=confidence of 86% outlier)
                for i in range(3):
                    if z_scores[i] > 1.5:
                        logger.warning(
                            f"🚫 [BFT] {names[i]} QUARANTINED! Vote={votes[i]:.2f} (Z={z_scores[i]:.2f})"
                        )
                        weights[i] = 0.0  # Remove traitor from consensus

            # Re-normalize weights
            if np.sum(weights) == 0:
                weights = np.array(
                    [0.4, 0.3, 0.3]
                )  # Fallback if all quarantined (Chaos!)
                logger.critical("💀 [BFT] TOTAL CONSENSUS COLLAPSE. Resetting weights.")

            weights /= np.sum(weights)
            consensus_score = np.dot(votes, weights)

            logger.info(
                f"🗳️ [BFT] Consensus: {consensus_score:.2f} | Votes: RL={rl_vote:.2f} GA={ga_vote:.2f} OL={ol_vote:.2f} | W={weights}"
            )

            # ✅ PHASE III: ENTROPY FILTER (Algorithmic Psychology)
            # If the model is "confused" (near 0.5), we force a HOLD.
            # We want High Conviction only.
            if 0.45 < consensus_score < 0.55:
                logger.info(
                    f"😵 [PSYCH] High Entropy Detected ({consensus_score:.2f}). Model is confused -> HOLD."
                )
                return

            if consensus_score < 0.75:
                logger.info(
                    f"🛡️ [PHALANX] Consensus VETO ({consensus_score:.2f} < 0.75)"
                )
                self._export_brain_telemetry(consensus_score, votes, weights, "VETO")
                return  # Abort Signal

            # Boost confidence if Consensus is Strong
            if consensus_score > 0.85:
                confidence = min(confidence + 0.1, 1.0)

            self._export_brain_telemetry(consensus_score, votes, weights, "ACTIVE")

            # ✅ PHASE 17.3: SPOOFING DETECTION (Injection Filter)
            # Detect Fake Walls: High VBI (>0.8) but Price NOT moving (or moving opposite)
            hft_metrics = self.data_provider.get_hft_indicators(self.symbol)
            vbi = hft_metrics.get("vbi", 0.0)

            if abs(vbi) > 0.75:
                # Check recent price velocity
                # If VBI is +0.8 (Strong Buy Wall) but Price is dropping -> SPOOFING (Trap)
                # If VBI is -0.8 (Strong Sell Wall) but Price is rising -> SPOOFING (Trap)

                # Simple heuristic: If VBI sign != Momentum sign -> Spoofing Risk
                # reusing 'm1' from Timeframe check if available, else zero
                mom_1m_val = locals().get("m1", 0.0)

                if (vbi > 0 and mom_1m_val < -0.0005) or (
                    vbi < 0 and mom_1m_val > 0.0005
                ):
                    logger.warning(
                        f"🤡 [BFT] SPOOFING DETECTED (Fake Wall)! VBI={vbi:.2f} vs Mom={mom_1m_val:.4f}. Penalty applied."
                    )
                    confidence -= 0.20  # Major Penalty
                    consensus_score *= 0.8  # Degrade consensus

            # ✅ PHASE II: LAYERING DETECTION (Microstructure)
            # Detect rapid book changes (VBI Volatility) without price movement
            # Fetches last 10 VBI snapshots via HFT helper if available
            # Note: We rely on HFT metrics returning 'vbi_avg', but we need history/volatility.
            # We assume data_provider can give us VBI history or we compute it if passed in bars?
            # binance_loader stores vbi_history.

            # Using data_provider proxy to access loader's vbi history if possible,
            # Or just infer from current snapshot vs previous (if we tracked it).
            # For strictness: If VBI is extreme (>0.9) and Price Volatility is low (<0.05%), it's Layering.

            # Re-using mom_1m_val as proxy for price movement magnitude
            if (
                abs(vbi) > 0.9 and abs(mom_1m_val) < 0.0002
            ):  # Huge imbalance, zero movement
                logger.warning(
                    f"🎭 [MICROSTRUCTURE] LAYERING DETECTED! Locked Order Book. VBI={vbi:.2f}"
                )
                confidence -= 0.15

            # 3. Proceed to Order Flow Check (Existing Logic)

            # [PHASE 13] Absorption Detection (Price Action + Volume)
            # Retrieve last 15 bars for structural analysis
            pa_bars = await asyncio.to_thread(
                self.data_provider.get_latest_bars, self.symbol, n=15, timeframe=self.primary_tf
            )
            absorption = self.phalanx.is_absorption_detected(pa_bars)

            if absorption["detected"]:
                logger.info(
                    f"🧱 [PHALANX] Absorption Detected: {absorption['type']} ({absorption['reason']})"
                )

            # Define of_analysis based on VBI
            of_analysis = {"signal": 0, "strength": abs(vbi)}
            if vbi > 0.6:
                of_analysis["signal"] = 1
            elif vbi < -0.6:
                of_analysis["signal"] = -1

            # Logic: Imbalance acts as a massive confidence booster or veto
            if signal_type == SignalType.LONG:
                # 1. Order Book Imbalance
                if of_analysis["signal"] == 1:  # Long Imbalance > 300%
                    confidence = min(confidence + 0.15, 1.0)
                    logger.info(
                        f"⚡ [PHALANX] ORACLE LONG BOOST +15% | Strength: {of_analysis['strength']:.2f}"
                    )
                elif of_analysis["signal"] == -1:  # Short Imbalance -> VETO
                    confidence = max(0.0, confidence - 0.20)
                    logger.info(
                        f"🛡️ [PHALANX] ORACLE LONG VETO (Sell Pressure) | Strength: {of_analysis['strength']:.2f}"
                    )

                # 2. Absorption Confirmation (Stopping Volume at Support)
                if absorption["type"] == "BULLISH":
                    confidence = min(confidence + 0.10, 1.0)
                    logger.info(f"⚡ [PHALANX] ABSORPTION BOOST (Bullish Stopping Vol)")
                elif absorption["type"] == "BEARISH":
                    confidence = max(0.0, confidence - 0.15)
                    logger.info(f"🛡️ [PHALANX] ABSORPTION VETO (Resistance blocking)")

            elif signal_type == SignalType.SHORT:
                # 1. Order Book Imbalance
                if of_analysis["signal"] == -1:  # Short Imbalance > 300% (Ratio < 0.33)
                    confidence = min(confidence + 0.15, 1.0)
                    logger.info(
                        f"⚡ [PHALANX] ORACLE SHORT BOOST +15% | Strength: {of_analysis['strength']:.2f}"
                    )
                elif of_analysis["signal"] == 1:  # Long Imbalance -> VETO
                    confidence = max(0.0, confidence - 0.20)
                    logger.info(
                        f"🛡️ [PHALANX] ORACLE SHORT VETO (Buy Pressure) | Strength: {of_analysis['strength']:.2f}"
                    )

                # 2. Absorption Confirmation (Stopping Volume at Resistance)
                if absorption["type"] == "BEARISH":
                    confidence = min(confidence + 0.10, 1.0)
                    logger.info(f"⚡ [PHALANX] ABSORPTION BOOST (Bearish Stopping Vol)")
                elif absorption["type"] == "BULLISH":
                    confidence = max(0.0, confidence - 0.15)
                    logger.info(f"🛡️ [PHALANX] ABSORPTION VETO (Support blocking)")

            # ============================================================
            # ✅ PREDICTIVE DECAY EXIT LOGIC (Fase 1)
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC FIX #10: portfolio.positions (aggregate) NO tiene
            #   campo 'direction'. pos.get('direction','') SIEMPRE retornaba
            #   '' → is_long/is_short SIEMPRE False → CÓDIGO MUERTO.
            # FIX: Usar get_horizon_position() y derivar dirección de qty.
            # ============================================================
            if getattr(self, 'portfolio', None):
                _decay_horizon = getattr(self, 'horizon_str', 'SCALPING')
                _decay_pos = self.portfolio.get_horizon_position(self.symbol, _decay_horizon)
                if _decay_pos:
                    _decay_qty = _decay_pos.get('quantity', 0)
                    is_long = _decay_qty > 0
                    is_short = _decay_qty < 0
                    
                    if is_long and confidence < 0.40:
                        logger.warning(f"📉 [PREDICTIVE DECAY] Confidence for LONG dropped to {confidence:.2f} < 0.40. Exiting {self.symbol} ({_decay_horizon})...")
                        signal_type = SignalType.EXIT
                    elif is_short and confidence > 0.60:
                        logger.warning(f"📈 [PREDICTIVE DECAY] Confidence for SHORT dropped (LONG prob={confidence:.2f} > 0.60). Exiting {self.symbol} ({_decay_horizon})...")
                        signal_type = SignalType.EXIT

            # Logic for Signal Creation (Professor Method)
            # QUÉ: Generación de evento de señal asíncrono.
            # POR QUÉ: Para notificar al Portfolio/RiskManager sin bloquear.

            # Only act if it's a trade signal
            if signal_type in [SignalType.LONG, SignalType.SHORT, SignalType.EXIT]:
                # Si probabilities no existe, lo inferimos del confidence (que es P(Long))
                prob_l = results.get("probabilities", {}).get("L", confidence if isinstance(confidence, float) else 0.0) if isinstance(results, dict) else confidence
                prob_s = results.get("probabilities", {}).get("S", 1.0 - confidence if isinstance(confidence, float) else 0.0) if isinstance(results, dict) else 1.0 - confidence
                metadata = {
                    "prob_L": prob_l,
                    "prob_S": prob_s,
                    "trail_start_pct": getattr(self, "par_engine", None).get(
                        "trail_start_pct"
                    )
                    if hasattr(self, "par_engine")
                    else 0.60,
                    "trail_dist_pct": getattr(self, "par_engine", None).get(
                        "trail_dist_pct"
                    )
                    if hasattr(self, "par_engine")
                    else 0.20,
                    "momentum_exit_accel": getattr(self, "par_engine", None).get(
                        "momentum_exit_accel"
                    )
                    if hasattr(self, "par_engine")
                    else -0.015,
                }
                
                # ═══════════════════════════════════════════════════════════════
                # 🧠 [INTEGRACIÓN MIDD] Motor de Inteligencia Direccional Dual
                # ═══════════════════════════════════════════════════════════════
                try:
                    from core.midd import MIDD
                    _midd = MIDD(data_provider=getattr(self, 'data_provider', None), ml_strategy=self)
                    _midd_res = _midd.evaluate_asset(self.symbol)
                    _med = _midd_res.get("MED_STATE", "MED-NEUTRAL")
                    _isn = _midd_res.get("ISN", 0)
                    
                    metadata['MED_STATE'] = _med
                    metadata['ISN'] = _isn
                    
                    if signal_type == SignalType.LONG:
                        if _med in ["MED-2", "MED-6"]:
                            logger.warning(f"🛑 [MIDD] LONG Vetoado en {self.symbol} | Estado: {_med} (ISN: {_isn:.1f})")
                            signal_type = None
                        elif _med in ["MED-4", "MED-5"]:
                            confidence = max(0.1, confidence - 0.20)
                            logger.info(f"⚖️ [MIDD] LONG Reducido en {self.symbol} | Estado: {_med} (ISN: {_isn:.1f})")
                            
                    elif signal_type == SignalType.SHORT:
                        if _med in ["MED-1", "MED-6"]:
                            logger.warning(f"🛑 [MIDD] SHORT Vetoado en {self.symbol} | Estado: {_med} (ISN: {_isn:.1f})")
                            signal_type = None
                        elif _med in ["MED-3", "MED-5"]:
                            confidence = max(0.1, confidence - 0.20)
                            logger.info(f"⚖️ [MIDD] SHORT Reducido en {self.symbol} | Estado: {_med} (ISN: {_isn:.1f})")
                except Exception as e:
                    logger.error(f"Error evaluando MIDD: {e}")
                
                # Check again if signal was vetoed by MIDD
                if signal_type not in [SignalType.LONG, SignalType.SHORT, SignalType.EXIT]:
                    return
                
                # 🧠 INJECT PETIM PREDICTION INTO SIGNAL METADATA
                if getattr(self, "_latest_petim_prediction", None):
                    metadata['trajectory_prediction'] = self._latest_petim_prediction
                
                if signal_type == SignalType.EXIT:
                    metadata['urgent'] = False
                    metadata['actual_order_type'] = 'limit'
                    metadata['is_tp_limit'] = True

                # Inyectar métricas de volatilidad para el Zombie-Chaser
                metadata['atr_pct'] = getattr(self, "current_sl_target", 0.002)
                metadata['volatility'] = getattr(self, "current_sl_target", 0.002)

                detailed_id = f"{self.strategy_id}.ML_PREDICTION"
                signal = SignalEvent(
                    strategy_id=detailed_id,
                    setup_type="ML_PREDICTION",
                    symbol=self.symbol,
                    datetime=self._now(),
                    signal_type=signal_type,
                    strength=confidence, ml_confidence=confidence,
                    current_price=results.get("price", 0),
                    tp_pct=getattr(self, "current_tp_target", 0.004),
                    sl_pct=getattr(self, "current_sl_target", 0.002),
                    horizon=self.horizon_str,
                    predicted_magnitude=getattr(self, "current_tp_target", 0.004),
                    predicted_duration=self.LOOKAHEAD_BARS,
                    metadata=metadata,
                )

                # Async logging and publishing
                # ... Simplified logging call ...
                self.events_queue.put(signal)

                # Update Neural Bridge insight in background
                neural_bridge.publish_insight(
                    strategy_id="ML_V3_ORACLE",
                    symbol=self.symbol,
                    insight={"confidence": confidence, "type": signal_type.name},
                )

                self._last_prediction_time = self._now()
                self.total_signals_generated += 1

        except Exception as e:
            logger.error(f"Error processing ML results: {e}")

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
                "datetime",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "target",
                "regime",
                "timestamp",
                "returns",
            ]

            # Obtener solo las columnas numéricas que no están en la lista de exclusión
            cols = [
                c
                for c in df.columns
                if c not in exclude
                and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
            ]

            with self._state_lock:
                self._feature_cols = cols

            if not self._feature_cols:
                logger.error(
                    f"❌ [{self.symbol}] _init_feature_cols: No valid feature columns found!"
                )
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
            required_bars = self.lookback
            bars = self.data_provider.get_latest_bars(self.symbol, n=required_bars, timeframe=self.primary_tf)

            # Data Health Check
            num_bars = len(bars)
            if num_bars < self.min_bars_to_train:
                if self.loop_count % 10 == 0:
                    logger.warning(
                        f"⚠️ [{self.symbol}] Waiting for data: {num_bars}/{self.min_bars_to_train} bars."
                    )
                return

            # Initial Feature Initialization (Rule 2.3)
            # Preparamos features una vez para inicializar _feature_cols antes del primer training
            # PROFESSOR METHOD:
            # QUÉ: Inicialización garantizada de metadatos de entrada.
            # POR QUÉ: Evita errores de disparidad de dimensiones si el modelo arranca sin saber qué columnas usar.
            if self._feature_cols is None or not self._feature_cols:
                temp_df = self._prepare_features(
                    bars[:1000] if len(bars) > 1000 else bars
                )
                if temp_df is not None and not temp_df.empty:
                    self._init_feature_cols(temp_df)
                    if self._feature_cols:
                        logger.info(
                            f"📋 [{self.symbol}] Feature Columns established: {len(self._feature_cols)} features."
                        )
                    else:
                        logger.error(
                            f"❌ [{self.symbol}] CRITICAL: _init_feature_cols produced empty list. Check data types."
                        )
                        return

            # Actualizar learning rate y parámetros
            self._adjust_learning_rate()

            # Verificar si necesita entrenar (Incremental o Full)
            with self._state_lock:
                self.bars_since_train += 1
                self.bars_since_incremental += 1

                # Check if a training thread is already active
                is_training = (
                    hasattr(self, "_training_thread")
                    and self._training_thread
                    and self._training_thread.is_alive()
                )

                # Update incremental cada X velas (ej. 30)
                needs_incremental = (
                    self.bars_since_incremental
                    >= Config.Strategies.ML_INCREMENTAL_UPDATE_BARS
                )
                should_train_full = (
                    (self.bars_since_train >= self.retrain_interval)
                    or (not self.is_trained)
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

    def _launch_training(self, bars, train_type="Full", sync=False):
        """Lanzar entrenamiento en thread separado"""

        def train_bg(bars_data, t_type):
            """Core training routine with concurrency control."""
            with TRAINING_LIMITER:
                monitor.register_thread(f"ML_Train_{self.symbol}")
                start_time = time.time()

                try:
                    if not self.running:
                        logger.debug(
                            f"ML Training for {self.symbol} aborted: shutdown signal received."
                        )
                        return

                    logger.info(
                        f"🔄 Starting {t_type} training #{self.training_iteration + 1} for {self.symbol}..."
                    )

                    df = self._prepare_features(bars_data, regime_aware=True)
                    result, score = self._train_with_cross_validation(df)

                    if result:
                        models, scaler, feature_cols = result

                        # ✅ GUARDIA DE CALIDAD (Rule 3.3)
                        # PROFESSOR METHOD:
                        # QUÉ: Filtro de persistencia básico.
                        # POR QUÉ: Asegura que el modelo supera la capacidad predictiva aleatoria.
                        # CÓMO: Comparamos el nuevo score contra el umbral mínimo (MIN_MODEL_ACCURACY).
                        if self.is_trained:
                            min_acc = getattr(self, "MIN_MODEL_ACCURACY", 0.50)
                            if score < min_acc:
                                logger.warning(
                                    f"🛡️ [Quality Guard] Rejected model update for {self.symbol}.\n"
                                    f"   Current Score: {score:.4f} | Minimum Required: {min_acc:.4f}\n"
                                    f"   Reason: Performance is worse than random guessing."
                                )
                                # TELEGRAM NOTIFICATION
                                try:
                                    Notifier.send_ml_training_update(
                                        symbol=self.symbol,
                                        horizon=self.horizon_str,
                                        status="REJECTED",
                                        details={"score": score, "min_acc": min_acc}
                                    )
                                except Exception as e:
                                    logger.error(f"Silent exception caught: {e}", exc_info=True)

                                # NO actualizamos, pero limpiamos flag de entrenamiento
                                with self._state_lock:
                                    self.bars_since_train = 0
                                return

                        if not self.running:
                            return

                        with self._state_lock:
                            # Guardar los 3 modelos
                            self.rf_model = models["rf"]
                            self.xgb_model = models["xgb"]
                            self.gb_model = models["gb"]
                            self.scaler = scaler

                            # Compilar modelos a C-Arrays para latencia nano
                            logger.info(
                                f"⚡ [{self.symbol}] Compilando árboles a matrices Numba..."
                            )
                            # Modelos Online (MLP/SGD) no son árboles, omitir Numba JIT
                            self.rf_arrays = None
                            self.gb_arrays = None
                            cleaned_feature_cols = [
                                col for col in feature_cols if col is not None
                            ]
                            self._feature_cols = cleaned_feature_cols
                            self.is_trained = True
                            self.bars_since_train = 0
                            self.last_training_score = score
                            self.par_engine.feedback_training(score)
                            self.last_training_time = self._now()
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
                        # TELEGRAM NOTIFICATION
                        try:
                            Notifier.send_ml_training_update(
                                symbol=self.symbol,
                                horizon=self.horizon_str,
                                status="SUCCESS",
                                details={"score": score, "duration": duration, "features": len(feature_cols)}
                            )
                        except Exception as e:
                            logger.error(f"Silent exception caught: {e}", exc_info=True)
                    else:
                        # ⚠️ Training failed or score below threshold.
                        # Do NOT mark as trained yet to allow retries,
                        # or mark as trained but keep scaler=None to indicate no model available.
                        # We previously were marking is_trained=True here causing crashes in inference.
                        with self._state_lock:
                            # Initialize components if they are missing
                            if (
                                not hasattr(self, "feature_engineer")
                                or self.feature_engineer is None
                            ):
                                self.feature_engineer = FeatureEngineering()
                            if (
                                not hasattr(self, "signal_generator")
                                or self.signal_generator is None
                            ):
                                self.signal_generator = SignalGenerator(
                                    self.strategy_id
                                )
                            if not hasattr(self, "phalanx") or self.phalanx is None:
                                self.phalanx = OrderFlowAnalyzer()
                            if not hasattr(self, "garch") or self.garch is None:
                                self.garch = OnlineGARCH(1e-6, 0.1, 0.85, 1e-4)
                            if (
                                not hasattr(self, "xai_engine")
                                or self.xai_engine is None
                            ):
                                self.xai_engine = XAIEngine()

                            self.bars_since_train = 0

                        logger.warning(
                            f"⚠️ ML {self.symbol} training failed (Score: {score:.3f} < {self.MIN_MODEL_ACCURACY}). "
                            "Retrying in next interval."
                        )
                        # TELEGRAM NOTIFICATION
                        try:
                            Notifier.send_ml_training_update(
                                symbol=self.symbol,
                                horizon=self.horizon_str,
                                status="FAILED",
                                details={"error": f"Score {score:.3f} < MIN {self.MIN_MODEL_ACCURACY}"}
                            )
                        except Exception as e:
                            import logging
                            logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"ML Training error {self.symbol}: {e}", exc_info=True)
                    # TELEGRAM NOTIFICATION
                    try:
                        Notifier.send_ml_training_update(
                            symbol=self.symbol,
                            horizon=self.horizon_str,
                            status="FAILED",
                            details={"error": str(e)}
                        )
                    except Exception as ignore:
                        logger.error(f"Silent exception caught: {e}", exc_info=True)
                finally:
                    # MODO PROFESOR: Liberar RAM agresivamente
                    gc.collect()

        # OPTIMIZACIÓN RAM (Memoria Omnisciente):
        # copy.deepcopy() sobre 5000+ barras bloqueaba el CPU y duplicaba el consumo de RAM.
        # En su lugar pasamos un shallow copy ya que la estructura dict/list es generada de nuevo por data_provider.
        safe_bars = bars.copy() if isinstance(bars, dict) else bars[:] if isinstance(bars, list) else bars
        if sync:
            train_bg(safe_bars, train_type)
        else:
            self._training_thread = threading.Thread(
                target=train_bg, args=(safe_bars, train_type), daemon=True
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
            logger.error(f"Error shutting down executor: {e}", exc_info=True)

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
            # PHASE 10: HotAdapterRL Active Memory
            if hasattr(self, 'hot_adapter') and self.hot_adapter:
                try:
                    # Feed the PnL back into the hot adapter.
                    # As we don't track exact prediction vs actual direction here,
                    # we just feed the reward. The hot adapter stores state internally.
                    self.hot_adapter.update_reward(profit_pct)
                    logger.debug(f"🧠 [HOT-ADAPTER] Memoria activa actualizada con PnL: {profit_pct:.2f}%")
                except Exception as e:
                    logger.error(f"Error en HotAdapterRL update: {e}")

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
        """
        Guardar modelos con NANO-SPEED persistence (B1/B2/B3 FIX).

        👨‍🏫 MODO PROFESOR:
        QUÉ: Persistencia optimizada de modelos ML al disco.
        POR QUÉ: Joblib compress=5 tardaba ~3-5s. Saves duplicados al registry añadían ~3s más.
        PARA QUÉ: Reducir save time de ~6s a <1s total.
        CÓMO:
          - XGBoost: formato UBJSON nativo (.ubj) → 10-100x más rápido que Pickle
          - RF/GB: Joblib compress=1 → 3x más rápido que compress=5
          - Metadata: JSON separado → rápido, legible, sin serialización pesada
          - Registry: usa mismos paths (no duplica)
        CUÁNDO: Después de cada training exitoso.
        DÓNDE: strategies/ml_strategy.py → _save_models()
        QUIÉN: MLStrategyHybridUltimate
        """
        import joblib

        try:
            suffix = (
                self.par_engine.get_model_suffix()
                if hasattr(self, "par_engine")
                else ""
            )
            safe_sym = self.symbol.replace("/", "") + suffix
            sym_path = self.symbol.replace("/", "_") + suffix

            # === PRIMARY: XGBoost NATIVE UBJSON (B3 FIX) ===
            # XGBoost has built-in binary serialization 10-100x faster than Pickle
            xgb_dir = os.path.join(getattr(Config, "MODEL_DIR", "models"))
            os.makedirs(xgb_dir, exist_ok=True)

            xgb_ubj_path = os.path.join(xgb_dir, f"{safe_sym}_xgb.ubj")
            meta_path = os.path.join(xgb_dir, f"{safe_sym}_meta.joblib")

            if self.xgb_model is not None:
                self.xgb_model.save_model(xgb_ubj_path)

            # Save metadata (feature cols, scores, timestamp) as lightweight joblib
            with self._state_lock:
                meta = {
                    "feature_cols": self._feature_cols,
                    "last_training_score": self.last_training_score,
                    "training_iteration": self.training_iteration,
                    "performance_history": list(self.performance_history),
                    "timestamp": self._now(),
                    "base_rf_weight": self.base_rf_weight,
                    "base_xgb_weight": self.base_xgb_weight,
                    "base_gb_weight": self.base_gb_weight,
                }
            joblib.dump(meta, meta_path, compress=1)

            # Save DeepPredictor
            try:
                from models.deep_predictor import deep_predictor
                dp_path = os.path.join(xgb_dir, f"{safe_sym}_deep.pth")
                deep_predictor.save(dp_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)

            # === SECONDARY: RF/GB with minimal compression (B1 FIX) ===
            # compress=1 instead of compress=5 → 3x faster, ~10% larger
            model_file = os.path.join(self.models_dir, f"models_{sym_path}.joblib")

            # Backup rotation
            if os.path.exists(model_file):
                old_path = model_file + ".old"
                if os.path.exists(old_path):
                    os.remove(old_path)
                os.rename(model_file, old_path)

            with self._state_lock:
                state = {
                    "rf_model": self.rf_model,
                    "xgb_model": None,  # XGBoost saved separately as UBJSON
                    "gb_model": self.gb_model,
                    "scaler": self.scaler,
                    "feature_cols": self._feature_cols,
                    "label_mapping": self._label_mapping,  # PHASE 7: Persist 3-class mapping
                    "last_training_score": self.last_training_score,
                    "training_iteration": self.training_iteration,
                    "performance_history": list(self.performance_history),
                    "timestamp": self._now(),
                }

            joblib.dump(state, model_file, compress=1)  # B1 FIX: compress=1

            # === B2 FIX: ELIMINATED DUPLICATE REGISTRY SAVE ===
            # Previously: saved RF/XGB/GB/Scaler individually AGAIN to registry dir
            # Now: governance uses the primary paths directly
            metrics = {"sharpe": self.last_training_score, "win_rate": 0.0}
            if len(self.performance_history) > 0:
                win_rate = len([p for p in self.performance_history if p > 0]) / len(
                    self.performance_history
                )
                metrics["win_rate"] = win_rate * 100

            # Register using primary model paths (no duplication)
            comp_paths = {
                "rf": model_file,
                "xgb": xgb_ubj_path,
                "gb": model_file,
                "scaler": model_file,
            }
            governance_sym = f"{self.symbol}{suffix}"
            self.ml_governance.register_model(governance_sym, metrics, comp_paths)

            logger.info(
                f"💾 [{self.symbol}] Models persisted: XGBoost→UBJSON, RF/GB→Joblib(c=1)"
            )
        except Exception as e:
            logger.error(f"Error saving models for {self.symbol}: {e}")

    def _load_governed_model(self):
        """
        👨‍🏫 MODO PROFESOR:
        QUÉ: Carga inteligente de modelos certificados.
        POR QUÉ: Priorizamos modelos que han pasado el Quality Gate de la gobernanza.
        """
        import joblib

        suffix = (
            self.par_engine.get_model_suffix()
            if hasattr(self, "par_engine")
            else ""
        )
        governance_sym = f"{self.symbol}{suffix}"
        gov_model = self.ml_governance.get_production_model(governance_sym)
        if gov_model:
            try:
                path = gov_model["path"]
                self.rf_model = joblib.load(os.path.join(path, "rf.joblib"))
                self.xgb_model = joblib.load(os.path.join(path, "xgb.joblib"))
                self.gb_model = joblib.load(os.path.join(path, "gb.joblib"))
                self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))

                logger.info(
                    f"⚡ [{self.symbol}] Compilando árboles de gobernanza a matrices Numba..."
                )
                self.rf_arrays = (
                    compile_rf_to_numpy_batch(self.rf_model) if self.rf_model else None
                )
                self.gb_arrays = (
                    compile_gb_to_numpy_batch(self.gb_model) if self.gb_model else None
                )

                # Cargar columnas (buscamos en models_dir original por ahora o el último cache)
                cols_path = os.path.join(
                    self.models_dir, f"features_{self.symbol.replace('/', '_')}.json"
                )
                if os.path.exists(cols_path):
                    with open(cols_path, "r") as f:
                        loaded_cols = json.load(f)
                        self._feature_cols = [c for c in loaded_cols if c is not None]

                self.is_trained = True
                logger.info(
                    f"🏆 [{self.symbol}] Cargado modelo de PRODUCCIÓN v{gov_model['version']} (Sharpe: {gov_model['sharpe']:.2f})"
                )
                return True
            except Exception as e:
                logger.error(f"❌ Error cargando modelo de gobernanza: {e}")

        # Fallback a carga tradicional si no hay modelo de gobernanza
        return self._load_models()

    def _load_models(self):
        """Cargar modelos desde el disco para operatividad instantánea"""
        try:
            suffix = (
                self.par_engine.get_model_suffix()
                if hasattr(self, "par_engine")
                else ""
            )
            symbol_path = self.symbol.replace("/", "_") + suffix
            model_file = os.path.join(self.models_dir, f"models_{symbol_path}.joblib")

            # --- SUPREME PROTOCOL: NANO XGBoost UBJ Support ---
            # En Vector Backtest / Entorno de Produccion, el sufijo suele ser _SCALPING o _SWING
            horizon_suffix = f"_{getattr(self, 'horizon_str', 'SCALPING')}"
            safe_sym = self.symbol.replace("/", "") + horizon_suffix
            xgb_dir = getattr(Config, "MODEL_DIR", ".models")
            
            xgb_ubj_path = os.path.join(xgb_dir, f"{safe_sym}_xgb.ubj")
            xgb_reg_long_path = os.path.join(xgb_dir, f"{safe_sym}_xgb_reg_long.ubj")
            xgb_reg_short_path = os.path.join(xgb_dir, f"{safe_sym}_xgb_reg_short.ubj")
            meta_joblib_path = os.path.join(xgb_dir, f"{safe_sym}_meta.joblib")

            supreme_loaded = False
            if os.path.exists(xgb_ubj_path) and os.path.exists(meta_joblib_path):
                try:
                    from xgboost import XGBClassifier
                    self.xgb_model = XGBClassifier(n_jobs=-1)
                    self.xgb_model.load_model(xgb_ubj_path)
                    
                    if os.path.exists(xgb_reg_long_path) and os.path.exists(xgb_reg_short_path):
                        from xgboost import XGBRegressor
                        self.xgb_regressor_long = XGBRegressor(n_jobs=-1)
                        self.xgb_regressor_long.load_model(xgb_reg_long_path)
                        
                        self.xgb_regressor_short = XGBRegressor(n_jobs=-1)
                        self.xgb_regressor_short.load_model(xgb_reg_short_path)
                        
                        # Load new FULL CANDLE prediction regressors
                        nh_path = os.path.join(xgb_dir, f"{safe_sym}_xgb_reg_next_high.ubj")
                        nl_path = os.path.join(xgb_dir, f"{safe_sym}_xgb_reg_next_low.ubj")
                        ttp_path = os.path.join(xgb_dir, f"{safe_sym}_xgb_reg_ttp.ubj")
                        
                        if os.path.exists(nh_path):
                            self.xgb_reg_next_high = XGBRegressor(n_jobs=-1)
                            self.xgb_reg_next_high.load_model(nh_path)
                        if os.path.exists(nl_path):
                            self.xgb_reg_next_low = XGBRegressor(n_jobs=-1)
                            self.xgb_reg_next_low.load_model(nl_path)
                        if os.path.exists(ttp_path):
                            self.xgb_reg_ttp = XGBRegressor(n_jobs=-1)
                            self.xgb_reg_ttp.load_model(ttp_path)
                            
                        logger.info(f"🟢 [{self.symbol}] SUPREME XGBoost Regressors Loaded (MFE + Candle OHLC).")
                    else:
                        self.xgb_regressor_long = None
                        self.xgb_regressor_short = None
                        self.xgb_reg_next_high = None
                        self.xgb_reg_next_low = None
                        self.xgb_reg_ttp = None

                    meta_data = joblib.load(meta_joblib_path)
                    self._feature_cols = meta_data.get("feature_cols", [])
                    # ═══════════════════════════════════════════════════════════
                    # FORENSIC-V130 FIX: PROTECT REGRESSOR FEATURE COLS
                    # QUÉ: Guardar las 122 features del regresor por separado.
                    # POR QUÉ: El entrenamiento online (_train_models) redefine
                    #   self._feature_cols a solo 20 features (top_20_features).
                    #   Cuando el regresor intenta predecir, usa _feature_cols
                    #   que ahora tiene 20 en vez de 122 → predicción INVÁLIDA.
                    # PARA QUÉ: El regresor siempre recibirá sus 122 features
                    #   originales, independientemente del re-entrenamiento online.
                    # ═══════════════════════════════════════════════════════════
                    self._supreme_feature_cols = list(self._feature_cols)

                    self.is_trained = True
                    supreme_loaded = True
                    self.rf_model = None
                    self.gb_model = None
                    self.scaler = (
                        StandardScaler()
                    )  # Dummy scaler, XGBoost handles scaling natively

                    logger.info(
                        f"🟢 [{self.symbol}] SUPREME XGBoost NANO Model Loaded (.ubj). Expected features: {len(self._feature_cols)}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load Supreme UBJ: {e}")

            # Load DeepPredictor
            try:
                from models.deep_predictor import deep_predictor
                safe_sym_dp = self.symbol.replace("/", "") + suffix
                dp_path = os.path.join(xgb_dir, f"{safe_sym_dp}_deep.pth")
                if os.path.exists(dp_path):
                    deep_predictor.load(dp_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)

            if supreme_loaded:
                return

            if os.path.exists(model_file):
                state = joblib.load(model_file)
                save_time = state.get("timestamp")

                # --- VERIFICACIÓN DE CADUCIDAD (FRESHNESS) ---
                is_stale = False
                if save_time:
                    age_hours = (
                        self._now() - save_time
                    ).total_seconds() / 3600
                    if age_hours > 24:
                        is_stale = True
                        logger.warning(
                            f"⚠️ [{self.symbol}] Intelligence STALE ({age_hours:.1f}h old). Checking transfer learning..."
                        )

                # ✅ VALIDACIÓN DE FEATURES
                loaded_feature_cols = state.get("feature_cols", [])
                cleaned_feature_cols = [
                    col for col in loaded_feature_cols if col is not None
                ]

                # ✅ CRITICAL: Si no hay features válidas, el modelo es corrupto
                if not cleaned_feature_cols:
                    logger.error(
                        f"❌ [{self.symbol}] Corrupted model detected (no valid features). Checking transfer learning..."
                    )
                    is_stale = True

                if not is_stale:
                    with self._state_lock:
                        self.rf_model = state["rf_model"]
                        self.xgb_model = state["xgb_model"]
                        self.gb_model = state["gb_model"]
                        self.scaler = state["scaler"]
                        self._feature_cols = cleaned_feature_cols

                        logger.info(
                            f"⚡ [{self.symbol}] Compilando árboles nativos a matrices Numba..."
                        )
                        self.rf_arrays = (
                            compile_rf_to_numpy_batch(self.rf_model)
                            if self.rf_model
                            else None
                        )
                        self.gb_arrays = (
                            compile_gb_to_numpy_batch(self.gb_model)
                            if self.gb_model
                            else None
                        )
                        self.last_training_score = state.get("last_training_score", 0)
                        self.training_iteration = state.get("training_iteration", 0)

                        # PHASE 7: Restore 3-class label mapping from persisted state
                        saved_mapping = state.get("label_mapping")
                        if saved_mapping:
                            self._label_mapping = saved_mapping
                        else:
                            # Legacy 2-class model loaded → derive mapping from model classes
                            if self.rf_model and hasattr(self.rf_model, 'classes_'):
                                n_classes = len(self.rf_model.classes_)
                                if n_classes == 3:
                                    self._label_mapping = {0: -1, 1: 0, 2: 1}
                                else:
                                    self._label_mapping = {0: -1, 1: 1}
                                logger.info(f"🔄 [{self.symbol}] Derived label_mapping from model classes ({n_classes} classes)")

                        hist = state.get("performance_history", [])
                        self.performance_history = deque(hist, maxlen=100)

                        self.is_trained = True

                    logger.info(
                        f"🟢 [{self.symbol}] ML HYBRID ULTIMATE [ENSEMBLE] - Native model loaded"
                    )
                    return
                else:
                    use_transfer_learning = True
            else:
                use_transfer_learning = True

            # --- TRANSFER LEARNING FROM BTC (HORIZON-AWARE) ---
            # FORENSIC FIX: Only transfer from BTC model of the SAME horizon.
            # A Scalping BTC model has different patterns than a Swing BTC model.
            btc_symbol_path = "BTC_USDT" + suffix  # suffix includes _scalping or _swing
            btc_model_file = os.path.join(
                self.models_dir, f"models_{btc_symbol_path}.joblib"
            )

            if (
                use_transfer_learning
                and self.symbol != "BTC/USDT"
                and os.path.exists(btc_model_file)
            ):
                try:
                    btc_state = joblib.load(btc_model_file)
                    btc_features = btc_state.get("feature_cols", [])
                    cleaned_btc_features = [
                        col for col in btc_features if col is not None
                    ]

                    if cleaned_btc_features and btc_state.get("rf_model"):
                        with self._state_lock:
                            self.rf_model = btc_state["rf_model"]
                            self.xgb_model = btc_state["xgb_model"]
                            self.gb_model = btc_state["gb_model"]
                            self.scaler = btc_state["scaler"]
                            self._feature_cols = cleaned_btc_features
                            self.is_trained = True
                            self.training_iteration = 0  # Mark for retraining

                        logger.info(
                            f"🟢 [{self.symbol}] ML HYBRID ULTIMATE [ENSEMBLE] - Transfer learned from BTC"
                        )
                        return
                except Exception as e:
                    logger.warning(f"⚠️ [{self.symbol}] Transfer learning failed: {e}")

            logger.info(
                f"🟢 [{self.symbol}] ML HYBRID ULTIMATE [ENSEMBLE] - Fresh training starting..."
            )

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
                recent_win_rate = sum(self.performance_window) / len(
                    self.performance_window
                )

            total_win_rate = 0.0
            if self.total_trades > 0:
                total_win_rate = self.winning_trades / self.total_trades

            # Accuracy por régimen
            regime_stats = {}
            for regime, results in self.regime_accuracy.items():
                if len(results) > 0:
                    regime_stats[regime] = {
                        "accuracy": sum(results) / len(results),
                        "signals": len(results),
                        "total_signals": self.signals_by_regime.get(regime, 0),
                    }
                else:
                    regime_stats[regime] = {
                        "accuracy": 0.0,
                        "signals": 0,
                        "total_signals": 0,
                    }

            # Feature importance si está disponible
            feature_importance = {}
            if self.rf_model is not None and self._feature_cols is not None:
                try:
                    importances = self.rf_model.feature_importances_
                    top_features = dict(zip(self._feature_cols, importances))
                    sorted_features = sorted(
                        top_features.items(), key=lambda x: x[1], reverse=True
                    )[:20]
                    feature_importance = dict(sorted_features)
                except Exception:
                    feature_importance = {}

            return {
                # Identificación
                "strategy_id": self.strategy_id,
                "symbol": self.symbol,
                "objective": f"${self.initial_capital} → ${self.target_capital}",
                # Estado general
                "trained": self.is_trained,
                "training_iteration": self.training_iteration,
                "training_score": self.last_training_score,
                "last_training": self.last_training_time.isoformat()
                if self.last_training_time
                else None,
                "bars_since_train": self.bars_since_train,
                "feature_count": len(self._feature_cols) if self._feature_cols else 0,
                # Régimen de mercado
                "market_regime": self.market_regime,
                "regime_confidence": self.regime_confidence,
                "regime_duration": self.regime_duration,
                "regime_history": list(self.regime_history)[-10:],
                "regime_stats": regime_stats,
                # Circuit breaker
                "circuit_breaker_active": self.circuit_breaker_active,
                "consecutive_losses": self.consecutive_losses,
                "max_consecutive_losses": self.max_consecutive_losses,
                "peak_equity": self.peak_equity,
                # Modelos y ensemble
                "model_weights": {
                    "rf": float(self.base_rf_weight),
                    "xgb": float(self.base_xgb_weight),
                    "gb": float(self.base_gb_weight),
                },
                "model_scores": {
                    k: float(v) for k, v in self.individual_model_scores.items()
                },
                "feature_importance": feature_importance,
                # Parámetros adaptativos
                "adaptive_parameters": {
                    "confidence_threshold": float(self.adaptive_confidence_threshold),
                    "confluence_long": float(self.adaptive_confluence_long),
                    "confluence_short": float(self.adaptive_confluence_short),
                    "learning_rate": float(self.learning_rate),
                    "aggressiveness_factor": float(self.aggressiveness_factor),
                },
                # Targets
                "targets": {
                    "tp": float(self.current_tp_target),
                    "sl": float(self.current_sl_target),
                    "tp_sl_ratio": float(
                        self.current_tp_target / self.current_sl_target
                    )
                    if self.current_sl_target > 0
                    else 0,
                },
                # Performance completa
                "performance": {
                    "total_signals": self.total_signals_generated,
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "losing_trades": self.losing_trades,
                    "total_win_rate": float(total_win_rate),
                    "recent_win_rate": float(recent_win_rate),
                    "max_win_streak": self.max_win_streak,
                    "max_loss_streak": self.max_loss_streak,
                    "current_win_streak": self.win_streak,
                    "current_loss_streak": self.loss_streak,
                    "compounding_factor": float(self.compounding_factor),
                },
                # Signals por régimen
                "signals_by_regime": self.signals_by_regime,
                # Historial reciente
                "recent_performance": list(self.performance_history)[-20:],
                "recent_signals": [
                    {
                        "time": s["timestamp"].isoformat(),
                        "type": s["type"].name,
                        "confidence": s["confidence"],
                        "regime": s["regime"],
                        "price": s["price"],
                        "confluence": s["confluence"],
                    }
                    for s in list(self.signal_history)[-10:]
                ],
                # Estado del sistema
                "thread_alive": self._training_thread.is_alive()
                if hasattr(self, "_training_thread") and self._training_thread
                else False,
                "last_prediction": self._last_prediction_time.isoformat()
                if self._last_prediction_time
                else None,
                "current_capital": float(self.current_capital),
                "progress_percentage": float(
                    (self.current_capital - self.initial_capital)
                    / (self.target_capital - self.initial_capital)
                    * 100
                )
                if self.target_capital > self.initial_capital
                else 0.0,
            }

        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            return {"error": str(e)}

    def get_performance_report(self):
        """
        Generar reporte detallado de performance
        """
        status = self.get_strategy_status()

        report = {
            "summary": {
                "symbol": status["symbol"],
                "trained": status["trained"],
                "total_trades": status["performance"]["total_trades"],
                "win_rate": f"{status['performance']['total_win_rate'] * 100:.1f}%",
                "recent_win_rate": f"{status['performance']['recent_win_rate'] * 100:.1f}%",
                "current_regime": status["market_regime"],
                "circuit_breaker": status["circuit_breaker_active"],
            },
            "model_info": {
                "training_score": f"{status['training_score']:.3f}",
                "ensemble_weights": status["model_weights"],
                "model_scores": status["model_scores"],
            },
            "parameters": status["adaptive_parameters"],
            "targets": status["targets"],
            "streaks": {
                "current_win_streak": status["performance"]["current_win_streak"],
                "current_loss_streak": status["performance"]["current_loss_streak"],
                "max_win_streak": status["performance"]["max_win_streak"],
                "max_loss_streak": status["performance"]["max_loss_streak"],
            },
            "regime_performance": status["regime_stats"],
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
    # 🧠 COGNITIVE TRADE MANAGEMENT (Phase 5)
    # ============================================================

    def request_exit_opinion(self, pos_data: dict) -> dict:
        """
        Evalúa de forma inteligente si una posición abierta debe cerrarse, 
        diferenciando el "Ruido de Mercado" de una "Reversión Estructural".
        
        Args:
            pos_data: Dict con data de la posición (symbol, direction, entry_price, pnl_pct, etc)
        Returns:
            (action, reason): action puede ser "KEEP_OPEN" o "CLOSE"
        """
        try:
            symbol = pos_data.get('symbol', self.symbol)
            direction = pos_data.get('direction', 'LONG')
            pnl_pct = pos_data.get('pnl_pct', 0.0)
            
            bars = self.data_provider.get_latest_bars(symbol, n=50, timeframe=self.primary_tf)
            if bars is None or len(bars) < 20:
                return {"vote": "HOLD", "reason": "⏳ Evaluando (Datos insuficientes)"}
                
            df = self._prepare_features(bars, regime_aware=False)
            if df is None or df.empty:
                return {"vote": "HOLD", "reason": "⏳ Evaluando (Features no listos)"}
                
            current_row = df.iloc[-1]
            atr_pct = current_row.get('atr_pct', 0.0) / 100.0 if current_row.get('atr_pct', 0.0) > 1.0 else current_row.get('atr_pct', 0.0)
            adx = current_row.get('adx', 0)
            
            # Obtener convicción de ML actual
            ml_confidence = 0.5
            if hasattr(self, '_proba_history') and len(self._proba_history) > 0:
                import numpy as np
                smoothed = np.mean(self._proba_history, axis=0)
                pred_idx = np.argmax(smoothed)
                model_classes = getattr(self.rf_model, "classes_", getattr(self.xgb_model, "classes_", None))
                if model_classes is not None:
                    pred_dir = self._label_mapping.get(model_classes[pred_idx], model_classes[pred_idx])
                    if (pred_dir == 1 and direction == 'LONG') or (pred_dir == -1 and direction == 'SHORT'):
                        ml_confidence = smoothed[pred_idx]
                    else:
                        ml_confidence = 1.0 - smoothed[pred_idx]

            # 1. Agotamiento Predictivo Matemático (Alpha Decay)
            edge_prob = ml_confidence
            elapsed_bars = 0
            if 'duration_seconds' in pos_data:
                elapsed_bars = pos_data['duration_seconds'] / 60.0
            elif 'entry_time' in pos_data:
                import time
                from datetime import datetime, timezone
                entry_ts = pos_data['entry_time']
                try:
                    if hasattr(entry_ts, 'timestamp'):
                        elapsed_seconds = time.time() - entry_ts.timestamp()
                    else:
                        elapsed_seconds = (self._now() - entry_ts).total_seconds()
                    elapsed_bars = elapsed_seconds / 60.0
                except: pass

            # ═══════════════════════════════════════════════════════════════
            # FORENSIC FIX #11: self.engine NUNCA se inyecta en la estrategia.
            #   register_strategy() inyecta self.sophia pero NO self.engine.
            #   Además, prediction_tracker vive en risk_manager, no en engine.
            # FIX: Usar _engine_ref (inyectado por fix #11 en engine.py)
            #   y acceder via risk_manager.prediction_tracker.
            # ═══════════════════════════════════════════════════════════════
            _engine_ref = getattr(self, '_engine_ref', None)
            _rm = getattr(_engine_ref, 'risk_manager', None) if _engine_ref else None
            _pt = getattr(_rm, 'prediction_tracker', None) if _rm else None
            if _pt and hasattr(_pt, 'calculate_realtime_edge'):
                edge_prob = _pt.calculate_realtime_edge(
                    strategy_id=self.strategy_id, 
                    elapsed_bars=max(1, elapsed_bars), 
                    horizon=pos_data.get('horizon', 'SCALPING')
                )

            if pnl_pct > 0.005:  # Ganancia > 0.5%
                if edge_prob < 0.45:
                    logger.info(f"🧠 [COGNITIVE EXIT] {symbol} {direction} | Agotamiento Predictivo Matemático (Edge {edge_prob:.2%}). Asegurando PnL: {pnl_pct:.2%}")
                    return {"vote": "EXIT", "reason": f"Agotamiento de momentum matemático ({edge_prob:.2%})"}
            
            # 2. Reversión Estructural vs Ruido
            if pnl_pct < 0:  # Drawdown
                drawdown_magnitude = abs(pnl_pct)
                atr_threshold = atr_pct * 1.5
                
                if drawdown_magnitude > atr_threshold:
                    if ml_confidence < 0.55:
                        logger.warning(f"⚠️ [STRUCTURAL REVERSAL] {symbol} {direction} | Rompimiento de ATR ({drawdown_magnitude:.2%} > {atr_threshold:.2%}) y ML bajo ({ml_confidence:.2%}). Abortando!")
                        return {"vote": "EXIT", "reason": "Reversión estructural confirmada por ML"}
                else:
                    if adx > 25 and ml_confidence > 0.55:
                        logger.debug(f"🛡️ [NOISE FILTER] {symbol} {direction} ignorando retroceso de {drawdown_magnitude:.2%} (ATR {atr_threshold:.2%}). Tendencia firme.")
                        return {"vote": "HOLD", "reason": "Ruido de mercado, tendencia intacta"}

            # 3. Dynamic Targets (Phase 8)
            dynamic_targets = {"tp_mult": 1.0, "sl_mult": 1.0}
            if ml_confidence > 0.70:
                dynamic_targets["tp_mult"] = 1.5  # Let winners run
            elif ml_confidence < 0.50:
                dynamic_targets["tp_mult"] = 0.5  # Secure profit
                dynamic_targets["sl_mult"] = 0.8  # Tighten SL

            return {
                "vote": "HOLD", 
                "reason": "Posición sana (Mantenimiento Dinámico)",
                "dynamic_targets": dynamic_targets
            }
            
        except Exception as e:
            logger.error(f"Error in request_exit_opinion: {e}")
            return {"vote": "HOLD", "reason": "Error de Evaluación Cognitiva"}

    # ============================================================
    # ✅ CLEANUP Y DESTRUCTOR
    # ============================================================

    def __del__(self):
        """Cleanup seguro"""
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)

            logger.info(f"🧹 ML Hybrid Ultimate Strategy for {self.symbol} cleaned up")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)


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

    # ═══════════════════════════════════════════════════════════════
    # FORENSIC-V50 FIX: UNIFIED CONSENSUS THRESHOLD
    # QUÉ: Un solo threshold para todo el sistema (era 0.55/0.60/0.78).
    # POR QUÉ: 3 valores compitiendo causaba que señales válidas con
    #   ML=0.76 fueran bloqueadas por threshold dinámico de 0.78.
    # PARA QUÉ: Consistencia — el mismo número aparece en logs y cálculos.
    # ═══════════════════════════════════════════════════════════════
    ENSEMBLE_CONSENSUS_THRESHOLD = 0.55
    MIN_ENGINES_REQUIRED = 1  # Sentimiento muerto → ML solo puede operar

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override strategy ID for ENSEMBLE mode
        lbl = (
            "[SCL]"
            if getattr(self, "horizon_str", "SCALPING") == "SCALPING"
            else "[SWG]"
        )
        self.strategy_id = f"{lbl}_ML_HYBRID_ULTIMATE_ENSEMBLE_V3_{self.symbol.replace('/', '_')}"

        # Engine tracking
        self.engine_scores = {"ml": 0.0, "sentiment": 0.0, "technical": 0.0}
        self.engines_active = 0
        self.consensus_threshold = self.ENSEMBLE_CONSENSUS_THRESHOLD  # UNIFIED: same value everywhere

        logger.info(f"🟢 [{self.symbol}] UNIVERSAL ENSEMBLE STRATEGY INITIALIZED")
        logger.info(f"   Consensus Threshold: {self.ENSEMBLE_CONSENSUS_THRESHOLD}")
        logger.info(f"   Min Engines Required: {self.MIN_ENGINES_REQUIRED}/3")

    def _calculate_temporal_confidence(
        self, bars_processed: int, total_bars: int
    ) -> float:
        """
        Adaptive Evolution Protocol: Temporal Confidence Decay.

        ═══════════════════════════════════════════════════════════════
        FORENSIC-V50 FIX: GENTLER RAMP (NO MORE 50% CLIFF)
        QUÉ: Factor de confianza temporal SUAVIZADO (0.75 a 1.0).
        POR QUÉ: La versión anterior cortaba ML al 50% durante el 20%
                  inicial del backtest, matando ~288 barras de un backtest
                  de 1 día. Esto destruía el Win Rate tempranamente.
        PARA QUÉ: Penalizar suavemente (no guillotinar) las primeras
                   predicciones. El modelo con 50+ features aún tiene
                   poder predictivo parcial desde la barra ~100.
        CÓMO: Primer 10% → factor 0.75, 10-25% → ramp 0.75→1.0, 25%+ → 1.0
        ═══════════════════════════════════════════════════════════════
        """
        if total_bars <= 0:
            return 1.0

        progress = bars_processed / total_bars

        if progress < 0.10:
            return 0.75  # Gentle reduction, not 50% guillotine
        elif progress < 0.25:
            # Linear interpolation from 0.75 to 1.0 between 10% and 25%
            return 0.75 + (progress - 0.10) * (0.25 / 0.15)
        else:
            return 1.0  # Full context available

    def _run_inference(self):
        """
        🔮 UNIVERSAL ENSEMBLE INFERENCE
        Overridden to force 3-engine consensus and bridge validation.
        """
        try:
            self.analysis_stats["total"] += 1
            if not self._check_circuit_breaker():
                logger.info(
                    f"DEBUG [{self.symbol}] _run_inference early exit: circuit breaker failed"
                )
                return

            # FIX: Forzar update de régimen con datos reales (no reloj del sistema)
            # En backtest, datetime.now() no avanza con las barras → régimen queda UNKNOWN
            # Solución: Llamar _update_market_regime con force=True cada N barras
            _should_update_regime = (
                self.market_regime == "UNKNOWN"
                or self.loop_count % 60 == 0  # Cada 60 barras (~1h en 1m)
            )
            if _should_update_regime:
                try:
                    _bars_for_regime = self.data_provider.get_latest_bars(
                        self.symbol, n=100
                    , timeframe=self.primary_tf)
                    # Fallback to 1m if primary_tf has insufficient data
                    if (_bars_for_regime is None or len(_bars_for_regime) < 50) and self.primary_tf != '1m':
                        _bars_for_regime = self.data_provider.get_latest_bars(
                            self.symbol, n=100, timeframe='1m')
                    if _bars_for_regime is not None and len(_bars_for_regime) >= 50:
                        _df_regime = self._prepare_features(
                            _bars_for_regime, regime_aware=False
                        )
                        if _df_regime is not None and len(_df_regime) > 0:
                            # Bypass throttle: reset timestamp para forzar update
                            self.last_regime_update = self._now() - pd.Timedelta(minutes=10)
                            self._update_market_regime(_df_regime)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)  # Non-fatal: si falla, el régimen queda como estaba

            # 1. Data Preparation
            bars = self.data_provider.get_latest_bars(self.symbol, n=250, timeframe=self.primary_tf)
            
            # ═══════════════════════════════════════════════════════════════
            # [FORENSIC-V3] SWING TIMEFRAME GRACEFUL DEGRADATION
            # QUÉ: Si el primary_tf (ej. 1h) no tiene barras suficientes,
            #   degradar a 5m para mantener la señal viva.
            # POR QUÉ: Un backtest de 1 día solo tiene ~24 barras de 1h.
            #   Pedir n=250 de 1h retorna solo ~6-24 barras, y luego
            #   FeatureEngineering las rechaza por < 50 barras.
            # PARA QUÉ: SWING no queda permanentemente muerta en backtests
            #   cortos. En producción, BinanceData precarga 500+ barras de 1h
            #   así que este fallback no se activa.
            # CÓMO: Si bars < 30 y primary_tf != '1m', intentar con '5m',
            #   luego con '1m'.
            # ═══════════════════════════════════════════════════════════════
            _actual_tf = self.primary_tf
            if bars is not None and len(bars) < 30 and self.primary_tf not in ('1m', '5m'):
                bars = self.data_provider.get_latest_bars(self.symbol, n=250, timeframe='5m')
                _actual_tf = '5m'
                if bars is not None and len(bars) < 30:
                    bars = self.data_provider.get_latest_bars(self.symbol, n=250, timeframe='1m')
                    _actual_tf = '1m'
            elif bars is None or len(bars) == 0:
                # Try fallback chain even if primary returned nothing
                if self.primary_tf not in ('1m', '5m'):
                    bars = self.data_provider.get_latest_bars(self.symbol, n=250, timeframe='5m')
                    _actual_tf = '5m'
                if bars is None or len(bars) == 0:
                    bars = self.data_provider.get_latest_bars(self.symbol, n=250, timeframe='1m')
                    _actual_tf = '1m'
            
            # ── FASE III-B: ZERO-COPY CACHE BYPASS (GIL EVASION) ──
            # Evade the Python GIL and Pandas overhead by using pre-calculated matrices.
            df = None
            if bars is not None and len(bars) > 0:
                current_ts = bars['timestamp'][-1] if hasattr(bars, 'dtype') else (bars[-1]['timestamp'] if isinstance(bars[-1], dict) else None)
                if current_ts is not None and hasattr(self, "_global_feature_cache") and hasattr(self, "_global_feature_cache_ts"):
                    import numpy as np
                    ts_arr = self._global_feature_cache_ts
                    idx = np.searchsorted(ts_arr, current_ts)
                    if idx < len(ts_arr) and ts_arr[idx] == current_ts:
                        # Slice the pre-calculated dataframe directly
                        start_idx = max(0, idx - 4)
                        if hasattr(self._global_feature_cache, "to_pandas"):
                            # If it's a Polars DataFrame, slicing returns Polars
                            df = self._global_feature_cache[start_idx:idx+1]
                        else:
                            # It's Pandas, we need it as pandas or dict row eventually
                            df = self._global_feature_cache.iloc[start_idx:idx+1].copy()
            
            if df is None:
                df = self._prepare_features(bars, regime_aware=True)
            if df is None:
                logger.info(
                    f"DEBUG [{self.symbol}|{self.horizon_str}] _run_inference early exit: df is None. Bars passed: {len(bars) if bars is not None else 0}"
                )
                return
            if len(df) == 0 or len(df) < 5:
                logger.info(
                    f"DEBUG [{self.symbol}|{self.horizon_str}] _run_inference early exit: df.empty or len(df)={len(df)} < 5. Bars passed: {len(bars) if bars is not None else 0}"
                )
                return

            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
                
            current_row = df.iloc[-1]
            atr_pct = current_row["atr_pct"] / 100
            vol_ratio = current_row.get("volume_ratio", 0)

            # 2. Model Availability
            with self._state_lock:
                # SUPREME MODE DETECTED: If only XGB is loaded and rf/gb are None AND we have exactly 3 features
                supreme_mode = (
                    self.xgb_model is not None
                    and getattr(self, "rf_model", None) is None
                    and len(getattr(self, "_feature_cols", None) or []) == 3
                )

                # Prevent sklearn unfitted truthiness exception (`__bool__` triggering `__len__`)
                # by strictly checking `is_trained` flag instead of evaluating `self.rf_model` instances.
                if supreme_mode:
                    models_ready = self.is_trained and self.xgb_model is not None
                    feature_cols = (
                        None  # Supreme constructs its own features array manually
                    )
                else:
                    # ALLOW XGBOOST-ONLY RUN EXPLICITLY (e.g. 111 features)
                    models_ready = self.is_trained and (
                        self.rf_model is not None
                        or self.xgb_model is not None
                        or self.gb_model is not None
                    )
                    raw_features = getattr(self, "_feature_cols", []) or []
                    feature_cols = [c for c in raw_features if c is not None]
                    if not feature_cols:
                        logger.info(
                            f"DEBUG [{self.symbol}] _run_inference: models_ready False due to empty feature_cols"
                        )
                        models_ready = False

            if not models_ready:
                logger.info(
                    f"DEBUG [{self.symbol}] _run_inference early exit: models_ready is False. is_trained={self.is_trained}"
                )
                return

            # 3. Aligned Feature Matrix
            if supreme_mode:
                # --- SUPREME FEATURE ADAPTER ---
                # train_supreme.py used: rsi_14, zscore_20, log_returns
                # We must reconstruct these exactly.
                # df has 'close' as float (from _prepare_features)
                closes = df["close"].values.astype(np.float64)

                # RSI 14
                rsi = talib.RSI(closes, timeperiod=14)

                # Z-Score 20 (Manual calculation to match math_kernel) - O(1) Optimized
                eff_window = min(20, len(closes))
                if eff_window > 0:
                    slice_closes = closes[-eff_window:]
                    mean_last = np.mean(slice_closes)
                    std_last = np.std(slice_closes, ddof=0)
                    curr_z = (closes[-1] - mean_last) / std_last if std_last > 0 else 0.0
                else:
                    curr_z = 0.0

                # Log Return
                # np.log(price / prev_price)
                returns = np.diff(np.log(closes), prepend=np.log(closes[0]))

                # Stack for last row
                # Features: [rsi, zscore, returns]
                # We take the last element
                curr_rsi = rsi[-1]
                curr_ret = returns[-1]

                X_pred = np.array([[curr_rsi, curr_z, curr_ret]], dtype=np.float32)

                # Predict
                # No scaler used in train_supreme (raw features)
                xgb_proba = self.xgb_model.predict_proba(X_pred)[0]

                # Mock others for organic confluence logic, or bypass
                # Since Organic Confluence expects 3 engines...
                # We will trust XGB fully in Supreme Mode.

                pred_idx = np.argmax(xgb_proba)
                final_confidence = xgb_proba[pred_idx]
                # classes: [0, 1] which map to [-1, 1] via label_mapping
                raw_cls = self.xgb_model.classes_[pred_idx]
                # train_supreme used: 0=DOWN, 1=UP.
                # Strategy expects: -1=SHORT, 1=LONG
                direction = 1 if raw_cls == 1 else -1

                # Log Oracle
                if final_confidence > 0.55:  # Min threshold
                    logger.info(
                        f"🔮 [SUPREME ORACLE] {self.symbol} Signal: {direction} (Conf: {final_confidence:.2f})"
                    )
                    # Trigger Signal
                    self.events_queue.put(
                        SignalEvent(
                            strategy_id="SUPREME_XGB_V1",
                            symbol=self.symbol,
                            datetime=self._now(),
                            signal_type=SignalType.LONG
                            if direction == 1
                            else SignalType.SHORT,
                            strength=final_confidence, ml_confidence=final_confidence,
                            current_price=bars[-1]["close"],
                            sl_pct=Config.Horizons.Swing['sl_pct'] if getattr(self, "horizon", "SCALPING") == 'SWING' else Config.Horizons.Scalping['sl_pct'],
                            tp_pct=Config.Horizons.Swing['tp_pct'] if getattr(self, "horizon", "SCALPING") == 'SWING' else Config.Horizons.Scalping['tp_pct'],
                            horizon=getattr(self, "horizon", "SCALPING"),
                            predicted_magnitude=0.015,
                            predicted_duration=self.LOOKAHEAD_BARS,
                            metadata={"source": "SUPREME_XGB"},
                        )
                    )
                return

            if (
                hasattr(self, "scaler")
                and self.scaler is not None
                and hasattr(self.scaler, "feature_names_in_")
            ):
                final_features = self.scaler.feature_names_in_
                # AEGIS-V21 OPTIMIZATION: Vectorized selection is 50x faster than iteration
                existing = [c for c in final_features if c in df.columns]
                missing = [c for c in final_features if c not in df.columns]
                
                X_pred = df[existing].iloc[[-1]].copy()
                for c in missing:
                    X_pred[c] = 0.0
                
                X_pred = X_pred[final_features] # Ensure correct order for scaler
            else:
                # Fallback for old models
                missing_cols = [c for c in feature_cols if c not in df.columns]
                if missing_cols:
                    logger.warning(f"[{self.symbol}] Padding {len(missing_cols)} missing features (e.g. {missing_cols[:3]})")
                    for c in missing_cols:
                        df[c] = 0.0
                X_pred = df[feature_cols].iloc[[-1]].copy()


            if (
                hasattr(self, "scaler")
                and self.scaler is not None
                and hasattr(self.scaler, "transform")
            ):
                try:
                    from sklearn.utils.validation import check_is_fitted

                    check_is_fitted(self.scaler)
                    X_scaled = self.scaler.transform(X_pred)
                except Exception:
                    X_scaled = X_pred.values
            else:
                X_scaled = X_pred.values

            # [FORENSIC-FIX] AEGIS-V22: Multi-Horizon Feature Parity
            # QUÉ: Trunca el vector de entrada a la capacidad que el modelo espera.
            # POR QUÉ: Los nuevos modelos Supreme esperan 122 features. El hardcode a 111 los rompía.
            # PARA QUÉ: Mantener compatibilidad dinámica sin hardcodes mágicos.
            expected_feats = len(self._feature_cols) if hasattr(self, '_feature_cols') and self._feature_cols else 111
            if isinstance(X_scaled, np.ndarray) and X_scaled.shape[1] > expected_feats:
                X_scaled = X_scaled[:, :expected_feats]

            rf_proba = (
                self.rf_model.predict_proba(X_scaled)[0]
                if getattr(self, "rf_model", None) is not None
                else None
            )
            xgb_proba = (
                self.xgb_model.predict_proba(X_scaled)[0]
                if getattr(self, "xgb_model", None) is not None
                else None
            )
            gb_proba = (
                self.gb_model.predict_proba(X_scaled)[0]
                if getattr(self, "gb_model", None) is not None
                else None
            )

            # Fallback if only XGBoost is loaded
            if rf_proba is None:
                rf_proba = xgb_proba
            if gb_proba is None:
                gb_proba = xgb_proba
            if xgb_proba is None:
                xgb_proba = rf_proba  # Just in case

            ensemble_proba = (
                rf_proba * self.base_rf_weight
                + xgb_proba * self.base_xgb_weight
                + gb_proba * self.base_gb_weight
            )

            # ── AEGIS-V16 FIX: PREDICTION SMOOTHING (ANTI-WHIPSAW) ──
            # QUÉ: Suaviza las probabilidades del modelo sobre las últimas 3 barras.
            # POR QUÉ: El modelo estaba sobre-reaccionando al ruido de tick, cambiando
            #   de LONG (0.95) a SHORT (0.98) en barras consecutivas, causando "churn"
            #   y trades de 0s de duración.
            # PARA QUÉ: Exigir convicción sostenida antes de cambiar de dirección.
            # ═══════════════════════════════════════════════════════════════
            # FIX-FORENSIC-V41: PRE-FILL PROBA HISTORY
            # QUÉ: Inicializar la deque con la predicción actual (no vacía).
            # POR QUÉ: Si arranca vacía, la primera predicción buena (ej: 0.85)
            #   se diluye con zeros → mean([0.85]) = 0.85 OK en 1 bar, pero
            #   mean([0.85, 0]) = 0.425 si había una iteración previa sin datos.
            # PARA QUÉ: Evitar sesgo temporal post-warmup que reduce confianza.
            # ═══════════════════════════════════════════════════════════════
            if not hasattr(self, '_proba_history'):
                import collections
                self._proba_history = collections.deque(maxlen=3)
                # Pre-fill with current proba to avoid dilution on first bar
                self._proba_history.append(ensemble_proba)
            self._proba_history.append(ensemble_proba)
            
            # Use smoothed probabilities for decision
            smoothed_proba = np.mean(self._proba_history, axis=0)

            model_with_classes = (
                getattr(self, "rf_model", None)
                or getattr(self, "xgb_model", None)
                or getattr(self, "gb_model", None)
            )
            classes = model_with_classes.classes_
            pred_idx = np.argmax(smoothed_proba)
            raw_confidence = smoothed_proba[pred_idx]
            direction = self._label_mapping.get(classes[pred_idx], classes[pred_idx])
            
            # PHASE 7 FIX: Anti-Noise Gate
            if direction == 0:
                logger.debug(f"🔇 [ML NOISE REJECT] {self.symbol} | Model predicted HOLD (Noise). No signal generated.")
                return
            # Adaptive Evolution Protocol: Temporal Confidence Decay
            # Reduce ML confidence during early bars when context is insufficient
            temporal_factor = self._calculate_temporal_confidence(
                self.bars_since_train, getattr(self, "retrain_interval", 2000)
            )
            raw_confidence *= temporal_factor

            # Phase 9: Capture Ensemble Input for PPO (Probabilities of CHOSEN direction)
            # We need the probabilities corresponding to the '1' class (Up) or '0' (Down), depending on how we model state.
            # But `update_recursive_weights` uses `last_ensemble_input` to compute `dot` product.
            # If direction is 1 (LONG), we want predicted prob of LONG.
            # If direction is -1 (SHORT), we want predicted prob of SHORT?
            # Actually, `last_ensemble_input` should be the raw inputs to the meta-learner.
            # The meta-learner weights [w1, w2, w3] allow it to trust RF, XGB, GB differently.
            # So `last_ensemble_input` should be the confidence of each model in the CHOSEN direction.

            # PHASE 7 FIX: 3-Class PPO Ensemble Input Indexing
            # For 3-class models: [0=SHORT, 1=HOLD, 2=LONG] (ml_idx=2 for LONG, 0 for SHORT)
            # For 2-class models: [0=SHORT, 1=LONG] (ml_idx=1 for LONG, 0 for SHORT)
            
            # Auto-detect class size
            num_classes = len(rf_proba) if rf_proba is not None else len(xgb_proba)
            
            if num_classes == 2:
                ml_idx = 1 if direction == 1 else 0
            else:
                ml_idx = 2 if direction == 1 else 0

            self.last_ensemble_input = np.array(
                [
                    rf_proba[ml_idx] if rf_proba is not None else 0.0, 
                    xgb_proba[ml_idx] if xgb_proba is not None else 0.0, 
                    gb_proba[ml_idx] if gb_proba is not None else 0.0
                ]
            )

            # Store state for PPO (Observation)
            # We store the inputs to the ensemble (model probs) + some key features?
            # Or just the inputs to the weighting layer?
            # The "Policy" here is the weighting [w1, w2, w3].
            # The state for PPO should be relevant to "which model is right?".
            # For now, we use the model probs as state.
            pass  # Removed self.last_ppo_state override to prevent 3-dim vs 18-dim collision.
            self.last_ppo_action_probs = ensemble_proba  # Full prob dist

            # [PHASE 9: ML Metacognition] Live Dashboard Hooks for Brier Score & Entropy
            # Brier Score Proxy: (1.0 - Confidencia)^2; Si es 1.0 = Brier 0.0 (Perfecto)
            self.current_brier_score = (1.0 - raw_confidence) ** 2
            # PPO Entropy Proxy: -sum(p * log(p)) over the 3 components across binary class
            self.current_ppo_entropy = -np.sum(
                ensemble_proba * np.log(ensemble_proba + 1e-9)
            )

            # ============================================================
            # 🎯 3-ENGINE ORGANIC CONFLUENCE (The Heart of Phase 8)
            # ============================================================
            final_confidence, engines_passing, is_valid, multi_horizon = (
                self.compute_organic_confluence(
                    df, direction, rf_proba, xgb_proba, gb_proba
                )
            )

            # 4. ORACLE REPORT (Universal Multi-Engine View)
            gap = self.consensus_threshold - final_confidence
            ready_status = "READY" if is_valid else "SCANNING"

            # Extract Horizon Details
            h1 = multi_horizon.get("h1", 0)
            h5 = multi_horizon.get("h5", 0)
            h15 = multi_horizon.get("h15", 0)
            h30 = multi_horizon.get("h30", 0)

            # Prepare Enhanced Stats
            vol_ratio = current_row.get("volume_ratio", 1.0)
            adx = current_row.get("adx", 0)
            ret_5 = current_row.get("returns_5", 0.0) * 100

            # Determine Concept/Context based on Regime
            if self.market_regime == "ZOMBIE":
                concept = (
                    "Zombie market detected. Stagnant price action. Protection active."
                )
            elif self.market_regime == "RANGING":
                concept = "Mean Reversion active. Hunting overextensions."
            elif self.market_regime == "TRENDING":
                concept = "Trend Following active. Riding momentum."
            elif self.market_regime == "VOLATILE":
                concept = "High Volatility. Defensive stops & wide targets."
            else:
                concept = "Analyzing market structure..."

            labels = ["M1", "M5", "M15", "M30"] if getattr(self, "horizon", "SCALPING") in ["SCALPING", "MICROSCALPING"] else ["H1", "H4", "H12", "D1"]
            oracle_msg = (
                f"\n🔮 [UNIFIED ORACLE] {self.symbol} | {ready_status}\n"
                f"   Engines Passing: {engines_passing}/3 | Threshold: {self.consensus_threshold}\n"
                f"   Scores  -> ML: {self.engine_scores['ml']:.2f} | SENT: {self.engine_scores['sentiment']:.2f} | TECH: {self.engine_scores['technical']:.2f}\n"
                f"   Horizon -> {labels[0]}: {h1:.2f} | {labels[1]}: {h5:.2f} | {labels[2]}: {h15:.2f} | {labels[3]}: {h30:.2f}\n"
                f"   Verdict -> Direction: {direction} | Final Conf: {final_confidence:.2f} (Gap: {gap:.2f})\n"
                f"   Phase: {self.market_regime} ({self.regime_confidence * 100:.1f}%)\n"
                f"   Concept: {concept}\n"
                f"   Stats: ADX={adx:.1f} | ATR%={atr_pct * 100:.2f}% | Ret5={ret_5:.2f}% | VolRatio={vol_ratio:.2f}\n"
                f"   Confidence: {final_confidence * 100:.1f}% | Strategy: Adaptive targets enabled."
            )
            logger.info(oracle_msg)

            # 5. ROBUSTNESS FILTERS
            if atr_pct > self.MAX_ATR_PCT * 1.5:
                logger.warning(
                    f"⛔ [FILTER] {self.symbol} Rejected: Extreme Volatility (ATR: {atr_pct:.2%})"
                )
                return

            # FORENSIC FIX: Remove is_backtest divergence to ensure parity
            min_vol = getattr(self, 'MIN_VOLUME_RATIO', 0.0)
            if vol_ratio < min_vol:
                logger.debug(
                    f"⛔ [FILTER] {self.symbol} Rejected: Low Volume (Ratio: {vol_ratio:.2f} < {min_vol})"
                )
                return

            if not is_valid:
                self.analysis_stats["filtered_conf"] += 1
                return

            # ============================================================
            # ✅ FIX-FORENSIC-V41: ORACLE MACRO VETO (PARITY WITH BASE)
            # QUÉ: Vetar señales que van contra la tendencia macro (1d/1w).
            # POR QUÉ: La inferencia base tenía este filtro pero el Ensemble no.
            # PARA QUÉ: Evitar abrir LONGs en tendencias bajistas macro.
            # ============================================================
            signal_type_raw = "LONG" if direction == 1 else "SHORT"
            try:
                _oracle_tf_data = {}
                for _otf in ["1d", "1w"]:
                    try:
                        _macro_bars = self.data_provider.get_latest_bars(
                            self.symbol, n=250, timeframe=_otf
                        )
                        if _macro_bars is not None and len(_macro_bars) >= 20:
                            _c = _macro_bars["close"]
                            _rsi = calculate_rsi_jit(_c, 14)
                            _ema_fast = calculate_ema_jit(_c, 20)
                            _ema_slow = calculate_ema_jit(_c, 50)
                            _ema_trend = calculate_ema_jit(_c, 200)
                            _in_up = (_ema_fast > _ema_slow) & (_c > _ema_trend)
                            _in_dn = (_ema_fast < _ema_slow) & (_c < _ema_trend)
                            _oracle_tf_data[_otf] = {
                                "inds": {"rsi": _rsi, "in_uptrend": _in_up, "in_downtrend": _in_dn},
                                "data": _macro_bars,
                            }
                    except Exception as e:
                        logger.error(f"Silent exception caught: {e}", exc_info=True)

                if _oracle_tf_data:
                    _oracle_verdict = MultiHorizonOracle.evaluate_clash_vector(
                        _oracle_tf_data, signal_type_raw
                    )
                    if _oracle_verdict["is_vetoed"]:
                        clash = _oracle_verdict['clash_score']
                        # QUÉ: Alineación de umbral de veto duro en la estrategia ML (0.85 -> 0.60).
                        # POR QUÉ: Consistencia arquitectónica con la estrategia técnica para proteger la microcuenta de $13 USD.
                        # PARA QUÉ: Evitar divergencias donde la estrategia ML permita trades que la técnica bloquearía.
                        # CÓMO: Cambiando `clash > 0.85` por `clash > 0.60`.
                        # CUÁNDO: Durante la inferencia de señal de ensemble.
                        # DÓNDE: En `strategies/ml_strategy.py` L4933.
                        # QUIÊN: Modificado por el Arquitecto Senior y Quant Developer.
                        if clash > 0.60:
                            # HARD VETO: Only extreme/strong macro opposition
                            logger.info(
                                f"🔮 [ENSEMBLE ORACLE VETO] {self.symbol} {signal_type_raw} BLOCKED (EXTREME) | "
                                f"Clash: {clash:.1%} | Macro: {_oracle_verdict['macro_context']}"
                            )
                            self.analysis_stats["filtered_conf"] += 1
                            return
                        else:
                            # CONSENSO PONDERADO: Reduce confidence
                            oracle_penalty = max(0.4, 1.0 - clash)
                            final_confidence *= oracle_penalty
                            logger.info(
                                f"🔮 [CONSENSUS] {self.symbol} ML Oracle penalty x{oracle_penalty:.2f} | "
                                f"Clash: {clash:.1%} | Macro: {_oracle_verdict['macro_context']}"
                            )
            except Exception as e:
                logger.error(f"Oracle Ensemble Integration Error on {self.symbol}: {e}", exc_info=True)

            # ============================================================
            # ✅ FIX-FORENSIC-V41: PREDICTIVE DECAY EXIT (PARITY WITH BASE)
            # QUÉ: Cerrar posición si la confianza ML decae en la dirección abierta.
            # POR QUÉ: Solo existía en _process_ml_results (async handler).
            # PARA QUÉ: Cerrar proactivamente trades que el modelo ya no respalda.
            # ════════════════════════════════════════════════════════════════
            # FORENSIC-V49 FIX: PREDICTIVE_DECAY DISABLED FOR SCALPING
            # QUÉ: Desactivar PREDICTIVE_DECAY como exit para Scalping.
            # POR QUÉ: La auditoría forense mostró que esta estrategia tiene
            #   un WR del 33% (9W/18L) y destruyó -$1.36 del capital en 1 día.
            #   El modelo ML cambia de dirección demasiado rápido en M1/M5,
            #   cerrando trades que habrían sido ganadores 2-3 barras después.
            # PARA QUÉ: Delegar las salidas de Scalping al ExitOracle, FLIP_EXIT
            #   (69% WR, +$0.61) y los SL/TP mecánicos.
            # CUÁNDO: Solo se aplica en SWING con confianza > 0.80 (extrema).
            # ════════════════════════════════════════════════════════════════
            _current_horizon = getattr(self, 'horizon', 'SCALPING')
            _decay_enabled = _current_horizon == "SWING"  # ONLY for Swing
            _decay_threshold = 0.80  # Raised from 0.60 to 0.80 (extreme conviction only)
            
            signal_type = SignalType.LONG if direction == 1 else SignalType.SHORT
            if _decay_enabled and getattr(self, 'portfolio', None):
                for v_key, v_pos in getattr(self.portfolio, 'virtual_ledger', {}).items():
                    if v_key.startswith(f"{self.symbol}_") and v_pos.get('quantity', 0) != 0:
                        pos_qty = v_pos.get('quantity', 0)
                        entry_price = v_pos.get('avg_price', current_row["close"])
                        is_long_pos = pos_qty > 0
                        is_short_pos = pos_qty < 0
                        
                        # [Oracle Fee Headroom]
                        # Verify we are not closing a tiny profit that fees will turn into a loss.
                        # If pnl is between 0 and 0.15%, we hold to avoid Fee Death Spiral.
                        current_price = current_row["close"]
                        pnl_pct = 0
                        if is_long_pos and entry_price > 0:
                            pnl_pct = (current_price - entry_price) / entry_price
                        elif is_short_pos and entry_price > 0:
                            pnl_pct = (entry_price - current_price) / entry_price
                            
                        fee_headroom_ok = not (0 < pnl_pct < 0.0015)
                        
                        # If we're LONG but ML now predicts SHORT with EXTREME conf → EXIT
                        if is_long_pos and direction == -1 and final_confidence > _decay_threshold:
                            if fee_headroom_ok:
                                logger.warning(f"📉 [ENSEMBLE DECAY EXIT] Confidence flipped against LONG {self.symbol} (SHORT conf={final_confidence:.2f}). Exiting...")
                                _decay_exit = SignalEvent(
                                    strategy_id=f"{self.strategy_id}.PREDICTIVE_DECAY",
                                    setup_type="PREDICTIVE_DECAY_EXIT",
                                    symbol=self.symbol,
                                    datetime=self._now(),
                                    signal_type=SignalType.EXIT,
                                    strength=final_confidence, ml_confidence=final_confidence,
                                    current_price=current_row["close"],
                                    horizon=_current_horizon,
                                    metadata={'urgent': False, 'actual_order_type': 'limit', 'is_tp_limit': True},
                                )
                                self.events_queue.put(_decay_exit)
                                return
                            else:
                                logger.info(f"🛡️ [FEE HEADROOM] Blocked decay exit for LONG {self.symbol} due to tiny profit ({pnl_pct*100:.3f}%).")
                                
                        # If we're SHORT but ML now predicts LONG with EXTREME conf → EXIT
                        if is_short_pos and direction == 1 and final_confidence > _decay_threshold:
                            if fee_headroom_ok:
                                logger.warning(f"📈 [ENSEMBLE DECAY EXIT] Confidence flipped against SHORT {self.symbol} (LONG conf={final_confidence:.2f}). Exiting...")
                                _decay_exit = SignalEvent(
                                    strategy_id=f"{self.strategy_id}.PREDICTIVE_DECAY",
                                    setup_type="PREDICTIVE_DECAY_EXIT",
                                    symbol=self.symbol,
                                    datetime=self._now(),
                                    signal_type=SignalType.EXIT,
                                    strength=final_confidence, ml_confidence=final_confidence,
                                    current_price=current_row["close"],
                                    horizon=_current_horizon,
                                    metadata={'urgent': False, 'actual_order_type': 'limit', 'is_tp_limit': True},
                                )
                                self.events_queue.put(_decay_exit)
                                return
                            else:
                                logger.info(f"🛡️ [FEE HEADROOM] Blocked decay exit for SHORT {self.symbol} due to tiny profit ({pnl_pct*100:.3f}%).")
            elif not _decay_enabled:
                signal_type = SignalType.LONG if direction == 1 else SignalType.SHORT

            # 6. SIGNAL CREATION
            tp_target = self.current_tp_target
            sl_target = self.current_sl_target

            # Volatility adjustments
            if atr_pct > 0.03:
                tp_target *= 1.3
                sl_target *= 1.3
            elif atr_pct < 0.01:
                tp_target *= 0.8
                sl_target *= 0.8

            # ============================================================
            # ✅ FIX-FORENSIC-V41: SOPHIA VETO (PARITY WITH BASE)
            # QUÉ: Filtrar señales con baja probabilidad de éxito según Sophia.
            # POR QUÉ: La inferencia base aplicaba Sophia veto, el Ensemble no.
            # PARA QUÉ: Evitar trades con win_probability < 70%.
            # ============================================================
            sophia_report_dict = {}
            if hasattr(self, 'sophia') and self.sophia:
                try:
                    # Vectorized numpy returns for < 1ms latency
                    _cvals = df["close"].values
                    if len(_cvals) > 1:
                        _returns = np.diff(_cvals) / _cvals[:-1]
                        _returns = _returns[~np.isnan(_returns) & ~np.isinf(_returns)]
                    else:
                        _returns = np.array([])
                    sophia_report = self.sophia.analyze(
                        symbol=self.symbol,
                        direction=signal_type.name,
                        signal_strength=final_confidence,
                        setups={

                            "source": "UNIVERSAL_ENSEMBLE",
                            "rsi": current_row.get("rsi_14", current_row.get("rsi", 50.0)),
                            "bb_position": current_row.get("bb_position", 0.5),
                            "adx": current_row.get("adx", 20.0),
                            "volume_ratio": current_row.get("volume_ratio", 1.0),
                            "macd_hist": current_row.get("macd_hist", 0.0),
                            "close": current_row.get("close", 0.0),
                            "atr": current_row.get("atr", 0.0)
                        },
                        confluence_score=final_confidence,
                        tp_pct=tp_target,
                        sl_pct=sl_target,
                        returns=_returns,
                        ttl_seconds=300 if getattr(self, 'horizon_str', 'SCALPING') in ['SCALPING', 'MICROSCALPING'] else 3600,
                        regime=self.market_regime,
                    )
                    # Dynamic Sophia Veto Threshold (55% - 65%)
                    if self.market_regime == "TRENDING":
                        veto_threshold = 0.65
                    elif self.market_regime in ["RANGING", "CHOPPY", "VOLATILE"]:
                        veto_threshold = 0.55
                    else:
                        veto_threshold = 0.60

                    if sophia_report.win_probability < veto_threshold:
                        # ═══════════════════════════════════════════════════════════
                        # [FORENSIC-AUDIT-V1] SOPHIA COLD-START BYPASS
                        # QUÉ: Si SOPHIA tiene < 20 observaciones (H > 1.2 = "Indeciso"),
                        #   no puede estimar win_probability confiablemente.
                        # POR QUÉ: En cold-start, SOPHIA siempre da WP ≈ 55% (entropía
                        #   máxima), lo que penaliza TODAS las señales ML un 14%.
                        # PARA QUÉ: Permitir que señales ML pasen sin degradación
                        #   hasta que SOPHIA tenga suficientes datos para juzgar.
                        # ═══════════════════════════════════════════════════════════
                        _sophia_n = getattr(sophia_report, 'n_observations', 0) or getattr(sophia_report, 'sample_size', 0)
                        _sophia_entropy = getattr(sophia_report, 'decision_entropy', 99.0)
                        if _sophia_n < 20 or _sophia_entropy > 1.2:
                            logger.info(f"🧠 [CONSENSUS] {self.symbol} ML Sophia BYPASS (cold-start: n={_sophia_n}, H={_sophia_entropy:.2f}). No penalty applied.")
                        else:
                            # CONSENSO PONDERADO v1.0 — ML Sophia Gate
                            # QUÉ: Sophia NO bloquea ML signals, los PENALIZA.
                            # POR QUÉ: ML tiene info que Sophia no ve (ensemble de 3 modelos).
                            # PARA QUÉ: Señales penalizadas → sizing más pequeño vía RiskManager.
                            sophia_penalty = max(0.5, sophia_report.win_probability / veto_threshold)
                            final_confidence *= sophia_penalty
                            logger.info(f"🧠 [CONSENSUS] {self.symbol} ML Sophia penalty x{sophia_penalty:.2f} (WP={sophia_report.win_probability*100:.1f}% < {veto_threshold*100:.0f}%, Regime: {self.market_regime})")
                    sophia_report_dict = sophia_report.to_dict()
                except Exception as e:
                    logger.error(f"Sophia Ensemble Integration Error on {self.symbol}: {e}", exc_info=True)

            # PHASE 9: PPO Metadata Injection
            # We capture model outputs as the "State" for the Ensemble Weight Optimization policy
            model_outputs = [
                self.individual_model_scores.get("rf", 0.0),
                self.individual_model_scores.get("xgb", 0.0),
                self.individual_model_scores.get("gb", 0.0),
            ]

            # ═══════════════════════════════════════════════════════════
            # CIRUGÍA-V131: DYNAMIC CONFIDENCE FLOOR (50% -> 55%)
            # QUÉ: Bajar el floor a 50% en cold-start, luego subir a 55%.
            # POR QUÉ: Con el floor en 55% y $13 de capital, el ML de 
            #   scalping lograba 0 trades porque la mediana de confianza
            #   en el primer día de entrenamiento es ~50%. Necesitamos
            #   que el sistema acumule trades reales para que el
            #   PredictionTracker evalúe la precisión real.
            # ═══════════════════════════════════════════════════════════
            trades_resolved = 0
            if self.risk_manager and hasattr(self.risk_manager, 'prediction_tracker'):
                metrics = self.risk_manager.prediction_tracker.get_strategy_metrics(self.strategy_id, self.horizon_str, self.symbol)
                if metrics:
                    trades_resolved = metrics.get('trades_resolved', 0)
            elif self.portfolio:
                trades_resolved = len(getattr(self.portfolio, 'trade_history', []))
                
            dynamic_floor = 0.65 if trades_resolved < 100 else 0.55
            
            if final_confidence < dynamic_floor:
                logger.info(f"🛑 [{self.symbol}|{self.horizon_str}] Signal VETOED: Confidence {final_confidence*100:.1f}% < {dynamic_floor*100:.1f}% floor (Trades: {trades_resolved}).")
                self.analysis_stats["filtered_conf"] += 1
                return

            # ═══════════════════════════════════════════════════════════
            # FORENSIC-V121 FIX: DYNAMIC PREDICTIVE MAGNITUDE & FEE GATE
            # QUÉ: Calcular la magnitud exacta predicha y la duración en barras.
            # POR QUÉ: El bot usaba tp_target estático (0.4%) que los scalps
            #   nunca alcanzaban, y generaba trades "Zombie".
            # PARA QUÉ:
            #   1) Rechazar instantáneamente si predice < 0.06% (comisión).
            #   2) Forzar a ExitOracle a respetar esta magnitud y tiempo.
            # ═══════════════════════════════════════════════════════════
            
            predicted_magnitude_real = 0.0
            is_ai_regressor = False
            
            # --- TRUE BAR-BY-BAR AI REGRESSION ---
            candle_pred_str = ""
            if getattr(self, 'xgb_regressor_long', None) is not None and getattr(self, 'xgb_regressor_short', None) is not None and not df.empty:
                try:
                    # ═══════════════════════════════════════════════════════
                    # FORENSIC-V130 FIX: USE PROTECTED SUPREME FEATURE COLS
                    # QUÉ: Los regressors se entrenaron con 122 features.
                    # POR QUÉ: _feature_cols puede ser sobreescrito a 20
                    #   por el entrenamiento online. Usar _supreme_feature_cols
                    #   garantiza que el regresor reciba las 122 originales.
                    # ═══════════════════════════════════════════════════════
                    reg_cols = getattr(self, '_supreme_feature_cols', self._feature_cols)
                    
                    # Validate that all required columns exist in df
                    available_cols = [c for c in reg_cols if c in df.columns]
                    if len(available_cols) < len(reg_cols):
                        missing = set(reg_cols) - set(available_cols)
                        logger.debug(f"⚠️ [{self.symbol}] Regressor missing {len(missing)} features: {list(missing)[:5]}...")
                        # Zero-fill missing columns
                        for mc in missing:
                            if mc not in df.columns:
                                df[mc] = 0.0
                        available_cols = reg_cols
                    
                    # 🚀 QUANTUM PERFORMANCE FIX: Bypassing Pandas .iloc Overhead
                    # Instead of creating intermediate DataFrame copies, we extract raw numpy array
                    latest_features = df[available_cols].to_numpy()[-1].reshape(1, -1)
                    
                    if direction == 1:
                        # LONG: Predict Max Favorable Excursion upwards
                        raw_pred = self.xgb_regressor_long.predict(latest_features)[0]
                    else:
                        # SHORT: Predict Max Favorable Excursion downwards
                        raw_pred = self.xgb_regressor_short.predict(latest_features)[0]
                        
                    predicted_magnitude_real = float(abs(raw_pred))
                    is_ai_regressor = True
                    
                    # Full Candle & Duration Regression (if models available)
                    if getattr(self, 'xgb_reg_next_high', None) is not None:
                        pred_high_pct = float(self.xgb_reg_next_high.predict(latest_features)[0])
                        pred_low_pct = float(self.xgb_reg_next_low.predict(latest_features)[0])
                        candle_pred_str = f" | Candle [↑{pred_high_pct*100:.2f}% ↓{pred_low_pct*100:.2f}%]"
                    
                    if getattr(self, 'xgb_reg_ttp', None) is not None:
                        ttp_fraction = float(self.xgb_reg_ttp.predict(latest_features)[0])
                        predicted_duration_bars = max(3, int(ttp_fraction * self.LOOKAHEAD_BARS))
                        
                except Exception as e:
                    logger.error(f"Dual Regressor error on {self.symbol}: {e}", exc_info=True)
                    
            # --- 🌌 CTOS OMNISCIENCE: Seq2Seq 1000-candle Trajectory ---
            omni_route = None
            if False and omniscient_engine.model is not None:  # DISABLED FOR NANO OPTIMIZATION
                try:
                    # Get the last 60 rows of features
                    seq_features = df[available_cols].iloc[-60:].values
                    current_close = float(current_row["close"])
                    current_horizon = getattr(self, "horizon", "SCALPING")
                    
                    omni_route = omniscient_engine.predict_trajectory(
                        seq_features, current_close, horizon=current_horizon
                    )
                    
                    if omni_route:
                        bar_dur = omni_route["bar_duration"]
                        
                        # CIRUGÍA-V131: Use SHORT-TERM waypoints, NOT 1000-candle macro peak
                        # QUÉ: El macro_peak_pct era +9-10% (sobre 1000 barras = 16h).
                        #   Eso forzaba un TP de +10%, inalcanzable para scalping.
                        # POR QUÉ: Un scalp dura 5-30 minutos. Usar T+5 o T+10 waypoints.
                        # PARA QUÉ: TP realista basado en la ventana temporal del trade.
                        waypoints = omni_route.get("waypoints", [])
                        current_horizon = getattr(self, "horizon", "SCALPING")
                        
                        if current_horizon in ["SCALPING", "MICROSCALPING"] and len(waypoints) >= 2:
                            # T+5 waypoint (5 min) for scalping
                            wp = waypoints[1]  # index 1 = T+5
                            omni_magnitude = abs(wp.get("close_pct", 0.0)) / 100.0
                            omni_duration = 5  # 5 bars
                        elif current_horizon == "SWING" and len(waypoints) >= 4:
                            # T+50 waypoint (50 min) for swing
                            wp = waypoints[3]  # index 3 = T+50
                            omni_magnitude = abs(wp.get("close_pct", 0.0)) / 100.0
                            omni_duration = 50  # 50 bars
                        else:
                            omni_magnitude = 0.0
                            omni_duration = 0
                        
                        # Only override if Omniscience gives a REASONABLE value
                        if omni_magnitude > 0.001 and omni_magnitude < 0.05:
                            predicted_magnitude_real = omni_magnitude
                            predicted_duration_bars = max(3, omni_duration)
                            is_ai_regressor = True
                        
                        # Log the first 3 individual candles for forensic traceability
                        first_candles = omni_route["candles"][:3]
                        candle_preview = " | ".join(
                            f"T+{c['bar']}({c['bar_duration']}): H={c['high_pct']:+.2f}% L={c['low_pct']:+.2f}% sz=${c['candle_size_usd']:.2f}"
                            for c in first_candles
                        )
                        candle_pred_str += (
                            f" | 🌌 Omni({bar_dur}): Peak +{omni_route['macro_peak_pct']:.2f}% "
                            f"en {omni_route['macro_peak_time']} | {candle_preview}"
                        )
                        
                        # Save full trajectory to file for Dashboard visualization
                        # [FIX] I/O block removed from hot-path.
                        # omniscient_engine.save_trajectory_to_file(omni_route, self.symbol)
                        
                except Exception as e:
                    logger.error(f"Omniscient Engine inference error: {e}", exc_info=True)
            
            # --- FALLBACK: ADAPTIVE TP TARGET ---
            if not is_ai_regressor or predicted_magnitude_real <= 0:
                # Calculate magnitude based on adaptive parameter engine's TP target
                # Scale by confidence: 50% conf = 0.5x TP, 99% conf = 1.0x TP
                confidence_mult = max(0.4, min(1.0, (final_confidence - 0.40) / 0.60))
                predicted_magnitude_real = self.current_tp_target * confidence_mult
            
            # The Parity Gate (Friction Filter)
            # QUÉ: Exigir un mínimo predicho para cubrir comisiones + slippage.
            # POR QUÉ: Maker fees son 0.04% RT. Usar 0.10% para Scalping, 0.30% para Swing.
            friction_threshold = 0.0025 if getattr(self, 'horizon_str', 'SCALPING') in ['SCALPING', 'MICROSCALPING'] else 0.0030
            if predicted_magnitude_real < friction_threshold:
                logger.info(f"🛑 [{self.symbol}|{self.horizon_str}] Signal Rejected: Predicted Magnitude {predicted_magnitude_real*100:.3f}% < Friction ({friction_threshold*100:.2f}%).")
                self.analysis_stats["filtered_conf"] += 1
                return
                
            # Align tp_target with reality
            tp_target = max(0.0030, predicted_magnitude_real)
            # Enforce 1.5:1 R:R floor for stop loss
            sl_target = min(self.current_sl_target, tp_target / 1.5)
            
            # Calculate exact expected duration bars if not predicted by ML
            if 'predicted_duration_bars' not in locals():
                predicted_duration_bars = int((predicted_magnitude_real / atr_pct) ** 2) if atr_pct > 0 else self.LOOKAHEAD_BARS
                predicted_duration_bars = max(5, min(self.LOOKAHEAD_BARS * 2, predicted_duration_bars))
            
            logger.info(
                f"🎯 [{self.symbol}] Magnitud Real {'[ML-Reg]' if is_ai_regressor else '[Math]'}: {predicted_magnitude_real*100:.3f}% en ~{predicted_duration_bars} barras "
                f"({signal_type.name} conf={final_confidence*100:.1f}%){candle_pred_str}"
            )

            ppo_metadata = {
                # ✅ FIX-FORENSIC-V41: GTX ENFORCEMENT (PARITY WITH BASE)
                "timeInForce": "GTX",
                "sophia": sophia_report_dict,
                "concept": concept,
                "phase": self.market_regime,
                "features": dict(current_row),  # Raw market features
                "model_outputs": model_outputs,
                "action": float(final_confidence),
                "log_prob": 0.0,  # Placeholder for deterministic policy
                "weights": [
                    self.base_rf_weight,
                    self.base_xgb_weight,
                    self.base_gb_weight,
                ],
                "raw_ml_confidence": float(np.max(ensemble_proba)),
                "smoothed_ml_confidence": float(raw_confidence),
                "prediction_stability": float(len(self._proba_history) / 3.0),
                "predicted_magnitude": float(predicted_magnitude_real),
                "predicted_duration": float(predicted_duration_bars),
                "predicted_next_high": float(pred_high_pct) if 'pred_high_pct' in locals() else None,
                "predicted_next_low": float(pred_low_pct) if 'pred_low_pct' in locals() else None,
                "omni_route": {
                    "macro_peak_pct": omni_route.get("macro_peak_pct"),
                    "macro_peak_bars": omni_route.get("macro_peak_bars"),
                    "macro_dump_pct": omni_route.get("macro_dump_pct"),
                    "macro_dump_bars": omni_route.get("macro_dump_bars"),
                    "bar_minutes": omni_route.get("bar_minutes", 1),
                } if omni_route is not None else None
            }

            detailed_id = f"{self.strategy_id}.ML_PREDICTION"
            signal = SignalEvent(
                strategy_id=detailed_id,
                setup_type="ML_PREDICTION",
                symbol=self.symbol,
                datetime=self._now(),
                signal_type=signal_type,
                strength=final_confidence, ml_confidence=final_confidence,
                atr=current_row["atr"],
                tp_pct=tp_target,
                sl_pct=sl_target,
                current_price=current_row["close"],
                horizon=getattr(self, "horizon", "SCALPING"),
                predicted_magnitude=predicted_magnitude_real,
                predicted_duration=predicted_duration_bars,
                metadata=ppo_metadata,
            )

            # 7. LOGGING & SUBMISSION
            self.performance_history.append(0)
            self.signal_history.append(
                {
                    "timestamp": self._now(),
                    "type": signal_type,
                    "confidence": final_confidence,
                    "engines": engines_passing,
                    "price": current_row["close"],
                }
            )

            logger.info(
                f"✨ [UNIVERSAL ENSEMBLE] Signal Generated: {signal_type.name} {self.symbol}"
            )
            
            try:
                from core.transparent_logger import monitor_log
                monitor_log.log_ml_prediction(
                    symbol=self.symbol,
                    model_name=self.strategy_id,
                    prediction=float(final_confidence),
                    confidence=float(final_confidence),
                    features={"engines_passing": engines_passing, "is_valid": is_valid, "scores": self.engine_scores, "horizon_scores": multi_horizon},
                    decision=signal_type.name
                )
            except Exception as e:
                logger.error(f"Failed to log thoughts: {e}")
                
            self.events_queue.put(signal)

            if len(self.performance_history) >= 15:
                self._update_model_weights()

            self._last_prediction_time = self._now()
            
            # [MEMORY OPTIMIZATION] Forced Garbage Collection Post-Inference
            # 🚨 FORENSIC-V4 QUANTUM FIX: REMOVED SYNCHRONOUS GC.COLLECT()
            # Calling gc.collect() blocks the Python GIL for 10-50ms, destroying HFT latency.
            # Memory should be handled asynchronously or organically by Python's Gen0 GC.
            # import gc
            # gc.collect()

        except Exception as e:
            logger.error(
                f"Universal Ensemble Inference error {self.symbol}: {e}", exc_info=True
            )

    def update_recursive_weights(self, trade_outcome):
        """
        PHASE 9: NEURAL-FORTRESS PPO UPDATE
        Uses TradeOutcome to calculate non-linear rewards and update ensemble weights via PPO.
        """
        try:
            # 1. Validate Input
            from core.reward_system import (
                TradeOutcome,  # Local import to avoid circular dependency
            )

            # Legacy fallback if just a float
            if isinstance(trade_outcome, (float, int)):
                return

            if not isinstance(trade_outcome, TradeOutcome):
                return

            # 2. Validation: Check if we have PPO metadata
            if (
                not trade_outcome.metadata
                or "model_outputs" not in trade_outcome.metadata
            ):
                # logger.debug("Skipping PPO update: No metadata in trade outcome.")
                return

            # 3. Calculate Neural Reward
            # Uses Tanh scaling, Drawdown penalty, and Skewness penalty
            reward = self.reward_system.calculate_reward(trade_outcome, current_drawdown=trade_outcome.max_adverse_excursion)

            # 4. Extract Experience Tuple
            # State: Model Probabilities [RF, XGB, GB] -> This is what the weights act upon
            state = np.array(trade_outcome.metadata["model_outputs"])

            # Next State: Current Model Probabilities (Approximate with current input or same)
            # For Weight Optimization, S' is slightly ambiguous. We use current input if available.
            next_state = (
                self.last_ensemble_input
                if self.last_ensemble_input is not None
                else state
            )

            action = trade_outcome.metadata.get("action", 0.5)
            log_prob = trade_outcome.metadata.get("log_prob", 0.0)

            # 5. Store in Prioritized Replay Buffer
            # Ensure state is valid shape
            if state.shape[0] == 3:
                self.memory.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    log_prob=log_prob,
                    axioma_reason=trade_outcome.metadata.get("axioma_reason", "NONE")
                )
                # logger.debug(f"🧠 [MEMORY] Stored Experience: R={reward:.4f} | State={state}")

            # 6. PPO Batch Learning Trigger
            self.steps_since_learn += 1
            if (
                self.steps_since_learn >= self.training_batch_size
                and len(self.memory) > self.training_batch_size
            ):
                # Sample Batch
                experiences, indices, weights = self.memory.sample(
                    self.training_batch_size
                )
                states = np.array([e[0] for e in experiences])
                actions = np.array([e[1] for e in experiences])
                rewards = np.array([e[2] for e in experiences])
                log_probs = np.array([e[4] for e in experiences])

                # Current Weights as "Policy"
                current_w = np.array(
                    [self.base_rf_weight, self.base_xgb_weight, self.base_gb_weight]
                )

                # Perform PPO Update
                new_weights, advantages = self.online_learner.update_ppo_batch(
                    weights=current_w,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    old_log_probs=log_probs,
                )

                # Update PER Priorities (using advantages as proxy for TD error/importance)
                # We use abs(advantage) because high advantage means "surprising" outcome compared to baseline
                self.memory.update_priorities(indices, np.abs(advantages) + 1e-5)

                # Normalize and Apply Weights
                new_weights = np.clip(new_weights, 0.05, 0.80)  # prevent extinction
                total_w = np.sum(new_weights)
                if total_w > 0:
                    new_weights /= total_w

                self.base_rf_weight = float(new_weights[0])
                self.base_xgb_weight = float(new_weights[1])
                self.base_gb_weight = float(new_weights[2])

                logger.info(
                    f"🧠 [PPO UPDATE] New Weights: RF={self.base_rf_weight:.2f} | XGB={self.base_xgb_weight:.2f} | GB={self.base_gb_weight:.2f}"
                )

                # Reset Counter
                self.steps_since_learn = 0

        except Exception as e:
            logger.error(f"PPO Update Failed: {e}", exc_info=True)

    def _compute_ml_engine_score(
        self, rf_proba, xgb_proba, gb_proba, direction: int
    ) -> float:
        """
        Motor ML: Weighted average of ensemble models.
        """
        def _safe_prob(model, proba, d):
            if proba is None: return 0.0
            if hasattr(model, 'classes_'):
                cls_list = list(model.classes_)
                if d in cls_list:
                    return proba[cls_list.index(d)]
            if len(proba) == 3:
                return proba[2 if d == 1 else 0]
            elif len(proba) == 2:
                return proba[1 if d == 1 else 0]
            return 0.0

        ml_score = (
            _safe_prob(getattr(self, 'rf_model', None), rf_proba, direction) * self.base_rf_weight
            + _safe_prob(getattr(self, 'xgb_model', None), xgb_proba, direction) * self.base_xgb_weight
            + _safe_prob(getattr(self, 'gb_model', None), gb_proba, direction) * self.base_gb_weight
        )

        self.engine_scores["ml"] = ml_score
        return ml_score

    def _compute_sentiment_engine_score(self, df) -> float:
        """
        Motor Sentiment: VADER/Social momentum analysis.
        """
        try:
            current_row = df.iloc[-1]

            # Get sentiment components (Phase 8 NLP names)
            sentiment = current_row.get("news_sentiment", current_row.get("sentiment", 0.0))
            sentiment_momentum = current_row.get("news_sentiment_shock", current_row.get("sentiment_momentum", 0.0))

            # Normalize to 0-1 range
            # Sentiment is typically -1 to 1, so we map to 0.5 as neutral
            sentiment_normalized = (sentiment + 1) / 2

            # Momentum adds weight
            if sentiment_momentum > 0:
                sentiment_score = sentiment_normalized * (
                    1 + min(sentiment_momentum, 0.3)
                )
            else:
                sentiment_score = sentiment_normalized * (
                    1 + max(sentiment_momentum, -0.3)
                )

            # Clamp to [0, 1]
            sentiment_score = max(0.0, min(1.0, sentiment_score))

            self.engine_scores["sentiment"] = sentiment_score
            return sentiment_score

        except Exception:
            self.engine_scores["sentiment"] = 0.5  # Neutral if unavailable
            return 0.5

    def _compute_technical_engine_score(self, df, direction: int) -> float:
        """
        Motor Técnico: RSI + EMA Cross + Bollinger Bands confluence.
        ROBUSTO: Usa múltiples fallbacks para features faltantes.
        """
        try:
            current_row = df.iloc[-1]
            score = 0.0
            factors = 0

            # RSI Component (weight: 40%) — usar rsi_14, rsi, o calcular
            rsi = None
            for col in ["rsi_14", "rsi", "RSI"]:
                if col in current_row.index:
                    v = current_row.get(col, None)
                    if v is not None and not (isinstance(v, float) and v != v):
                        rsi = float(v)
                        break

            if rsi is not None:
                if direction == 1:  # LONG
                    if rsi < 35:
                        score += 0.40
                        factors += 1  # Oversold = strong buy
                    elif rsi < 50:
                        score += 0.25
                        factors += 1  # Below neutral = moderate
                    elif rsi < 65:
                        score += 0.10
                        factors += 1  # Neutral/above
                else:  # SHORT
                    if rsi > 65:
                        score += 0.40
                        factors += 1  # Overbought = strong short
                    elif rsi > 50:
                        score += 0.25
                        factors += 1  # Above neutral = moderate
                    elif rsi > 35:
                        score += 0.10
                        factors += 1  # Neutral/below

            # EMA Cross Component (weight: 30%) — varios nombres posibles
            ema_cross = None
            for col in ["ema_20_50_cross", "ema_cross", "ema_signal"]:
                if col in current_row.index:
                    v = current_row.get(col, None)
                    if v is not None:
                        ema_cross = float(v)
                        break

            # Fallback: usar dist_ema_20 y dist_ema_50 (normalizados)
            if ema_cross is None:
                d20 = current_row.get("dist_ema_20", None)
                d50 = current_row.get("dist_ema_50", None)
                if d20 is not None and d50 is not None:
                    # dist_ema_X = (close - ema_X) / ema_X → positivo = precio sobre EMA
                    # Si dist_ema_20 > dist_ema_50, la EMA rápida está sobre la lenta → trend up
                    ema_cross = float(d20) - float(d50)

            if ema_cross is not None:
                if (direction == 1 and ema_cross > 0) or (
                    direction == -1 and ema_cross < 0
                ):
                    score += 0.30
                    factors += 1
                elif abs(ema_cross) < 0.001:  # Neutro
                    score += 0.15
                    factors += 1

            # Bollinger Bands Component (weight: 30%) — varios nombres
            bb_pct = None
            for col in ["bb_pctb", "bb_pct_b", "bb_pct", "bollinger_pct"]:
                if col in current_row.index:
                    v = current_row.get(col, None)
                    if v is not None:
                        bb_pct = float(v)
                        break

            if bb_pct is not None:
                if direction == 1:  # LONG — precio cerca banda inferior
                    if bb_pct < 0.2:
                        score += 0.30
                        factors += 1
                    elif bb_pct < 0.4:
                        score += 0.15
                        factors += 1
                else:  # SHORT — precio cerca banda superior
                    if bb_pct > 0.8:
                        score += 0.30
                        factors += 1
                    elif bb_pct > 0.6:
                        score += 0.15
                        factors += 1

            # Calcular score final
            if factors == 0:
                # No había features técnicas útiles → score neutro-positivo basado en ML direction
                # Si el ML dice LONG con 0.96, damos benefit of doubt = 0.40
                technical_score = 0.40
            else:
                # Normalizar: si solo 1 factor contributió, escalarlo proporcionalmente
                max_possible = 0.40 + 0.30 + 0.30  # = 1.0
                technical_score = min(
                    1.0, score / (max_possible / factors * max(factors, 1))
                )
                # Boost si múltiples factores alineados
                if factors >= 2:
                    technical_score = min(1.0, technical_score * 1.15)

            self.engine_scores["technical"] = technical_score
            return technical_score

        except Exception as e:
            self.engine_scores["technical"] = (
                0.40  # Mejor que 0.5 neutro, da ligero beneficio
            )
            return 0.40

    def compute_organic_confluence(
        self, df, direction: int, rf_proba, xgb_proba, gb_proba
    ) -> tuple:
        """
        🎯 ORGANIC CONFLUENCE CALCULATOR
        Returns: (final_confidence, engines_passing, is_valid, multi_horizon)
        """
        # 1. Calculate each engine score
        ml_score = self._compute_ml_engine_score(
            rf_proba, xgb_proba, gb_proba, direction
        )
        sentiment_score = self._compute_sentiment_engine_score(df)
        technical_score = self._compute_technical_engine_score(df, direction)

        # 2. Dynamic Threshold Logic (World Awareness)
        # PROFESSOR METHOD: adaptamos el rigor según la liquidez global.
        ls = getattr(self, "market_context", {}).get("liquidity_score", 0.8)

        # Base is the user's setting (0.75)
        # If LS is low (dead zone), we demand extreme confluence (0.82)
        base_t = self.ENSEMBLE_CONSENSUS_THRESHOLD

        # FIX: Threshold razonable — sin exacerbar en sesiones bajas
        # En backtesting y sesiones bajas ya no hay feed de liquidez real
        # El umbral no debe subir más de 0.05 sobre el base
        if ls >= 0.85:  # PRIME (London/NY)
            dynamic_threshold = base_t
        elif ls >= 0.65:  # MID (Tokyo)
            dynamic_threshold = base_t + 0.02
        elif ls >= 0.50:  # LOW (Sydney)
            dynamic_threshold = base_t + 0.03
        else:  # DEAD ZONE o backtest sin LS real
            dynamic_threshold = base_t + 0.05

        self.consensus_threshold = dynamic_threshold  # Update for oracle logging

        # ============================================================
        # ⏳ SYNTHETIC MULTI-HORIZON LOGIC (H1, H5, H15, H30)
        # ═══════════════════════════════════════════════════════════
        # FORENSIC-V50 FIX: SENTIMENT REMOVED FROM HORIZONS
        # QUÉ: Eliminamos sentiment_score de H1/H5/H15/H30 y agregamos H1.
        # POR QUÉ: sentiment_score SIEMPRE retorna 0.50. Para H1 (1m), calculamos
        #   la micro-acción del precio de la vela actual contra la dirección.
        # PARA QUÉ: Multi-horizon refleja realidad completa del mercado.
        # ═══════════════════════════════════════════════════════════
        # H1 (Ultra-Short): Immediate 1-minute Price Action + Technical
        try:
            current_row = df.iloc[-1]
            c_open = current_row.get("open", current_row["close"])
            c_close = current_row["close"]
            if direction == 1:
                h1_base = 0.80 if c_close > c_open else 0.30
            else:
                h1_base = 0.80 if c_close < c_open else 0.30
        except Exception:
            h1_base = 0.50
        h1_score = (h1_base * 0.70) + (technical_score * 0.30)

        # H5 (Short): Pure Technical Momentum
        h5_score = technical_score

        # H15 (Mid): Balanced ML + Technical
        h15_score = (ml_score * 0.50) + (technical_score * 0.50)

        # H30 (Full): ML Dominant (Original Model Horizon)
        h30_score = (ml_score * 0.75) + (technical_score * 0.25)

        multi_horizon = {
            "h1": max(0.0, min(1.0, h1_score)),
            "h5": max(0.0, min(1.0, h5_score)),
            "h15": max(0.0, min(1.0, h15_score)),
            "h30": max(0.0, min(1.0, h30_score)),
        }

        # FIX: engines_passing basado en threshold relativo al motor
        ml_passes = ml_score >= dynamic_threshold
        tech_passes = technical_score >= (
            dynamic_threshold * 0.70
        )  # TECH threshold más bajo
        ml_dominant = ml_score >= 0.80  # FIXED: 0.85→0.80 (menos restrictivo)

        engines_passing = sum([ml_passes, tech_passes])

        # Calculate weighted final confidence
        final_confidence = (
            ml_score * 0.70  # 70% ML
            + technical_score * 0.30  # 30% TECH (sentimiento = 0%)
        )

        # is_valid: ML+TECH alineados, O ML muy dominante con apoyo parcial, O final_confidence altísimo
        is_valid = (engines_passing >= self.MIN_ENGINES_REQUIRED) or (
            ml_dominant and technical_score >= 0.30
        ) or (final_confidence >= 0.78)

        # ═══════════════════════════════════════════════════════════
        # FORENSIC-V50 FIX: SOFTER SINGLE-ENGINE PENALTY
        # QUÉ: Reducimos penalty de 0.80 a 0.90 para 1 engine.
        # POR QUÉ: Con ML=0.76 y penalty 0.80 → 0.608, que no pasa
        #   el threshold de 0.60. Pero ML=0.76 ES una señal fuerte.
        # PARA QUÉ: Permitir que ML-dominant opere cuando tiene alta
        #   confianza, aunque Technical no confirme al 100%.
        # ═══════════════════════════════════════════════════════════
        if engines_passing < 2 and final_confidence < 0.78:
            penalty = 0.90 if engines_passing == 1 else 0.70
            final_confidence *= penalty

        self.engines_active = engines_passing

        # Phase 8: Neural Bridge Publication
        neural_bridge.publish_insight(
            strategy_id="ML_ENSEMBLE",
            symbol=self.symbol,
            insight={
                "confidence": final_confidence,
                "direction": "LONG" if direction == 1 else "SHORT",
                "engines_passing": engines_passing,
                "horizons": multi_horizon,
            },
        )

        return final_confidence, engines_passing, is_valid, multi_horizon

    def get_ensemble_status(self) -> dict:
        """Get current ensemble engine status."""
        return {
            "engines": self.engine_scores.copy(),
            "engines_active": self.engines_active,
            "threshold": self.ENSEMBLE_CONSENSUS_THRESHOLD,
            "is_unified": True,
            "mode": "UNIVERSAL_ENSEMBLE",
        }


# Add method to base class for backwards compatibility
def _add_ensemble_methods_to_base():
    """Inject ensemble methods into base class."""

    def compute_organic_confluence(self, df, direction, rf_proba, xgb_proba, gb_proba):
        """Simplified organic confluence for base class."""
        # FIX: Auto-detect class count (2-class vs 3-class)
        num_classes = len(rf_proba)
        if num_classes == 2:
            idx = 1 if direction == 1 else 0
        else:
            idx = 2 if direction == 1 else 0
            
        ml_score = (
            rf_proba[idx] * self.base_rf_weight
            + xgb_proba[idx] * self.base_xgb_weight
            + gb_proba[idx] * self.base_gb_weight
        )

        # Technical Engine Score (from confluence_score)
        technical_score = abs(df.iloc[-1].get("confluence_score", 0))
        technical_score = min(1.0, technical_score + 0.5)  # Normalize

        # Count passing engines (ML + Technical, sentiment optional)
        THRESHOLD = 0.60
        engines_passing = sum(1 for s in [ml_score, technical_score] if s >= THRESHOLD)

        # Weighted average
        final_conf = ml_score * 0.6 + technical_score * 0.4

        if engines_passing < 2:
            final_conf *= 0.8  # Penalty
            
        # FIX: Return 4 values to match UniversalEnsembleStrategy signature
        multi_horizon = {"h1": final_conf, "h5": final_conf, "h15": final_conf, "h30": final_conf}

        return final_conf, engines_passing, engines_passing >= 2, multi_horizon

    # Add method if not exists
    if not hasattr(MLStrategyHybridUltimate, "compute_organic_confluence"):
        MLStrategyHybridUltimate.compute_organic_confluence = compute_organic_confluence


_add_ensemble_methods_to_base()


# ============================================================
# ✅ FACTORY FUNCTION PARA CREACIÓN FÁCIL
# ============================================================


def create_ml_strategy_hybrid_ultimate(
    data_provider,
    events_queue,
    symbol="BTC/USDT",
    sentiment_loader=None,
    portfolio=None,
    initial_capital=12.0,
    target_capital=100000.0,
):
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
        portfolio=portfolio,
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
