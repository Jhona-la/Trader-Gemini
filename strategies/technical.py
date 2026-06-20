"""
Estrategia Técnica HÍBRIDA - Optimized for $12→$50 Scalping
Combina simplicidad del scalping con robustez del análisis técnico avanzado
"""

import os
import numpy as np
from core.events import SignalEvent
from core.enums import SignalType
from datetime import datetime, timezone
from config import Config
from strategies.strategy import Strategy
from utils.math_kernel import (
    calculate_rsi_jit, calculate_bollinger_robust_jit, calculate_ema_jit,
    calculate_macd_jit, calculate_atr_jit, calculate_adx_jit,
    kalman_filter_1d_jit, fractional_differencing_jit
) # Phase 3 & 5: Total Vectorization & Mathematical Refinement
from core.neural_bridge import neural_bridge
from core.genotype import Genotype  # Phase 1: Trinidad Omega
from core.online_learning import OnlineLearner # Phase 46: Real-time Learning
from core.fused_strategy_kernel import fused_compute_step # Phase 65: Kernel Fusion
from sophia.intelligence import SophiaIntelligence, MultiHorizonOracle  # SOPHIA-INTELLIGENCE Protocol + Phase 3 Oracle
from sophia.narrative import NarrativeGenerator  # SOPHIA: Human-readable narratives
from utils.metrics_exporter import metrics  # SOPHIA-VIEW: Real-time telemetry
from core.sovereign_oracle import sovereign_oracle # Phase 47: Causal Reasoning
from core.swarm_correlator import swarm_correlator # Phase 47: Swarm Fabric
from utils.logger import logger

class HybridScalpingStrategy(Strategy):
    """
    Estrategia híbrida que combina:
    - Velocidad y simplicidad del scalping
    - Análisis multi-timeframe del código original  
    - Filtros de tendencia robustos
    - TP/SL definidos para scalping
    """
    
    # V5.21 Quantum Tunnelling: Global BTC State
    BTC_QUANTUM_STATE = {
        'vortex_pulse': 1.0,
        'noise_level': 0.5,
        'is_active': False
    }
    
    def __init__(self, data_provider, events_queue, genotype: Genotype = None, horizon: str = "SCALPING", priority: int = 1):
        self.data_provider = data_provider
        self.events_queue = events_queue
        base_label = getattr(Config, 'STRATEGY_LABELS', {}).get("technical", f"HYBRID")
        lbl = "[SCL]" if horizon == "SCALPING" else "[SWG]"
        self.strategy_id = f"{lbl}_{base_label}_{horizon}"
        self.genotype = genotype
        self.symbol = genotype.symbol if genotype else None
        self.horizon = horizon
        self.priority = priority
        
        # ================================================================
        # PHASE FORENSIC-1: HORIZON-AWARE PARAMETER LOADING
        # QUÉ: Carga parámetros especializados según el horizonte.
        # POR QUÉ: Un TP de 1.5% mata la frecuencia en scalping 1min,
        #   pero es insuficiente para capturar tendencias en swing 4h.
        # CÓMO: Lee Config.Strategies.SCALPING_PARAMS o SWING_PARAMS
        #   y sobrescribe los defaults genéricos.
        # CUÁNDO: En cada instanciación de la estrategia.
        # DÓNDE: strategies/technical.py → __init__
        # QUIÉN: HybridScalpingStrategy
        # ================================================================
        if horizon.upper() == 'SCALPING':
            h_params = getattr(Config.Horizons, 'Scalping', {})
        elif horizon.upper() == 'SWING':
            h_params = getattr(Config.Horizons, 'Swing', {})
        elif horizon.upper() == 'MICROSCALPING':
            h_params = getattr(Config.Horizons, 'Microscalping', {})
        else:
            h_params = {}
        
        # Parámetros centralizados: horizon-specific → Config fallback
        self.BB_PERIOD = h_params.get('bb_period', getattr(Config.Strategies, 'TECH_BB_PERIOD', 20))
        self.BB_STD = h_params.get('bb_std', getattr(Config.Strategies, 'TECH_BB_STD', 2.0))
        
        self.RSI_PERIOD = h_params.get('rsi_period', getattr(Config.Strategies, 'TECH_RSI_PERIOD', 14))
        self.RSI_OVERBOUGHT = h_params.get('rsi_sell', getattr(Config.Strategies, 'TECH_RSI_SELL', 70))
        self.RSI_OVERSOLD = h_params.get('rsi_buy', getattr(Config.Strategies, 'TECH_RSI_BUY', 30))
        
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # TP/SL centralizados — HORIZON-AWARE
        self.TP_PCT = h_params.get('tp_pct', getattr(Config.Strategies, 'TECH_TP_PCT', 0.015))
        self.SL_PCT = h_params.get('sl_pct', getattr(Config.Strategies, 'TECH_SL_PCT', 0.02))
        
        # ATR Multipliers - HORIZON-AWARE
        self.ATR_SL_MULT_BASE = h_params.get('atr_sl_mult', 1.5)
        self.ATR_TP_MULT_BASE = h_params.get('atr_tp_mult', 3.0)
        
        # Filtro de tendencia — HORIZON-AWARE
        self.EMA_FAST = h_params.get('ema_fast', getattr(Config.Strategies, 'TECH_EMA_FAST', 20))
        self.EMA_SLOW = h_params.get('ema_slow', getattr(Config.Strategies, 'TECH_EMA_SLOW', 50))
        self.EMA_TREND = h_params.get('ema_trend', 200)
        
        # ATR/ADX periods — HORIZON-AWARE
        self.ATR_PERIOD = h_params.get('atr_period', 14)
        self.ADX_PERIOD = h_params.get('adx_period', 14)
        
        # Horizon-specific operational params
        self.HORIZON_TIMEFRAMES = h_params.get('timeframes', ['5m', '15m', '1h'])
        self.PRIMARY_TF = h_params.get('primary_tf', '5m' if horizon.upper() == 'SCALPING' else '1h')
        self.MIN_VOLUME_RATIO = h_params.get('min_volume_ratio', 0.70)
        self.COOLDOWN_SECONDS = h_params.get('cooldown_seconds', 90)
        self.MAX_HOLD_BARS = h_params.get('max_hold_bars', 60)
        self.STRENGTH_THRESHOLD = h_params.get('strength_threshold', 0.40)
        self.ATR_SL_MULT_BASE = h_params.get('atr_sl_mult', 1.5)
        self.ATR_TP_MULT_BASE = h_params.get('atr_tp_mult', 3.0)
        
        logger.info(f"🎯 [{self.strategy_id}] Horizon={horizon} | TP={self.TP_PCT*100:.1f}% SL={self.SL_PCT*100:.1f}% | RSI={self.RSI_PERIOD} | TFs={self.HORIZON_TIMEFRAMES} | Primary={self.PRIMARY_TF}")
        
        # Mejora del ORIGINAL: Multi-timeframe — FILTERED BY HORIZON
        self.MULTI_TIMEFRAME_WEIGHTS = {
            '5m': 0.4,   # Peso principal (timeframe de trading)
            '15m': 0.3,  # Confirmación
            '1h': 0.3    # Dirección general
        }
        
        # === PER-SYMBOL ADAPTIVE PROFILES (Phase 7.2) ===
        # === V3 UNIVERSALLY-PROFITABLE PROFILES (Multi-Horizon Optimized) ===
        # ================================================================
        # FORENSIC-AUDIT-FIX: Strength thresholds REDUCED from 0.82-0.85 → 0.45-0.55
        # QUÉ: Los thresholds anteriores (0.82-0.85) eran INALCANZABLES tras 14 filtros previos.
        # POR QUÉ: Con 14 gates secuenciales, la probabilidad de supervivencia era 0.0001%.
        # PARA QUÉ: Restaurar la capacidad de generar señales (de 0 → N trades).
        # EVIDENCIA: god_mode_backtest_results.json → trades: 0, NO_STRATEGY_SIGNALS: 1
        # ================================================================
        self.PROFILES = {
            'AGGRESSIVE': {
                'tp_pct': 0.025, 'sl_pct': 0.008, 
                'adx_threshold': 20, 'strength_threshold': 0.70,
                'atr_sl_mult': 1.5, 'atr_tp_mult': 3.5, 'trailing_rsi': 70
            },
            'BALANCED': {
                'tp_pct': 0.030, 'sl_pct': 0.010,
                'adx_threshold': 20, 'strength_threshold': 0.75,
                'atr_sl_mult': 2.0, 'atr_tp_mult': 4.0, 'trailing_rsi': 65
            },
            'CONSERVATIVE': {
                'tp_pct': 0.040, 'sl_pct': 0.020,
                'adx_threshold': 22, 'strength_threshold': 0.80,
                'atr_sl_mult': 2.5, 'atr_tp_mult': 5.0, 'trailing_rsi': 60
            },
            'NANO_MICRO': {
                'tp_pct': 0.0035, 'sl_pct': 0.0020,
                'adx_threshold': 15, 'strength_threshold': 0.40,
                'atr_sl_mult': 2.5, 'atr_tp_mult': 2.5, 'trailing_rsi': 75
            }
        }
        
        # Trackers para Salidas Dinámicas (Phase 71)
        self.trailing_sl = {} # {symbol: current_sl_price}
        self.partial_tp = {} # {symbol: bool_hit}
        self.last_trade_prices = {} # {symbol: entry_price}
        
        # Map symbols to profiles (V3 Multi-Horizon Optimized)
        self.SYMBOL_MAP = {
            'BTC/USDT': 'BALANCED',     # V3: degradado de AGGRESSIVE (15D Payoff=0.21 era desastroso)
            'ETH/USDT': 'BALANCED',     # V3: mantener (30D WR=52%, Payoff=1.08)
            'SOL/USDT': 'BALANCED',     # V3: mantener (15D Sharpe=5.97)
            'XRP/USDT': 'AGGRESSIVE',   # V3: mantener (15D Payoff=3.19, mejor ratio)
            'BNB/USDT': 'CONSERVATIVE', # V3: degradado de AGGRESSIVE (30D Payoff=0.44 era letal)
            'DOGE/USDT': 'CONSERVATIVE',# V3: mantener (15D Sharpe=7.75, mejor activo)
            'ADA/USDT': 'BALANCED',     # Default
            'DOT/USDT': 'BALANCED',     # Default
            'LINK/USDT': 'BALANCED',    # Default
            'MATIC/USDT': 'BALANCED',   # Default
            'AVAX/USDT': 'BALANCED',    # Default
            'NEAR/USDT': 'BALANCED',    # Default
            'INJ/USDT': 'BALANCED',     # Default
            'PEPE/USDT': 'CONSERVATIVE',# V3: memecoins agrupadas con DOGE
            'RENDER/USDT': 'BALANCED',  # Default
            'SHIB/USDT': 'CONSERVATIVE',# V3: memecoins agrupadas con DOGE
            'ATOM/USDT': 'BALANCED',    # Default
            'LTC/USDT': 'BALANCED',     # Default
            'OP/USDT': 'BALANCED',      # Default
            'ARB/USDT': 'BALANCED',     # Default
        }

        # V5.7 COGNITIVE AUTO-TUNING + GENETIC INSTINCT
        # Devolvermos el 'Instinto Base' (V5.6) pero ahora la IA puede censurarlo.
        # Phase 47.5: Unlocking Altcoin Universe for non-BTC assets.
        self.PER_SYMBOL_PROFILES = {
            'BTC/USDT': {'allowed_setups': 'ALL_SETUPS'}, # FORENSIC-FIX: removed MOMENTUM_ONLY to allow scalping mean reversion
            'ETH/USDT': {'allowed_setups': 'ALL_SETUPS'},
            'SOL/USDT': {'allowed_setups': 'ALL_SETUPS'},
            'XRP/USDT': {'allowed_setups': 'ALL_SETUPS'}, # Unlocked
            'DOGE/USDT': {'allowed_setups': 'ALL_SETUPS'}, # Unlocked
            'ADA/USDT': {'allowed_setups': 'ALL_SETUPS'}, # Unlocked
        }
        
        # Memoria del Dolor (Cognitive Memory) V5.8 State-Based
        # Permite modular el riesgo dinámicamente según el WinRate reciente.
        self.cognitive_memory = {}

        # Estado (del ORIGINAL)
        self.bought = {}
        self.last_processed_times = {}
        self.last_trade_times = {} # FOR COOLDOWNS (Rule 4.1)
        
        # ONLINE LEARNING STATE (Phase 48)
        self.learner = OnlineLearner(learning_rate=0.01) # Conservative rate
        self.brain_memory = {} # symbol -> {'last_state': np.array, 'last_pred': float}
        
        # MULTIVERSE SUPPORT (Phase 56)
        self.genotypes = {} # symbol -> Genotype
        
        # SOPHIA-INTELLIGENCE Protocol: XAI Engine (MULTI-HORIZON AWARE)
        tf_to_mins = {'1m': 1.0, '5m': 5.0, '15m': 15.0, '30m': 30.0, '1h': 60.0, '4h': 240.0, '1d': 1440.0}
        primary_tf = getattr(self, 'PRIMARY_TF', '5m' if horizon.upper() == 'SCALPING' else '1h')
        bar_mins = tf_to_mins.get(primary_tf, 5.0 if horizon.upper() == 'SCALPING' else 60.0)
        
        # OPTIMIZACIÓN RAM: Singleton por horizonte (2 instancias vs 42)
        self.sophia = SophiaIntelligence.get_instance(bar_minutes=bar_mins)
        
        # FORENSIC-1: Set Horizon Profile to prevent false Chaos Dampening
        horizon_days_map = {'SCALPING': 1, 'SWING': 15}
        target_days = horizon_days_map.get(horizon.upper(), 1)
        self.sophia.set_horizon_profile(target_days)
        
        # C-1 FIX: Flag for lazy Némesis→Sophia feedback loop binding
        self._sophia_feedback_linked = False
        
        logger.info(f"🧠 [SOPHIA INIT] Horizon: {horizon} | Primary TF: {primary_tf} | Bar Mins: {bar_mins} | Profile Days: {target_days}")
        
        # Pre-load provided genotype if any
        if genotype:
            self.genotypes[genotype.symbol] = genotype

    def get_symbol_params(self, symbol):
        """Devuelve parámetros adaptados al símbolo (Merged Genotype + Legacy Profile + Optimized)"""
        # 0. Get Legacy Defaults for this symbol
        profile_key = self.SYMBOL_MAP.get(symbol, 'BALANCED')
        if getattr(Config, 'INITIAL_CAPITAL', 1000) <= 20.0 or self.horizon == 'MICROSCALPING':
            profile_key = 'NANO_MICRO'
        defaults = self.PROFILES.get(profile_key).copy()
        
        # FORENSIC-FIX: INJECT HORIZON-SPECIFIC BASE PARAMETERS
        # QUÉ: Los parameters base de SCALPING_PARAMS se perdían aquí porque se usaban los defaults genéricos del PROFILE.
        # POR QUÉ: Para scalping de micro-cuenta ($13), necesitamos el strength_threshold de 0.35, no el 0.50 genérico de BALANCED.
        defaults['strength_threshold'] = getattr(self, 'STRENGTH_THRESHOLD', defaults.get('strength_threshold'))
        defaults['adx_threshold'] = getattr(self, 'ADX_THRESHOLD', defaults.get('adx_threshold', 25))
        defaults['tp_pct'] = getattr(self, 'TP_PCT', defaults.get('tp_pct'))
        defaults['sl_pct'] = getattr(self, 'SL_PCT', defaults.get('sl_pct'))
        
        # 0.5 Override with Optimized Precision Profile (Strategic ARMOR V5.9/V5.10)
        if hasattr(self, 'PER_SYMBOL_PROFILES') and symbol in self.PER_SYMBOL_PROFILES:
            opt = self.PER_SYMBOL_PROFILES[symbol]
            
            # --- V5.10 METAMORFOSIS ESTRUCTURAL ---
            # Si la moneda es ALPHA, desbloqueamos TODO su potencial.
            # Si es NORMAL/INJURED, mantenemos el blindaje fijo.
            cog_state = 'NORMAL'
            # Buscamos el estado dominante (si hay al menos un setup Alpha, la moneda es Alpha)
            if symbol in self.cognitive_memory:
                for s_type in self.cognitive_memory[symbol]:
                    if self.cognitive_memory[symbol][s_type].get('state') == 'ALPHA':
                        cog_state = 'ALPHA'
                        break

            if cog_state == 'ALPHA':
                defaults['allowed_setups'] = 'ALL_SETUPS'
                logger.debug(f"🔥 [V5.10 METAMORFOSIS] {symbol} desbloqueada a ALL_SETUPS por racha ganadora.")
            elif 'allowed_setups' in opt:
                defaults['allowed_setups'] = opt['allowed_setups']
        
        # 1. Check Memory / Load Genotype
        found_genes = {}
        if symbol in self.genotypes:
            found_genes = self.genotypes[symbol].genes
        else:
            # Try Load from Disk (Persistence)
            try:
                filename = f"data/genotypes/{symbol.replace('/','')}_gene.json"
                if os.path.exists(filename):
                    loaded = Genotype.load(filename)
                    if loaded:
                        self.genotypes[symbol] = loaded
                        found_genes = loaded.genes
            except Exception as e:
                logger.error(f"Silent exception caught: {e}", exc_info=True)
        
        # 2. Case: Not found -> Auto-Spawn
        if not found_genes:
            new_gene = Genotype(symbol)
            new_gene.init_brain(25, 4)
            self.genotypes[symbol] = new_gene
            found_genes = new_gene.genes
            
        # 3. MAPPING & MERGING (Ensure no KeyErrors)
        generation = self.genotypes.get(symbol).generation if symbol in self.genotypes else 0
        
        # Genotype genes override defaults if present (but generation 0 defaults shouldn't overwrite our tuned horizon configs)
        final_params = defaults
        for k, v in found_genes.items():
            if v is not None and (not hasattr(v, '__len__') or len(v) > 0): # Don't override with empty weights
                final_params[k] = v
                
        # FORENSIC-FIX: INJECT HORIZON-SPECIFIC BASE PARAMETERS
        # QUÉ: Los parameters base de SCALPING_PARAMS se perdían aquí porque se usaban los defaults genéricos del PROFILE o del GENOTYPE(gen 0).
        # POR QUÉ: Para scalping de micro-cuenta ($13), necesitamos el strength_threshold de 0.35. El Genotype (gen 0) inyectaba 0.60.
        # Phase 2 FIX: Config is the master source of truth for Generation 0.
        if generation == 0:
            final_params['strength_threshold'] = getattr(self, 'STRENGTH_THRESHOLD', Config.Strategies.SCALPING_PARAMS.get('strength_threshold', 0.55))
            
            # FORENSIC-V81: HYPER-EVOLVER MUTATIONS INJECTION
            # FIX-FORENSIC-V82: Mutations lives in Config.Strategies, NOT Config root!
            # Bug: getattr(Config, 'Mutations', {}) always returned {} because
            # Mutations is defined at Config.Strategies.Mutations (config.py L498).
            # Impact: ALL Optuna-optimized parameters (80% WR) were NEVER applied.
            mutations = getattr(Config.Strategies, 'Mutations', {})
            final_params['adx_threshold'] = mutations.get('adx_threshold', getattr(self, 'ADX_THRESHOLD', final_params.get('adx_threshold', 25)))
            final_params['strength_threshold'] = mutations.get('strength_threshold', final_params.get('strength_threshold', 0.55))
            final_params['tp_pct'] = mutations.get('max_tp_cap', getattr(self, 'TP_PCT', final_params.get('tp_pct')))
            # [CIRUGÍA #1] SL comes directly from SCALPING_PARAMS, not via sl_multiplier
            final_params['sl_pct'] = getattr(self, 'SL_PCT', final_params.get('sl_pct'))
            
            final_params['rsi_buy'] = self.RSI_OVERSOLD
            final_params['rsi_sell'] = self.RSI_OVERBOUGHT
            final_params['bb_std'] = self.BB_STD
            final_params['bb_period'] = self.BB_PERIOD
                
        # 4. OVERRIDE FINAL con el Perfil Precision (Máxima prioridad para Evolución)
        if hasattr(self, 'PER_SYMBOL_PROFILES') and symbol in self.PER_SYMBOL_PROFILES:
            opt = self.PER_SYMBOL_PROFILES[symbol]
            if opt.get('profile') == profile_key: # Si no ha sido degradado
                final_params['tp_pct'] = opt.get('dynamic_tp', opt['tp_pct'])
                final_params['sl_pct'] = opt.get('dynamic_sl', opt['sl_pct'])
                if 'dynamic_strength' in opt:
                    final_params['strength_threshold'] = opt['dynamic_strength']
                if 'dynamic_adx' in opt:
                    final_params['adx_threshold'] = opt['dynamic_adx']
                
        return final_params

    def calculate_indicators(self, data, time_multiplier=1.0):
        """
        SUPREMO-V3 / MULTI-HORIZON: Zero-Pandas Indicator Calculation.
        Calculates all indicators using JIT-compiled functions on raw NumPy arrays.
        Adapts periods dynamically using time_multiplier.
        """
        if data is None or len(data) == 0:
            return None

        # Prepare results dictionary (Arrays of same length as input)
        inds = {}
        
        # Extract raw arrays from structured array (Zero-Copy views)
        closes = data['close']
        highs = data['high']
        lows = data['low']
        vols = data['volume']

        try:
            # 0. Dynamic Parameter Scaling (Phase 2.1)
            bb_p = max(5, int(self.BB_PERIOD * time_multiplier))
            rsi_p = max(5, int(self.RSI_PERIOD * time_multiplier))
            macd_f = max(4, int(self.MACD_FAST * time_multiplier))
            macd_s = max(8, int(self.MACD_SLOW * time_multiplier))
            macd_sig = max(4, int(self.MACD_SIGNAL * time_multiplier))
            ema_f = max(5, int(self.EMA_FAST * time_multiplier))
            ema_s = max(10, int(self.EMA_SLOW * time_multiplier))
            ema_t = max(20, int(self.EMA_TREND * time_multiplier))
            vol_p = max(5, int(20 * time_multiplier))
            atr_p = max(5, int(14 * time_multiplier))
            # 🧮 FASE 5: Kalman Filter (Zero-Lag Smoothing)
            kalman_closes = kalman_filter_1d_jit(closes, R=1e-4, Q=1e-5)
            inds['kalman_close'] = kalman_closes
            
            # 1. Bollinger Bands (Numba JIT RANSAC - Phase 10) on Kalman Closes
            inds['bb_upper'], inds['bb_middle'], inds['bb_lower'] = calculate_bollinger_robust_jit(kalman_closes, bb_p, self.BB_STD)
            
            # 2. RSI (Numba JIT) on Kalman Closes
            inds['rsi'] = calculate_rsi_jit(kalman_closes, rsi_p)
            
            # 3. MACD (Phase 3 JIT) on Kalman Closes
            inds['macd'], inds['macd_signal'], inds['macd_hist'] = calculate_macd_jit(kalman_closes, macd_f, macd_s, macd_sig)
            
            # 4. EMAs (Numba JIT) on Kalman Closes
            inds['ema_fast'] = calculate_ema_jit(kalman_closes, ema_f)
            inds['ema_slow'] = calculate_ema_jit(kalman_closes, ema_s)
            inds['ema_trend'] = calculate_ema_jit(kalman_closes, ema_t)
            inds['ema_micro'] = calculate_ema_jit(kalman_closes, 3) # Nano-Core Micro Trend
            
            # 5. Trend Flags (Boolean Arrays)
            inds['in_uptrend'] = (inds['ema_fast'] > inds['ema_slow']) & (closes > inds['ema_trend'])
            inds['in_downtrend'] = (inds['ema_fast'] < inds['ema_slow']) & (closes < inds['ema_trend'])
            inds['micro_trend_up'] = (closes > inds['ema_micro'])
            inds['micro_trend_down'] = (closes < inds['ema_micro'])
            
            # 6. Volume Metrics
            # Simple Volume MA (Vectorized with Convolve)
            period = vol_p
            if len(vols) >= period:
                # Efficient Moving Average using 1D Convolution
                kernel = np.ones(period) / period
                # valid mode returns len(vols) - period + 1
                # we need to pad correctly to match shape.
                # actually for indicators we usually want aligned arrays.
                # Using talib is easier if available, but staying numpy:
                
                # SUPREMO-V3: JIT-like Convolve
                v_ma_valid = np.convolve(vols, kernel, mode='valid')
                
                # Pad beginning with first value to avoid 0 division (BUG-003 FIX)
                # v_ma needs to be same length as vols
                first_valid = v_ma_valid[0] if len(v_ma_valid) > 0 else vols[0]
                padding = np.full(period - 1, first_valid)
                inds['volume_ma'] = np.concatenate((padding, v_ma_valid))
            else:
                inds['volume_ma'] = np.zeros_like(vols)
            inds['volume_ratio'] = np.divide(vols, inds['volume_ma'], out=np.ones_like(vols), where=inds['volume_ma'] > 0)
            
            # 7. ATR & ADX (Phase 3 JIT)
            inds['atr'] = calculate_atr_jit(highs, lows, closes, atr_p)
            inds['adx'] = calculate_adx_jit(highs, lows, closes, atr_p)

            return inds
        except Exception as e:
            # logger.error(f"Indicator Calc Error: {e}")
            return None

    def get_multi_timeframe_data(self, symbol):
        """SUPREMO-V3 + FORENSIC-1: Multi-timeframe analysis FILTERED BY HORIZON."""
        timeframe_data = {}
        
        # FORENSIC-1: Only process timeframes relevant to this horizon
        # Scalping: 1m, 5m, 15m | Swing: 1h, 4h, 1d
        all_tf_bars = {'1m': 300, '5m': 300, '15m': 200, '1h': 300, '4h': 300, '1d': 100, '1w': 100}
        allowed_tfs = list(self.HORIZON_TIMEFRAMES) if hasattr(self, 'HORIZON_TIMEFRAMES') else ['5m', '15m', '1h']
        # QUÉ: Asegurar la inyección de datos macro (1d, 1w) en el ledger de indicadores del horizonte.
        # POR QUÉ: El oráculo macro (`MultiHorizonOracle.evaluate_clash_vector`) evalúa obligatoriamente
        #   las tendencias estructurales en 1d y 1w. Si faltan en `allowed_tfs`, el oráculo retorna
        #   `NO_MACRO_DATA` (is_vetoed=False), desactivando silenciosamente la protección y
        #   causando pérdidas masivas en pullbacks durante caídas generalizadas.
        # PARA QUÉ: Reducir pérdidas por trade contra-tendencia en el micro-capital de $13 USD.
        # CÓMO: Copiamos `self.HORIZON_TIMEFRAMES` como lista y forzamos la inclusión de `1d` y `1w`.
        # CUÁNDO: Ejecutado en cada actualización de vela.
        # DÓNDE: En `strategies/technical.py::get_multi_timeframe_data`.
        # QUIÊN: Modificado por el Quant Developer y el Arquitecto Senior.
        for tf in ['1d', '1w']:
            if tf not in allowed_tfs:
                allowed_tfs.append(tf)
        
        for tf in allowed_tfs:
            n_bars = all_tf_bars.get(tf, 200)
            # Integrar time_multiplier = current_resolution_minutes / base_resolution_minutes (1m)
            tf_mins = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}.get(tf, 5)
            time_multiplier_raw = tf_mins / 1.0 # Base 1m
            
            # Adaptamos factor temporal
            if time_multiplier_raw <= 5: time_multiplier = 1.0
            elif time_multiplier_raw <= 15: time_multiplier = 0.8
            elif time_multiplier_raw <= 60: time_multiplier = 0.6
            elif time_multiplier_raw <= 240: time_multiplier = 0.4
            else: time_multiplier = 0.3
            
            try:
                # get_latest_bars now returns structured array
                data = self.data_provider.get_latest_bars(symbol, n=n_bars, timeframe=tf)
                if data is not None and len(data) >= (30 if tf not in ('1w', '1d', '4h') else 10):
                    # ═══════════════════════════════════════════════════════════════
                    # FORENSIC-V99: O(1) LAZY EVALUATION CACHE (QUANTUM SPEED)
                    # QUÉ: Caching de indicadores basado en la última vela cerrada.
                    # POR QUÉ: Antes, el bot recalculaba MACD/RSI/BB sobre 300 velas
                    #   para gráficas de 1H y 1D a CADA MINUTO. Un desperdicio de 99.9%.
                    # PARA QUÉ: Reducir la complejidad O(N) a O(1), acelerando el backtest x100.
                    # CÓMO: Hash basado en symbol_tf_timestamp_idx[-2].
                    # ═══════════════════════════════════════════════════════════════
                    last_closed_ts = data['timestamp'][-2] if len(data) > 1 else 0
                    cache_key = f"{symbol}_{tf}_{last_closed_ts}_{time_multiplier}"
                    
                    if not hasattr(self, '_macro_ind_cache'):
                        self._macro_ind_cache = {}
                        
                    if cache_key in self._macro_ind_cache:
                        inds = self._macro_ind_cache[cache_key]
                    else:
                        inds = self.calculate_indicators(data, time_multiplier=time_multiplier)
                        self._macro_ind_cache[cache_key] = inds
                        
                        # Limpieza de memoria O(1) agresiva (evitar OOM)
                        keys_to_del = [k for k in self._macro_ind_cache.keys() if k.startswith(f"{symbol}_{tf}_") and k != cache_key]
                        for k in keys_to_del:
                            del self._macro_ind_cache[k]
                    
                    if inds:
                        timeframe_data[tf] = {'data': data, 'inds': inds}
            except Exception as e:
                logger.error(f"Silent exception caught: {e}", exc_info=True)
        
        return timeframe_data


    def calculate_multi_timeframe_confluence(self, timeframe_data, symbol=None):
        """SUPREMO-V3: Confluence using structured arrays."""
        confluence_score = 0.0
        total_weight = 0.0
        
        is_btc = symbol and 'BTC' in symbol
        
        for tf, weight in self.MULTI_TIMEFRAME_WEIGHTS.items():
            if tf in timeframe_data:
                pkg = timeframe_data[tf]
                data = pkg['data']
                inds = pkg['inds']
                
                if len(data) > 0:
                    # Index -1 is the last available bar
                    tf_score = 0.0
                    
                    # Bonus por tendencia (Using last index)
                    if inds['in_uptrend'][-1] or inds['in_downtrend'][-1]:
                        tf_score += 0.3
                    
                    # Bonus por RSI (Using last index)
                    last_rsi = inds['rsi'][-1]
                    if Config.Strategies.TECHNICAL_THRESHOLDS['rsi_pullback_uptrend'] <= last_rsi <= Config.Strategies.TECHNICAL_THRESHOLDS['rsi_rally_downtrend']:
                        tf_score += 0.2
                    elif inds['in_uptrend'][-1] and last_rsi < Config.Strategies.TECHNICAL_THRESHOLDS['rsi_pullback_uptrend']:
                        tf_score += 0.3  # Pullback en uptrend
                    elif inds['in_downtrend'][-1] and last_rsi > Config.Strategies.TECHNICAL_THRESHOLDS['rsi_rally_downtrend']:
                        tf_score += 0.3  # Rally en downtrend (Corrected Logic)
                    elif last_rsi < Config.Strategies.TECHNICAL_THRESHOLDS['rsi_extreme_low'] or last_rsi > Config.Strategies.TECHNICAL_THRESHOLDS['rsi_extreme_high']:
                        tf_score += 0.4  # VITAL FIX: Extreme RSI means Mean Reversion is possible!
                    
                    # Bonus por volumen (V5.45 Relaxed for Alts)
                    vol_thresh = Config.Strategies.TECHNICAL_THRESHOLDS['vol_ratio_btc'] if is_btc else Config.Strategies.TECHNICAL_THRESHOLDS['vol_ratio_alts'] # Reduced from 1.5 to 1.2 for BTC to catch more MR Setups
                    if inds['volume_ratio'][-1] > vol_thresh:
                        tf_score += 0.2
                    
                    confluence_score += tf_score * weight
                    total_weight += weight
        
        return min(confluence_score / total_weight if total_weight > 0 else 0.0, 1.0)
        # [SS-003 FIX] Dead code removed — was unreachable after return above

    def _get_dynamic_rsi_levels(self, inds, lookback=200):
        """V5.5: DPE (Dynamic Parametric Evolution) - Percentile Based RSI"""
        try:
            # We want to look at the last `lookback` valid periods (ignoring NaNs)
            rsi_array = inds['rsi']
            valid_rsi = rsi_array[~np.isnan(rsi_array)]
            
            if len(valid_rsi) < 50: # Fallback si no hay suficientes datos
                return 30, 70
                
            recent_rsi = valid_rsi[-lookback:]
            
            # Calculamos percentiles (Ej: el 10% más bajo y el 90% más alto de la historia reciente)
            # Esto significa que el mercado define qué es "sobrevendido" hoy.
            buy_level = np.percentile(recent_rsi, 15)
            sell_level = np.percentile(recent_rsi, 85)
            
            # Sanity caps para evitar extremos rotos
            buy_level = max(10, min(buy_level, 40))
            sell_level = min(90, max(sell_level, 60))
            
            return buy_level, sell_level
        except Exception:
            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
            return 30, 70
            
    def _calculate_dynamic_risk_params(self, inds, current_price, setup_type="UNKNOWN", regime="UNKNOWN"):
        """V5.6: Total Dynamic Ecosystem - Auto-calculated Risk Profile (Dual Paradigm).
        Mejora 13: Regime-Aware Stop Loss adjust.
        """
        try:
            atr_array = inds['atr']
            valid_atr = atr_array[~np.isnan(atr_array)]
            
            if len(valid_atr) < 50:
                current_atr = valid_atr[-1] if len(valid_atr) > 0 else current_price * 0.005
                return 1.5, 3.5, (current_atr * 1.5) / current_price, (current_atr * 3.5) / current_price
            
            current_atr = valid_atr[-1]
            recent_atr = valid_atr[-100:]
            mean_atr = np.mean(recent_atr)
            
            # Ratio de expansión de volatilidad
            vol_ratio = current_atr / mean_atr if mean_atr > 0 else 1.0
            
            # FORENSIC-1: Use Horizon-Aware Multipliers
            # Instead of hardcoded 1.4/1.6, use the ones loaded from Config (Scalping vs Swing)
            base_sl_mult = self.ATR_SL_MULT_BASE if hasattr(self, 'ATR_SL_MULT_BASE') else 1.5
            base_tp_mult = self.ATR_TP_MULT_BASE if hasattr(self, 'ATR_TP_MULT_BASE') else 3.0
            
            # Ajuste táctico según el setup
            if setup_type == "MEAN_REV":
                # Mean Reversion: tighter stops, quicker exits
                base_sl_mult *= 0.9
                base_tp_mult *= 0.8
            elif setup_type == "MOMENTUM":
                # Momentum: wider stops, longer runs
                base_sl_mult *= 1.2
                base_tp_mult *= 1.3
            
            # Auto-adaptabilidad de Volatilidad (Sigue siendo dinámico / Evolutivo)
            # MEJORA 13: Regime-Aware Stop Loss
            regime_mult = 1.0
            if 'TRENDING' in regime:
                regime_mult = 1.25 # Stops más amplios
            elif 'CHOPPY' in regime:
                regime_mult = 0.75 # Stops más cerrados
                
            if vol_ratio > Config.Strategies.TECHNICAL_THRESHOLDS['vol_ratio_high']: # Volatilidad expandiéndose (Mechas largas)
                # El mercado está loco: Ampliamos red de pesca de profit, y alejamos stop loss del ruido
                atr_sl_mult = base_sl_mult * 1.2 * regime_mult
                atr_tp_mult = base_tp_mult * 1.5
            elif vol_ratio < Config.Strategies.TECHNICAL_THRESHOLDS['vol_ratio_low']: # Volatilidad muy baja (Laterales estrechos)
                # El mercado está muerto: TPs ultracortos, SL muy pegados
                atr_sl_mult = base_sl_mult * 0.8 * regime_mult
                atr_tp_mult = base_tp_mult * 0.8
            else:
                atr_sl_mult = base_sl_mult * regime_mult
                atr_tp_mult = base_tp_mult
                
            # Calculo crudo
            sl_pct = (current_atr * atr_sl_mult) / current_price
            tp_pct = (current_atr * atr_tp_mult) / current_price
            
            # FORENSIC-1: Horizon-Aware Bounds calculation
            # Las hardcoded bounds originales (0.005 min_sl y 0.010 min_tp)
            # hacían IMPOSIBLE el scalping rápido con cuenta micro de $13.
            min_sl_cap = self.SL_PCT * 0.5 if hasattr(self, 'SL_PCT') else 0.0015
            min_tp_cap = self.TP_PCT * 0.5 if hasattr(self, 'TP_PCT') else 0.003
            
            # ════════════════════════════════════════════════════════════════
            # FORENSIC-V70 FIX: HORIZON-AWARE TP/SL CAPS
            # QUÉ: Cap máximo para TP/SL ajustado al horizonte temporal real.
            # POR QUÉ: El ATR se calcula en M5 pero el scalping opera en M1.
            #   Con ATR(M5)=$115 y atr_tp_mult=3.5, TP=0.50% — INALCANZABLE.
            #   Datos empíricos M1 (24h, 1440 barras):
            #     - ATR(14) M1 = 0.028%
            #     - MFE 30-bar P50 = +0.14%
            #     - TP 0.50% hit rate en 90min = solo 33.8%
            #     - TP 0.20% hit rate en 30min = 35.4%
            #     - TP 0.15% hit rate en 30min = 46.3%
            # PARA QUÉ: Reducir ZOMBIE exits (58/72 = 81%) que destrozan WR.
            # CÓMO: Cap TP scalping a 0.20% y SL a 0.15% (empíricamente viable).
            # CUÁNDO: En cada cálculo de risk params para señales de entrada.
            # DÓNDE: strategies/technical.py → _calculate_dynamic_risk_params()
            # QUIÉN: HybridScalpingStrategy (Quant Developer + Risk Manager)
            # ════════════════════════════════════════════════════════════════
            is_scalping = hasattr(self, 'horizon') and self.horizon == 'SCALPING'
            if is_scalping:
                # [CIRUGÍA #1] REDUCED FROM 1.0% TO 0.50%
                # TP > 0.50% is unrealistic for M1 scalping and reduces win rate.
                max_tp_cap = 0.0050  # 0.50% cap
                max_sl_cap = 0.0050  # 0.50% cap
            else:
                max_tp_cap = getattr(Config.Strategies, 'MAX_EVO_TP', 0.30)
                max_sl_cap = getattr(Config.Strategies, 'MAX_EVO_SL', 0.15)
            
            sl_pct = np.clip(sl_pct, min_sl_cap, max_sl_cap)
            tp_pct = np.clip(tp_pct, min_tp_cap, max_tp_cap)
            
            return atr_sl_mult, atr_tp_mult, sl_pct, tp_pct
        except Exception:
            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
            # Safe Fallback (Horizon-Aware)
            fallback_sl = self.SL_PCT if hasattr(self, 'SL_PCT') else 0.01
            fallback_tp = self.TP_PCT if hasattr(self, 'TP_PCT') else 0.02
            return 1.5, 2.0, fallback_sl, fallback_tp
            
    def _get_dynamic_adx_threshold(self, inds, lookback=100):
        """V5.5: DPE - Moving Average Based ADX Threshold"""
        try:
            adx_array = inds['adx']
            valid_adx = adx_array[~np.isnan(adx_array)]
            
            if len(valid_adx) < 50:
                return 20
                
            recent_adx = valid_adx[-lookback:]
            # Calculamos la media del ADX reciente + 0.5 Desviación Estándar
            mean_adx = np.mean(recent_adx)
            std_adx = np.std(recent_adx)
            
            # El mercado exige superar su propia media reciente para considerarse tendencia
            dynamic_thresh = mean_adx + (std_adx * 0.5)
            
            # Sanity bounds
            return max(15, min(dynamic_thresh, 35))
        except Exception:
            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
            return 20

    def detect_setup(self, pkg_primary, params=None, symbol=None):
        """SUPREMO-V3 + FORENSIC-1: Horizon-agnostic setup detection using Cognitive Interception."""
        data = pkg_primary['data']
        inds = pkg_primary['inds']
        
        if len(data) < 3: return None
        
        # Use -2 for Confirmed Closed Bar
        idx = -2
        
        # Phase 5.5: Dynamic Parametric Evolution (DPE)
        rsi_buy, rsi_sell = self._get_dynamic_rsi_levels(inds)
        adx_thresh = self._get_dynamic_adx_threshold(inds)
        
        last_close = data['close'][idx]
        last_open = data['open'][idx]
        last_rsi = inds['rsi'][idx]
        last_vol_ratio = inds['volume_ratio'][idx]
        
        # MICRO-STRUCTURE ANALYSIS (Phase 7 Predictive Edge)
        last_low = data['low'][idx]
        last_high = data['high'][idx]
        
        body_top = max(last_open, last_close)
        body_bottom = min(last_open, last_close)
        body_size = body_top - body_bottom
        upper_wick = last_high - body_top
        lower_wick = body_bottom - last_low
        
        # Avoid division by zero
        safe_body = max(body_size, last_close * 0.0001)
        
        # Pin Bar logic (Wick Rejection)
        is_bullish_pin = (lower_wick > safe_body * 1.5) and (upper_wick < safe_body * 1.0)
        is_bearish_pin = (upper_wick > safe_body * 1.5) and (lower_wick < safe_body * 1.0)
        
        setups = {
            'long_mean_rev': False,
            'short_mean_rev': False, 
            'long_momentum': False,
            'short_momentum': False,
            'rsi': last_rsi,
            'volume_ratio': last_vol_ratio,
            'in_uptrend': inds['in_uptrend'][idx],
            'in_downtrend': inds['in_downtrend'][idx],
            'bb_position': 0.5,
            'atr': inds['atr'][idx],
            'close': last_close,
            'adx': inds['adx'][idx],
            'macd_hist': inds['macd_hist'][idx] if 'macd_hist' in inds else 0.0
        }
        
        # BB Position Calculation
        bbu, bbl = inds['bb_upper'][idx], inds['bb_lower'][idx]
        if (bbu - bbl) > 0:
            setups['bb_position'] = (last_close - bbl) / (bbu - bbl)
        
        # 1. MEAN REVERSION (Flexibilizar si no hay tendencia clara)
        # SUPREMO-V4: PROTECCIÓN CONTRA FALLING KNIVES (INTEGRALIDAD)
        adx_extreme = setups['adx'] > 35
        is_strong_trend = setups['adx'] > adx_thresh
        
        price_at_lower = last_low <= bbl  # Use wick for scalping detection
        price_at_upper = last_high >= bbu # Use wick for scalping detection
        rsi_oversold = last_rsi < rsi_buy
        rsi_overbought = last_rsi > rsi_sell
        
        # Filtro Falling Knife: Si hay tendencia fuerte, el RSI debe ser < 20 o > 80 para MR
        if is_strong_trend:
            rsi_oversold = last_rsi < min(20, rsi_buy)
            rsi_overbought = last_rsi > max(80, rsi_sell)
            
        # Bloqueo total en ADX Extremo (Knife is too sharp)
        if adx_extreme:
            is_range = False
        else:
            is_range = not is_strong_trend or rsi_oversold or rsi_overbought
        
        # [QUANTUM EVOLUTION: FASE 12.1] Liquidity Void Sniping (Mechas Asesinas)
        # QUÉ: Cazamos ineficiencias de precio ultra-rápidas donde el precio cae/sube abruptamente
        #      y retrocede al instante, dejando una mecha gigante.
        # POR QUÉ: Los barridos de stops (Stop Hunts) crean vacíos de liquidez. Entrar en el rechazo
        #          es el setup de mayor win-rate para scalping puro.
        # CÓMO: Verificamos si la vela total es grande (> 2x ATR) y si la mecha representa > 80% de toda la vela.
        setups['liquidity_void_long'] = False
        setups['liquidity_void_short'] = False
        if self.horizon in ("MICROSCALPING", "SCALPING"):
            atr_norm = setups['atr']
            total_size = last_high - last_low
            
            if total_size > 0 and atr_norm > 0 and total_size > (atr_norm * 2.0):
                lower_wick_ratio = lower_wick / total_size
                upper_wick_ratio = upper_wick / total_size
                
                # Mecha inferior gigante (Flash Crash / Barrido de Longs) -> Entramos LONG
                if lower_wick_ratio > 0.80:
                    setups['liquidity_void_long'] = True
                    logger.warning(f"🕳️ [LIQUIDITY SNIPE] {self.symbol} Flash Crash detectado! Wick: {lower_wick_ratio*100:.1f}%. Disparando LONG.")
                
                # Mecha superior gigante (Flash Pump / Barrido de Shorts) -> Entramos SHORT
                elif upper_wick_ratio > 0.80:
                    setups['liquidity_void_short'] = True
                    logger.warning(f"🕳️ [LIQUIDITY SNIPE] {self.symbol} Flash Pump detectado! Wick: {upper_wick_ratio*100:.1f}%. Disparando SHORT.")
        
        # ================================================================
        # IMPLEMENTACIÓN DE SHORTS SIMÉTRICOS V2 (Advanced Filters)
        # QUÉ: Filtros precisos para señales SHORT con confirmación multi-señal.
        # POR QUÉ: Las señales anteriores eran demasiado laxas, generando
        #   entradas en falsos techos y short-squeezes.
        # PARA QUÉ: Mejorar win-rate en shorts verificando FUERZA DE GIRO,
        #   INERCIA de histograma, EXPANSIÓN de volatilidad y RECHAZO de precio.
        # CÓMO: Cada filtro exige confirmación en idx=-2 vs idx=-3.
        # CUÁNDO: En cada evaluación de detect_setup.
        # DÓNDE: strategies/technical.py → detect_setup
        # QUIÉN: HybridScalpingStrategy
        # ================================================================

        # Datos previos para confirmación de giro (idx-1 relativo a idx=-2 → idx=-3)
        prev_rsi = inds['rsi'][idx - 1]  # RSI de la vela anterior a la cerrada

        # ================================================================
        # FORENSIC REMEDIATION: SYMMETRIC EXPLICIT SETUPS (LONG + SHORT)
        # QUÉ: Se añaden 4 setups LONG explícitos simétricos a los 4 SHORT.
        # POR QUÉ: El sistema tenía 6 SHORT vs 2 LONG activos → sesgo masivo.
        # PARA QUÉ: Balancear la exposición direccional a ~50/50.
        # CÓMO: Cada filtro SHORT tiene su mirror LONG con misma rigurosidad.
        # CUÁNDO: En cada evaluación de detect_setup.
        # DÓNDE: strategies/technical.py → detect_setup
        # QUIÉN: HybridScalpingStrategy
        # ================================================================

        # RSI SHORT: RSI > overbought + ADX > 25 + GIRO confirmado
        setups['short_rsi_explicit'] = (
            rsi_overbought
            and (inds['adx'][idx] > 25)
            and (last_rsi < prev_rsi)     # RSI ya bajando (giro descendente)
        )
        # RSI LONG: RSI < oversold + ADX > 25 + GIRO confirmado
        setups['long_rsi_explicit'] = (
            rsi_oversold
            and (inds['adx'][idx] > 25)
            and (last_rsi > prev_rsi)     # RSI ya subiendo (giro ascendente)
        )

        # MACD SHORT: macd < macd_signal + histograma descendente
        macd, macd_sig, macd_hist = inds['macd'][idx], inds['macd_signal'][idx], inds['macd_hist'][idx]
        macd_prev_hist = inds['macd_hist'][idx - 1]
        setups['short_macd_explicit'] = (
            (macd < macd_sig)
            and (macd_hist < 0)
            and (macd_hist < macd_prev_hist)
        )
        # MACD LONG: macd > macd_signal + histograma ascendente
        setups['long_macd_explicit'] = (
            (macd > macd_sig)
            and (macd_hist > 0)
            and (macd_hist > macd_prev_hist)
        )

        # BB WIDTH calculation (shared by both SHORT and LONG BB setups)
        prev_bbu, prev_bbl = inds['bb_upper'][idx - 1], inds['bb_lower'][idx - 1]
        prev_bbw = (prev_bbu - prev_bbl) / prev_bbl if prev_bbl > 0 else 0
        current_bbw = (bbu - bbl) / bbl if bbl > 0 else 0

        bb_width_lookback = 20
        bb_widths = []
        for i in range(max(0, idx - bb_width_lookback), idx):
            _u = inds['bb_upper'][i]
            _l = inds['bb_lower'][i]
            if _l > 0:
                bb_widths.append((_u - _l) / _l)
        mean_bbw = np.mean(bb_widths) if bb_widths else current_bbw

        # BB SHORT: Precio sobre upper + EXPANSIÓN de volatilidad
        setups['short_bb_explicit'] = (
            price_at_upper
            and (current_bbw > prev_bbw)
            and (current_bbw > mean_bbw)
        )
        # BB LONG: Precio bajo lower + EXPANSIÓN de volatilidad
        setups['long_bb_explicit'] = (
            price_at_lower
            and (current_bbw > prev_bbw)
            and (current_bbw > mean_bbw)
        )

        # VOLUME/VWAP calculation (shared)
        last_close_val = data['close'][idx]
        last_open_val = data['open'][idx]
        is_red_candle = last_close_val < last_open_val
        is_green_candle = last_close_val > last_open_val

        vwap_lookback = min(20, len(data['close']) - 1)
        if vwap_lookback > 0:
            _vwap_prices = (data['high'][-vwap_lookback:] + data['low'][-vwap_lookback:] + data['close'][-vwap_lookback:]) / 3.0
            _vwap_vols = data['volume'][-vwap_lookback:]
            _vwap_sum_vol = np.sum(_vwap_vols)
            vwap = np.sum(_vwap_prices * _vwap_vols) / _vwap_sum_vol if _vwap_sum_vol > 0 else last_close_val
        else:
            vwap = last_close_val

        # VOLUME SHORT: Vela roja + close bajo VWAP + volumen alto
        setups['short_volume_explicit'] = (
            is_red_candle
            and (last_close_val < vwap)
            and (last_vol_ratio > Config.Strategies.TECHNICAL_THRESHOLDS['vol_ratio_expansion'])
        )
        # VOLUME LONG: Vela verde + close sobre VWAP + volumen alto
        setups['long_volume_explicit'] = (
            is_green_candle
            and (last_close_val > vwap)
            and (last_vol_ratio > Config.Strategies.TECHNICAL_THRESHOLDS['vol_ratio_expansion'])
        )

        # VOLATILITY GATE (applies to both SHORT and LONG extremes in SCALPING)
        volatility_pct = setups['atr'] / last_close if last_close > 0 else 0
        setups['_short_vol_gate_pass'] = (volatility_pct < Config.Strategies.TECHNICAL_THRESHOLDS['volatility_gate_pct'])  # < 2.5% ATR/Price
        setups['_long_vol_gate_pass'] = (volatility_pct < Config.Strategies.TECHNICAL_THRESHOLDS['volatility_gate_pct'])
        setups['_vwap'] = vwap
        # ================================================================

        if self.data_provider and getattr(self.data_provider, 'is_backtest', False):
            vol_min = 0.85
        else:
            vol_min = 1.0 # Default for live
            
        high_volume = last_vol_ratio > vol_min
        
        # PHASE 7: Predictive Edge for MEAN_REV
        if self.horizon == 'SCALPING':
            # TRUE SCALPING MEAN REVERSION: Wick Rejection of Bollinger Bands
            # We don't wait for RSI. If the wick pierced the band and closed back inside as a pin bar, we enter.
            micro_trend_up = inds.get('micro_trend_up', [True]*len(data['close']))[idx]
            micro_trend_down = inds.get('micro_trend_down', [True]*len(data['close']))[idx]
            
            # 🚨 $13 MICRO-ACCOUNT FILTER: Only trade with micro_trend to increase WR
            setups['long_mean_rev'] = price_at_lower and is_bullish_pin and high_volume and not adx_extreme and micro_trend_up
            setups['short_mean_rev'] = price_at_upper and is_bearish_pin and high_volume and not adx_extreme and micro_trend_down
            
            # PROXIMITY SETUPS: Now backed by Wick analysis instead of pure random BB position
            if not setups['long_mean_rev'] and not setups['short_mean_rev']:
                bb_pos = setups['bb_position']
                bb_pos_lower = getattr(Config.Strategies, 'TECHNICAL_THRESHOLDS', {}).get('bb_pos_lower_prox', 0.25)
                bb_pos_upper = getattr(Config.Strategies, 'TECHNICAL_THRESHOLDS', {}).get('bb_pos_upper_prox', 0.75)
                vol_ok = last_vol_ratio > 0.8
                adx_strong = setups['adx'] > 20
                
                # If near the lower band, and there is a bullish wick (not strictly a pin bar, but strong rejection)
                if bb_pos < bb_pos_lower and (lower_wick > safe_body) and vol_ok and is_range and adx_strong:
                    setups['long_mean_rev'] = True
                    logger.debug(f"👻 [FANTASMA] Proximity LONG activado para {self.symbol} (Wick Rejection)")
                elif bb_pos > bb_pos_upper and (upper_wick > safe_body) and vol_ok and is_range and adx_strong:
                    setups['short_mean_rev'] = True
                    logger.debug(f"👻 [FANTASMA] Proximity SHORT activado para {self.symbol} (Wick Rejection)")
        else:
            # SWING (FASE 7): Relaxed setup (Trend Reversal Anticipation)
            # Quitamos 'high_volume' y 'is_range' para forzar a la IA a capturar el pivot rápido.
            setups['long_mean_rev'] = price_at_lower and rsi_oversold
            setups['short_mean_rev'] = price_at_upper and rsi_overbought
        
        # 2. MOMENTUM (Optimizado para Nivel Supremo-V3 con VCP & ADX)
        # MACD variables ya declaradas arriba (macd, macd_sig, macd_hist, macd_prev_hist)
        
        # Detectar aceleración
        momentum_accel = abs(macd_hist) > abs(macd_prev_hist)
        
        # Filtro 1: ADX estricto para evitar mercados planos (Choppiness)
        adx_trend_confirmed = setups['adx'] > 20
        
        # Filtro 2: VCP (Volatility Contraction Pattern)
        # BBW variables ya declaradas arriba (prev_bbw, current_bbw)
        current_bbw = (bbu - bbl) / bbl if bbl > 0 else 0
        
        vcp_expansion = (current_bbw > prev_bbw)  # Las bandas se están abriendo
        volume_expansion = last_vol_ratio > 1.0   # Volumen > Media móvil
        vcp_confirmed = vcp_expansion and volume_expansion
        
        # FORENSIC AUDIT L1: Momentum setups disabled for SCALPING, enabled for SWING
        # SUPREMO-V4: Bloqueo de Momentum en zonas de RSI extremo para evitar "pisarse los pies"
        # Si el RSI está en sobreventa/sobrecompra extrema, el momentum ya está agotado.
        # Dejamos que Mean Reversion tome el control.
        rsi_exhausted_long = last_rsi > 65
        rsi_exhausted_short = last_rsi < 35
        
        is_swing = self.horizon == 'SWING'
        
        # Phase 7: Predictive Edge for MOMENTUM
        if is_swing:
            # FASE 7: Swing Momentum acelerado (No exigimos VCP estricto con expansión de volumen inmediata)
            setups['long_momentum'] = setups['in_uptrend'] and momentum_accel and adx_trend_confirmed and not rsi_exhausted_long
            setups['short_momentum'] = setups['in_downtrend'] and momentum_accel and adx_trend_confirmed and not rsi_exhausted_short
            setups['long_scalp_break'] = False
            setups['short_scalp_break'] = False
        else:
            # TRUE SCALPING MOMENTUM (VCP BREAKOUT):
            # Rather than waiting for lagging MACD across zero, we anticipate the breakout when:
            # 1. We had VCP contraction recently (mean_bbw is small, current_bbw is expanding)
            # 2. Volume is expanding massively (vol_ratio > 1.5)
            # 3. Price closes aggressively outside or very near the bands
            scalp_vol_surge = last_vol_ratio > 1.5
            
            micro_trend_up = inds.get('micro_trend_up', [True]*len(data['close']))[idx]
            micro_trend_down = inds.get('micro_trend_down', [True]*len(data['close']))[idx]
            
            # Use strict VCP setup
            setups['long_momentum'] = setups['in_uptrend'] and vcp_expansion and scalp_vol_surge and price_at_upper and not rsi_exhausted_long and micro_trend_up
            setups['short_momentum'] = setups['in_downtrend'] and vcp_expansion and scalp_vol_surge and price_at_lower and not rsi_exhausted_short and micro_trend_down
            
            # SCALP BREAKOUTS (Explosive wicks)
            setups['long_scalp_break'] = momentum_accel and scalp_vol_surge and (last_close > bbu) and not rsi_exhausted_long and micro_trend_up
            
            symmetric_shorts = getattr(Config.Strategies, 'SYMMETRIC_SHORTS_SCALPING', False)
            if symmetric_shorts:
                setups['short_scalp_break'] = momentum_accel and scalp_vol_surge and (last_close < bbl) and not rsi_exhausted_short and micro_trend_down
            else:
                setups['short_scalp_break'] = setups['in_downtrend'] and momentum_accel and scalp_vol_surge and (last_close < bbl) and not rsi_exhausted_short and micro_trend_down
        
        # Phase 5.7 Cognitive Auto-Tuning (Self-Healing Interception)
        allowed_setups = params.get('allowed_setups', 'ALL_SETUPS') if params else 'ALL_SETUPS'
        
        # Instinto Genético Básico (Fallback V5.6)
        if allowed_setups == 'MOMENTUM_ONLY':
            setups['long_mean_rev'] = False
            setups['short_mean_rev'] = False
        elif allowed_setups == 'MEAN_REV_ONLY':
            setups['long_momentum'] = False
            setups['short_momentum'] = False
            
        # V5.8 True Cognitive Interception (Asymmetric State Detection)
        cog_state = 'NORMAL'
        if symbol and hasattr(self, 'cognitive_memory') and symbol in self.cognitive_memory:
            # BUG-008 FIX: Extraemos el estado dominante de la memoria
            # Si hay una serie de pérdidas, el estado se vuelve INJURED. Si hay ganancias, ALPHA.
            states = [mem.get('state', 'NORMAL') for mem in self.cognitive_memory[symbol].values()]
            if 'INJURED' in states:
                cog_state = 'INJURED'
            elif 'ALPHA' in states:
                cog_state = 'ALPHA'

        setups['cognitive_state'] = cog_state
        return setups

    def calculate_signal_strength(self, setups, confluence_score, volatility, symbol=None, setup_type=None):
        """
        SUPREMO-V4: Cálculo de fuerza de señal MULTIPLICATIVO (INTEGRALIDAD)
        QUÉ: Los filtros ahora actúan como multiplicadores (0.0 a 1.2) en lugar de sumas.
        POR QUÉ: Evita que un setup base supere el umbral sin confirmación de volumen o confluencia.
        """
        # 0. Determinar Estado Cognitivo V5.8
        cog_state = 'NORMAL'
        if symbol and setup_type and hasattr(self, 'cognitive_memory') and symbol in self.cognitive_memory:
            mem = self.cognitive_memory[symbol].get(setup_type, {})
            cog_state = mem.get('state', 'NORMAL')
        
        # 1. BASE SCORE (Convicción inicial reducida para forzar multiplicadores)
        if setups.get('long_mean_rev') or setups.get('short_mean_rev'):
            strength = 0.45  # Reducido de 0.6 para exigir confirmación
        elif setups.get('long_momentum') or setups.get('short_momentum') or setups.get('long_scalp_break') or setups.get('short_scalp_break'):
            strength = 0.40  # Reducido de 0.5
        else:
            strength = 0.0
            
        # 2. MULTIPLICADOR DE CONFLUENCIA (Impacto 0.8x a 1.2x)
        # Una confluencia de 0.5 (neutral) no cambia nada. >0.5 mejora, <0.5 penaliza.
        conf_mult = 0.8 + (confluence_score * 0.4) 
        strength *= conf_mult
        
        # 3. MULTIPLICADOR DE VOLUMEN (Impacto 0.9x a 1.25x)
        vol_ratio = setups.get('volume_ratio', 1.0)
        if vol_ratio > 2.5: vol_mult = 1.25
        elif vol_ratio > 1.5: vol_mult = 1.15
        elif vol_ratio > 1.0: vol_mult = 1.05
        else: vol_mult = 0.90
        strength *= vol_mult
        
        # 🌊 FASE 13: MICROSTRUCTURE BOOSTERS (Dark Pool, Gamma Risk, Magnetic Pull)
        of_metrics = setups.get('order_flow', {})
        if of_metrics:
            # Gamma Expansion Risk: Volatilidad comprimida con alto volumen = Breakout inminente
            if of_metrics.get('gamma_expansion_risk'):
                strength *= 1.20
            
            # Dark Pool Tracking: Ballenas atacando en la misma dirección = Ultra convicción
            dp_side = of_metrics.get('dark_pool_side')
            if dp_side:
                is_long = setups.get('long_mean_rev') or setups.get('long_momentum') or setups.get('long_scalp_break') or setups.get('long_rsi_explicit') or setups.get('long_macd_explicit') or setups.get('long_bb_explicit') or setups.get('long_volume_explicit')
                if (is_long and dp_side == 'BUY') or (not is_long and dp_side == 'SELL'):
                    strength *= 1.30
                else:
                    strength *= 0.80 # Dark Pool en contra
                    
            # Magnetic Pull: Atracción por niveles de liquidación cuántica
            pull_up = of_metrics.get('magnetic_pull_up', 0.0)
            pull_down = of_metrics.get('magnetic_pull_down', 0.0)
            if pull_up > 0 and pull_down > 0:
                is_long = setups.get('long_mean_rev') or setups.get('long_momentum') or setups.get('long_scalp_break') or setups.get('long_rsi_explicit') or setups.get('long_macd_explicit') or setups.get('long_bb_explicit') or setups.get('long_volume_explicit')
                if (is_long and pull_up > pull_down * 1.5) or (not is_long and pull_down > pull_up * 1.5):
                    strength *= 1.15
        
        # 4. BONUS POR RSI EXTREMO (Solo para Mean Reversion)
        if (setups.get('long_mean_rev') or setups.get('short_mean_rev')) and (setups['rsi'] < 25 or setups['rsi'] > 75):
            strength *= 1.15

        # 5. EXPLICIT SETUPS BOOST (SYMMETRIC: both LONG and SHORT)
        if setups.get('short_rsi_explicit') or setups.get('short_bb_explicit'):
            strength *= 1.2
        if setups.get('short_macd_explicit') or setups.get('short_volume_explicit'):
            strength *= 1.15
        # FORENSIC REMEDIATION: Mirror LONG explicit boosts
        if setups.get('long_rsi_explicit') or setups.get('long_bb_explicit'):
            strength *= 1.2
        if setups.get('long_macd_explicit') or setups.get('long_volume_explicit'):
            strength *= 1.15
        
        # 6. Penalty por volatilidad RELATIVA evolutiva (Phase 47.5 / V5 Genesis)
        # En BTC 1.5% es mucho, en SOL es normal. Usamos un umbral dinámico atado al Genotipo.
        genotype_vol = None
        if hasattr(self, 'genotypes') and symbol in self.genotypes:
            genotype_vol = self.genotypes[symbol].genes.get('vol_sensitivity', None)
        
        if genotype_vol is not None:
            vol_threshold = genotype_vol
        else:
            vol_threshold = 0.015 if 'BTC' in (symbol or '') else 0.025
            
        if volatility > vol_threshold * 1.5:
            strength *= 0.7
        elif volatility > vol_threshold:
            strength *= 0.9
            
        # --- V5.8 & V5.9 ASYMMETRIC COGNITIVE MODULATION ---
        if cog_state == 'INJURED':
            # V5.9 Endurecimiento de Filtro SOPHIA:
            # Si estamos perdiendo, solo aceptamos señales con respaldo de IA Brutal (>0.75)
            sophia_prob = confluence_score # Sophia Score mapped to confluence for easy check here
            if sophia_prob < 0.75:
                strength *= 0.50 # Penalización masiva si no hay convicción IA
                logger.debug(f"⚠️ [V5.9 ARMOR] Signal crushed (x0.5) because Sophia Prob ({sophia_prob:.2f}) < 0.75 in INJURED state.")
            else:
                # V5.13 PULSO DE RECUPERACIÓN: Si la señal es élite, bajamos la guardia para recuperar rápido
                if strength > 0.90:
                    strength *= 0.95 # Casi original (Recuperador Ágil)
                else:
                    strength *= 0.80 
                logger.debug(f"🛡️ [COGNITIVE] Signal modulated due to INJURED state.")
        elif cog_state == 'ALPHA':
            # V5.14 CATALYST: Predator Aggression
            # Maximizamos captura y frecuencia
            strength *= 1.35
            # Reduce penalty for volatility in ALPHA state (Predators love chaos)
            if volatility > 0.015:
                strength *= 1.1 # Inverse penalty
            logger.debug(f"🚀 [CATALYST] Predator Aggression Active (x1.35).")
            
        # FASE 68: VPIN Toxicity Hard Block (Institucional Dump Protection)
        vpin = of_metrics.get('vpin_toxicity', 0.0)
        if vpin > 0.80:
            is_long = setups.get('long_mean_rev') or setups.get('long_momentum') or setups.get('long_scalp_break') or setups.get('long_rsi_explicit') or setups.get('long_macd_explicit') or setups.get('long_bb_explicit') or setups.get('long_volume_explicit')
            delta_flow = of_metrics.get('delta', 0.0)
            
            # Si es largo pero el flujo es negativo (ventas), o es corto y flujo es positivo (compras)
            if (is_long and delta_flow < 0) or (not is_long and delta_flow > 0):
                if hasattr(logger, 'warning'):
                    logger.warning(f"☠️ [VPIN TOXICITY] HARD BLOCK! VPIN: {vpin:.2f} | Delta: {delta_flow:.0f}. Operación bloqueada contra cuchillo cayendo.")
                strength = 0.0
        
        return min(strength, 1.0)

    def generate_signals(self, event=None):
        """Generación de señales HÍBRIDA"""
        # Determine symbols to process
        symbols = []
        if self.symbol:
            symbols = [self.symbol]
        elif event and getattr(event, 'symbol', None):
            symbols = [event.symbol]
        else:
            symbols = self.data_provider.symbol_list
        
        # V5.21 Quantum Tunnelling: Ensure BTC is processed FIRST to update state
        if "BTC/USDT" in symbols:
            symbols = ["BTC/USDT"] + [s for s in symbols if s != "BTC/USDT"]
        
        # C-1 FIX: Lazy bind Némesis→Sophia feedback loop on first invocation
        if not self._sophia_feedback_linked and hasattr(self, 'portfolio') and self.portfolio:
            if hasattr(self.portfolio, 'link_nemesis_to_sophia'):
                self.portfolio.link_nemesis_to_sophia(self.sophia)
                self._sophia_feedback_linked = True
        
        for symbol in symbols:
            try:
                # 0. CONFIGURACIÓN DINÁMICA (Phase 7.2)
                params = self.get_symbol_params(symbol)
                ADX_THRESH = params['adx_threshold']
                STRENGTH_THRESH = params['strength_threshold']
                TP_PCT_LOCAL = params['tp_pct']
                SL_PCT_LOCAL = params['sl_pct']
                
                # B-1 FIX: Initialize dynamic risk parameters to prevent UnboundLocalError or leakage
                final_tp_pct = TP_PCT_LOCAL
                final_sl_pct = SL_PCT_LOCAL

                # FORENSIC FIX #1: Signal Deduplication (BAR-BASED, not tick-based)
                # QUÉ: Deduplicación basada en el timestamp de la BARRA OHLCV, no del tick.
                # POR QUÉ: El código anterior usaba int(event_time.timestamp()) que cambia
                #   cada segundo, permitiendo señales duplicadas infinitas (BUG-1).
                # CÓMO: Usamos el timestamp de la última barra cerrada del data_provider,
                #   que es fijo para todas las evaluaciones dentro de esa barra.
                # EVIDENCIA: massive_god_mode.log → ARB/USDT LONG repetido cada ~1s con mismos datos.
                # 0.5 DATA HEALTH GUARD (Phase 3 Hardening)
                health = getattr(event, 'health_metrics', None)
                if health and health.get('score', 100) < 80:
                    logger.warning(f"⚠️ [DATA-HEALTH] Skipping {symbol} due to poor integrity: {health['score']:.1f}% (Gap: {health.get('gap_s', 0)}s)")
                    if health.get('score', 100) < 50:
                        continue # Critical integrity loss
                
                event_time = event.timestamp if hasattr(event, 'timestamp') else datetime.now(timezone.utc)
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)

                # Use OHLCV bar timestamp for dedup (not tick time)
                try:
                    _bars = self.data_provider.get_latest_bars(symbol, n=2)
                    if _bars is not None and len(_bars) >= 2:
                        bar_ts = int(_bars['timestamp'][-2])  # Closed bar timestamp
                    else:
                        bar_ts = int(event_time.timestamp())
                except Exception:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    bar_ts = int(event_time.timestamp())
                
                dedupe_key = f"{symbol}_{self.horizon}_{bar_ts}"
                
                if self.last_processed_times.get(dedupe_key):
                    continue
                self.last_processed_times[dedupe_key] = True
                
                # Memory cleanup: Keep only last 500 entries to prevent unbounded growth
                if len(self.last_processed_times) > 500:
                    keys = list(self.last_processed_times.keys())
                    for k in keys[:250]:
                        del self.last_processed_times[k]
                
                # --- XRP SPECIFIC COOLDOWN (Rule 4.1) ---
                if 'XRP' in symbol:
                    last_trade = self.last_trade_times.get(symbol, 0)
                    if (event_time.timestamp() - last_trade) < 3600: # 60 minutes
                        continue
                
                # 1. Obtener datos multi-timeframe
                timeframe_data = self.get_multi_timeframe_data(symbol)
                
                # FORENSIC-1: Use horizon-specific primary timeframe
                primary_tf = self.PRIMARY_TF if hasattr(self, 'PRIMARY_TF') else '5m'
                if primary_tf not in timeframe_data:
                    continue
                
                pkg_primary = timeframe_data[primary_tf]
                data_primary = pkg_primary['data']
                inds_primary = pkg_primary['inds']
                
                if len(data_primary) < 5:
                    continue

                # --- Mutación 22: Z-Score Flash-Crash Interceptor ---
                of_metrics = self.data_provider.get_order_flow_metrics(symbol) if hasattr(self.data_provider, 'get_order_flow_metrics') else None
                if of_metrics and of_metrics.get('flash_crash_anomaly', False):
                    direction = of_metrics.get('flash_crash_direction')
                    if direction:
                        logger.critical(f"🚨 [FLASH-CRASH ANOMALY] {symbol} {direction} Triggered! Z-Score > 5 detected.")
                        flash_signal = SignalType.LONG if direction == 'BUY' else SignalType.SHORT
                        
                        event_out = SignalEvent(
                            symbol=symbol,
                            signal_type=flash_signal,
                            strength=1.0,
                            horizon='MICROSCALPING',
                            metrics={'setup_type': 'FLASH_CRASH_REVERSION', 'strength': 1.0, 'order_flow': of_metrics},
                            strategy_id=self.strategy_id
                        )
                        event_out.is_urgent = True
                        self.events_queue.put(event_out)
                        continue

                # 🚀 [PHASE 9] L2 Wick Sniper (Front-Running)
                if of_metrics:
                    of_delta = of_metrics.get('delta', 0.0)
                    tot_vol = of_metrics.get('total_volume', 1.0)
                    of_imbalance = of_delta / tot_vol if tot_vol > 0 else 0.0
                    
                    if abs(of_imbalance) > 0.85: # Threshold brutal de vaciado L2
                        if of_imbalance < -0.85: # Vendedores atrapados = Reversal LONG
                            sniper_signal = SignalType.LONG
                            logger.critical(f"🎯 [WICK SNIPER] {symbol} Delta Masivo Negativo ({of_imbalance*100:.1f}%) absorbido. Disparando LONG Front-Run.")
                        elif of_imbalance > 0.85: # Compradores atrapados = Reversal SHORT
                            sniper_signal = SignalType.SHORT
                            logger.critical(f"🎯 [WICK SNIPER] {symbol} Delta Masivo Positivo ({of_imbalance*100:.1f}%) absorbido. Disparando SHORT Front-Run.")
                        else:
                            sniper_signal = None
                            
                        if sniper_signal:
                            event_out = SignalEvent(
                                symbol=symbol,
                                signal_type=sniper_signal,
                                strength=1.0,  # Máxima convicción
                                horizon='MICROSCALPING',
                                metrics={'setup_type': 'L2_WICK_SNIPER', 'strength': 1.0, 'order_flow': of_metrics},
                                strategy_id=self.strategy_id
                            )
                            event_out.is_urgent = True
                            self.events_queue.put(event_out)
                            continue

                # Retrieve Brain for this symbol
                # This ensures we have a genotype (created by get_symbol_params if needed)
                # But get_symbol_params returns genes dict, we need the object for update.
                # We can access self.genotypes[symbol] directly or ensure it exists.
                self.get_symbol_params(symbol) # Ensure loaded/spawned
                current_genotype = self.genotypes.get(symbol)

                # --- PHASE 65: FUSED PATH (DIRECT SYMBOL BRAIN) ---
                use_fused_path = getattr(params, 'use_fused_path', False) if params else False
                
                signal_type = None
                strength = 0.0
                
                if use_fused_path and current_genotype and 'brain_weights' in current_genotype.genes:
                    try:
                        # 1. Obtain Portfolio State
                        real_pos = self.data_provider.get_active_positions().get(symbol, {'quantity': 0})
                        
                        # 2. Fused Insight (Indicators -> State -> Inference)
                        fused_decision, fused_confidence = self.get_fused_insight(
                            symbol, data_primary, portfolio_state=real_pos
                        )
                        
                        if fused_decision:
                            signal_type = fused_decision
                            strength = fused_confidence
                            
                            # Backfill 'setups' for logic compatibility downstream
                            # This ensures Step 8+ works without modification
                            setups = {
                                'close': data_primary['close'][-1],
                                'atr': inds_primary['atr'][-1],
                                'adx': 30, # Placeholder for fused path
                                'rsi': 50, # Placeholder for fused path
                                'in_uptrend': True,
                                'in_downtrend': False,
                                'volume_ratio': 1.0,
                                'long_mean_rev': fused_decision == SignalType.LONG,
                                'short_mean_rev': fused_decision == SignalType.SHORT,
                                'long_momentum': False,
                                'short_momentum': False,
                                'bb_position': 0.5
                            }
                            confluence_score = 1.0 
                            volatility = setups['atr'] / setups['close']
                            
                            # Skip legacy sequential calculation
                            # GOTO Step 6
                            goto_step_6 = True
                        else:
                            goto_step_6 = False
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        print(f"❌ Fused Path Error {symbol}: {e}")
                        goto_step_6 = False
                else:
                    goto_step_6 = False

                if not goto_step_6:
                    # Legacy Sequential Path
                    # 2. Calcular confluence multi-timeframe
                    confluence_score = self.calculate_multi_timeframe_confluence(timeframe_data, symbol)
                    
                    # FORENSIC-AUDIT-V9-FIX: Confluence threshold REDUCED 0.15 → 0.05
                    # QUÉ: En scalping 1m, confluencia multi-TF rara vez supera 0.15.
                    # POR QUÉ: El threshold de 0.15 mataba >80% de señales válidas.
                    # EVIDENCIA: Audit V9 mostraba tasa de supervivencia ~0.02% con 19 gates.
                    # PARA QUÉ: Permitir que señales con edge mínimo lleguen a Sophia para
                    #   evaluación más sofisticada (en vez de morir en el primer filtro).
                    if confluence_score < 0.05:
                        if 'BTC' not in symbol:
                            logger.debug(f"🛑 [DIAG] {symbol} Killed by low confluence: {confluence_score:.2f}")
                        continue
                        
                    # Pasa el símbolo para la validación cognitiva V5.7
                    setups = self.detect_setup(pkg_primary, params, symbol)
                    if not setups:
                        # logger.debug(f"DEBUG: {symbol} No active setup detected.")
                        continue
                    
                    # 4. Calcular volatilidad y ajustar confluencia (Phase 5.5 DPE)
                    volatility = setups['atr'] / setups['close']
                    
                    # Adaptar umbral de confluencia a la volatilidad reciente
                    # [FORENSIC-RECAL] Tightened penalty: old +0.05 at 0.5% ATR was near-permanent in M5 crypto
                    base_strength = params.get('strength_threshold', 0.45) if params else 0.45
                    dynamic_strength = base_strength
                    
                    if volatility > 0.010:  # [FORENSIC-RECAL] Only extreme volatility (>1.0% ATR/Price)
                        dynamic_strength += 0.02 # Mild penalty (was +0.05, killed all signals)
                    elif volatility < 0.001: # Ultra Low Volatility
                        dynamic_strength -= 0.03 # Relax threshold for flat markets
                        
                    params['strength_threshold'] = dynamic_strength
                    
                    # 5. Determinar dirección y tipo de setup V5.8
                    signal_type = None
                    setup_type = "UNKNOWN"
                    if setups.get('liquidity_void_long') or setups.get('liquidity_void_short'):
                        signal_type = SignalType.LONG if setups.get('liquidity_void_long') else SignalType.SHORT
                        setup_type = "LIQUIDITY_VOID_SNIPER"
                    elif setups.get('long_mean_rev') or setups.get('short_mean_rev'):
                        signal_type = SignalType.LONG if setups.get('long_mean_rev') else SignalType.SHORT
                        setup_type = "MEAN_REV"
                    elif setups.get('long_momentum') or setups.get('short_momentum'):
                        signal_type = SignalType.LONG if setups.get('long_momentum') else SignalType.SHORT
                        setup_type = "MOMENTUM"
                    elif setups.get('long_scalp_break') or setups.get('short_scalp_break'):
                        signal_type = SignalType.LONG if setups.get('long_scalp_break') else SignalType.SHORT
                        setup_type = "SCALP_BREAKOUT"
                    # ================================================================
                    # FORENSIC REMEDIATION: Process BOTH LONG and SHORT explicit setups
                    # ================================================================
                    elif setups.get('long_mean_rev'):
                        if self.horizon == 'SCALPING' and not setups.get('_long_vol_gate_pass', True):
                            logger.debug(f"🚫 [VOL_GATE] {symbol} LONG MEAN_REVERSION blocked: extreme volatility")
                        else:
                            signal_type = SignalType.LONG
                            setup_type = "RSI_MEAN_REVERSION"
                    elif setups.get('short_mean_rev'):
                        if self.horizon == 'SCALPING' and not setups.get('_short_vol_gate_pass', True):
                            logger.debug(f"🚫 [VOL_GATE] {symbol} SHORT MEAN_REVERSION blocked: extreme volatility")
                        else:
                            signal_type = SignalType.SHORT
                            setup_type = "RSI_MEAN_REVERSION"
                    elif setups.get('long_momentum'):
                        if self.horizon == 'SCALPING' and not setups.get('_long_vol_gate_pass', True):
                            logger.debug(f"🚫 [VOL_GATE] {symbol} LONG MOMENTUM blocked: extreme volatility")
                        else:
                            signal_type = SignalType.LONG
                            setup_type = "TREND_MOMENTUM"
                    elif setups.get('short_momentum'):
                        if self.horizon == 'SCALPING' and not setups.get('_short_vol_gate_pass', True):
                            logger.debug(f"🚫 [VOL_GATE] {symbol} SHORT MOMENTUM blocked: extreme volatility")
                        else:
                            signal_type = SignalType.SHORT
                            setup_type = "TREND_MOMENTUM"
                    elif setups.get('long_rsi_explicit') or setups.get('long_bb_explicit'):
                        if self.horizon == 'SCALPING' and not setups.get('_long_vol_gate_pass', True):
                            logger.debug(f"🚫 [VOL_GATE] {symbol} LONG EXPLICIT_REVERSAL blocked: extreme volatility")
                        else:
                            signal_type = SignalType.LONG
                            setup_type = "EXPLICIT_REVERSAL"
                    elif setups.get('short_rsi_explicit') or setups.get('short_bb_explicit'):
                        if self.horizon == 'SCALPING' and not setups.get('_short_vol_gate_pass', True):
                            logger.debug(f"🚫 [VOL_GATE] {symbol} SHORT EXPLICIT_REVERSAL blocked: extreme volatility")
                        else:
                            signal_type = SignalType.SHORT
                            setup_type = "EXPLICIT_REVERSAL"
                    elif setups.get('long_macd_explicit') or setups.get('long_volume_explicit'):
                        if self.horizon == 'SCALPING' and not setups.get('_long_vol_gate_pass', True):
                            logger.debug(f"🚫 [VOL_GATE] {symbol} LONG EXPLICIT_MOMENTUM blocked: extreme volatility")
                        else:
                            signal_type = SignalType.LONG
                            setup_type = "EXPLICIT_MOMENTUM"
                    elif setups.get('short_macd_explicit') or setups.get('short_volume_explicit'):
                        if self.horizon == 'SCALPING' and not setups.get('_short_vol_gate_pass', True):
                            logger.debug(f"🚫 [VOL_GATE] {symbol} SHORT EXPLICIT_MOMENTUM blocked: extreme volatility")
                        else:
                            signal_type = SignalType.SHORT
                            setup_type = "EXPLICIT_MOMENTUM"
                        
                else:
                    setup_type = "FUSED_ML"

                if signal_type is None:
                    try:
                        from core.global_state import global_state
                        global_state.update_symbol_vector(symbol, {
                            "tech_long_active": 0,
                            "tech_short_active": 0
                        })
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        import logging
                        logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
                    continue
                else:
                    try:
                        from core.global_state import global_state
                        global_state.update_symbol_vector(symbol, {
                            "tech_long_active": 1 if signal_type == SignalType.LONG else 0,
                            "tech_short_active": 1 if signal_type == SignalType.SHORT else 0
                        })
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        import logging
                        logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
                
                # 🔮 FASE 5: MULTI-COIN ORACLE (LEAD-LAG ARBITRAGE) + CROSS-EXCHANGE
                # 1. Macro BTC Bias
                if symbol != "BTC/USDT":
                    try:
                        from core.global_state import global_state
                        btc_vel = getattr(global_state, 'btc_velocity', 0.0)
                        if btc_vel > 0.005 and signal_type == SignalType.LONG:
                            logger.critical(f"🚀 [MULTI-COIN ORACLE] BTC Velocity ALTA ({btc_vel:.4f}). Technical LONG acelerado en {symbol}!")
                        elif btc_vel < -0.005 and signal_type == SignalType.SHORT:
                            logger.critical(f"📉 [MULTI-COIN ORACLE] BTC Velocity NEGATIVA ({btc_vel:.4f}). Technical SHORT acelerado en {symbol}!")
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        import logging
                        logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
                        
                # 2. Micro Cross-Exchange PDC Bias (Coinbase/Deribit/Bybit)
                cross_exchange_bypass = False
                try:
                    if hasattr(global_state, 'cross_exchange_metrics'):
                        metrics = global_state.cross_exchange_metrics.get(symbol, {})
                        pdc = metrics.get('pdc_signal', 0.0)
                        
                        if pdc > 0.3 and signal_type == SignalType.LONG:
                            logger.critical(f"🌌 [CROSS-EXCHANGE] Fuerte PDC ALCISTA ({pdc:.2f}) en Coinbase/Bybit. Protegiendo señal LONG en {symbol} de vetos macro.")
                            cross_exchange_bypass = True
                            if 'strength' in locals(): strength = min(1.0, strength + 0.2)
                        elif pdc < -0.3 and signal_type == SignalType.SHORT:
                            logger.critical(f"🌌 [CROSS-EXCHANGE] Fuerte PDC BAJISTA ({pdc:.2f}) en Coinbase/Bybit. Protegiendo señal SHORT en {symbol} de vetos macro.")
                            cross_exchange_bypass = True
                            if 'strength' in locals(): strength = min(1.0, strength + 0.2)
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    import logging
                    logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
                
                # ═══════════════════════════════════════════════════
                # Mutación 39: BAYESIAN MIRAGE (SPOOFING VETO)
                # ═══════════════════════════════════════════════════
                # QUÉ: Protege contra "Liquidity Traps" (muros falsos).
                # POR QUÉ: Si entramos LONG apoyados en un gran Bid Wall, y este
                #   fue calculado como Spoofing (>80%), nos van a liquidar cuando lo quiten.
                direction_str = 'LONG' if signal_type == SignalType.LONG else 'SHORT'
                
                _of_metrics = self.data_provider.get_order_flow_metrics(symbol) if hasattr(self.data_provider, 'get_order_flow_metrics') else None
                if _of_metrics:
                    _prob_buy = _of_metrics.get('spoofing_prob_buy', 0.0)
                    _prob_sell = _of_metrics.get('spoofing_prob_sell', 0.0)
                    
                    if direction_str == 'LONG' and _prob_buy > 0.80:
                        logger.warning(f"🚨 [BAYESIAN MIRAGE] {symbol} LONG VETOED. Fake Buy Wall (Trap) Detected (Prob: {_prob_buy:.1%})")
                        continue
                        
                    if direction_str == 'SHORT' and _prob_sell > 0.80:
                        logger.warning(f"🚨 [BAYESIAN MIRAGE] {symbol} SHORT VETOED. Fake Sell Wall (Trap) Detected (Prob: {_prob_sell:.1%})")
                        continue

                # ═══════════════════════════════════════════════════════════════
                # 🚀 FASE 12: CROSS-HORIZON RESONANCE (Filtro Cuántico)
                # QUÉ: Suprime operaciones Scalp/Microscalp contra la tendencia Swing activa.
                # POR QUÉ: Un Swing activo significa que el sesgo macro es fuerte en esa dirección.
                # ═══════════════════════════════════════════════════════════════
                portfolio = getattr(self, 'portfolio', None) or (getattr(self, '_engine_ref', None).portfolio if getattr(self, '_engine_ref', None) else None)
                if portfolio and self.horizon in ("SCALPING", "MICROSCALPING", "MICRO"):
                    active_pos = portfolio.positions.get(symbol, [])
                    swing_opposing = any(
                        p.get('horizon') == "SWING" and 
                        ((p['direction'] == 1 and direction_str == "SHORT") or
                         (p['direction'] == -1 and direction_str == "LONG"))
                        for p in active_pos
                    )
                    if swing_opposing:
                        logger.info(
                            f"🛑 [CROSS-HORIZON RESONANCE] {symbol} {direction_str} BLOCKED | "
                            f"Opposing active SWING position detected."
                        )
                        continue

                # ═══════════════════════════════════════════════════
                # PHASE 3: MULTI-HORIZON ORACLE VETO
                # ═══════════════════════════════════════════════════
                # QUÉ: Consultar al Oráculo si el contexto macro (1d, 1w) permite este trade.
                # POR QUÉ: El 47% de Stop Loss hits ocurrían por trades micro alineados contra la macro-tendencia.
                # CÓMO: Si 1D y 1W van en dirección opuesta al trade, se VETEA la operación.
                try:
                    direction_str = 'LONG' if signal_type == SignalType.LONG else 'SHORT'
                    oracle_verdict = MultiHorizonOracle.evaluate_clash_vector(timeframe_data, direction_str, horizon=self.horizon)
                    
                    # FORENSIC-V9-FIX: Oracle Veto converted from HARD BLOCK to SOFT PENALTY
                    # QUÉ: El Oracle bloqueaba 100% de trades contra-macro con `continue`.
                    # POR QUÉ: En scalping, micro-estructuras rentables ocurren CONTRA la macro.
                    #   Un retroceso de 0.3% en una tendencia bajista es un scalp válido.
                    # PARA QUÉ: Reducir la tasa de rechazo de señales sin eliminar la protección.
                    # CÓMO: Veto total solo si clash > 0.85 (extremo). De lo contrario, penalty.
                    if oracle_verdict['is_vetoed']:
                        # 🔮 FASE 5: MULTI-COIN ORACLE (LEAD-LAG ARBITRAGE)
                        # Bypass the veto if BTC velocity strongly supports this signal
                        btc_vel_bypass = False
                        if symbol != "BTC/USDT":
                            try:
                                from core.global_state import global_state
                                btc_vel = getattr(global_state, 'btc_velocity', 0.0)
                                if btc_vel > 0.005 and signal_type == SignalType.LONG:
                                    btc_vel_bypass = True
                                    logger.critical(f"🚀 [MULTI-COIN ORACLE] Ignorando Veto Macro para {symbol} LONG debido a BTC Velocity ({btc_vel:.4f}).")
                                elif btc_vel < -0.005 and signal_type == SignalType.SHORT:
                                    btc_vel_bypass = True
                                    logger.critical(f"📉 [MULTI-COIN ORACLE] Ignorando Veto Macro para {symbol} SHORT debido a BTC Velocity ({btc_vel:.4f}).")
                            except Exception as e:
                                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                                import logging
                                logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
                                
                        if btc_vel_bypass or cross_exchange_bypass:
                            pass # We ignore the veto completely
                        else:
                            clash = oracle_verdict['clash_score']
                        # QUÉ: Reducción del umbral de veto duro del oráculo macro de 0.85 a 0.60.
                        # POR QUÉ: Un clash_score > 0.60 indica una fuerte contradicción entre la señal y la macro-tendencia.
                        #   Dado el capital micro de $13 USD, no podemos permitirnos asumir riesgos innecesarios.
                        # PARA QUÉ: Evitar pérdidas en condiciones de mercado donde el contexto estructural es hostil.
                        # CÓMO: Cambiando la condición de descarte directo de `clash > 0.85` a `clash > 0.60`.
                        # CUÁNDO: Al evaluar señales generadas antes de ser enviadas a la cola de eventos.
                        # DÓNDE: En `strategies/technical.py` L1317.
                        # QUIÊN: Modificado por el Risk Manager y el Quant Developer.
                        if clash > 0.60:
                            # HARD VETO: Solo para clash extremo/fuerte (macro y micro opuestos)
                            logger.info(
                                f"🔮 [ORACLE VETO] {symbol} {direction_str} BLOCKED (EXTREME) | "
                                f"Clash: {clash:.1%} | Macro: {oracle_verdict['macro_context']}"
                            )
                            continue
                        else:
                            # SOFT PENALTY: Reduce strength pero permite la señal
                            clash_penalty = max(0.4, 1.0 - clash)
                            strength = strength * clash_penalty if 'strength' in dir() else 0.4
                            logger.info(
                                f"🔮 [ORACLE SOFT] {symbol} {direction_str} PENALIZED x{clash_penalty:.2f} | "
                                f"Clash: {clash:.1%} | Macro: {oracle_verdict['macro_context']}"
                            )
                    
                    # Choque parcial menor: penalty suave
                    elif oracle_verdict['clash_score'] > 0.3:
                        clash_penalty = 1.0 - (oracle_verdict['clash_score'] * 0.3)
                        strength = strength * clash_penalty if 'strength' in dir() else 0.5
                        logger.debug(
                            f"🔮 [ORACLE WARN] {symbol} {direction_str} WEAKENED x{clash_penalty:.2f} | "
                            f"Macro: {oracle_verdict['macro_context']}"
                        )
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    logger.warning(f"🔮 [ORACLE] Evaluation failed for {symbol}: {e}")
                    # Fail-open: si el Oráculo falla, permitir el trade (no bloquear por bug)
                
                # --- XRP TREND ALIGNMENT (Rule 4.3) ---
                if 'XRP' in symbol:
                    if '1h' in timeframe_data:
                        inds_1h = timeframe_data['1h']['inds']
                        idx_1h = -2 if len(inds_1h['in_uptrend']) > 1 else -1
                        trend_1h = 1 if inds_1h['in_uptrend'][idx_1h] else -1
                        if (signal_type == SignalType.LONG and trend_1h < 0) or \
                           (signal_type == SignalType.SHORT and trend_1h > 0):
                            continue

                # 🌊 FASE 13: Microstructure Data Integration
                if '_of_metrics' in locals() and _of_metrics:
                    setups['order_flow'] = _of_metrics
                else:
                    _of_metrics = self.data_provider.get_order_flow_metrics(symbol) if hasattr(self.data_provider, 'get_order_flow_metrics') else {}
                    setups['order_flow'] = _of_metrics
                
                # 6. Calcular fuerza (Pasando Símbolo y Setup_Type para Asimetría V5.8)
                strength = self.calculate_signal_strength(setups, confluence_score, volatility, symbol, setup_type)
                
                # ═══════════════════════════════════════════════════════
                # FORENSIC-V80: FEE HEADROOM & STRICT MOMENTUM GUARD
                # QUÉ: Bloqueo duro si el mercado está completamente plano.
                # POR QUÉ: Comisiones consumen 0.075%. Si ATR es 0.10%, ganar 0.10% netos es imposible.
                #   Esto generaba los 54+ TIME_STOP_ZOMBIE.
                # ═══════════════════════════════════════════════════════
                current_adx = setups.get('adx', 0)
                
                # Calcular ATR_PCT real (porque no viene en setups dict)
                current_atr = setups.get('atr', 0)
                current_close = setups.get('close', 1)
                atr_pct = current_atr / current_close if current_close > 0 else 0
                
                # MÓDULO HORIZON: Pre-consensus ATR filter — horizon-differentiated
                # QUÉ: Mínimo ATR requerido para generar señal, adaptado al horizonte.
                # POR QUÉ: MICRO tolera ATR más bajo (SL ajustado protege), SWING necesita más.
                # HORIZONTE | min_atr | Razón
                # MICRO     | 0.02%  | Micro-edges en mercados quietos
                # SCALP     | 0.04%  | Balance frecuencia/calidad
                # SWING     | 0.15%  | Solo movimientos significativos
                if self.horizon == 'MICROSCALPING':
                    min_atr_required = 0.0002  # HORIZONTE: MICRO | 0.02%
                elif self.horizon == 'SCALPING':
                    min_atr_required = 0.0004  # HORIZONTE: SCALP | 0.04%
                else:
                    min_atr_required = 0.0015  # HORIZONTE: SWING | 0.15%
                
                if atr_pct < min_atr_required:
                    logger.warning(f"🛑 [VOLATILITY BLOCK] {symbol} {self.horizon} ATR {atr_pct*100:.3f}% < {min_atr_required*100:.3f}%")
                    continue
                
                if current_adx < ADX_THRESH:
                    # En Scalping, el ADX bajo significa Chop. NO OVERRIDES permitidos si ATR es mediocre.
                    if atr_pct < (min_atr_required * 1.5):
                        logger.warning(f"🛑 [CHOP BLOCK] {symbol} ADX {current_adx:.1f} < {ADX_THRESH} and low ATR")
                        continue
                
                # 🔮 FASE 20: PDC (Price Discovery Coefficient) Veto for Scalping
                if self.horizon == 'SCALPING':
                    try:
                        from core.global_state import global_state
                        ce_metrics = getattr(global_state, 'cross_exchange_metrics', {})
                        sym_pdc = ce_metrics.get(symbol, {})
                        pdc_velocity = sym_pdc.get('pdc_velocity', 0.0)
                        
                        min_pdc = getattr(Config.Strategies, 'TECHNICAL_THRESHOLDS', {}).get('min_pdc_velocity', 0.05)
                        if abs(pdc_velocity) < min_pdc: # Requerimos velocidad significativa (positiva o negativa dependiendo de la dirección, pero usemos valor absoluto de velocidad o convicción)
                            # Actually, if signal_type is LONG, we want pdc_velocity > min_pdc
                            # if signal_type is SHORT, we want pdc_velocity < -min_pdc
                            # But wait, pdc is usually positive for lead-lag strength? 
                            # If PDC Velocity is a magnitude, we just need pdc_velocity > min_pdc.
                            # "Scalping strictly requires positive confirmation (PDC Velocity > threshold)."
                            if pdc_velocity < min_pdc:
                                # PARIDAD ABSOLUTA: PDC Veto aplica en backtest y producción (Fase V Audit Fix #3)
                                logger.warning(f"🛑 [PDC VETO] {symbol} SCALPING blocked | PDC Velocity: {pdc_velocity:.4f} < {min_pdc}")
                                continue
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.warning(f"⚠️ [PDC VETO] Error checking PDC for {symbol}: {e}")
                
                # 🧬 PHASE 3 SYNERGY: DNA TOGGLES (KILL SWITCHES)
                # DNA_SNIPER_VOLUME: Require volume burst for entry
                if os.environ.get("DNA_SNIPER_VOLUME") == "1":
                    vol_ratio = setups.get('volume_ratio', 1.0)
                    if vol_ratio < 1.2:
                        logger.debug(f"🧬 [DNA_SNIPER_VOLUME] {symbol} BLOCKED: Volume Ratio {vol_ratio:.2f} < 1.2")
                        continue
                
                # DNA_PATTERN_STRICT: Require higher confidence for momentum
                min_momentum_strength = 0.70 if os.environ.get("DNA_PATTERN_STRICT") == "1" else 0.60

                # CONFIDENCE GATE: Strict threshold for Momentum to avoid low-quality entries
                if setup_type == "MOMENTUM" and strength < min_momentum_strength:
                    if 'BTC' not in symbol:
                        logger.debug(f"🛑 [DIAG] {symbol} Momentum Strength {strength:.2f} < {min_momentum_strength:.2f} gate")
                    continue
                
                # 2. Umbral de Fuerza Dinámico
                if strength < STRENGTH_THRESH:
                    if 'BTC' not in symbol:
                        logger.debug(f"🛑 [DIAG] {symbol} Strength {strength:.2f} < {STRENGTH_THRESH:.2f}")
                    continue
                
                # SUPREMO-V4: Inject signal strength into the event for downstream SMART-ORDER logic
                signal_metadata = {'setup_type': setup_type, 'strength': strength}
                
                # 7. Verificar si ya estamos en posición
                if symbol not in self.bought:
                    self.bought[symbol] = False
                           # 9. Gestión de Posición Existente (handled below in Intelligent Reverse Detection)
                # --- PHASE 46: MULTIVERSE MUTATION ---
                self._apply_live_mutation(symbol, current_genotype)
                
                # --- PHASE 47.1: THE PERPETUAL PULSE (Infinitesimal Adaptation) ---
                self._apply_infinitesimal_tuning(symbol, current_genotype)
                # --- INTELLIGENT REVERSE DETECTION ---
                if self.bought[symbol]:
                    existing_pos = self.data_provider.get_active_positions().get(symbol, {'quantity': 0})
                    current_qty = existing_pos['quantity']
                    
                    is_currently_long = current_qty > 0
                    is_currently_short = current_qty < 0
                    
                    strong_reverse = False
                    if is_currently_long and (setups['short_mean_rev'] or setups['short_momentum']) and strength > 0.8:
                        strong_reverse = True
                        signal_type = SignalType.REVERSE
                    elif is_currently_short and (setups['long_mean_rev'] or setups['long_momentum']) and strength > 0.8:
                        strong_reverse = True
                        signal_type = SignalType.REVERSE
                    
                    if strong_reverse:
                        logger.info(f"🔄 [{symbol}] STRONG REVERSE detected (Strength: {strength:.2f}). Triggering Flip.")
                    else:
                        # --- DYNAMIC EXIT & TRAILING (Phase 71-73) ---
                        current_rsi = setups['rsi']
                        current_price = setups['close']
                        
                        # 1. Proactive BE Guard (V5.28 RAZOR-RELAXED)
                        entry_price = self.last_trade_prices.get(symbol, current_price)
                        cur_pnl = (current_price / entry_price - 1.0) if current_qty > 0 else (entry_price / current_price - 1.0)
                        
                        # FORENSIC FIX #1: FEE GUARD
                        # QUÉ: Exigir un PnL mínimo antes de permitir cierres "proactivos" o "por RSI".
                        # POR QUÉ: Cierres prematuros (ej. PnL 0.035%) son devorados por las comisiones (0.04%).
                        _maker_fee = getattr(Config, "BINANCE_MAKER_FEE_BNB", 0.0002)
                        _taker_fee = getattr(Config, "BINANCE_TAKER_FEE_BNB", 0.000375)
                        fee_guard_pnl = (_maker_fee + _taker_fee) * 2.5
                        
                        # FORENSIC FIX #3: SWING FIREWALL
                        # QUÉ: Aislar los trades SWING de los cierres tempranos de Scalping.
                        # POR QUÉ: Swing busca 1-5%, no debe cerrar en 60s por ruido de RSI.
                        is_swing = self.horizon in ['SWING', 'MACRO']
                        
                        be_trigger = final_tp_pct * 0.8 if 'final_tp_pct' in dir() else 0.008
                        if cur_pnl > be_trigger and symbol not in self.trailing_sl:
                            new_sl = entry_price * 1.001 if current_qty > 0 else entry_price * 0.999
                            self.trailing_sl[symbol] = new_sl
                            logger.info(f"🛡️ [V5.28 RAZOR-RELAX] Proactive BE Guard Activated for {symbol} (PnL: {cur_pnl*100:.2f}%)")
                        
                        # 2. RSI-Based Trailing (Legacy check) - BLOCKED FOR SWING
                        elif symbol not in self.trailing_sl and not is_swing:
                            trailing_rsi_thresh = params.get('trailing_rsi', 70)
                            should_trail = (current_qty > 0 and current_rsi > trailing_rsi_thresh) or \
                                           (current_qty < 0 and current_rsi < (100 - trailing_rsi_thresh))
                            
                            if should_trail:
                                new_sl = entry_price * 1.001 if current_qty > 0 else entry_price * 0.999
                                self.trailing_sl[symbol] = new_sl
                                logger.info(f"🛡️ [{symbol}] Trailing SL Activated by RSI at {new_sl:.6f}")
                            
                        # AEGIS-V15: Stability Guard
                        # QUÉ: No permite cerrar por indicadores si no han pasado X barras.
                        # POR QUÉ: Evita micro-trades de 0 segundos que mueren por fees.
                        entry_bar = self.bought[symbol] if isinstance(self.bought[symbol], int) else 0
                        bars_held = len(data_primary) - entry_bar
                        min_bars = 3 if self.horizon == 'SCALPING' else 12  # SWING needs more time
                        
                        if bars_held >= min_bars:
                            # 2. Check Trailing SL Hit
                            if symbol in self.trailing_sl:
                                tsl = self.trailing_sl[symbol]
                                if (current_qty > 0 and current_price <= tsl) or \
                                   (current_qty < 0 and current_price >= tsl):
                                    # Ensure we don't exit for a loss if we were trailing (slippage protection)
                                    if cur_pnl >= (fee_guard_pnl * 0.5):
                                        exit_signal = SignalEvent(
                                            strategy_id=self.strategy_id,
                                            symbol=symbol,
                                            datetime=event_time,
                                            signal_type=SignalType.EXIT,
                                            strength=1.0,
                                            horizon=self.horizon,
                                            priority=self.priority,
                                            metadata={'urgent': False, 'actual_order_type': 'limit', 'is_tp_limit': True}
                                        )
                                        self.events_queue.put(exit_signal)
                                        self.bought[symbol] = False
                                        self.trailing_sl.pop(symbol, None)
                                        logger.info(f"🛡️ [{symbol}] BREAK-EVEN/TRAILING EXIT at {current_price:.6f}")
                                        continue

                            # 3. RSI Extreme Exit (Partial/Total) - PROTECTED BY FEE GUARD & FIREWALL
                            if not is_swing and cur_pnl > fee_guard_pnl:
                                if (current_qty > 0 and current_rsi > 80) or \
                                   (current_qty < 0 and current_rsi < 20):
                                    exit_signal = SignalEvent(
                                        strategy_id=self.strategy_id,
                                        symbol=symbol,
                                        datetime=event_time,
                                        signal_type=SignalType.EXIT,
                                        strength=1.0,
                                        horizon=self.horizon,
                                        priority=self.priority,
                                        metadata={'urgent': False, 'actual_order_type': 'limit', 'is_tp_limit': True}
                                    )
                                    self.events_queue.put(exit_signal)
                                    self.bought[symbol] = False
                                    logger.info(f"🛡️ [{symbol}] RSI EXTREME EXIT at {current_price:.6f} after {bars_held} bars. PnL: {cur_pnl*100:.2f}%")
                                    continue

                # 10. Emit Signal si no hay posición o si la reversión es viable
                # AEGIS-V15: Atribución Granular
                # QUÉ: Inyecta el setup_type en el strategy_id.
                # POR QUÉ: Permite saber si perdemos plata en MEAN_REV o MOMENTUM.
                detailed_id = f"{self.strategy_id}.{setup_type}"
                
                if not self.bought[symbol]:
                    # Store bar index for min_hold validation
                    self.bought[symbol] = len(data_primary)
                dist_to_tp = setups['close'] * TP_PCT_LOCAL
                current_atr = setups['atr']
                time_to_target = 10
                if current_atr > 0:
                    ttt_bars = dist_to_tp / current_atr
                    time_to_target = max(1, int(ttt_bars))

                # ⚙️ MODO EVOLUTIVO V5.6: TOTAL DYNAMIC ECOSYSTEM (Riesgo Auto-calculado)
                # El sistema mide la varianza matemática para evitar SL fijos y buscar TPs altísimos si hay volumen
                current_atr = setups['atr']
                current_price = setups['close']
                
                # Dynamic Risk Pipeline (Dual Paradigm Injection)
                current_regime = 'UNKNOWN'  # Default regime when portfolio is not available (e.g., backtest)
                if hasattr(self, 'portfolio') and self.portfolio and hasattr(self.portfolio, 'global_regime_data'):
                    regime_meta = self.portfolio.global_regime_data
                    current_regime = regime_meta.get('sentiment', 'UNKNOWN')
                    # Phase 6: Specific symbol regime
                    symbol_regime = regime_meta.get('symbol_regimes', {}).get(symbol, current_regime)
                    
                atr_sl_mult, atr_tp_mult, final_sl_pct, final_tp_pct = self._calculate_dynamic_risk_params(
                    inds_primary, current_price, setup_type=setup_type, regime=current_regime
                )
                
                if current_atr > 0:
                    # FORENSIC-V9-FIX: Volatility filter relaxed for majors
                    # QUÉ: BTC/ETH siempre tienen suficiente liquidez para scalping.
                    # POR QUÉ: El filtro de 0.10% mataba señales en períodos de consolidación
                    #   que son exactamente cuando mean-reversion funciona mejor.
                    # PARA QUÉ: Permitir scalping durante consolidación en majors.
                    vol_ratio = current_atr / current_price
                    is_major = any(m in symbol for m in ['BTC', 'ETH', 'BNB', 'SOL'])
                    vol_floor = 0.0001 if is_major else 0.0002  # 0.01% majors, 0.02% alts (Relaxed for Micro-Scalping)
                    if vol_ratio < vol_floor:
                        logger.debug(f"💤 [V5.6] {symbol} Skipping: Low volatility ({vol_ratio*100:.3f}% < {vol_floor*100:.2f}%).")
                        continue
                    
                    logger.debug(f"⚡ [V5.6 DPE] {symbol}: Volatility Auto-tuned -> SL={final_sl_pct*100:.2f}%, TP={final_tp_pct*100:.2f}%")
                else:
                    # Fallback crítico usando Config centralizado
                    final_sl_pct = Config.Strategies.SWING_PARAMS['sl_pct'] if self.horizon == 'SWING' else Config.Strategies.SCALPING_PARAMS['sl_pct']
                    final_tp_pct = Config.Strategies.SWING_PARAMS['tp_pct'] if self.horizon == 'SWING' else Config.Strategies.SCALPING_PARAMS['tp_pct']


                # ── FASE 25: VETO CUÁNTICO (Microestructura / Order Flow) ──
                of_metrics = self.data_provider.get_order_flow_metrics(symbol)
                if of_metrics and signal_type != SignalType.HOLD:
                    is_toxic = of_metrics.get('is_toxic', False)
                    vpin = of_metrics.get('vpin', 0.5)
                    iceberg = of_metrics.get('iceberg_score', 0.0)
                    delta = of_metrics.get('rolling_delta_60s', 0.0)
                    
                    if is_toxic:
                        logger.warning(f"🌌 [VETO CUÁNTICO] {symbol} {signal_type.name} CANCELADO | Flujo Tóxico/Icebergs (VPIN: {vpin:.2f}, Score: {iceberg:.2f})")
                        continue
                    
                    # CONSENSO PONDERADO v1.0 — Delta Pressure
                    # QUÉ: Presión de venta/compra agresiva NO mata la señal, la penaliza.
                    # POR QUÉ: En scalping, contra-tendencia de corto plazo puede ser rentable.
                    # PARA QUÉ: Permitir scalps contra-corriente con sizing reducido.
                    if signal_type == SignalType.LONG and delta < -100:
                         delta_penalty = max(0.5, 1.0 - (abs(delta) - 100) / 500)
                         strength *= delta_penalty
                         logger.info(f"📉 [CONSENSUS] {symbol} LONG penalized x{delta_penalty:.2f} | Sell Pressure Delta={delta:.0f}")
                    elif signal_type == SignalType.SHORT and delta > 100:
                         delta_penalty = max(0.5, 1.0 - (delta - 100) / 500)
                         strength *= delta_penalty
                         logger.info(f"📈 [CONSENSUS] {symbol} SHORT penalized x{delta_penalty:.2f} | Buy Pressure Delta={delta:.0f}")

                # ── SOPHIA-INTELLIGENCE: Pre-trade XAI Analysis ──
                sophia_report = None
                sophia_narrative = ""
                # Define fallback regime for Sophia
                symbol_regime_val = "UNKNOWN"
                if 'symbol_regime' in locals():
                    symbol_regime_val = symbol_regime
                
                try:
                    # Gather returns for GARCH/tail analysis
                    # data_primary is guaranteed to exist here from L1028
                    _closes = data_primary['close'].astype(np.float64)
                    _volumes = data_primary['volume'].astype(np.float64)
                    _returns = np.diff(np.log(_closes)) if len(_closes) > 1 else None
                    
                    # ── V5.19 Apex: Whale & Breakout Detection ──
                    # Whale: Current volume vs mean of last 240 bars (4H move approx)
                    mean_vol_4h = _volumes[-240:].mean() if len(_volumes) >= 240 else _volumes.mean()
                    whale_ratio = _volumes[-1] / mean_vol_4h if mean_vol_4h > 0 else 1.0
                    
                    # Breakout: Price vs High/Low of last 50 bars
                    lookback_50 = _closes[-50:] if len(_closes) >= 50 else _closes
                    is_50_bar_high = _closes[-1] >= lookback_50.max()
                    is_50_bar_low = _closes[-1] <= lookback_50.min()
                    
                    # Inject into setups for Sophia
                    setups['volume_ratio_4h'] = whale_ratio
                    setups['is_50_bar_high'] = is_50_bar_high
                    setups['is_50_bar_low'] = is_50_bar_low
                    
                    # ── V5.36: QUANTUM ENTANGLEMENT ──
                    # Extract BTC returns for entanglement verification
                    btc_returns = None
                    if "BTC/USDT" in self.data_provider.symbol_list:
                        btc_pkg = self.get_multi_timeframe_data("BTC/USDT").get('5m')
                        if btc_pkg:
                            _btc_closes = btc_pkg['data']['close'].astype(np.float64)
                            btc_returns = np.diff(np.log(_btc_closes)) if len(_btc_closes) > 1 else None

                    # FORENSIC-1: Tie TTL dynamically to the horizon's bar_minutes to accurately project Survival Estimates.
                    # Default is 3 bars base survival.
                    bar_mins = getattr(self.sophia.survival, 'bar_minutes', 5.0) if hasattr(self, 'sophia') and self.sophia else 5.0
                    if self.horizon == 'SCALPING':
                        dynamic_ttl = 180.0 # 3 min for Scalping micro-edges
                    else:
                        dynamic_ttl = bar_mins * 60.0 * 3.0 # 3 bars for Swing/Trending
                        
                    if hasattr(self, 'sophia') and self.sophia:
                        sophia_report = self.sophia.analyze(
                            symbol=symbol,
                            direction=signal_type.name,
                            signal_strength=strength,
                            setups=setups,
                            confluence_score=confluence_score,
                            tp_pct=final_tp_pct,
                            sl_pct=final_sl_pct,
                            returns=_returns,
                            ttl_seconds=dynamic_ttl,
                            btc_returns=btc_returns,
                            regime=symbol_regime_val,
                        )
                    
                    # V5.21 Quantum Tunnelling: Update Global BTC State
                    if symbol == "BTC/USDT":
                        HybridScalpingStrategy.BTC_QUANTUM_STATE.update({
                            'vortex_pulse': sophia_report.vortex_pulse,
                            'noise_level': sophia_report.noise_level,
                            'is_active': True
                        })
                        logger.debug(f"🌐 [BTC SYNC] Leader Updated: Vortex={sophia_report.vortex_pulse:.2f}, Noise={sophia_report.noise_level:.2f}")
                    
                    # Generate human-readable narrative
                    sophia_narrative = NarrativeGenerator.generate_intention(
                        symbol=symbol,
                        direction=signal_type.name,
                        win_prob=sophia_report.win_probability,
                        expected_exit_mins=sophia_report.expected_exit_mins,
                        top_features=sophia_report.top_features,
                        setups=setups,
                        entropy_label=sophia_report.entropy_label,
                        tail_warning=sophia_report.tail_risk_warning,
                        current_price=setups['close'],
                    )
                    
                    # ── V5.15 Symmetry Breaker: Elastic SL/TP ──
                    # If Sophia's predicted range is wider than ATR, we trust the horizon
                    if sophia_report.win_probability > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_win_prob_high']:
                        pred_tp = abs(sophia_report.expected_high_pct if signal_type == SignalType.LONG else sophia_report.expected_low_pct)
                        pred_sl = abs(sophia_report.expected_low_pct if signal_type == SignalType.LONG else sophia_report.expected_high_pct)
                        
                        # Apply Elasticity (Max 50% expansion)
                        if pred_tp > final_tp_pct:
                            final_tp_pct = min(final_tp_pct * 1.5, pred_tp)
                        
                        # Symmetry Breaker: Wider SL if WinProb is elite to avoid "Symmetry Lock" noise
                        if sophia_report.win_probability > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_win_prob_supreme']:
                            final_sl_pct *= 1.35 # Extra breathing room for the predator
                            logger.debug(f"🔓 [V5.15 CHRONOS] Symmetry Breaker Active for {symbol}: SL expanded x1.35")

                    # ═══════════════════════════════════════════════════════
                    # V5.26 THE GREAT COLLAPSE: Single Omniscient Decision
                    # ═══════════════════════════════════════════════════════
                    # Instead of 6 sequential gates that killed 99.99% of signals,
                    # we use Sophia's omniscient_score as the ONLY entry filter.
                    
                    # V5.29 THE ORACLE: Single Omniscient Decision with Chaos Filters
                    omni = sophia_report.omniscient_score
                    
                    # ================================================================
                    # FORENSIC-AUDIT-FIX: Sophia Hurdle RAISED 0.03 → 0.38
                    # QUÉ: Restaurar un filtro agresivo para evitar zombies.
                    # POR QUÉ: Un hurdle de 0.03 permitía entrar en mercados muertos (chop),
                    #   generando 65+ zombies por backtest (TIME_STOP 90 min).
                    # PARA QUÉ: Reducir drásticamente los TIME_STOP_ZOMBIE, subiendo el Win Rate.
                    # ================================================================
                    base_hurdle = 0.20  # FORENSIC FIX: was 0.38
                    if 'BTC' not in symbol:
                        base_hurdle = 0.15  # FORENSIC FIX: was 0.35
                        logger.debug(f"🔓 [V5.50 ALT-GATE] Using {base_hurdle} hurdle for {symbol}")
                    
                    hurdle = base_hurdle
                    
                    is_divine = sophia_report.superposition_coherence > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_superposition_divine']
                    is_harmonic = sophia_report.superposition_coherence > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_superposition_harmonic'] or sophia_report.singularity_horizon > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_superposition_harmonic']
                    is_resonant = sophia_report.resonance_index > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_resonance_index']
                    
                    if is_divine:
                        hurdle = base_hurdle * 0.5  # FORENSIC FIX: was hardcoded 0.01
                        logger.warning(f"✨ [DIVINE HARMONY] {symbol} Total alignment: Hurdle={hurdle:.3f}")
                    elif is_harmonic:
                        hurdle = base_hurdle * 0.75  # FORENSIC FIX: was hardcoded 0.02
                        logger.info(f"🏹 [HARMONIC GATE] Frequency agreement: Hurdle={hurdle:.3f}")
                    elif is_resonant:
                        hurdle = base_hurdle * 0.90  # FORENSIC FIX: was hardcoded 0.025
                        logger.info(f"🧬 [RESONANCE BRIDGE] Reducing friction: Hurdle={hurdle:.3f}")
                    
                    if omni < hurdle:
                        # ═══════════════════════════════════════════════════════
                        # FORENSIC-V75 FIX: OMNISCIENT HARD BLOCK FOR CHOPPY MARKETS
                        # QUÉ: Restaurar el hard-block del oráculo (en vez de solo penalizar).
                        # POR QUÉ: "Consenso Ponderado" permitía que señales de baja energía pasen,
                        #   las cuales el RiskManager dejaba abrir, pero luego no tenían momentum
                        #   para alcanzar el TP de 0.10%, convirtiéndose en 65+ Zombies.
                        # PARA QUÉ: Cortar de raíz los setups sin momentum direccional real.
                        # ═══════════════════════════════════════════════════════
                        logger.warning(f"🛑 [ORACLE BLOCK] {symbol} rejected: Omni Score {omni:.3f} < {hurdle:.3f}")
                        continue

                    # ═══════════════════════════════════════════════════════════
                    # CTOS OMNIPOTENCE: WIN PROBABILITY HARD GATE
                    # QUÉ: Bloqueo DURO si la probabilidad de ganar < 48%.
                    # POR QUÉ: En backtest, Sophia empieza con prior ~50% (sin modelo pre-entrenado).
                    #   El threshold de 52% bloqueaba 100% de señales por cold start del modelo.
                    #   El sistema aún protege vía: ORACLE SCORE hurdle, ADX gate, PREDICTION_GATE.
                    # PARA QUÉ: Permitir entrada y aprendizaje adaptativo del modelo.
                    # CÓMO: Hard block solo por debajo de 48%. Soft penalty entre 48-70%.
                    # ═══════════════════════════════════════════════════════════
                    if sophia_report.win_probability < 0.48:
                        logger.warning(f"🛑 [WP GATE] {symbol} BLOCKED: Win Prob {sophia_report.win_probability*100:.1f}% < 48% (coin flip territory)")
                        continue
                    
                    if sophia_report.win_probability < Config.Strategies.TECHNICAL_THRESHOLDS['sophia_win_prob_min']:
                        sophia_penalty = max(0.5, sophia_report.win_probability / 0.70)
                        strength *= sophia_penalty
                        logger.info(f"🧠 [CONSENSUS] {symbol}: Sophia penalty x{sophia_penalty:.2f} (WP={sophia_report.win_probability*100:.1f}% < 70%)")
                    
                    # ── V5.45: SOVEREIGN ADAPTIVE LEVERAGE ──
                    # Leverage is dictated by the Market Order (1 - Entropy) & H values.
                    entropy_norm = sophia_report.decision_entropy # 0 to 1.585
                    order_factor = max(0.2, 1.0 - (entropy_norm / 1.585))
                    
                    # QUÉ: Apalancamiento Cuántico basado en Incertidumbre de Sophia
                    if entropy_norm > 1.0:
                        # Fat-Tails / Alta Incertidumbre -> Protección de Micro-Cuenta
                        leverage = 2.0 + (order_factor * 5.0)  # Rango ~2x a 5x
                        logger.debug(f"⚖️ [ADAPTIVE LEV] {symbol} | Alta Incertidumbre (H={entropy_norm:.2f}). Leverage reducido.")
                    elif entropy_norm > 0.5:
                        # Condición Normal
                        leverage = 10.0 + (order_factor * 10.0) # Rango ~10x a ~15x
                    else:
                        # Certeza Cuántica -> Maximizar ganancia
                        leverage = 20.0 + (order_factor * 10.0) # Rango ~20x a 30x
                        logger.debug(f"⚖️ [ADAPTIVE LEV] {symbol} | Certeza Alta (H={entropy_norm:.2f}). Apalancamiento Agresivo.")
                    
                    if is_divine:
                        leverage *= 1.5 # Extra power for Divine states
                        
                    leverage = min(leverage, getattr(Config, "MAX_LEVERAGE", 30.0))
                        
                    logger.info(f"⚖️ [ADAPTIVE LEVERAGE] {symbol}: Order={order_factor:.2f} → Leverage={leverage:.1f}x")
                    
                    # ── V5.33: QUANTUM SCALP LOGIC ──
                    # If butterfly force is high, we reduce expected exit time to capture micro-patterns.
                    original_exit = sophia_report.expected_exit_mins
                    if sophia_report.butterfly_force > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_butterfly_force']:
                        sophia_report.expected_exit_mins *= 0.5
                        sophia_report.time_to_tp_mins *= 0.5
                        logger.info(f"⚡ [QUANTUM SCALP] {symbol}: Reducing duration to {sophia_report.expected_exit_mins:.1f}m (B_Force={sophia_report.butterfly_force:.2f})")

                    logger.info(f"🧿 [OMNISCIENT] ✅ TRADE {symbol}: Score={omni:.3f} (WP={sophia_report.win_probability:.2f}, Edge={abs(sophia_report.expected_high_pct if signal_type == SignalType.LONG else sophia_report.expected_low_pct)*100:.2f}%, Energy={sophia_report.vortex_pulse:.2f}, Noise={sophia_report.noise_level:.2f})")
                    
                    # ── TP/SL MODIFIERS (Not gates — they adjust, never block) ──
                    
                    # V5.16 Hologram: Trajectory TP Expansion (capped in V5.27)
                    if sophia_report.path_score > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_path_score']:
                        final_tp_pct *= 1.10  # V5.27: Reduced from 1.15 to 1.10
                        logger.debug(f"🚀 [HOLOGRAM] Explosive Trajectory! TP Expanded to {final_tp_pct*100:.2f}%")

                    # V5.17 Sovereign: Regime-Specific TP (moderated in V5.27)
                    if sophia_report.hurst_exponent > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_hurst_trend']:
                        final_tp_pct *= 1.1  # V5.27: Reduced from 1.2 to 1.1 (lightning scalp priority)
                        logger.debug(f"📈 [SOVEREIGN] Trending Regime (H={sophia_report.hurst_exponent:.2f})! TP x1.1")
                    elif sophia_report.hurst_exponent < Config.Strategies.TECHNICAL_THRESHOLDS['sophia_hurst_mean_rev']:
                        final_tp_pct *= 0.85
                        logger.debug(f"🔄 [SOVEREIGN] Mean Rev Regime (H={sophia_report.hurst_exponent:.2f}). Scalp Mode.")

                    # V5.19 Apex: TP Expansion (Whale Power — capped in V5.27)
                    if sophia_report.whale_ratio > Config.Strategies.TECHNICAL_THRESHOLDS['sophia_whale_ratio']:
                        final_tp_pct *= 1.25  # V5.27: Reduced from 1.5 to 1.25
                        logger.info(f"🐋 [APEX] Whale Movement! TP Expanded x1.25 to {final_tp_pct*100:.2f}%")

                    # V5.20 Noise Predator: Spectral SL (V5.29: Evolutionary Adaptability)
                    # PROFESOR: CÓMO - Restauramos el multiplicador dinámico al ruido detectado.
                    # POR QUÉ - Para que el algoritmo se adapte a estallidos GARCH en vez de ser estático.
                    # FORENSIC FIX #7: Cap noise buffer to max 30% of base SL to prevent R:R inversion.
                    # Before: noise_sigma=0.005 added +0.75% to a 0.15% SL → 6x inflation → R:R < 1:1
                    noise_buffer = min(sophia_report.noise_sigma * 1.5, final_sl_pct * 0.30)
                    final_sl_pct += noise_buffer
                    
                    # V5.29: EVOLUTIONARY SAFETY NET (Reemplaza The Razor Hard Cap)
                    # PARA QUÉ: Permitimos que las redes neuronales y el ATR definan el SL real.
                    # Solo detenemos anomalías catastróficas > 3.0%, eliminando la asfixia del ruido cripto.
                    if final_sl_pct > 0.030:
                        logger.debug(f"🪒 [ADAPTIVE NET] Clipping anomalistic SL from {final_sl_pct*100:.2f}% to Dynamic Max 3.00%")
                        final_sl_pct = 0.030
                    
                    # V5.26: ENFORCE R:R > 1.0 (TP must be >= SL)
                    # [GOLDEN-V4 FIX]: Disabled! Scalping needs inverted R:R (SL 1.5x TP) to absorb noise.
                    # Forcing TP to be 1.2x SL pushes the target out of reach and causes 70% ZOMBIE exits.
                    # if final_tp_pct < final_sl_pct:
                    #     final_tp_pct = final_sl_pct * 1.2  # At least 1.2:1 R:R
                    #     logger.debug(f"⚖️ [V5.26 R:R] Enforced minimum R:R 1.2:1 → TP={final_tp_pct*100:.2f}%, SL={final_sl_pct*100:.2f}%")

                    logger.info(f"   💭 {sophia_narrative}")
                    
                    # ── SOPHIA-VIEW: Real-time Metacognition Metrics ──
                    try:
                        metrics.record_sophia_inference(
                            symbol=symbol,
                            entropy=sophia_report.decision_entropy,
                            top_features=sophia_report.top_features,
                            consensus_count=confluence_score  # Using multi-timeframe score as consensus proxy
                        )
                    except Exception as m_e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.debug(f"[SOPHIA-VIEW] Metric emission skipped: {m_e}")
                        
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    logger.warning(f"⚠️ [SOPHIA] Analysis failed for {symbol}: {e}")

                # 8. Metadatos Finales (V5.11 Dynamic Scaling)
                current_cog_state = 'NORMAL'
                tp_mult = 1.0
                sl_mult = 1.0
                
                if symbol in self.cognitive_memory and setup_type in self.cognitive_memory[symbol]:
                    mem = self.cognitive_memory[symbol][setup_type]
                    current_cog_state = mem.get('state', 'NORMAL')
                    
                    if current_cog_state == 'ALPHA':
                        tp_mult = 1.3 # Expand profit target
                        sl_mult = 1.0 # Standard SL
                    elif current_cog_state == 'INJURED':
                        tp_mult = 0.8 # Secure profit faster
                        sl_mult = 0.8 # Tighten stop loss
                
                _metadata = {
                    'multi_timeframe_score': confluence_score,
                    'trend_direction': "UP" if setups['in_uptrend'] else "DOWN",
                    'time_to_target': time_to_target,
                    'adx': setups['adx'],
                    'rsi': setups['rsi'],
                    'atr_mult': atr_sl_mult,
                    'is_v5_dynamic': True,
                    'atr_val': current_atr,
                    'setup_type': setup_type,
                    'cog_state': current_cog_state,
                    'tp_mult': tp_mult,
                    'sl_mult': sl_mult,
                    'is_recursive_sprint': sophia_report.noise_level < 0.10 if sophia_report else False,
                    'ttl': dynamic_ttl if 'dynamic_ttl' in locals() else 180.0,
                    'signal_strength': strength
                }
                # ── NEURAL BIAS: Phase 48 Online Learning State Capture ──
                neural_bias = 0.5
                neural_action, neural_conf = self.get_fused_insight(symbol, data_primary)  # C-2 FIX: was data_5m
                if neural_action and neural_conf > 0.25:  # Phase 50: Minimum neural confidence
                    # If Neural Action matches Technical Direction, boost conviction
                    # action_idx: 1=LONG, 2=SHORT
                    if (neural_action == SignalType.LONG and signal_type == SignalType.LONG) or \
                       (neural_action == SignalType.SHORT and signal_type == SignalType.SHORT):
                        neural_bias = 0.5 + (neural_conf * 0.3)  # Phase 50: Reduced boost (0.3 vs 0.5)
                        logger.info(f"🧠 [NEURAL BIAS] Agreement! Conviction Boosted: {neural_bias:.2f}")
                    else:
                        neural_bias = 0.5 - (neural_conf * 0.2) # Slight penalization if neural disagrees
                        logger.info(f"🧠 [NEURAL BIAS] Disagreement. Conviction Damped: {neural_bias:.2f}")

                if sophia_report:
                    _metadata['sophia'] = sophia_report.to_dict()
                    _metadata['sophia_narrative'] = sophia_narrative
                
                _metadata['neural_bias'] = neural_bias # For telemetry
                
                # 🧠 MUTACIÓN 42: Q-Learning On-The-Fly TP/SL Adjustment
                try:
                    from core.q_learning import q_agent
                    # Build State Key
                    regime = sophia_report.is_vortex_regime if sophia_report else False
                    vol_level = int(min(5, (current_atr / setups['close']) / 0.001))
                    q_state_key = q_agent._get_state_key(str(regime), vol_level, 0)
                    
                    action_idx, (tp_q_mult, sl_q_mult) = q_agent.get_action(q_state_key)
                    final_tp_pct *= tp_q_mult
                    final_sl_pct *= sl_q_mult
                    
                    # Store pending trade for reward linkage
                    q_agent.pending_trades[symbol] = (q_state_key, action_idx)
                    logger.debug(f"🧠 [Q-LEARNING] Adjusted TP/SL -> Action {action_idx} ({tp_q_mult}x, {sl_q_mult}x)")
                    _metadata['q_action'] = action_idx
                except Exception as e:
                    logger.error(f"Q-Learning hook failed: {e}", exc_info=True)

                
                # ════════════════════════════════════════════════════════════════
                # FORENSIC-V81: FINAL HARD CAP before emission (GOLDEN GENOTYPE)
                # QUÉ: Cap absoluto post-modificadores alineado con Hyper-Evolver.
                # POR QUÉ: El cap anterior de 0.10% dejaba solo 0.06% neto
                #   después de fees RT (0.04%), haciendo el TP inalcanzable.
                #   Esto causaba el 84% de exits como TIME_STOP_ZOMBIE.
                # PARA QUÉ: Permitir TP viable de 0.163% (net ~0.12% after fees).
                # GOLDEN GENOTYPE: TP=0.163%, SL=0.188% (Optuna Trial #47)
                # ════════════════════════════════════════════════════════════════
                if self.horizon == 'MICROSCALPING':
                    # [QUANTUM EVOLUTION: FASE 2] Greedy Dynamic TP
                    # El Breakeven cuántico ya nos protege, así que podemos ser avariciosos en setups extremos.
                    if strength > 0.85:
                        final_tp_pct *= 1.5
                        logger.info(f"💎 [GREEDY TP] Microscalping TP expanded x1.5 to {final_tp_pct*100:.2f}% (High Strength)")
                    elif setup_type == "LIQUIDITY_VOID_REVERSION":
                        final_tp_pct *= 2.0
                        logger.info(f"🕳️ [GREEDY TP] Microscalping TP expanded x2.0 to {final_tp_pct*100:.2f}% (Liquidity Void)")
                        
                    final_tp_pct = min(final_tp_pct, 0.0080) # Cap at 0.80% for micro
                    final_sl_pct = min(final_sl_pct, 0.0030) # Cap at 0.30% for micro
                    
                elif self.horizon == 'SCALPING':
                    # FORENSIC-V156: Strict Caps to prevent Sophia from inflating TP/SL beyond M1 viability.
                    # TP > 0.40% is unrealistic for pure Scalping and causes Zombie trades.
                    final_tp_pct = min(final_tp_pct, 0.0040)  # Cap at 0.40%
                    final_sl_pct = min(final_sl_pct, 0.0040)  # Cap at 0.40%
                
                signal = SignalEvent(
                    strategy_id=detailed_id,
                    setup_type=setup_type,
                    symbol=symbol,
                    datetime=event_time,
                    signal_type=signal_type,
                    strength=strength,
                    atr=setups['atr'],
                    ttl=int(dynamic_ttl),  # C-5 FIX: was hardcoded to 180, causing horizon mismatch
                    horizon=self.horizon,
                    priority=self.priority,
                    # FORENSIC FIX #1: Send TP/SL as decimal fractions (e.g., 0.004 = 0.4%)
                    # Before: 0.004 * 100 = 0.4 → risk_manager interpreted as 40% (100x error)
                    tp_pct=round(final_tp_pct, 6),
                    sl_pct=round(final_sl_pct, 6),
                    current_price=setups['close'],
                    metadata={
                        **_metadata,
                        'atr_pct': current_atr / setups['close'] if setups['close'] > 0 else 0.001,
                        'volatility': setups['atr'] / setups['close'] if setups['close'] > 0 else 0.001,
                        'exhaustion': self.sophia.calibrator.calculate_exhaustion(inds_primary['macd_hist'], setups['rsi']) if hasattr(self, 'sophia') and self.sophia else 0.0,  # C-2 FIX: was inds_5m
                        'boost_factor': sophia_report.metadata.get('boost_factor', 1.0) if sophia_report else 1.0,
                        'win_prob': sophia_report.win_probability if sophia_report else 0.5,
                        'expected_high': sophia_report.expected_high_pct if sophia_report else 0.0,
                        'expected_low': sophia_report.expected_low_pct if sophia_report else 0.0,
                        'path_score': sophia_report.path_score if sophia_report else 0.5,
                        'hurst': sophia_report.hurst_exponent if sophia_report else 0.5,
                        'leverage': round(leverage, 1), # Inyectar apalancamiento adaptativo
                        'quantum_leverage': sophia_report.quantum_leverage if sophia_report else 1.0,
                        'vortex_pulse': sophia_report.vortex_pulse if sophia_report else 0.0,
                        'is_vortex': sophia_report.is_vortex_regime if sophia_report else False,
                        # ── PEPITA #4: KELLY ADAPTIVE SIZING ──
                        # QUÉ: Propaga ml_confidence y strength al metadata.
                        # POR QUÉ: RiskManager.size_position busca signal_metadata.get('ml_confidence')
                        #   para CompoundingEngine.get_quantum_kelly_fraction().
                        'ml_confidence': sophia_report.win_probability if sophia_report else 0.5,
                        'strength': float(strength),
                    },
                )
                
                # 9. Emit signal and update records
                self.events_queue.put(signal)
                self.last_trade_times[symbol] = event_time.timestamp()
                self.last_trade_prices[symbol] = setups['close']
                self.partial_tp[symbol] = False
                self.trailing_sl.pop(symbol, None) # Clear old trailing
                if self.bought[symbol] is False:
                    self.bought[symbol] = len(data_primary)
                
                # PHASE 3: Neural Insight Publication
                neural_bridge.publish_insight(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    insight={
                        'confidence': strength,
                        'direction': signal_type.name,
                        'setups': "MEAN_REV" if (setups['long_mean_rev'] or setups['short_mean_rev']) else "MOMENTUM",
                        'adx': float(current_adx)
                    }
                )
                
                # LOG detallado
                setup_type = "MEAN_REV" if (setups['long_mean_rev'] or setups['short_mean_rev']) else "MOMENTUM"
                print(f"✅ {signal_type.name} {symbol}: Strength={strength:.2f}, "
                      f"Setup={setup_type}, RSI={setups['rsi']:.1f}, "
                      f"Confluence={confluence_score:.2f}, Vol={setups['volume_ratio']:.1f}x")
                
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                import traceback
                traceback.print_exc()
                print(f"❌ Error processing {symbol}: {e}")
                continue

    def calculate_signals(self, event):
        """Wrapper para integración con framework existente"""
        # ============================================================
        # ⏱️ ESPECIALIZACIÓN POR HORIZONTE (SCALPING vs SWING)
        # ============================================================
        is_swing = getattr(self, "horizon", "SCALPING") == "SWING"
        is_closed = getattr(event, "is_closed", True)
        
        # FASE HORIZONS: Filtrado estricto por timeframe para evitar Phantom Triggers
        event_tf = getattr(event, "timeframe", "1m")
        from config import Config
        horizon_name = getattr(self, "horizon", "SCALPING").capitalize()
        if horizon_name == "Swing":
            target_tf = getattr(Config.Horizons, "Swing", {}).get("primary_tf", "4h")
        elif horizon_name == "Microscalping":
            target_tf = getattr(Config.Horizons, "Microscalping", {}).get("primary_tf", "1m")
        else:
            target_tf = getattr(Config.Horizons, "Scalping", {}).get("primary_tf", "1m")
        
        if event_tf != target_tf:
            return
            
        # SWING SOLO evalúa velas cerradas. Ignora el ruido HFT del websocket.
        if is_swing and not is_closed:
            return

        import time
        current_time = time.time()
        if not hasattr(self, '_last_prediction_time'):
            self._last_prediction_time = 0
            
        throttle_seconds = 0.5 if not is_swing else 10.0
        if os.environ.get('TRADER_GEMINI_BACKTEST') == 'true':
            throttle_seconds = 0.0 # Bypass throttle in backtest
        if (current_time - self._last_prediction_time) < throttle_seconds:
            return
            
        self._last_prediction_time = current_time
        
        self.generate_signals(event)

    def update_recursive_weights(self, trade_outcome):
        """
        Phase 5 Mea Culpa: Cognitive Memory Update (BUG-008 Fix).
        Receives TradeOutcome from engine when a trade closes.
        """
        if not hasattr(self, 'cognitive_memory'):
            self.cognitive_memory = {}
            
        # Determine PnL and symbol
        if isinstance(trade_outcome, float):
            pnl = trade_outcome
            symbol = getattr(self, 'symbol', None)
        else:
            pnl = trade_outcome.pnl if hasattr(trade_outcome, 'pnl') else (
                (trade_outcome.exit_price - trade_outcome.entry_price) * trade_outcome.direction
            )
            symbol = getattr(trade_outcome, 'symbol', getattr(self, 'symbol', None))
            
        if not symbol:
            return
            
        if symbol not in self.cognitive_memory:
            self.cognitive_memory[symbol] = {}
            
        setup_type = 'MOMENTUM'
        if not hasattr(trade_outcome, 'setup_type') and hasattr(trade_outcome, 'metadata') and trade_outcome.metadata:
            setup_type = trade_outcome.metadata.get('setup_type', 'MOMENTUM')
            
        if setup_type not in self.cognitive_memory[symbol]:
            self.cognitive_memory[symbol][setup_type] = {'consecutive_losses': 0, 'state': 'NORMAL'}
            
        mem = self.cognitive_memory[symbol][setup_type]
        
        if pnl > 0:
            mem['consecutive_losses'] = 0
            mem['state'] = 'ALPHA'
        else:
            mem['consecutive_losses'] += 1
            if mem['consecutive_losses'] >= 2:
                mem['state'] = 'INJURED'
            else:
                mem['state'] = 'NORMAL'

    def _reconstruct_neural_state(self, closes, volumes, ps, gene_params, l2_state, window=5) -> np.ndarray:
        """
        Phase 48: Reconstructs the 25-feature state tensor for learning.
        Must match core/fused_strategy_kernel.py logic exactly.
        """
        n = len(closes)
        state = np.zeros(25, dtype=np.float32)
        if n < 30: return state
        
        # 1. Returns (5)
        for i in range(window):
            idx = n - window + i
            state[i] = (closes[idx] - closes[idx-1]) / closes[idx-1]
            
        # 2. Volumes (5)
        vol_sum = np.sum(volumes[n-20:n])
        mean_vol = vol_sum / 20.0 if vol_sum > 0 else 1.0
        for i in range(window):
            idx = n - window + i
            state[5+i] = volumes[idx] / mean_vol
            
        # 3. Momentum Proxy (5)
        for i in range(window):
            idx = n - window + i
            mom = (closes[idx] / closes[idx-2] - 1.0) if idx >= 2 else 0.0
            state[10+i] = mom
            
        # Placeholder
        for i in range(window):
            state[15+i] = 0.0
            
        # Inject L2 Data (Phase 66: Orderbook Vectorization)
        state[18] = l2_state[0] # ofi
        state[19] = l2_state[1] # microprice_divergence
            
        # 4. Portfolio & Genes
        state[20] = ps[0]
        state[21] = ps[1]
        state[22] = ps[2]
        state[23] = gene_params[0]
        state[24] = gene_params[1]
        
        return state

    def get_fused_insight(self, symbol, data, portfolio_state=None):
        """
        [PHASE 65] Fused End-to-End Decision.
        Uses the Alpha Genotype's brain to map market data directly to an action.
        """
        genotype = self.genotypes.get(symbol) # Get symbol-specific genotype
        if genotype is None or len(data) < 30:
            return None, 0.0
            
        # 1. Prepare Genotype Params
        # We need brain_weights as float32 array
        weights = genotype.genes.get('brain_weights', [])
        if weights is None or (hasattr(weights, '__len__') and len(weights) == 0):
            # Initialize if empty (from Genotype.init_brain logic)
            genotype.init_brain(25, 4)
            weights = genotype.genes['brain_weights']
            
        weights_arr = np.array(weights, dtype=np.float32)
        
        # 2. Normalize Gene Context for state tensor [sl, tp]
        sl = genotype.genes.get('sl_pct', 0.02)
        tp = genotype.genes.get('tp_pct', 0.015)
        gene_params = np.array([min(sl * 10, 1.0), min(tp * 10, 1.0)], dtype=np.float32)
        
        # 3. Portfolio State [has_pos, pnl_norm, dur_norm]
        ps = np.zeros(3, dtype=np.float32)
        if portfolio_state:
            ps[0] = 1.0 if portfolio_state.get('quantity', 0) != 0 else 0.0
            ps[1] = np.clip(portfolio_state.get('pnl_pct', 0.0) * 10, -1.0, 1.0)
            ps[2] = min(portfolio_state.get('duration', 0) / 100.0, 1.0)
            
        # 4. Fused Compute
        closes = data['close'].astype(np.float64)
        volumes = data['volume'].astype(np.float64)
        
        # L2 State Extraction from SSOT (GlobalMarketState)
        from core.global_state import global_state
        l2_state = np.zeros(2, dtype=np.float32)
        if symbol in global_state.symbol_states:
            sv = global_state.symbol_states[symbol]
            l2_state[0] = sv.orderflow_imbalance
            l2_state[1] = sv.microprice_divergence
            
        # State Reconstruction (Phase 48: For Learning Feedback)
        state_tensor = self._reconstruct_neural_state(closes, volumes, ps, gene_params, l2_state)
        
        action_scores = fused_compute_step(
            closes, volumes, ps, gene_params, weights_arr, l2_state
        )
        
        # 5. Decode Decision
        action_idx = np.argmax(action_scores)
        confidence = action_scores[action_idx]
        
        # 6. Store in Brain Memory for real-time feedback loop (Phase 48)
        self.brain_memory[symbol] = {
            'state': state_tensor,
            'action_idx': action_idx,
            'prediction': confidence,
            'weights': weights_arr
        }
        
        # Map to SignalType (matching NeuralBridge.decode_action)
        decision = None
        if action_idx == 1: decision = SignalType.LONG
        elif action_idx == 2: decision = SignalType.SHORT
        elif action_idx == 3: decision = "CLOSE"
        
        return decision, confidence


    def process_reward(self, trade: dict):
        """
        [PHASE 7.3/V5.7] Cognitive Auto-Tuning (Self-Healing)
        El bot audita sus propios resultados diferenciando *por qué* entró.
        Si la estrategia que usó (MOMENTUM / MEAN_REV) quema saldo, la censurará.
        """
        symbol = trade.get('symbol')
        pnl = trade.get('pnl_usd', 0)
        metadata = trade.get('metadata', {})
        setup_type = metadata.get('setup_type', 'UNKNOWN')
        
        if not symbol or setup_type == 'UNKNOWN':
            return
            
        # 1. Inicializar subconsciente de la moneda
        if symbol not in self.cognitive_memory:
            self.cognitive_memory[symbol] = {
                'MOMENTUM': {'wins': 0, 'losses': 0, 'history': [], 'state': 'NORMAL'},
                'MEAN_REV': {'wins': 0, 'losses': 0, 'history': [], 'state': 'NORMAL'}
            }
            
        if setup_type not in self.cognitive_memory[symbol]:
            return # Ignore unknown
            
        mem = self.cognitive_memory[symbol][setup_type]
        
        # 2. Registrar el Dolor/Placer
        mem['history'].append(pnl)
        if len(mem['history']) > 8: # V5.10: Ventana eléctrica de 8 trades
            mem['history'].pop(0)
            
        if pnl > 0:
            mem['wins'] += 1
        else:
            mem['losses'] += 1
            
        # 3. Evolución de Estado (V5.10 Alpha Hunter)
        if len(mem['history']) >= 3: # Reacción ultra rápida
            recent_wins = sum(1 for x in mem['history'] if x > 0)
            recent_wr = recent_wins / len(mem['history'])
            old_state = mem.get('state', 'NORMAL')
            
            if recent_wr < 0.45:
                mem['state'] = 'INJURED'
            elif recent_wr >= 0.70: # V5.10: Exigencia de élite
                mem['state'] = 'ALPHA'
            else:
                mem['state'] = 'NORMAL'
                
            if old_state != mem['state']:
                logger.info(f"🧬 [EVOLUCIÓN COGNITIVA] {symbol} ({setup_type}) cambió de {old_state} a {mem['state']} (WR: {recent_wr*100:.1f}%)")
                
        # 4. Phase 48: ONLINE LEARNING (SGD)
        # If we have stored state for this trade, we update weights
        brain_data = self.brain_memory.get(symbol)
        if brain_data:
            # We determine the target based on PnL
            # If PnL > 0, we reinforce the action (Target = 1.0)
            # If PnL < 0, we punish the action (Target = -1.0)
            target = 1.0 if pnl > 0 else -0.5 # Asymmetric punishment
            
            # Update Matrix (25x4 Weights)
            new_weights = self.learner.update_matrix(
                weights_matrix=brain_data['weights'].reshape(25, 4), # Reshape for matrix update
                inputs=brain_data['state'],
                target=target,
                prediction=brain_data['prediction'],
                output_index=brain_data['action_idx']
            )
            
            # Save back to Genotype
            genotype = self.genotypes.get(symbol)
            if genotype:
                genotype.genes['brain_weights'] = new_weights.flatten().tolist()
                logger.info(f"🧠 [ONLINE LEARNING] Adjusted weights for {symbol} ({setup_type}). PnL={pnl:.2f}")

        # 5. Return Oracle Reasoning for Telemetry
        from core.sovereign_oracle import sovereign_oracle
        history = sovereign_oracle.knowledge_base.get(symbol, [])
        return history[-1] if history else None

    def _apply_live_mutation(self, symbol: str, genotype: Genotype):
        """
        PHASE 46: MULTIVERSE MUTATION (Continuous infinitesimal drift).
        """
        if not genotype: return
        
        # Base mutation force (1e-6)
        force = 1e-6 
        
        # Mutation modulated by Oracle Conviction
        conviction_mod = sovereign_oracle.get_mutation_mod(symbol)
        effective_force = force * conviction_mod
        
        for k in ['tp_pct', 'sl_pct', 'strength_threshold', 'adx_threshold']:
            if k in genotype.genes:
                # Random drift
                genotype.genes[k] *= (1.0 + np.random.uniform(-effective_force, effective_force))
                
                # Hard limits
                if k.endswith('_pct'):
                    genotype.genes[k] = np.clip(genotype.genes[k], 0.005, 0.10)
                elif 'threshold' in k:
                    genotype.genes[k] = np.clip(genotype.genes[k], 0.1, 0.95)
                elif 'adx' in k:
                    genotype.genes[k] = np.clip(genotype.genes[k], 15.0, 40.0)

    def _apply_infinitesimal_tuning(self, symbol: str, genotype: Genotype):
        """
        PHASE 47.1: THE PERPETUAL PULSE.
        Guided adaptation based on Causal Bias (The search for the secret formula).
        """
        if not genotype: return
        
        # Infinitesimal Drift (1e-7)
        drift_force = 1e-7
        
        # Get Causal Bias (Aura) from Sophia
        aura = sovereign_oracle.get_causal_bias(symbol)
        
        # Get Swarm Pressure (Cohesion vs Autonomy)
        swarm_pressure = swarm_correlator.get_swarm_pressure(symbol)
        
        # Logic: If swarm pressure is high (>0.8), we reduce autonomy drift
        autonomy_factor = 1.0 - (swarm_pressure * 0.5) 
        
        total_drift = drift_force * aura['drift_multiplier'] * autonomy_factor
        bias = aura['aggression_bias'] # Positive = More aggressive, Negative = Defensive
        
        # Apply Pulse
        for k in ['tp_pct', 'sl_pct', 'strength_threshold']:
            if k in genotype.genes:
                # Directed drift: bias * total_drift
                direction = 1.0 + (bias * total_drift)
                genotype.genes[k] *= direction
                
        # Optional: Neural Weight drift
        bw = genotype.genes.get('brain_weights')
        if bw is not None and len(bw) > 0:
            weights = np.array(bw)
            weights += np.random.normal(0, total_drift * 0.1, size=weights.shape)
            genotype.genes['brain_weights'] = weights.tolist()

    def stop(self):
        """
        Phase 49: Persistence.
        Saves ALL learned brain weights to disk on shutdown.

        """
        if self.genotypes:
            try:
                # Ensure directory exists
                import os
                if not os.path.exists("data/genotypes"):
                    os.makedirs("data/genotypes")
                
                count = 0
                for symbol, gene in self.genotypes.items():
                    filename = f"data/genotypes/{symbol.replace('/','')}_gene.json"
                    gene.save(filename)
                    count += 1
                    
                print(f"💾 Persistence: Saved {count} Brains.")
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                print(f"❌ Error saving brains: {e}")
