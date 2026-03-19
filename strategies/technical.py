"""
Estrategia Técnica HÍBRIDA - Optimized for $12→$50 Scalping
Combina simplicidad del scalping con robustez del análisis técnico avanzado
"""

import numpy as np
import pandas as pd
import talib
from core.events import SignalEvent
from core.enums import SignalType
from datetime import datetime, timezone
from config import Config
from strategies.strategy import Strategy
from utils.math_kernel import (
    calculate_rsi_jit, calculate_bollinger_robust_jit, calculate_ema_jit,
    calculate_macd_jit, calculate_atr_jit, calculate_adx_jit
) # Phase 3: Total Vectorization
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
    
    def __init__(self, data_provider, events_queue, genotype: Genotype = None):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.strategy_id = "HYBRID_SCALPING"
        self.genotype = genotype
        self.symbol = genotype.symbol if genotype else None
        
        # Parámetros centralizados en Config.Strategies
        self.BB_PERIOD = getattr(Config.Strategies, 'TECH_BB_PERIOD', 20)
        self.BB_STD = getattr(Config.Strategies, 'TECH_BB_STD', 2.0)
        
        self.RSI_PERIOD = getattr(Config.Strategies, 'TECH_RSI_PERIOD', 14)
        self.RSI_OVERBOUGHT = getattr(Config.Strategies, 'TECH_RSI_SELL', 70)
        self.RSI_OVERSOLD = getattr(Config.Strategies, 'TECH_RSI_BUY', 30)
        
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # TP/SL centralizados
        self.TP_PCT = getattr(Config.Strategies, 'TECH_TP_PCT', 0.015)
        self.SL_PCT = getattr(Config.Strategies, 'TECH_SL_PCT', 0.02)
        
        # Filtro de tendencia centralizado
        self.EMA_FAST = getattr(Config.Strategies, 'TECH_EMA_FAST', 20)
        self.EMA_SLOW = getattr(Config.Strategies, 'TECH_EMA_SLOW', 50)
        self.EMA_TREND = 200 # Fixed Golden Filter for "Smart" logic
        
        # Mejora del ORIGINAL: Multi-timeframe
        self.MULTI_TIMEFRAME_WEIGHTS = {
            '5m': 0.4,   # Peso principal (timeframe de trading)
            '15m': 0.3,  # Confirmación
            '1h': 0.3    # Dirección general
        }
        
        # === PER-SYMBOL ADAPTIVE PROFILES (Phase 7.2) ===
        # === V3 UNIVERSALLY-PROFITABLE PROFILES (Multi-Horizon Optimized) ===
        self.PROFILES = {
            'AGGRESSIVE': {
                'tp_pct': 0.025, 'sl_pct': 0.008, 
                'adx_threshold': 22, 'strength_threshold': 0.82,
                'atr_sl_mult': 1.5, 'atr_tp_mult': 3.5, 'trailing_rsi': 70
            },
            'BALANCED': {
                'tp_pct': 0.030, 'sl_pct': 0.010,
                'adx_threshold': 22, 'strength_threshold': 0.85,
                'atr_sl_mult': 2.0, 'atr_tp_mult': 4.0, 'trailing_rsi': 65
            },
            'CONSERVATIVE': {
                'tp_pct': 0.040, 'sl_pct': 0.020,
                'adx_threshold': 22, 'strength_threshold': 0.85,
                'atr_sl_mult': 2.5, 'atr_tp_mult': 5.0, 'trailing_rsi': 60
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
            'BTC/USDT': {'allowed_setups': 'MOMENTUM_ONLY'},
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
        
        # SOPHIA-INTELLIGENCE Protocol: XAI Engine
        self.sophia = SophiaIntelligence(bar_minutes=5.0)
        
        # Pre-load provided genotype if any
        if genotype:
            self.genotypes[genotype.symbol] = genotype

    def get_symbol_params(self, symbol):
        """Devuelve parámetros adaptados al símbolo (Merged Genotype + Legacy Profile + Optimized)"""
        # 0. Get Legacy Defaults for this symbol
        profile_key = self.SYMBOL_MAP.get(symbol, 'BALANCED')
        defaults = self.PROFILES.get(profile_key).copy()
        
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
            except Exception:
                pass
        
        # 2. Case: Not found -> Auto-Spawn
        if not found_genes:
            new_gene = Genotype(symbol)
            new_gene.init_brain(25, 4)
            self.genotypes[symbol] = new_gene
            found_genes = new_gene.genes
            
        # 3. MAPPING & MERGING (Ensure no KeyErrors)
        # Genotype genes override defaults if present
        final_params = defaults
        for k, v in found_genes.items():
            if v is not None and (not hasattr(v, '__len__') or len(v) > 0): # Don't override with empty weights
                final_params[k] = v
                
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

    def calculate_indicators(self, data):
        """
        SUPREMO-V3: Zero-Pandas Indicator Calculation.
        Calculates all indicators using JIT-compiled functions on raw NumPy arrays.
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
            # 1. Bollinger Bands (Numba JIT RANSAC - Phase 10)
            inds['bb_upper'], inds['bb_middle'], inds['bb_lower'] = calculate_bollinger_robust_jit(closes, self.BB_PERIOD, self.BB_STD)
            
            # 2. RSI (Numba JIT)
            inds['rsi'] = calculate_rsi_jit(closes, self.RSI_PERIOD)
            
            # 3. MACD (Phase 3 JIT)
            inds['macd'], inds['macd_signal'], inds['macd_hist'] = calculate_macd_jit(closes, self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL)
            
            # 4. EMAs (Numba JIT)
            inds['ema_fast'] = calculate_ema_jit(closes, self.EMA_FAST)
            inds['ema_slow'] = calculate_ema_jit(closes, self.EMA_SLOW)
            inds['ema_trend'] = calculate_ema_jit(closes, self.EMA_TREND)
            
            # 5. Trend Flags (Boolean Arrays)
            inds['in_uptrend'] = (inds['ema_fast'] > inds['ema_slow']) & (closes > inds['ema_trend'])
            inds['in_downtrend'] = (inds['ema_fast'] < inds['ema_slow']) & (closes < inds['ema_trend'])
            
            # 6. Volume Metrics
            # Simple Volume MA (Vectorized with Convolve)
            period = 20
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
            inds['atr'] = calculate_atr_jit(highs, lows, closes, 14)
            inds['adx'] = calculate_adx_jit(highs, lows, closes, 14)

            return inds
        except Exception as e:
            # logger.error(f"Indicator Calc Error: {e}")
            return None

    def get_multi_timeframe_data(self, symbol):
        """SUPREMO-V3: Multi-timeframe analysis using structured arrays."""
        timeframe_data = {}
        
        # Phase 3 Expansion: Fetch 1d and 1w horizons as well
        for tf, n_bars in [('5m', 300), ('15m', 200), ('1h', 300), ('1d', 100), ('1w', 100)]:
            try:
                # get_latest_bars now returns structured array
                data = self.data_provider.get_latest_bars(symbol, n=n_bars, timeframe=tf)
                if data is not None and len(data) >= (30 if tf not in ('1w', '1d') else 10):
                    inds = self.calculate_indicators(data)
                    if inds:
                        timeframe_data[tf] = {'data': data, 'inds': inds}
            except Exception as e:
                pass
        
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
                    if 40 <= last_rsi <= 60:
                        tf_score += 0.2
                    elif inds['in_uptrend'][-1] and last_rsi < 40:
                        tf_score += 0.3  # Pullback en uptrend
                    elif inds['in_downtrend'][-1] and last_rsi > 60:
                        tf_score += 0.3  # Rally en downtrend (Corrected Logic)
                    
                    # Bonus por volumen (V5.45 Relaxed for Alts)
                    vol_thresh = 1.5 if is_btc else 1.1 # Reduced from 1.5 for Alts
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
            
            # DUAL PARADIGM ARCHITECTURE
            if setup_type == "MEAN_REV":
                # SCALPER PARADIGM: Fast hits, high win rate, tight leash.
                base_sl_mult = 1.4
                base_tp_mult = 1.6
            elif setup_type == "MOMENTUM":
                # TREND FOLLOWER PARADIGM: Give it room to breathe, target huge runners.
                base_sl_mult = 2.0
                base_tp_mult = 4.0
            else:
                # DEFAULT FALLBACK
                base_sl_mult = 1.5
                base_tp_mult = 2.0
            
            # Auto-adaptabilidad de Volatilidad (Sigue siendo dinámico / Evolutivo)
            # MEJORA 13: Regime-Aware Stop Loss
            regime_mult = 1.0
            if 'TRENDING' in regime:
                regime_mult = 1.25 # Stops más amplios
            elif 'CHOPPY' in regime:
                regime_mult = 0.75 # Stops más cerrados
                
            if vol_ratio > 1.2: # Volatilidad expandiéndose (Mechas largas)
                # El mercado está loco: Ampliamos red de pesca de profit, y alejamos stop loss del ruido
                atr_sl_mult = base_sl_mult * 1.2 * regime_mult
                atr_tp_mult = base_tp_mult * 1.5
            elif vol_ratio < 0.8: # Volatilidad muy baja (Laterales estrechos)
                # El mercado está muerto: TPs ultracortos, SL muy pegados
                atr_sl_mult = base_sl_mult * 0.8 * regime_mult
                atr_tp_mult = base_tp_mult * 0.8
            else:
                atr_sl_mult = base_sl_mult * regime_mult
                atr_tp_mult = base_tp_mult
                
            # Calculo crudo
            sl_pct = (current_atr * atr_sl_mult) / current_price
            tp_pct = (current_atr * atr_tp_mult) / current_price
            
            # Topes de cordura probabilística evolutivos
            sl_pct = np.clip(sl_pct, 0.008, 0.035) # Max 3.5% SL
            
            # En Scalping el TP máximo es más conservador que en Tendencia
            max_tp_cap = 0.03 if setup_type == "MEAN_REV" else 0.10
            tp_pct = np.clip(tp_pct, 0.015, max_tp_cap) 
            
            return atr_sl_mult, atr_tp_mult, sl_pct, tp_pct
        except Exception:
            # Safe Fallback
            return 1.5, 2.0, 0.01, 0.02
            
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
            return 20

    def detect_scalping_setup(self, pkg_5m, params=None, symbol=None):
        """SUPREMO-V3: Scalping setup detection with V5.7 Cognitive Interception."""
        data = pkg_5m['data']
        inds = pkg_5m['inds']
        
        if len(data) < 3: return None
        
        # Use -2 for Confirmed Closed Bar
        idx = -2
        
        # Phase 5.5: Dynamic Parametric Evolution (DPE)
        rsi_buy, rsi_sell = self._get_dynamic_rsi_levels(inds)
        adx_thresh = self._get_dynamic_adx_threshold(inds)
        
        last_close = data['close'][idx]
        last_rsi = inds['rsi'][idx]
        last_vol_ratio = inds['volume_ratio'][idx]
        
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
            'adx': inds['adx'][idx]
        }
        
        # BB Position Calculation
        bbu, bbl = inds['bb_upper'][idx], inds['bb_lower'][idx]
        if (bbu - bbl) > 0:
            setups['bb_position'] = (last_close - bbl) / (bbu - bbl)
        
        # 1. MEAN REVERSION (Flexibilizar si no hay tendencia clara)
        is_range = setups['adx'] < adx_thresh
        
        # DEFINICIÓN DE SETUPS (Optimizado para SUPREMO-V3)
        price_at_lower = last_close <= bbl
        price_at_upper = last_close >= bbu
        rsi_oversold = last_rsi < rsi_buy
        rsi_overbought = last_rsi > rsi_sell
        high_volume = last_vol_ratio > 0.7 # FREQUENTIST: Relaxed from 1.1 to 0.7
        
        setups['long_mean_rev'] = price_at_lower and rsi_oversold and high_volume and (setups['in_uptrend'] or is_range)
        setups['short_mean_rev'] = price_at_upper and rsi_overbought and high_volume and (setups['in_downtrend'] or is_range)
        
        # 2. MOMENTUM (Optimizado para Nivel Supremo-V3 con VCP & ADX)
        macd, macd_sig, macd_hist = inds['macd'][idx], inds['macd_signal'][idx], inds['macd_hist'][idx]
        macd_prev_hist = inds['macd_hist'][idx-1]
        
        # Detectar aceleración
        momentum_accel = abs(macd_hist) > abs(macd_prev_hist)
        
        # Filtro 1: ADX estricto para evitar mercados planos (Choppiness)
        adx_trend_confirmed = setups['adx'] > 20
        
        # Filtro 2: VCP (Volatility Contraction Pattern)
        # Requerimos expansión de las Bandas de Bollinger respecto a la vela anterior + Volumen Real
        prev_bbu, prev_bbl = inds['bb_upper'][idx-1], inds['bb_lower'][idx-1]
        prev_bbw = (prev_bbu - prev_bbl) / prev_bbl if prev_bbl > 0 else 0
        current_bbw = (bbu - bbl) / bbl if bbl > 0 else 0
        
        vcp_expansion = (current_bbw > prev_bbw)  # Las bandas se están abriendo
        volume_expansion = last_vol_ratio > 1.0   # Volumen > Media móvil
        vcp_confirmed = vcp_expansion and volume_expansion
        
        setups['long_momentum'] = (macd > macd_sig) and (macd_hist > 0) and momentum_accel and setups['in_uptrend'] and adx_trend_confirmed and vcp_confirmed
        # BUG-007 FIX: No shortar cuando RSI es extremo bajo (oversold) en bear market
        setups['short_momentum'] = (macd < macd_sig) and (macd_hist < 0) and momentum_accel and setups['in_downtrend'] and inds['rsi'][idx] > 35 and adx_trend_confirmed and vcp_confirmed
        
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
        """Cálculo de fuerza de señal COMBINADO con V5.8 Asymmetric Impact"""
        strength = 0.0
        
        # 0. Determinar Estado Cognitivo V5.8
        cog_state = 'NORMAL'
        if symbol and setup_type and hasattr(self, 'cognitive_memory') and symbol in self.cognitive_memory:
            mem = self.cognitive_memory[symbol].get(setup_type, {})
            cog_state = mem.get('state', 'NORMAL')
        
        # BASE SCORE por tipo de setup
        if setups['long_mean_rev'] or setups['short_mean_rev']:
            strength += 0.6  # Mean reversion tiene mayor convicción
            
            # Bonus por RSI extremo
            if setups['rsi'] < 25 or setups['rsi'] > 75:
                strength += 0.15
            
        elif setups['long_momentum'] or setups['short_momentum']:
            strength += 0.5  # Aumentado de 0.4 para facilitar disparos de calidad
        
        # MEJORA del ORIGINAL: Multi-timeframe confluence
        strength += confluence_score * 0.3
        
        # MEJORA del ORIGINAL: Volume boost (Alpha-Max Aggression)
        if setups['volume_ratio'] > 3.0:
            strength += 0.2
        elif setups['volume_ratio'] > 2.0:
            strength += 0.15
        elif setups['volume_ratio'] > 1.5:
            strength += 0.08
        
        # Penalty por volatilidad RELATIVA (Phase 47.5)
        # En BTC 1.5% es mucho, en SOL es normal. Usamos un umbral dinámico.
        vol_threshold = 0.015
        if 'BTC' not in (symbol or ''):
            vol_threshold = 0.025 # Umbral más alto para Alts
            
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
        
        for symbol in symbols:
            try:
                # 0. CONFIGURACIÓN DINÁMICA (Phase 7.2)
                params = self.get_symbol_params(symbol)
                ADX_THRESH = params['adx_threshold']
                STRENGTH_THRESH = params['strength_threshold']
                TP_PCT_LOCAL = params['tp_pct']
                SL_PCT_LOCAL = params['sl_pct']

                # MEJORA del ORIGINAL: Deduplicación
                # FIXED: Use event.timestamp instead of datetime.now for backtest parity
                event_time = event.timestamp if hasattr(event, 'timestamp') else datetime.now(timezone.utc)
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
                
                dedupe_key = f"{symbol}-{int(event_time.timestamp())}"  # Unique per bar (timestamp-based)
                
                if self.last_processed_times.get(dedupe_key):
                    continue
                self.last_processed_times[dedupe_key] = True
                
                # --- XRP SPECIFIC COOLDOWN (Rule 4.1) ---
                if 'XRP' in symbol:
                    last_trade = self.last_trade_times.get(symbol, 0)
                    if (event_time.timestamp() - last_trade) < 3600: # 60 minutes
                        continue
                
                # 1. Obtener datos multi-timeframe
                timeframe_data = self.get_multi_timeframe_data(symbol)
                
                if '5m' not in timeframe_data:
                    continue
                
                pkg_5m = timeframe_data['5m']
                data_5m = pkg_5m['data']
                inds_5m = pkg_5m['inds']
                
                if len(data_5m) < 5:
                    continue

                # Retrieve Brain for this symbol
                # This ensures we have a genotype (created by get_symbol_params if needed)
                # But get_symbol_params returns genes dict, we need the object for update.
                # We can access self.genotypes[symbol] directly or ensure it exists.
                self.get_symbol_params(symbol) # Ensure loaded/spawned
                current_genotype = self.genotypes.get(symbol)

                # --- PHASE 65: FUSED PATH (DIRECT SYMBOL BRAIN) ---
                if current_genotype and 'brain_weights' in current_genotype.genes:
                    try:
                        # 1. Obtain Portfolio State
                        real_pos = self.data_provider.get_active_positions().get(symbol, {'quantity': 0})
                        
                        # 2. Fused Insight (Indicators -> State -> Inference)
                        fused_decision, fused_confidence = self.get_fused_insight(
                            symbol, data_5m, portfolio_state=real_pos
                        )
                        
                        if fused_decision:
                            signal_type = fused_decision
                            strength = fused_confidence
                            
                            # Backfill 'setups' for logic compatibility downstream
                            # This ensures Step 8+ works without modification
                            setups = {
                                'close': data_5m['close'][-1],
                                'atr': inds_5m['atr'][-1],
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
                        print(f"❌ Fused Path Error {symbol}: {e}")
                        goto_step_6 = False
                else:
                    goto_step_6 = False

                if not goto_step_6:
                    # Legacy Sequential Path
                    # 2. Calcular confluence multi-timeframe
                    confluence_score = self.calculate_multi_timeframe_confluence(timeframe_data, symbol)
                    
                    if confluence_score < 0.3: # RELAXED from 0.5 for diagnostics
                        if 'BTC' not in symbol:
                            logger.debug(f"🛑 [DIAG] {symbol} Killed by low confluence: {confluence_score:.2f}")
                        continue
                        
                    # Pasa el símbolo para la validación cognitiva V5.7
                    setups = self.detect_scalping_setup(pkg_5m, params, symbol)
                    if not setups:
                        # logger.debug(f"DEBUG: {symbol} No active setup detected.")
                        continue
                    
                    # 4. Calcular volatilidad y ajustar confluencia (Phase 5.5 DPE)
                    volatility = setups['atr'] / setups['close']
                    
                    # Adaptar umbral de confluencia a la volatilidad reciente
                    # Si hay mucha volatilidad extrema, asumo spread más caro y reduzco ruido
                    base_strength = params.get('strength_threshold', 0.80) if params else 0.80
                    dynamic_strength = base_strength
                    
                    if volatility > 0.005:  # High recent volatility 
                        dynamic_strength += 0.05 # Requerir más confirmación (spread ancho, movimientos violentos)
                    elif volatility < 0.001: # Ultra Low Volatility
                        dynamic_strength -= 0.05 # Bajar exigencias porque no hay estallidos falsos
                        
                    params['strength_threshold'] = dynamic_strength
                    
                # 5. Determinar dirección y tipo de setup V5.8
                signal_type = None
                setup_type = "UNKNOWN"
                if setups['long_mean_rev'] or setups['short_mean_rev']:
                    signal_type = SignalType.LONG if setups['long_mean_rev'] else SignalType.SHORT
                    setup_type = "MEAN_REV"
                elif setups['long_momentum'] or setups['short_momentum']:
                    signal_type = SignalType.LONG if setups['long_momentum'] else SignalType.SHORT
                    setup_type = "MOMENTUM"
                
                if signal_type is None:
                    continue
                
                # ═══════════════════════════════════════════════════
                # PHASE 3: MULTI-HORIZON ORACLE VETO
                # ═══════════════════════════════════════════════════
                # QUÉ: Consultar al Oráculo si el contexto macro (1d, 1w) permite este trade.
                # POR QUÉ: El 47% de Stop Loss hits ocurrían por trades micro alineados contra la macro-tendencia.
                # CÓMO: Si 1D y 1W van en dirección opuesta al trade, se VETEA la operación.
                try:
                    direction_str = 'LONG' if signal_type == SignalType.LONG else 'SHORT'
                    oracle_verdict = MultiHorizonOracle.evaluate_clash_vector(timeframe_data, direction_str)
                    
                    if oracle_verdict['is_vetoed']:
                        logger.info(
                            f"🔮 [ORACLE VETO] {symbol} {direction_str} BLOCKED | "
                            f"Clash: {oracle_verdict['clash_score']:.1%} | "
                            f"Macro: {oracle_verdict['macro_context']}"
                        )
                        continue  # ← ABORTAR: El macro prohíbe este trade
                    
                    # Si no fue vetado pero hay choque parcial, reducir fuerza
                    if oracle_verdict['clash_score'] > 0.3:
                        clash_penalty = 1.0 - (oracle_verdict['clash_score'] * 0.4)
                        strength = strength * clash_penalty if 'strength' in dir() else 0.5
                        logger.debug(
                            f"🔮 [ORACLE WARN] {symbol} {direction_str} WEAKENED x{clash_penalty:.2f} | "
                            f"Macro: {oracle_verdict['macro_context']}"
                        )
                except Exception as e:
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

                # 6. Calcular fuerza (Pasando Símbolo y Setup_Type para Asimetría V5.8)
                strength = self.calculate_signal_strength(setups, confluence_score, volatility, symbol, setup_type)
                
                # OPTIMIZACIÓN DE FRECUENCIA
                # 1. Filtro ADX Dinámico
                current_adx = setups['adx']
                if current_adx < ADX_THRESH:
                    # Allow if RSI is extreme OR Strength is very high (Dynamic override)
                    is_rsi_extreme = setups['rsi'] < 15 or setups['rsi'] > 85
                    is_high_strength = strength > (STRENGTH_THRESH + 0.1) # Extra confidence
                    if not (is_rsi_extreme or is_high_strength):
                        continue
                
                # 2. Umbral de Fuerza Dinámico
                if strength < STRENGTH_THRESH:
                    if 'BTC' not in symbol:
                        logger.debug(f"🛑 [DIAG] {symbol} Strength {strength:.2f} < {STRENGTH_THRESH:.2f}")
                    continue
                
                # 7. Verificar si ya estamos en posición
                if symbol not in self.bought:
                    self.bought[symbol] = False
                           # 9. Gestión de Posición Existente (Exit/Trailing)
                if symbol in self.bought and self.bought[symbol]:
                    existing_pos = self.data_provider.get_active_positions().get(symbol, {'quantity': 0})
                    current_qty = existing_pos.get('quantity', 0)
                    
                    if current_qty != 0:
                        current_rsi = setups['rsi']
                        current_price = data_5m['close'][-1]
                        entry_price = self.last_trade_prices.get(symbol, current_price)
                        
                        # A. Calibración de PnL
                        is_long = current_qty > 0
                        cur_pnl = (current_price / entry_price - 1.0) if is_long else (entry_price / current_price - 1.0)
                        
                        # B. Proactive Break-Even Guard (V5.28 RAZOR-RELAXED)
                        # QUÉ: Mueve el SL a BE solo cuando el trade ha recorrido el 80% del camino al TP.
                        # POR QUÉ: Evita que el ruido de 0.3% asfixie un trade que tiene potencial de 1.5%+.
                        be_trigger = final_tp_pct * 0.8 if 'final_tp_pct' in dir() else 0.008
                        if cur_pnl > be_trigger and symbol not in self.trailing_sl:
                            new_sl = entry_price * 1.001 if is_long else entry_price * 0.999
                            self.trailing_sl[symbol] = new_sl
                            logger.info(f"🛡️ [V5.28 RAZOR-RELAX] BE Guard for {symbol} at {cur_pnl*100:.2f}% (Target: {be_trigger*100:.2f}%)")
                        
                        # Phase 6: Partial TP (50%) - Evolution Protocol
                        if not self.partial_tp.get(symbol, False):
                            # Usamos la mitad del target final como objetivo parcial
                            partial_target = final_tp_pct * 0.5 if 'final_tp_pct' in locals() else 0.005
                            if cur_pnl >= partial_target:
                                partial_signal = SignalEvent(
                                    strategy_id=self.strategy_id, symbol=symbol, datetime=event_time,
                                    signal_type=SignalType.EXIT, strength=0.5, current_price=current_price,
                                    metadata={'exit_reason': 'PARTIAL_TP_50'}
                                )
                                self.events_queue.put(partial_signal)
                                self.partial_tp[symbol] = True
                                logger.info(f"💰 [PARTIAL TP] {symbol} hit 50% target ({cur_pnl*100:.2f}%) - Closing half.")
                        
                        # C. Check Trailing SL Hit
                        if symbol in self.trailing_sl:
                            tsl = self.trailing_sl[symbol]
                            if (is_long and current_price <= tsl) or (not is_long and current_price >= tsl):
                                exit_signal = SignalEvent(
                                    strategy_id=self.strategy_id, symbol=symbol, datetime=event_time,
                                    signal_type=SignalType.EXIT, strength=1.0, current_price=current_price,
                                    metadata={'exit_reason': 'TRAILING_SL'}
                                )
                                self.events_queue.put(exit_signal)
                                self.bought[symbol] = False
                                self.trailing_sl.pop(symbol, None)
                                continue

                        # D. RSI Extreme Exit
                        if (is_long and current_rsi > 80) or (not is_long and current_rsi < 20):
                            exit_signal = SignalEvent(
                                strategy_id=self.strategy_id, symbol=symbol, datetime=event_time,
                                signal_type=SignalType.EXIT, strength=1.0, current_price=current_price,
                                metadata={'exit_reason': 'RSI_EXTREME'}
                            )
                            self.events_queue.put(exit_signal)
                            self.bought[symbol] = False
                            continue
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
                        
                        be_trigger = final_tp_pct * 0.8 if 'final_tp_pct' in dir() else 0.008
                        if cur_pnl > be_trigger and symbol not in self.trailing_sl:
                            new_sl = entry_price * 1.001 if current_qty > 0 else entry_price * 0.999
                            self.trailing_sl[symbol] = new_sl
                            logger.info(f"🛡️ [V5.28 RAZOR-RELAX] Proactive BE Guard Activated for {symbol} (PnL: {cur_pnl*100:.2f}%)")
                        
                        # 2. RSI-Based Trailing (Legacy check)
                        elif symbol not in self.trailing_sl:
                            trailing_rsi_thresh = params.get('trailing_rsi', 70)
                            should_trail = (current_qty > 0 and current_rsi > trailing_rsi_thresh) or \
                                           (current_qty < 0 and current_rsi < (100 - trailing_rsi_thresh))
                            
                            if should_trail:
                                new_sl = entry_price * 1.001 if current_qty > 0 else entry_price * 0.999
                                self.trailing_sl[symbol] = new_sl
                                logger.info(f"🛡️ [{symbol}] Trailing SL Activated by RSI at {new_sl:.6f}")
                        
                        # 2. Check Trailing SL Hit
                        if symbol in self.trailing_sl:
                            tsl = self.trailing_sl[symbol]
                            if (current_qty > 0 and current_price <= tsl) or \
                               (current_qty < 0 and current_price >= tsl):
                                exit_signal = SignalEvent(
                                    strategy_id=self.strategy_id,
                                    symbol=symbol,
                                    datetime=event_time,
                                    signal_type=SignalType.EXIT,
                                    strength=1.0
                                )
                                self.events_queue.put(exit_signal)
                                self.bought[symbol] = False
                                self.trailing_sl.pop(symbol, None)
                                logger.info(f"🛡️ [{symbol}] BREAK-EVEN/TRAILING EXIT at {current_price:.6f}")
                                continue

                        # 3. RSI Extreme Exit (Partial/Total)
                        if (current_qty > 0 and current_rsi > 80) or \
                           (current_qty < 0 and current_rsi < 20):
                            exit_signal = SignalEvent(
                                strategy_id=self.strategy_id,
                                symbol=symbol,
                                datetime=event_time,
                                signal_type=SignalType.EXIT,
                                strength=1.0
                            )
                            self.events_queue.put(exit_signal)
                            self.bought[symbol] = False
                            self.trailing_sl.pop(symbol, None)
                            print(f"🔄 EXIT {symbol}: RSI Extremo ({current_rsi:.1f})")
                        continue
                
                # PHASE 5: Time-to-Target (TTT) Analysis
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
                if hasattr(self, 'portfolio') and self.portfolio and hasattr(self.portfolio, 'global_regime_data'):
                    regime_meta = self.portfolio.global_regime_data
                    current_regime = regime_meta.get('sentiment', 'UNKNOWN')
                    # Phase 6: Specific symbol regime
                    symbol_regime = regime_meta.get('symbol_regimes', {}).get(symbol, current_regime)
                    
                atr_sl_mult, atr_tp_mult, final_sl_pct, final_tp_pct = self._calculate_dynamic_risk_params(
                    inds_5m, current_price, setup_type=setup_type, regime=current_regime
                )
                
                if current_atr > 0:
                    # Filtro de Volatilidad Mínima: Si el mercado no se mueve nada, no hay scalp.
                    if (current_atr / current_price) < 0.0010: # < 0.10% volatilidad relativa
                        logger.debug(f"💤 [V5.6] {symbol} Skipping: Low volatility.")
                        continue
                    
                    logger.debug(f"⚡ [V5.6 DPE] {symbol}: Volatility Auto-tuned -> SL={final_sl_pct*100:.2f}%, TP={final_tp_pct*100:.2f}%")
                else:
                    # Fallback crítico
                    final_sl_pct = 0.01
                    final_tp_pct = 0.02


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
                    
                    if signal_type == SignalType.LONG and delta < -100:
                         logger.warning(f"📉 [VETO CUÁNTICO] {symbol} LONG CANCELADO | Presión de Venta Agresiva Continua (Delta 60s: {delta:.2f})")
                         continue
                    elif signal_type == SignalType.SHORT and delta > 100:
                         logger.warning(f"📈 [VETO CUÁNTICO] {symbol} SHORT CANCELADO | Presión de Compra Agresiva Continua (Delta 60s: {delta:.2f})")
                         continue

                # ── SOPHIA-INTELLIGENCE: Pre-trade XAI Analysis ──
                sophia_report = None
                sophia_narrative = ""
                try:
                    # Gather returns for GARCH/tail analysis
                    _closes = data_5m['close'].astype(np.float64)
                    _volumes = data_5m['volume'].astype(np.float64)
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

                    sophia_report = self.sophia.analyze(
                        symbol=symbol,
                        direction=signal_type.name,
                        signal_strength=strength,
                        setups=setups,
                        confluence_score=confluence_score,
                        tp_pct=final_tp_pct,
                        sl_pct=final_sl_pct,
                        returns=_returns,
                        ttl_seconds=180.0,
                        btc_returns=btc_returns,
                        regime=symbol_regime if 'symbol_regime' in locals() else "UNKNOWN",
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
                    if sophia_report.win_probability > 0.85:
                        pred_tp = abs(sophia_report.expected_high_pct if signal_type == SignalType.LONG else sophia_report.expected_low_pct)
                        pred_sl = abs(sophia_report.expected_low_pct if signal_type == SignalType.LONG else sophia_report.expected_high_pct)
                        
                        # Apply Elasticity (Max 50% expansion)
                        if pred_tp > final_tp_pct:
                            final_tp_pct = min(final_tp_pct * 1.5, pred_tp)
                        
                        # Symmetry Breaker: Wider SL if WinProb is elite to avoid "Symmetry Lock" noise
                        if sophia_report.win_probability > 0.92:
                            final_sl_pct *= 1.35 # Extra breathing room for the predator
                            logger.debug(f"🔓 [V5.15 CHRONOS] Symmetry Breaker Active for {symbol}: SL expanded x1.35")

                    # ═══════════════════════════════════════════════════════
                    # V5.26 THE GREAT COLLAPSE: Single Omniscient Decision
                    # ═══════════════════════════════════════════════════════
                    # Instead of 6 sequential gates that killed 99.99% of signals,
                    # we use Sophia's omniscient_score as the ONLY entry filter.
                    
                    # V5.29 THE ORACLE: Single Omniscient Decision with Chaos Filters
                    omni = sophia_report.omniscient_score
                    
                    # Dynamic Threshold (V5.45) - THE HARMONY GATE:
                    # Normally 0.15 (BALANCED for Frequency & Precision).
                    # Phase 47.5: Sovereign Alt-Hurdle (Reduce by 50% for non-BTC symbols)
                    base_hurdle = 0.15
                    if 'BTC' not in symbol:
                        base_hurdle = 0.12 # Phase 50: Balanced Alt-Hurdle (was 0.08)
                        logger.debug(f"🔓 [V5.50 ALT-GATE] Using 0.12 hurdle for {symbol}")
                    
                    hurdle = base_hurdle
                    
                    is_divine = sophia_report.superposition_coherence > 0.85
                    is_harmonic = sophia_report.superposition_coherence > 0.7 or sophia_report.singularity_horizon > 0.7
                    is_resonant = sophia_report.resonance_index > 0.6
                    
                    if is_divine:
                        hurdle = 0.05
                        logger.warning(f"✨ [DIVINE HARMONY] {symbol} Total alignment: Hurdle=0.05")
                    elif is_harmonic:
                        hurdle = 0.10
                        logger.info(f"🏹 [HARMONIC GATE] Frequency agreement: Hurdle=0.10")
                    elif is_resonant:
                        hurdle = 0.12
                        logger.info(f"🧬 [RESONANCE BRIDGE] Reducing friction: Hurdle=0.12")
                    
                    if omni < hurdle:
                        status = "OPEN" if (is_divine or is_harmonic or is_resonant) else "CLOSED"
                        logger.info(f"🧿 [ORACLE] SKIP {symbol}: Score={omni:.3f} < {hurdle:.2f} (Gate={status})")
                        continue
                    
                    # ── V5.45: SOVEREIGN ADAPTIVE LEVERAGE ──
                    # Leverage is dictated by the Market Order (1 - Entropy).
                    # Higher order = More trust = Higher leverage.
                    entropy_norm = sophia_report.decision_entropy # 0 to 1.585
                    order_factor = max(0.2, 1.0 - (entropy_norm / 1.585))
                    
                    leverage = 10.0 + (order_factor * 20.0) # Adaptive from 10x to 30x
                    
                    if is_divine:
                        leverage *= 1.5 # Extra power for Divine states (up to 45x)
                        
                    logger.info(f"⚖️ [ADAPTIVE LEVERAGE] {symbol}: Order={order_factor:.2f} → Leverage={leverage:.1f}x")
                    
                    # ── V5.33: QUANTUM SCALP LOGIC ──
                    # If butterfly force is high, we reduce expected exit time to capture micro-patterns.
                    original_exit = sophia_report.expected_exit_mins
                    if sophia_report.butterfly_force > 1.5:
                        sophia_report.expected_exit_mins *= 0.5
                        sophia_report.time_to_tp_mins *= 0.5
                        logger.info(f"⚡ [QUANTUM SCALP] {symbol}: Reducing duration to {sophia_report.expected_exit_mins:.1f}m (B_Force={sophia_report.butterfly_force:.2f})")

                    logger.info(f"🧿 [OMNISCIENT] ✅ TRADE {symbol}: Score={omni:.3f} (WP={sophia_report.win_probability:.2f}, Edge={abs(sophia_report.expected_high_pct if signal_type == SignalType.LONG else sophia_report.expected_low_pct)*100:.2f}%, Energy={sophia_report.vortex_pulse:.2f}, Noise={sophia_report.noise_level:.2f})")
                    
                    # ── TP/SL MODIFIERS (Not gates — they adjust, never block) ──
                    
                    # V5.16 Hologram: Trajectory TP Expansion (capped in V5.27)
                    if sophia_report.path_score > 0.80:
                        final_tp_pct *= 1.10  # V5.27: Reduced from 1.15 to 1.10
                        logger.debug(f"🚀 [HOLOGRAM] Explosive Trajectory! TP Expanded to {final_tp_pct*100:.2f}%")

                    # V5.17 Sovereign: Regime-Specific TP (moderated in V5.27)
                    if sophia_report.hurst_exponent > 0.55:
                        final_tp_pct *= 1.1  # V5.27: Reduced from 1.2 to 1.1 (lightning scalp priority)
                        logger.debug(f"📈 [SOVEREIGN] Trending Regime (H={sophia_report.hurst_exponent:.2f})! TP x1.1")
                    elif sophia_report.hurst_exponent < 0.42:
                        final_tp_pct *= 0.85
                        logger.debug(f"🔄 [SOVEREIGN] Mean Rev Regime (H={sophia_report.hurst_exponent:.2f}). Scalp Mode.")

                    # V5.19 Apex: TP Expansion (Whale Power — capped in V5.27)
                    if sophia_report.whale_ratio > 5.0:
                        final_tp_pct *= 1.25  # V5.27: Reduced from 1.5 to 1.25
                        logger.info(f"🐋 [APEX] Whale Movement! TP Expanded x1.25 to {final_tp_pct*100:.2f}%")

                    # V5.20 Noise Predator: Spectral SL (V5.29: Evolutionary Adaptability)
                    # PROFESOR: CÓMO - Restauramos el multiplicador dinámico al ruido detectado.
                    # POR QUÉ - Para que el algoritmo se adapte a estallidos GARCH en vez de ser estático.
                    noise_buffer = sophia_report.noise_sigma * 1.5  
                    final_sl_pct += noise_buffer
                    
                    # V5.29: EVOLUTIONARY SAFETY NET (Reemplaza The Razor Hard Cap)
                    # PARA QUÉ: Permitimos que las redes neuronales y el ATR definan el SL real.
                    # Solo detenemos anomalías catastróficas > 3.0%, eliminando la asfixia del ruido cripto.
                    if final_sl_pct > 0.030:
                        logger.debug(f"🪒 [ADAPTIVE NET] Clipping anomalistic SL from {final_sl_pct*100:.2f}% to Dynamic Max 3.00%")
                        final_sl_pct = 0.030
                    
                    # V5.26: ENFORCE R:R > 1.0 (TP must be >= SL)
                    if final_tp_pct < final_sl_pct:
                        final_tp_pct = final_sl_pct * 1.2  # At least 1.2:1 R:R
                        logger.debug(f"⚖️ [V5.26 R:R] Enforced minimum R:R 1.2:1 → TP={final_tp_pct*100:.2f}%, SL={final_sl_pct*100:.2f}%")

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
                        logger.debug(f"[SOPHIA-VIEW] Metric emission skipped: {m_e}")
                        
                except Exception as e:
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
                    'is_recursive_sprint': sophia_report.noise_level < 0.10 # V5.21: Recursive expansion for pure signals
                }
                # ── NEURAL BIAS: Phase 48 Online Learning State Capture ──
                neural_bias = 0.5
                neural_action, neural_conf = self.get_fused_insight(symbol, data_5m)
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
                
                signal = SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    datetime=event_time,
                    signal_type=signal_type,
                    strength=strength,
                    atr=setups['atr'],
                    ttl=180,
                    tp_pct=round(final_tp_pct * 100, 4),
                    sl_pct=round(final_sl_pct * 100, 4),
                    current_price=setups['close'],
                    metadata={
                        **_metadata,
                        'exhaustion': self.sophia.calibrator.calculate_exhaustion(inds_5m['macd_hist'], setups['rsi']),
                        'boost_factor': sophia_report.metadata.get('boost_factor', 1.0) if sophia_report else 1.0,
                        'win_prob': sophia_report.win_probability if sophia_report else 0.5,
                        'expected_high': sophia_report.expected_high_pct if sophia_report else 0.0,
                        'expected_low': sophia_report.expected_low_pct if sophia_report else 0.0,
                        'path_score': sophia_report.path_score if sophia_report else 0.5,
                        'hurst': sophia_report.hurst_exponent if sophia_report else 0.5,
                        'quantum_leverage': sophia_report.quantum_leverage if sophia_report else 1.0,
                        'vortex_pulse': sophia_report.vortex_pulse if sophia_report else 0.0,
                        'is_vortex': sophia_report.is_vortex_regime if sophia_report else False
                    },
                )
                
                # 9. Emit signal and update records
                self.events_queue.put(signal)
                self.last_trade_times[symbol] = event_time.timestamp()
                self.last_trade_prices[symbol] = setups['close']
                self.partial_tp[symbol] = False
                self.trailing_sl.pop(symbol, None) # Clear old trailing
                self.bought[symbol] = True
                
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
                print(f"❌ Error processing {symbol}: {e}")
                continue

    def calculate_signals(self, event):
        """Wrapper para integración con framework existente"""
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
        closes = data['close'].astype(np.float32)
        volumes = data['volume'].astype(np.float32)
        
        # State Reconstruction (Phase 48: For Learning Feedback)
        state_tensor = self._reconstruct_neural_state(closes, volumes, ps, gene_params)
        
        action_scores = fused_compute_step(
            closes, volumes, ps, gene_params, weights_arr
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

    def _reconstruct_neural_state(self, closes, volumes, ps, gene_params, window=5) -> np.ndarray:
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
            
        # 3. Momentum (5)
        for i in range(window):
            idx = n - window + i
            state[10+i] = (closes[idx] / closes[idx-2] - 1.0) if idx >= 2 else 0.0
            
        # 4. Portfolio & Gene (5)
        state[20:23] = ps
        state[23:25] = gene_params
        return state

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
                print(f"❌ Error saving brains: {e}")
