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
from sophia.intelligence import SophiaIntelligence  # SOPHIA-INTELLIGENCE Protocol
from sophia.narrative import NarrativeGenerator  # SOPHIA: Human-readable narratives
from utils.metrics_exporter import metrics  # SOPHIA-VIEW: Real-time telemetry

class HybridScalpingStrategy(Strategy):
    """
    Estrategia híbrida que combina:
    - Velocidad y simplicidad del scalping
    - Análisis multi-timeframe del código original  
    - Filtros de tendencia robustos
    - TP/SL definidos para scalping
    """
    
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
        self.PROFILES = {
            'AGGRESSIVE': {
                'tp_pct': 0.050, 'sl_pct': 0.030, 
                'adx_threshold': 20, 'strength_threshold': 0.60,
                'atr_sl_mult': 1.5, 'trailing_rsi': 70
            },
            'BALANCED': {
                'tp_pct': 0.035, 'sl_pct': 0.020,
                'adx_threshold': 25, 'strength_threshold': 0.65,
                'atr_sl_mult': 2.0, 'trailing_rsi': 65
            },
            'CONSERVATIVE': {
                'tp_pct': 0.030, 'sl_pct': 0.015,
                'adx_threshold': 30, 'strength_threshold': 0.75,
                'atr_sl_mult': 2.5, 'trailing_rsi': 60
            }
        }
        
        # Trackers para Salidas Dinámicas (Phase 71)
        self.trailing_sl = {} # {symbol: current_sl_price}
        self.partial_tp = {} # {symbol: bool_hit}
        self.last_trade_prices = {} # {symbol: entry_price}
        
        # Map symbols to profiles (Expanded for 20 symbols)
        self.SYMBOL_MAP = {
            'BTC/USDT': 'CONSERVATIVE', 
            'BNB/USDT': 'CONSERVATIVE',
            'ETH/USDT': 'BALANCED', 
            'SOL/USDT': 'BALANCED',
            'ADA/USDT': 'BALANCED', 
            'XRP/USDT': 'BALANCED',
            'DOT/USDT': 'BALANCED',
            'LINK/USDT': 'BALANCED',
            'MATIC/USDT': 'BALANCED',
            'AVAX/USDT': 'BALANCED',
            'NEAR/USDT': 'AGGRESSIVE',
            'INJ/USDT': 'AGGRESSIVE',
            'PEPE/USDT': 'AGGRESSIVE',
            'RENDER/USDT': 'AGGRESSIVE',
            'SHIB/USDT': 'AGGRESSIVE',
            'DOGE/USDT': 'AGGRESSIVE',
            'ATOM/USDT': 'AGGRESSIVE',
            'LTC/USDT': 'AGGRESSIVE',
            'OP/USDT': 'AGGRESSIVE',
            'ARB/USDT': 'AGGRESSIVE'
        }

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
        """Devuelve parámetros adaptados al símbolo (Merged Genotype + Legacy Profile)"""
        # 0. Get Legacy Defaults for this symbol
        profile_key = self.SYMBOL_MAP.get(symbol, 'BALANCED')
        defaults = self.PROFILES.get(profile_key).copy()
        
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
            if v is not None and v != []: # Don't override with empty weights
                final_params[k] = v
                
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
                
                # Pad beginning with first value or NaN (using 0 here for safety)
                # v_ma needs to be same length as vols
                padding = np.zeros(period - 1)
                # Actually typically first (period-1) values are NaN or partials.
                # We will use 0.0 padding to match loop behavior implicitly.
                inds['volume_ma'] = np.concatenate((padding, v_ma_valid))
            else:
                inds['volume_ma'] = np.zeros_like(vols)
            inds['volume_ratio'] = np.where(inds['volume_ma'] > 0, vols / inds['volume_ma'], 1.0)
            
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
        
        for tf, n_bars in [('5m', 300), ('15m', 200), ('1h', 300)]:
            try:
                # get_latest_bars now returns structured array
                data = self.data_provider.get_latest_bars(symbol, n=n_bars, timeframe=tf)
                if data is not None and len(data) >= (30 if tf != '15m' else 20):
                    inds = self.calculate_indicators(data)
                    if inds:
                        timeframe_data[tf] = {'data': data, 'inds': inds}
            except Exception as e:
                pass
        
        return timeframe_data


    def calculate_multi_timeframe_confluence(self, timeframe_data):
        """SUPREMO-V3: Confluence using structured arrays."""
        confluence_score = 0.0
        total_weight = 0.0
        
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
                    
                    # Bonus por volumen
                    if inds['volume_ratio'][-1] > 1.5:
                        tf_score += 0.2
                    
                    confluence_score += tf_score * weight
                    total_weight += weight
        
        return min(confluence_score / total_weight if total_weight > 0 else 0.0, 1.0)
        # [SS-003 FIX] Dead code removed — was unreachable after return above

    def _get_dynamic_rsi_levels(self, data, inds):
        """SUPREMO-V3: Dynamic RSI Bands with NumPy."""
        try:
            last_close = data['close'][-1]
            last_atr = inds['atr'][-1]
            atr_pct = (last_atr / last_close) * 100
            
            # Base levels
            buy_level, sell_level = 30, 70
            
            if atr_pct > 0.3: # High volatility
                buy_level, sell_level = 20, 80
            elif atr_pct < 0.05: # Low volatility
                buy_level, sell_level = 35, 65
                
            return buy_level, sell_level
        except:
            return 30, 70

    def detect_scalping_setup(self, pkg_5m):
        """SUPREMO-V3: Scalping setup detection with optimized indexing."""
        data = pkg_5m['data']
        inds = pkg_5m['inds']
        
        if len(data) < 3: return None
        
        # Use -2 for Confirmed Closed Bar
        idx = -2
        
        # Phase 5: Dynamic RSI Levels
        rsi_buy, rsi_sell = self._get_dynamic_rsi_levels(data, inds)
        
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
        is_range = setups['adx'] < 20
        
        # DEFINICIÓN DE SETUPS (Optimizado para SUPREMO-V3)
        price_at_lower = last_close <= bbl
        price_at_upper = last_close >= bbu
        rsi_oversold = last_rsi < rsi_buy
        rsi_overbought = last_rsi > rsi_sell
        high_volume = last_vol_ratio > 1.1 # RELAXED from 1.5
        
        setups['long_mean_rev'] = price_at_lower and rsi_oversold and high_volume and (setups['in_uptrend'] or is_range)
        setups['short_mean_rev'] = price_at_upper and rsi_overbought and high_volume and (setups['in_downtrend'] or is_range)
        
        # 2. MOMENTUM (Optimizado para Nivel Supremo-V3)
        macd, macd_sig, macd_hist = inds['macd'][idx], inds['macd_signal'][idx], inds['macd_hist'][idx]
        macd_prev_hist = inds['macd_hist'][idx-1]
        
        # Detectar aceleración incluso si el volumen es estable
        momentum_accel = abs(macd_hist) > abs(macd_prev_hist)
        
        setups['long_momentum'] = (macd > macd_sig) and (macd_hist > 0) and momentum_accel and setups['in_uptrend']
        setups['short_momentum'] = (macd < macd_sig) and (macd_hist < 0) and momentum_accel and setups['in_downtrend']
        
        return setups

    def calculate_signal_strength(self, setups, confluence_score, volatility):
        """Cálculo de fuerza de señal COMBINADO"""
        strength = 0.0
        
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
        
        # Penalty por alta volatilidad (del SCALPING)
        if volatility > 0.025:
            strength *= 0.7
        elif volatility > 0.015:
            strength *= 0.9
        
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
                    confluence_score = self.calculate_multi_timeframe_confluence(timeframe_data)
                    
                    if confluence_score < 0.5: # Min threshold
                        continue
                    setups = self.detect_scalping_setup(pkg_5m)
                    if not setups:
                        continue
                    
                    # 4. Calcular volatilidad (usando setups calculados)
                    volatility = setups['atr'] / setups['close']
                    
                    # 5. Determinar dirección
                    signal_type = None
                    if setups['long_mean_rev'] or setups['long_momentum']:
                        signal_type = SignalType.LONG
                    elif setups['short_mean_rev'] or setups['short_momentum']:
                        signal_type = SignalType.SHORT
                    
                    if signal_type is None:
                        continue
                
                # --- XRP TREND ALIGNMENT (Rule 4.3) ---
                if 'XRP' in symbol:
                    if '1h' in timeframe_data:
                        inds_1h = timeframe_data['1h']['inds']
                        # Use -2 for 1h as well
                        idx_1h = -2 if len(inds_1h['in_uptrend']) > 1 else -1
                        trend_1h = 1 if inds_1h['in_uptrend'][idx_1h] else -1
                        if (signal_type == SignalType.LONG and trend_1h < 0) or \
                           (signal_type == SignalType.SHORT and trend_1h > 0):
                            continue
                
                # 6. Calcular fuerza
                strength = self.calculate_signal_strength(setups, confluence_score, volatility)
                
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
                    continue
                
                # 7. Verificar si ya estamos en posición
                if symbol not in self.bought:
                    self.bought[symbol] = False
                
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
                        
                        # 1. Trailing Activation
                        trailing_rsi_thresh = params.get('trailing_rsi', 70)
                        should_trail = (current_qty > 0 and current_rsi > trailing_rsi_thresh) or \
                                       (current_qty < 0 and current_rsi < (100 - trailing_rsi_thresh))
                        
                        if should_trail:
                            # Move SL to Entry + small buffer (Break-even protection)
                            entry_price = self.last_trade_prices.get(symbol, current_price)
                            buffer = entry_price * 0.001
                            new_sl = entry_price + buffer if current_qty > 0 else entry_price - buffer
                            
                            if symbol not in self.trailing_sl:
                                self.trailing_sl[symbol] = new_sl
                                logger.info(f"🛡️ [{symbol}] Trailing SL Activated at {new_sl:.6f}")
                        
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

                # ALPHA-MAX: ATR-Adaptive Stop Loss & Take Profit calculation
                atr_mult_sl = params.get('atr_sl_mult', 2.2)
                atr_mult_tp = 1.6 # Dynamic TP1 target (1.6x ATR)
                
                final_sl_pct = (setups['atr'] * atr_mult_sl) / setups['close']
                final_tp_pct = (setups['atr'] * atr_mult_tp) / setups['close']
                
                # Calibration for micro-scalping
                final_sl_pct = max(0.003, min(final_sl_pct, 0.015))
                final_tp_pct = max(0.005, min(final_tp_pct, 0.030))

                # ── SOPHIA-INTELLIGENCE: Pre-trade XAI Analysis ──
                sophia_report = None
                sophia_narrative = ""
                try:
                    # Gather returns for GARCH/tail analysis
                    _closes = data_5m['close'].astype(np.float64)
                    _returns = np.diff(np.log(_closes)) if len(_closes) > 1 else None
                    
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
                    )
                    
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

                # 8. Crear señal de entrada
                _metadata = {
                    'multi_timeframe_score': confluence_score,
                    'trend_direction': "UP" if setups['in_uptrend'] else "DOWN",
                    'time_to_target': time_to_target,
                    'adx': setups['adx'],
                    'rsi': setups['rsi'],
                    'atr_mult': atr_mult_sl,
                }
                if sophia_report:
                    _metadata['sophia'] = sophia_report.to_dict()
                    _metadata['sophia_narrative'] = sophia_narrative
                
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
                    metadata=_metadata,
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
                # print(f"✅ {signal_type.name} {symbol}: Strength={strength:.2f}, "
                #       f"Setup={setup_type}, RSI={setups['rsi']:.1f}, "
                #       f"Confluence={confluence_score:.2f}, Vol={setups['volume_ratio']:.1f}x")
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                continue

    def calculate_signals(self, event):
        """Wrapper para integración con framework existente"""
        self.generate_signals(event)

    def get_fused_insight(self, symbol, data, portfolio_state=None):
        """
        [PHASE 65] Fused End-to-End Decision.
        Uses the Alpha Genotype's brain to map market data directly to an action.
        """
        if self.genotype is None or len(data) < 30:
            return None, 0.0
            
        # 1. Prepare Genotype Params
        # We need brain_weights as float32 array
        weights = self.genotype.genes.get('brain_weights', [])
        if not weights:
            # Initialize if empty (from Genotype.init_brain logic)
            self.genotype.init_brain(25, 4)
            weights = self.genotype.genes['brain_weights']
            
        weights_arr = np.array(weights, dtype=np.float32)
        
        # 2. Normalize Gene Context for state tensor [sl, tp]
        sl = self.genotype.genes.get('sl_pct', 0.02)
        tp = self.genotype.genes.get('tp_pct', 0.015)
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
        
        action_scores = fused_compute_step(
            closes, volumes, ps, gene_params, weights_arr
        )
        
        # 5. Decode Decision
        action_idx = np.argmax(action_scores)
        confidence = action_scores[action_idx]
        
        # Map to SignalType (matching NeuralBridge.decode_action)
        decision = None
        if action_idx == 1: decision = SignalType.LONG
        elif action_idx == 2: decision = SignalType.SHORT
        elif action_idx == 3: decision = "CLOSE"
        
        return decision, confidence
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
