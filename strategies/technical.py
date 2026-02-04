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

class HybridScalpingStrategy:
    """
    Estrategia híbrida que combina:
    - Velocidad y simplicidad del scalping
    - Análisis multi-timeframe del código original  
    - Filtros de tendencia robustos
    - TP/SL definidos para scalping
    """
    
    def __init__(self, data_provider, events_queue):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.strategy_id = "HYBRID_SCALPING"
        
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
        
        # Mejora del ORIGINAL: Multi-timeframe
        self.MULTI_TIMEFRAME_WEIGHTS = {
            '5m': 0.4,   # Peso principal (timeframe de trading)
            '15m': 0.3,  # Confirmación
            '1h': 0.3    # Dirección general
        }
        
        # Estado (del ORIGINAL)
        self.bought = {}
        self.last_processed_times = {}
        self.last_trade_times = {} # FOR COOLDOWNS (Rule 4.1)

    def calculate_indicators(self, df, timeframe='5m'):
        """Calcular indicadores para un timeframe específico - COMBINADO"""
        # Bollinger Bands (del SCALPING)
        df['bb_middle'] = df['close'].rolling(self.BB_PERIOD).mean()
        df['bb_std'] = df['close'].rolling(self.BB_PERIOD).std()
        df['bb_upper'] = df['bb_middle'] + (self.BB_STD * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.BB_STD * df['bb_std'])
        
        # RSI (común a ambos)
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.RSI_PERIOD)
        
        # MACD (del SCALPING)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.MACD_FAST,
            slowperiod=self.MACD_SLOW, 
            signalperiod=self.MACD_SIGNAL
        )
        
        # MEJORA del ORIGINAL: EMAs para tendencia
        df['ema_fast'] = talib.EMA(df['close'], timeperiod=self.EMA_FAST)
        df['ema_slow'] = talib.EMA(df['close'], timeperiod=self.EMA_SLOW)
        df['in_uptrend'] = df['ema_fast'] > df['ema_slow']
        
        # Volume (común)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR para volatilidad (común)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ADX por solicitud del usuario (Tendencia > 25)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        return df

    def get_multi_timeframe_data(self, symbol):
        """MEJORA del ORIGINAL: Análisis multi-timeframe simplificado"""
        timeframe_data = {}
        
        # Timeframe principal (5m)
        try:
            bars_5m = self.data_provider.get_latest_bars(symbol, n=50)
            if len(bars_5m) >= 30:
                df_5m = self._bars_to_dataframe(bars_5m)
                df_5m = self.calculate_indicators(df_5m, '5m')
                timeframe_data['5m'] = df_5m
        except:
            pass
        
        # Timeframe 15m (confirmación)
        try:
            bars_15m = self.data_provider.get_latest_bars_15m(symbol, n=30)
            if len(bars_15m) >= 20:
                df_15m = self._bars_to_dataframe(bars_15m)
                df_15m = self.calculate_indicators(df_15m, '15m')
                timeframe_data['15m'] = df_15m
        except:
            pass
        
        # Timeframe 1h (tendencia general)
        try:
            bars_1h = self.data_provider.get_latest_bars_1h(symbol, n=50)
            if len(bars_1h) >= 30:
                df_1h = self._bars_to_dataframe(bars_1h)
                df_1h = self.calculate_indicators(df_1h, '1h')
                timeframe_data['1h'] = df_1h
        except:
            pass
        
        return timeframe_data

    def _bars_to_dataframe(self, bars):
        """Convertir bars a DataFrame - Útil para ambos enfoques"""
        data = []
        for b in bars:
            data.append({
                'datetime': b['datetime'],
                'open': b['open'],
                'high': b['high'], 
                'low': b['low'],
                'close': b['close'],
                'volume': b['volume']
            })
        return pd.DataFrame(data)

    def calculate_multi_timeframe_confluence(self, timeframe_data):
        """MEJORA del ORIGINAL: Confluence simplificado"""
        confluence_score = 0.0
        total_weight = 0.0
        
        for tf, weight in self.MULTI_TIMEFRAME_WEIGHTS.items():
            if tf in timeframe_data:
                df = timeframe_data[tf]
                if len(df) > 0:
                    last = df.iloc[-1]
                    
                    # Puntuación basada en tendencia y momentum
                    tf_score = 0.0
                    
                    # Bonus por alineación con tendencia
                    if last['in_uptrend']:
                        tf_score += 0.3
                    
                    # Bonus por RSI saludable (40-60)
                    if 40 <= last['rsi'] <= 60:
                        tf_score += 0.2
                    # Bonus extra por RSI en extremos con tendencia
                    elif last['in_uptrend'] and last['rsi'] < 40:
                        tf_score += 0.3  # Pullback en uptrend
                    elif not last['in_uptrend'] and last['rsi'] > 60:
                        tf_score += 0.3  # Rally en downtrend
                    
                    # Bonus por volumen
                    if last['volume_ratio'] > 1.5:
                        tf_score += 0.2
                    
                    confluence_score += tf_score * weight
                    total_weight += weight
        
        # Normalizar a 0-1
        if total_weight > 0:
            confluence_score /= total_weight
        
        return min(confluence_score, 1.0)

    def detect_scalping_setup(self, df_5m):
        """Detección de setups de scalping - del SCALPING mejorado"""
        if len(df_5m) < 2:
            return None
        
        last = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        
        # MEJORA: Combinar mean reversion y momentum
        setups = {
            'long_mean_rev': False,
            'short_mean_rev': False, 
            'long_momentum': False,
            'short_momentum': False,
            'rsi': last['rsi'],
            'volume_ratio': last['volume_ratio'],
            'in_uptrend': last['in_uptrend'],
            'bb_position': (last['close'] - last['bb_lower']) / (last['bb_upper'] - last['bb_lower']) if (last['bb_upper'] - last['bb_lower']) > 0 else 0.5
        }
        
        # 1. MEAN REVERSION (del SCALPING)
        price_at_lower = last['close'] <= last['bb_lower']
        price_at_upper = last['close'] >= last['bb_upper']
        rsi_oversold = last['rsi'] < self.RSI_OVERSOLD
        rsi_overbought = last['rsi'] > self.RSI_OVERBOUGHT
        high_volume = last['volume_ratio'] > 1.5
        
        # MEJORA: Requerir alineación con tendencia para mean reversion
        setups['long_mean_rev'] = price_at_lower and rsi_oversold and high_volume and last['in_uptrend']
        setups['short_mean_rev'] = price_at_upper and rsi_overbought and high_volume and not last['in_uptrend']
        
        # 2. MOMENTUM (del SCALPING + filtro tendencia)
        macd_bullish = last['macd'] > last['macd_signal'] and last['macd_hist'] > 0
        macd_bearish = last['macd'] < last['macd_signal'] and last['macd_hist'] < 0
        volume_increasing = last['volume'] > prev['volume']
        
        # MEJORA: Momentum solo en dirección de tendencia
        rsi_momentum_zone = 40 < last['rsi'] < 60
        
        setups['long_momentum'] = macd_bullish and rsi_momentum_zone and volume_increasing and last['in_uptrend']
        setups['short_momentum'] = macd_bearish and rsi_momentum_zone and volume_increasing and not last['in_uptrend']
        
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
            strength += 0.4  # Momentum tiene menor convicción en scalping
        
        # MEJORA del ORIGINAL: Multi-timeframe confluence
        strength += confluence_score * 0.3
        
        # MEJORA del ORIGINAL: Volume boost
        if setups['volume_ratio'] > 2.0:
            strength += 0.1
        elif setups['volume_ratio'] > 1.5:
            strength += 0.05
        
        # Penalty por alta volatilidad (del SCALPING)
        if volatility > 0.025:
            strength *= 0.7
        elif volatility > 0.015:
            strength *= 0.9
        
        return min(strength, 1.0)

    def generate_signals(self):
        """Generación de señales HÍBRIDA"""
        symbols = self.data_provider.symbol_list
        
        for symbol in symbols:
            try:
                # MEJORA del ORIGINAL: Deduplicación
                current_time = datetime.now(timezone.utc)
                dedupe_key = f"{symbol}-{current_time.minute//5}"  # Agrupar por bloques de 5min
                
                if self.last_processed_times.get(dedupe_key):
                    continue
                self.last_processed_times[dedupe_key] = True
                
                # --- XRP SPECIFIC COOLDOWN (Rule 4.1) ---
                if 'XRP' in symbol:
                    last_trade = self.last_trade_times.get(symbol, 0)
                    if (current_time.timestamp() - last_trade) < 3600: # 60 minutes
                        continue
                
                # 1. Obtener datos multi-timeframe (MEJORA del ORIGINAL)
                timeframe_data = self.get_multi_timeframe_data(symbol)
                
                if '5m' not in timeframe_data:
                    continue
                
                df_5m = timeframe_data['5m']
                if len(df_5m) < 5:
                    continue
                
                # 2. Calcular confluence multi-timeframe (MEJORA del ORIGINAL)
                confluence_score = self.calculate_multi_timeframe_confluence(timeframe_data)
                
                # 3. Detectar setups en 5m (del SCALPING)
                setups = self.detect_scalping_setup(df_5m)
                if not setups:
                    continue
                
                # 4. Calcular volatilidad
                volatility = df_5m.iloc[-1]['atr'] / df_5m.iloc[-1]['close']
                
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
                    # Enforce alignment with 1h trend for XRP
                    if '1h' in timeframe_data:
                        last_1h = timeframe_data['1h'].iloc[-1]
                        # trend_1h logic from ml_strategy logic or similar:
                        # HybridScalpingStrategy also has in_uptrend for 1h
                        trend_1h = 1 if last_1h['in_uptrend'] else -1
                        if (signal_type == SignalType.LONG and trend_1h < 0) or \
                           (signal_type == SignalType.SHORT and trend_1h > 0):
                            continue
                
                # 6. Calcular fuerza (COMBINADO)
                strength = self.calculate_signal_strength(setups, confluence_score, volatility)
                
                # OPTIMIZACIÓN DE FRECUENCIA (Quality over Quantity)
                # 1. Filtro ADX > 25 (Evitar mercados planos)
                current_adx = df_5m.iloc[-1]['adx']
                if current_adx < 25:
                    # Permitir solo si RSI es extremo (Mean Reversion puro en rango)
                    # Si no es extremo y ADX < 25 -> Ignorar
                    is_rsi_extreme = setups['rsi'] < 25 or setups['rsi'] > 75
                    if not is_rsi_extreme:
                        continue
                
                # 2. Umbral aumentado (0.6 -> 0.7)
                if strength < 0.7:
                    continue
                
                # 7. Verificar si ya estamos en posición (del ORIGINAL)
                if symbol not in self.bought:
                    self.bought[symbol] = False
                
                # --- INTELLIGENT REVERSE DETECTION (Phase 5) ---
                if self.bought[symbol]:
                    existing_pos = self.data_provider.get_active_positions().get(symbol, {'quantity': 0})
                    current_qty = existing_pos['quantity']
                    
                    # Check for Strong Opposite Signal
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
                        # Proceed to create SIGNAL (REVERSE) below
                    else:
                        # Standard Exit Logic...
                        current_rsi = setups['rsi']
                        if (current_qty > 0 and current_rsi > 70) or \
                           (current_qty < 0 and current_rsi < 30):
                            # Señal de salida
                            exit_signal = SignalEvent(
                                strategy_id=self.strategy_id,
                                symbol=symbol,
                                datetime=current_time,
                                signal_type=SignalType.EXIT,
                                strength=1.0
                            )
                            self.events_queue.put(exit_signal)
                            self.bought[symbol] = False
                            print(f"🔄 EXIT {symbol}: RSI extremo ({current_rsi:.1f})")
                        continue
                
                # 8. Crear señal de entrada
                signal = SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    datetime=current_time,
                    signal_type=signal_type,
                    strength=strength,
                    atr=df_5m.iloc[-1]['atr']
                )
                
                # METADATOS para Risk Manager (del SCALPING)
                signal.tp_pct = self.TP_PCT * 100
                signal.sl_pct = self.SL_PCT * 100
                signal.current_price = df_5m.iloc[-1]['close']
                
                # MEJORA: Información multi-timeframe para Risk Manager
                signal.multi_timeframe_score = confluence_score
                signal.trend_direction = "UP" if setups['in_uptrend'] else "DOWN"
                
                # 9. Update last trade time and finalize
                self.last_trade_times[symbol] = current_time.timestamp()
                self.events_queue.put(signal)
                self.bought[symbol] = True
                
                # LOG detallado (COMBINADO)
                setup_type = "MEAN_REV" if (setups['long_mean_rev'] or setups['short_mean_rev']) else "MOMENTUM"
                print(f"✅ {signal_type.name} {symbol}: Strength={strength:.2f}, "
                      f"Setup={setup_type}, RSI={setups['rsi']:.1f}, "
                      f"Confluence={confluence_score:.2f}, Vol={setups['volume_ratio']:.1f}x")
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                continue

    def calculate_signals(self, event):
        """Wrapper para integración con framework existente"""
        self.generate_signals()
