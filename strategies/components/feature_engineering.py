import pandas as pd
import numpy as np
import talib
from utils.logger import logger
from utils.debug_tracer import trace_execution
from utils.math_helpers import safe_div

class FeatureEngineering:
    """
    🏗️ COMPONENT: Feature Engineering
    Handles all technical indicator calculations and feature generation.
    Extracted from MLStrategy to improve modularity (Excelsior Phase I).
    """
    def __init__(self):
        pass

    @trace_execution
    def prepare_features(self, bars, market_regime="UNKNOWN", sentiment_loader=None, data_provider=None, symbol=None, feature_store=None):
        """
        Feature engineering completo con 80+ features adaptativos
        """
        if bars is None or len(bars) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(bars)
        
        # === [PHASE 12] FEATURE STORE LOOKUP ===
        if feature_store and len(df) > 100:
            try:
                start_ts = df['datetime'].min()
                end_ts = df['datetime'].max()
                cached_df = feature_store.get_features(symbol, start_ts, end_ts)
                if not cached_df.empty and len(cached_df) >= len(df) * 0.9:
                    full_df = pd.concat([df.set_index('datetime'), cached_df], axis=1)
                    return full_df.reset_index()
            except Exception as e:
                logger.warning(f"FeatureStore retrieval skipped: {e}")

        # Convert to numeric
        numeric_cols = ['close', 'open', 'high', 'low', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
        
        if len(df) < 50:
            return pd.DataFrame()

        # Numpy arrays for TA-Lib
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        open_ = df['open'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        
        # Batch dictionary for new features
        new_features = {}

        # ==================== PRICE ACTION ====================
        new_features['returns_1'] = df['close'].pct_change(1)
        new_features['returns_3'] = df['close'].pct_change(3)
        new_features['returns_5'] = df['close'].pct_change(5)
        new_features['returns_10'] = df['close'].pct_change(10)
        
        # High-Low ratios
        new_features['hl_range'] = (df['high'] - df['low']) / df['close']
        new_features['oc_range'] = abs(df['close'] - df['open']) / df['close']
        new_features['close_position'] = safe_div(df['close'] - df['low'], df['high'] - df['low'], 0.5)
        
        # Body to wick
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        new_features['body_to_wick'] = safe_div(abs(df['close'] - df['open']), upper_wick + lower_wick, 1.0)
        
        # ==================== MOMENTUM ====================
        for period in [3, 5, 8, 13, 21, 34]:
            new_features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        new_features['roc_5'] = df['close'].pct_change(5)
        new_features['roc_10'] = df['close'].pct_change(10)
        new_features['roc_20'] = df['close'].pct_change(20)
        
        # ==================== INDICADORES ====================
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
        
        # Bollinger
        upper, middle, lower_band = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        new_features['bb_upper'] = upper
        new_features['bb_middle'] = middle
        new_features['bb_lower'] = lower_band
        new_features['bb_position'] = safe_div(close - lower_band, upper - lower_band, 0.5)
        new_features['bb_width'] = safe_div(upper - lower_band, middle)
        
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
        
        # Volatility
        new_features['volatility_10'] = pd.Series(close).pct_change().rolling(10).std() * 100
        
        # Garman-Klass
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        new_features['gk_vol'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

        # --- MERGE BATCH ---
        features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, features_df], axis=1)

        # Post-merge complex calculations
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.5).astype(int)
        
        # Crossovers
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

        # Pattern Recognition
        df['up_bar'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['down_bar'] = (df['close'] < df['close'].shift(1)).astype(int)
        
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # ==================== REGIME AWARE FEATURES ====================
        df['trend_power'] = 0.0
        df['trend_alignment'] = 0.0
        df['range_extreme'] = 0.0
        df['mean_reversion_potential'] = 0.0
        df['volatility_regime'] = 1.0
        df['panic_index'] = 0.0

        if market_regime == "TRENDING":
            df['trend_power'] = df['adx'] * df['volume_ratio']
            df['trend_alignment'] = (
                np.where(df['ema_5_20_cross'] > 0, 1, -1) +
                np.where(df['ema_20_50_cross'] > 0, 1, -1) +
                np.where(df['ema_50_200_cross'] > 0, 1, -1)
            ) / 3
        elif market_regime == "RANGING":
            df['range_extreme'] = (df['rsi_14'] < 30).astype(int) - (df['rsi_14'] > 70).astype(int)
            df['mean_reversion_potential'] = abs(df['bb_position'] - 0.5) * 2
        elif market_regime == "VOLATILE":
            df['volatility_regime'] = df['atr_pct'].rolling(10).mean() / df['atr_pct'].rolling(50).mean()
            df['panic_index'] = df['volume_ratio'] * df['volatility_10']

        # ==================== SENTIMENT ====================
        if sentiment_loader:
            try:
                sentiment = sentiment_loader.get_sentiment(symbol)
                df['sentiment'] = float(sentiment) if sentiment is not None else 0.0
                df['sentiment_change'] = df['sentiment'].diff().fillna(0)
                df['sentiment_momentum'] = df['sentiment_change'].rolling(5).mean()
            except:
                df['sentiment'] = 0.0
                df['sentiment_change'] = 0.0
                df['sentiment_momentum'] = 0.0
        else:
            df['sentiment'] = 0.0
            df['sentiment_change'] = 0.0
            df['sentiment_momentum'] = 0.0

        # ==================== OMEGA MIND HFT ====================
        if data_provider:
            try:
                hft = data_provider.get_hft_indicators(symbol)
                df['vbi'] = hft.get('vbi', 0.0)
                df['vbi_avg'] = hft.get('vbi_avg', 0.0)
                df['liq_intensity'] = hft.get('liq_intensity', 0.0) / 100000.0
            except:
                df['vbi'] = 0.0
                df['vbi_avg'] = 0.0
                df['liq_intensity'] = 0.0
        else:
            df['vbi'] = 0.0
            df['vbi_avg'] = 0.0
            df['liq_intensity'] = 0.0

        # ==================== VALIDATE ====================
        df = self.validate_features(df)
        
        # [PHASE 12] SAVE TO STORE
        if feature_store and len(df) > 1:
            try:
                feature_store.store_features(symbol, df)
            except Exception as e:
                logger.debug(f"FeatureStore storage skipped: {e}")

        return df

    def validate_features(self, df):
        """Limpieza robusta de features"""
        if len(df) == 0: return df
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.ffill(limit=5, inplace=True)
        df.bfill(limit=5, inplace=True)
        return df
