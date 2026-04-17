import time  # A7 FIX: Module-level import
import pandas as pd
import numpy as np
import talib
from utils.logger import logger
from utils.debug_tracer import trace_execution
from utils.math_helpers import safe_div
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.math_kernel import calculate_zscore_jit, calculate_quantum_features_batch_jit

def fast_pct_change(arr, period):
    n = len(arr)
    res = np.full(n, np.nan, dtype=np.float64)
    if n > period:
        # Safe divide avoiding ZeroDivisionError natively handled by numpy with nan/inf
        res[period:] = np.where(arr[:-period] != 0, (arr[period:] - arr[:-period]) / arr[:-period], 0.0)
    return res

def fast_shift(arr, period):
    n = len(arr)
    res = np.full(n, np.nan, dtype=np.float64)
    if n > period:
        res[period:] = arr[:-period]
    return res

class FeatureEngineering:
    """
    🏗️ COMPONENT: Feature Engineering
    Handles all technical indicator calculations and feature generation.
    Extracted from MLStrategy to improve modularity (Excelsior Phase I).
    """
    def __init__(self):
        # [PHASE 3] Caching for Sophia KMeans to avoid computing on every tick
        self._kmeans_cache = {}
        self._scaler_cache = {}
        self._kmeans_last_fit = {}
        self._btc_cache = {'time': 0, 'data': None}

    @trace_execution
    def prepare_features(self, bars, market_regime="UNKNOWN", sentiment_loader=None, data_provider=None, symbol=None, feature_store=None, horizon="SCALPING"):
        """
        Feature engineering completo con 80+ features adaptativos.
        FORENSIC FIX: Now horizon-aware to let XGBoost learn horizon-specific patterns.
        """
        if bars is None or len(bars) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(bars)
        
        # === [PHASE 12] FEATURE STORE LOOKUP ===
        if feature_store and len(df) > 100:
            try:
                ts_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
                if ts_col in df.columns:
                    start_ts = df[ts_col].min()
                    end_ts = df[ts_col].max()
                    cached_df = feature_store.get_features(symbol, start_ts, end_ts)
                    if not cached_df.empty and len(cached_df) >= len(df) * 0.9:
                        idx_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
                        full_df = pd.concat([df.set_index(idx_col), cached_df], axis=1)
                        return full_df.reset_index()
            except Exception as e:
                logger.warning(f"FeatureStore retrieval skipped: {e}")

        # A4 FIX: Convert to numeric WITHOUT downcast (downcast forces type inspection
        # then L58-62 re-casts to float64 anyway — downcast was wasted work)
        numeric_cols = ['close', 'open', 'high', 'low', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
        new_features['returns_1'] = fast_pct_change(close, 1)
        new_features['returns_3'] = fast_pct_change(close, 3)
        new_features['returns_5'] = fast_pct_change(close, 5)
        new_features['returns_10'] = fast_pct_change(close, 10)
        
        # ==================== QUANTUM FEATURES V4 (A1 FIX: BATCH VECTORIZED) ===========
        # A1 FIX: Single JIT call replaces ~4980 individual Python→JIT calls
        # Previous: Python for-loop calling 3 separate JIT functions per bar = ~200ms
        # Now: One batch JIT call with inline logic = ~5ms
        z_scores = calculate_zscore_jit(close, period=20)
        returns_5_values = new_features['returns_5']
        
        hurst_arr, ransac_arr, bayes_arr = calculate_quantum_features_batch_jit(
            close, z_scores, returns_5_values, period=20
        )
        
        new_features['hurst_memory'] = pd.Series(hurst_arr)
        new_features['volatility_ransac'] = pd.Series(ransac_arr)
        new_features['bayesian_prior'] = pd.Series(bayes_arr)
        
        # ==================== MICROSTRUCTURE ====================
        # Amihud Illiquidity Ratio: abs(return) / volume
        new_features['amihud'] = np.where(volume != 0, abs(new_features['returns_1']) / volume, 0.0)
        
        # [PHASE 5] Microstructure (Order Flow Proxy)
        # Aproxima la presión de órdenes límite reconstruyendo el delta de volumen intravela
        hl_diff = high - low
        new_features['close_position'] = np.where(hl_diff != 0, (close - low) / hl_diff, 0.5)
        new_features['volume_imbalance'] = volume * (new_features['close_position'] * 2 - 1)
        
        v_imb_ma5 = talib.SMA(new_features['volume_imbalance'], timeperiod=5)
        v_ma5 = talib.SMA(volume, timeperiod=5)
        new_features['micro_imbalance'] = np.where(v_ma5 != 0, v_imb_ma5 / v_ma5, 0.0)
        
        # High-Low spread proxy
        new_features['hl_spread'] = np.where(close != 0, (high - low) / close, 0.0)
        new_features['oc_range'] = np.where(close != 0, abs(close - open_) / close, 0.0)
        
        # Body to wick
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        total_wick = upper_wick + lower_wick
        new_features['body_to_wick'] = np.where(total_wick != 0, abs(close - open_) / total_wick, 1.0)
        
        # ==================== MOMENTUM ====================
        for period in [3, 5, 8, 13, 21, 34]:
            shifted_close = fast_shift(close, period)
            new_features[f'momentum_{period}'] = np.where(shifted_close != 0, (close - shifted_close) / shifted_close, 0.0)
        
        new_features['roc_5'] = fast_pct_change(close, 5)
        new_features['roc_10'] = fast_pct_change(close, 10)
        new_features['roc_20'] = fast_pct_change(close, 20)
        
        # ==================== CROSS-SECTIONAL (Phase 6.3) ====================
        # QUÉ: Mide la divergencia de momentum contra el activo líder (BTC).
        # POR QUÉ: Los Alts que suben cuando BTC cae tienen alta "Fuerza Relativa" real.
        new_features['cross_spread_vs_btc'] = 0.0
        new_features['cross_relative_strength'] = 0.0
        
        try:
            if symbol and 'BTC' not in symbol and data_provider is not None:
                # A7 FIX: import time moved to module level
                current_time = time.time()
                if current_time - self._btc_cache.get('time', 0) > 300 or self._btc_cache.get('data') is None:
                    try:
                        self._btc_cache['data'] = data_provider.get_latest_bars('BTC/USDT', n=len(df), timeframe='5m')
                        self._btc_cache['time'] = current_time
                    except:
                        pass
                btc_bars = self._btc_cache.get('data')
                if btc_bars is not None and len(btc_bars) > 10:
                    btc_closes = btc_bars['close'].astype(np.float64)
                    btc_returns = pd.Series(btc_closes).pct_change()
                    
                    min_len = min(len(btc_returns), len(new_features['returns_1']))
                    
                    # Spread 1 vela (Momentum estallido)
                    spread = new_features['returns_1'].values[-min_len:] - btc_returns.values[-min_len:]
                    new_features['cross_spread_vs_btc'] = pd.Series(np.pad(spread, (len(df) - min_len, 0), constant_values=0))
                    
                    # Fuerza relativa 5 velas (Tendencia micro)
                    btc_ret_5 = talib.SMA(btc_returns.values, timeperiod=5)[-min_len:] * 5
                    my_ret_5 = new_features['returns_5'][-min_len:]
                    rs = my_ret_5 - btc_ret_5
                    new_features['cross_relative_strength'] = pd.Series(np.pad(rs, (len(df) - min_len, 0), constant_values=0))
        except Exception as e:
            pass # Falla silenciosa permitida si BTC no está cacheado
            
        
        # ==================== INDICADORES ====================
        # RSIs (Including Fast Microstructure for Scalping)
        new_features['rsi_3'] = talib.RSI(close, timeperiod=3)
        new_features['rsi_5'] = talib.RSI(close, timeperiod=5)
        new_features['rsi_7'] = talib.RSI(close, timeperiod=7)
        new_features['rsi_14'] = talib.RSI(close, timeperiod=14)
        new_features['rsi_21'] = talib.RSI(close, timeperiod=21)
        
        # ATR / ADX
        new_features['atr'] = talib.ATR(high, low, close, timeperiod=14)
        new_features['atr_pct'] = np.where(close != 0, (new_features['atr'] / close) * 100, 0.0)
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
        new_features['volume_ratio'] = np.where(new_features['volume_sma_20'] != 0, volume / new_features['volume_sma_20'], 0.0)
        
        # OBV
        new_features['obv'] = talib.OBV(close, volume)
        new_features['obv_sma'] = talib.SMA(new_features['obv'], timeperiod=20)
        new_features['obv_ratio'] = np.where(new_features['obv_sma'] != 0, new_features['obv'] / new_features['obv_sma'], 1.0)
        
        # Volatility
        vol_ret = fast_pct_change(close, 1)
        new_features['volatility_10'] = talib.STDDEV(vol_ret, timeperiod=10, nbdev=1) * 100
        
        # Garman-Klass
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        new_features['gk_vol'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

        # --- Merge point moved to end for O(N) optimization ---

        # --- POST-INDICATOR LOGIC (Consolidated in new_features for O(N) performance) ---
        bbw_ma20 = talib.SMA(new_features['bb_width'], timeperiod=20)
        new_features['bb_squeeze'] = np.where(new_features['bb_width'] < bbw_ma20 * 0.5, 1, 0)
        
        # Crossovers
        if 'ema_5' in new_features and 'ema_20' in new_features:
            new_features['ema_5_20_cross'] = np.where(new_features['ema_5'] > new_features['ema_20'], 1, -1)
        else:
            new_features['ema_5_20_cross'] = 0
             
        if 'ema_20' in new_features and 'ema_50' in new_features:
            new_features['ema_20_50_cross'] = np.where(new_features['ema_20'] > new_features['ema_50'], 1, -1)
        else:
            new_features['ema_20_50_cross'] = 0
            
        if 'ema_50' in new_features and 'ema_200' in new_features:
            new_features['ema_50_200_cross'] = np.where(new_features['ema_50'] > new_features['ema_200'], 1, -1)
        else:
            new_features['ema_50_200_cross'] = 0

        # Pattern Recognition
        new_features['up_bar'] = np.where(close > fast_shift(close, 1), 1, 0)
        new_features['down_bar'] = np.where(close < fast_shift(close, 1), 1, 0)
        
        new_features['higher_high'] = np.where(high > fast_shift(high, 1), 1, 0)
        new_features['lower_low'] = np.where(low < fast_shift(low, 1), 1, 0)
        
        # ==================== REGIME AWARE FEATURES ====================
        n_len = len(close)
        new_features['trend_power'] = np.zeros(n_len)
        new_features['trend_alignment'] = np.zeros(n_len)
        new_features['range_extreme'] = np.zeros(n_len)
        new_features['mean_reversion_potential'] = np.zeros(n_len)
        new_features['volatility_regime'] = np.ones(n_len)
        new_features['panic_index'] = np.zeros(n_len)

        if market_regime == "TRENDING":
            new_features['trend_power'] = new_features['adx'] * new_features['volume_ratio']
            new_features['trend_alignment'] = (
                np.where(new_features['ema_5_20_cross'] > 0, 1, -1) +
                np.where(new_features['ema_20_50_cross'] > 0, 1, -1) +
                np.where(new_features['ema_50_200_cross'] > 0, 1, -1)
            ) / 3
        elif market_regime == "RANGING":
            new_features['range_extreme'] = np.where(new_features['rsi_14'] < 30, 1, 0) - np.where(new_features['rsi_14'] > 70, 1, 0)
            new_features['mean_reversion_potential'] = abs(new_features['bb_position'] - 0.5) * 2
        elif market_regime == "VOLATILE":
            atr_pct_ma10 = talib.SMA(new_features['atr_pct'], timeperiod=10)
            atr_pct_ma50 = talib.SMA(new_features['atr_pct'], timeperiod=50)
            new_features['volatility_regime'] = np.where(atr_pct_ma50 != 0, atr_pct_ma10 / atr_pct_ma50, 1.0)
            new_features['panic_index'] = new_features['volume_ratio'] * new_features['volatility_10']

        # ==================== SENTIMENT ====================
        new_features['sentiment'] = np.zeros(n_len)
        new_features['sentiment_change'] = np.zeros(n_len)
        new_features['sentiment_momentum'] = np.zeros(n_len)

        if sentiment_loader:
            try:
                sentiment = sentiment_loader.get_sentiment(symbol)
                s_val = float(sentiment) if sentiment is not None else 0.0
                new_features['sentiment'] = np.full(n_len, s_val)
                sent_arr = new_features['sentiment']
                sent_shifted = fast_shift(sent_arr, 1)
                new_features['sentiment_change'] = np.where(np.isnan(sent_shifted), 0.0, sent_arr - sent_shifted)
                new_features['sentiment_momentum'] = talib.SMA(new_features['sentiment_change'], timeperiod=5)
            except:
                pass

        # ==================== OMEGA MIND HFT ====================
        new_features['vbi'] = np.zeros(n_len)
        new_features['vbi_avg'] = np.zeros(n_len)
        new_features['liq_intensity'] = np.zeros(n_len)

        if data_provider:
            try:
                hft = data_provider.get_hft_indicators(symbol)
                new_features['vbi'] = np.full(n_len, hft.get('vbi', 0.0))
                new_features['vbi_avg'] = np.full(n_len, hft.get('vbi_avg', 0.0))
                new_features['liq_intensity'] = np.full(n_len, hft.get('liq_intensity', 0.0) / 100000.0)
            except:
                pass

        # ==================== OMEGA DERIVATIVES ====================
        new_features['funding_rate'] = np.zeros(n_len)
        new_features['oi'] = np.zeros(n_len)
        new_features['oi_delta'] = np.zeros(n_len)
        new_features['funding_distortion'] = np.zeros(n_len)

        if data_provider and hasattr(data_provider, 'get_derivatives_metrics'):
            try:
                deriv = data_provider.get_derivatives_metrics(symbol)
                new_features['funding_rate'] = np.full(n_len, deriv.get('funding_rate', 0.0))
                new_features['oi'] = np.full(n_len, deriv.get('oi', 0.0))
                new_features['oi_delta'] = np.full(n_len, deriv.get('oi_delta', 0.0))
                
                # Synthetic derived proxy features for XGBoost
                # Disparidad de funding (1=alto pago de largos, -1=alto pago de cortos)
                fr_ma20 = talib.SMA(new_features['funding_rate'], timeperiod=20)
                new_features['funding_distortion'] = fr_ma20 / 0.0001
            except Exception as e:
                pass

        # ==================== VALIDATE ====================
        # ==================== PHASE 3: SOPHIA KMEANS CLUSTER ====================
        try:
            symbol_key = symbol if symbol else 'default'
            cluster_cols = ['rsi_14', 'atr_pct', 'volume_ratio', 'adx']
            
            if all(c in new_features for c in cluster_cols) and len(df) >= 50:
                # Use data from new_features for clustering
                feat_data = {c: new_features[c] for c in cluster_cols}
                features_array = pd.DataFrame(feat_data).fillna(0).values
                current_time = pd.Timestamp.utcnow() if 'datetime' not in df.columns else df['datetime'].iloc[-1]
                
                last_fit_time = self._kmeans_last_fit.get(symbol_key)
                fit_count = getattr(self, '_kmeans_fit_counter', {})
                current_count = fit_count.get(symbol_key, 0) + 1
                fit_count[symbol_key] = current_count
                self._kmeans_fit_counter = fit_count
                
                need_refit = False
                if last_fit_time is None or current_count % 50 == 0:
                    need_refit = True
                
                if need_refit:
                    scaler = StandardScaler()
                    scaled_fit = scaler.fit_transform(features_array)
                    kmeans = KMeans(n_clusters=4, random_state=42, n_init=2, max_iter=50)
                    kmeans.fit(scaled_fit)
                    
                    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                    cluster_map = {}
                    for i, centroid in enumerate(centroids):
                        rsi_c, atr_c, vol_c, adx_c = centroid[0], centroid[1], centroid[2], centroid[3]
                        regime_id = 1 if adx_c > 25 and rsi_c > 50 else (2 if adx_c > 25 else 0)
                        if adx_c <= 25 and atr_c > 0.015: regime_id = 3
                        cluster_map[i] = regime_id
                        
                    scaler.cluster_map = cluster_map
                    raw_clusters = kmeans.predict(scaled_fit)
                    anchored_clusters = np.vectorize(cluster_map.get)(raw_clusters, 0)
                    
                    self._kmeans_cache[symbol_key] = kmeans
                    self._scaler_cache[symbol_key] = scaler
                    self._kmeans_last_fit[symbol_key] = current_time
                    new_features['market_cluster'] = anchored_clusters
                else:
                    scaler = self._scaler_cache[symbol_key]
                    kmeans = self._kmeans_cache[symbol_key]
                    scaled_features = scaler.transform(features_array)
                    raw_clusters = kmeans.predict(scaled_features)
                    cluster_map = getattr(scaler, 'cluster_map', {0:0,1:1,2:2,3:3})
                    anchored_clusters = np.vectorize(cluster_map.get)(raw_clusters, 0)
                    new_features['market_cluster'] = anchored_clusters
                
                # One-Hot Encoding
                for i in range(4):
                    new_features[f'cluster_{i}'] = (new_features['market_cluster'] == i).astype(int)
            else:
                new_features['market_cluster'] = pd.Series(-1, index=df.index)
                for i in range(4): new_features[f'cluster_{i}'] = pd.Series(0, index=df.index)
                
        except Exception as e:
            logger.error(f"Error computing KMeans market clusters: {e}")
            new_features['market_cluster'] = pd.Series(-1, index=df.index)
            for i in range(4): new_features[f'cluster_{i}'] = pd.Series(0, index=df.index)
            
        # ==================== SCALPING MICROSTRUCTURE ====================
        new_features['micro_velocity_3'] = fast_pct_change(close, 3) / 3.0
        vol_ma_5 = talib.SMA(volume, timeperiod=5)
        vol_ma_15 = talib.SMA(volume, timeperiod=15)
        new_features['volume_accel'] = np.where(vol_ma_15 > 0, vol_ma_5 / vol_ma_15, 1.0)
        
        if 'bb_width' in new_features:
            bb_width_ma_20 = talib.SMA(new_features['bb_width'], timeperiod=20)
            new_features['spread_squeeze'] = np.where(bb_width_ma_20 > 0, new_features['bb_width'] / bb_width_ma_20, 1.0)
        else:
            new_features['spread_squeeze'] = np.ones(n_len)

        # Microstructure Labeling
        vol_mean_20 = talib.SMA(volume, timeperiod=20)
        high_vol = volume > (vol_mean_20 * 1.5)
        
        body_size = abs(close - open_)
        body_avg_20 = talib.SMA(body_size, timeperiod=20)
        small_body = body_size < (body_avg_20 * 0.5)
        
        new_features['micro_absorption'] = np.where(high_vol & small_body, 1, 0)
        
        wick_avg = talib.SMA(total_wick, timeperiod=20)
        
        close_shifted_1 = fast_shift(close, 1)
        ex_bear = (upper_wick > wick_avg) & (close < close_shifted_1)
        ex_bull = (lower_wick > wick_avg) & (close > close_shifted_1)
        new_features['micro_exhaustion'] = np.where(ex_bull, 1, np.where(ex_bear, -1, 0))
        
        big_body = body_size > (body_avg_20 * 1.5)
        sw_bull = high_vol & big_body & (close >= (high - (high - low) * 0.1))
        sw_bear = high_vol & big_body & (close <= (low + (high - low) * 0.1))
        new_features['micro_sweep'] = np.where(sw_bull, 1, np.where(sw_bear, -1, 0))
        
        conds = [new_features['micro_sweep'] == 1, new_features['micro_sweep'] == -1, new_features['micro_exhaustion'] == 1, new_features['micro_exhaustion'] == -1, new_features['micro_absorption'] == 1]
        new_features['micro_label'] = np.select(conds, [1, -1, 2, -2, 3], default=0)

        # ==================== HORIZON-SPECIFIC FEATURES (FORENSIC FIX) ====================
        # QUÉ: Features que codifican el contexto temporal del horizonte.
        # POR QUÉ: Scalping y Swing ven los mismos indicadores pero deben interpretarlos
        #          diferente. Un RSI de 30 es entrada inmediata en Scalping pero podría
        #          ser solo el inicio de una caída en Swing.
        # CÓMO: Indicador binario + features de velocidad/inercia calibrados por horizonte.
        is_swing = 1 if horizon.upper() == 'SWING' else 0
        new_features['is_swing_horizon'] = np.full(n_len, is_swing)
        
        if is_swing:
            # Swing-specific: Longer-term momentum ratios
            new_features['swing_momentum_ratio'] = np.where(
                new_features.get('momentum_5', np.zeros(n_len)) != 0,
                new_features.get('momentum_34', np.zeros(n_len)) / (new_features.get('momentum_5', np.ones(n_len)) + 1e-10),
                0.0
            )
            # Swing-specific: EMA slope over longer period
            if 'ema_50' in new_features:
                ema50_shifted = fast_shift(new_features['ema_50'], 10)
                new_features['swing_ema50_slope'] = np.where(
                    ema50_shifted != 0,
                    (new_features['ema_50'] - ema50_shifted) / (ema50_shifted + 1e-10),
                    0.0
                )
            else:
                new_features['swing_ema50_slope'] = np.zeros(n_len)
        else:
            # Scalping-specific: Ultra-short momentum burst
            new_features['scalp_velocity_1'] = fast_pct_change(close, 1)
            new_features['scalp_rsi_divergence'] = (
                new_features.get('rsi_3', np.full(n_len, 50)) - 
                new_features.get('rsi_14', np.full(n_len, 50))
            )
        
        # Fill missing horizon features with zeros (consistent shape)
        for feat_name in ['swing_momentum_ratio', 'swing_ema50_slope', 'scalp_velocity_1', 'scalp_rsi_divergence']:
            if feat_name not in new_features:
                new_features[feat_name] = np.zeros(n_len)

        # ==================== FINAL ATOMIC MERGE (SUPREMO-V3) ====================
        # This replaces iterative assignments with a single batch operation
        for col_name, col_data in new_features.items():
            if isinstance(col_data, (pd.Series, np.ndarray)):
                df[col_name] = col_data if isinstance(col_data, pd.Series) else pd.Series(col_data, index=df.index)
            else:
                df[col_name] = col_data

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
