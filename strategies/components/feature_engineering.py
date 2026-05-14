import time
import polars as pl
import pandas as pd
import numpy as np
import talib
from utils.logger import logger
from utils.debug_tracer import trace_execution
from utils.math_helpers import safe_div
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.math_kernel import calculate_zscore_jit, calculate_quantum_features_batch_jit
from core.swarm_correlator import swarm_correlator
from data.macro_loader import macro_loader
from data.onchain_loader import onchain_loader

def fast_pct_change(arr, period):
    n = len(arr)
    res = np.full(n, np.nan, dtype=np.float64)
    if n > period:
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
    🏗️ COMPONENT: Feature Engineering (POLARS/ARROW EDITION)
    Migrated to Polars for Zero-Copy IPC and Rust-based multithreading.
    """
    def __init__(self):
        self._kmeans_cache = {}
        self._scaler_cache = {}
        self._kmeans_last_fit = {}
        self._btc_cache = {'time': 0, 'data': None}

    @trace_execution
    def prepare_features(self, bars, market_regime="UNKNOWN", sentiment_loader=None, data_provider=None, symbol=None, feature_store=None, horizon="SCALPING"):
        if bars is None or len(bars) == 0:
            return pd.DataFrame()
            
        # [PHASE 12] FEATURE STORE LOOKUP
        # Se requiere Pandas para compatibilidad con la capa de persistencia actual
        df_pd = pd.DataFrame(bars)
        if feature_store and len(df_pd) > 100:
            try:
                ts_col = 'datetime' if 'datetime' in df_pd.columns else 'timestamp'
                if ts_col in df_pd.columns:
                    start_ts = df_pd[ts_col].min()
                    end_ts = df_pd[ts_col].max()
                    cached_df = feature_store.get_features(symbol, start_ts, end_ts)
                    if not cached_df.empty and len(cached_df) >= len(df_pd) * 0.9:
                        idx_col = 'datetime' if 'datetime' in df_pd.columns else 'timestamp'
                        full_df = pd.concat([df_pd.set_index(idx_col), cached_df], axis=1)
                        return full_df.reset_index()
            except Exception as e:
                logger.warning(f"FeatureStore retrieval skipped: {e}")

        # INGESTA POLARS: Zero-copy desde diccionarios/listas
        df = pl.DataFrame(bars)
        if len(df) < 50:
            return pd.DataFrame()

        numeric_cols = ['close', 'open', 'high', 'low', 'volume']
        cast_exprs = [pl.col(c).cast(pl.Float64) for c in numeric_cols if c in df.columns]
        if cast_exprs:
            df = df.with_columns(cast_exprs)

        # Arrays NumPy zero-copy para TA-Lib y JIT
        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        open_ = df['open'].to_numpy()
        volume = df['volume'].to_numpy()
        
        n_len = len(close)
        new_features = {}

        # ==================== PRICE ACTION & MOMENTUM (POLARS EXPRS) ====================
        exprs = []
        for p in [1, 3, 5, 10]:
            exprs.append(pl.col('close').pct_change(p).fill_null(0.0).alias(f'returns_{p}'))
        
        # XGBoost explicitly expects roc_5, roc_10, roc_20 as named columns
        for p in [5, 10, 20]:
            exprs.append(pl.col('close').pct_change(p).fill_null(0.0).alias(f'roc_{p}'))
        
        for p in [3, 5, 8, 13, 21, 34]:
            exprs.append(
                pl.when(pl.col('close').shift(p) != 0)
                .then((pl.col('close') - pl.col('close').shift(p)) / pl.col('close').shift(p))
                .otherwise(0.0)
                .alias(f'momentum_{p}')
            )
            
        # ==================== MICROSTRUCTURE (POLARS EXPRS) ====================
        exprs.append(
            pl.when(pl.col('volume') != 0)
            .then(pl.col('close').pct_change(1).fill_null(0.0).abs() / pl.col('volume'))
            .otherwise(0.0).alias('amihud')
        )
        
        hl_diff = pl.col('high') - pl.col('low')
        close_pos = pl.when(hl_diff != 0).then((pl.col('close') - pl.col('low')) / hl_diff).otherwise(0.5)
        exprs.append(close_pos.alias('close_position'))
        exprs.append((pl.col('volume') * (close_pos * 2 - 1)).alias('volume_imbalance'))
        
        exprs.append(pl.when(pl.col('close') != 0).then(hl_diff / pl.col('close')).otherwise(0.0).alias('hl_spread'))
        exprs.append(pl.when(pl.col('close') != 0).then((pl.col('close') - pl.col('open')).abs() / pl.col('close')).otherwise(0.0).alias('oc_range'))
        
        total_wick = (pl.col('high') - pl.max_horizontal('open', 'close')) + (pl.min_horizontal('open', 'close') - pl.col('low'))
        exprs.append(pl.when(total_wick != 0).then((pl.col('close') - pl.col('open')).abs() / total_wick).otherwise(1.0).alias('body_to_wick'))
        
        # Ejecutamos el primer bloque paralelizado
        df = df.with_columns(exprs)
        
        # Micro imbalance (depende de volume_imbalance recién calculado)
        v_imb_ma5 = df['volume_imbalance'].rolling_mean(window_size=5)
        v_ma5 = df['volume'].rolling_mean(window_size=5)
        df = df.with_columns([
            pl.when(v_ma5 != 0).then(v_imb_ma5 / v_ma5).otherwise(0.0).alias('micro_imbalance')
        ])

        # ==================== QUANTUM FEATURES V4 (JIT NumPy) ====================
        z_scores = calculate_zscore_jit(close, period=20)
        returns_5_arr = df['returns_5'].to_numpy()
        hurst_arr, ransac_arr, bayes_arr = calculate_quantum_features_batch_jit(close, z_scores, returns_5_arr, period=20)
        
        new_features['hurst_memory'] = hurst_arr
        new_features['volatility_ransac'] = ransac_arr
        new_features['bayesian_prior'] = bayes_arr

        # ==================== CROSS-SECTIONAL ====================
        new_features['cross_spread_vs_btc'] = np.zeros(n_len)
        new_features['cross_relative_strength'] = np.zeros(n_len)
        try:
            if symbol and 'BTC' not in symbol and data_provider is not None:
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
                    btc_returns = pd.Series(btc_closes).pct_change().fillna(0).values
                    returns_1_arr = df['returns_1'].to_numpy()
                    min_len = min(len(btc_returns), len(returns_1_arr))
                    
                    spread = returns_1_arr[-min_len:] - btc_returns[-min_len:]
                    new_features['cross_spread_vs_btc'] = np.pad(spread, (n_len - min_len, 0), constant_values=0)
                    
                    btc_ret_5 = talib.SMA(btc_returns, timeperiod=5)[-min_len:] * 5
                    rs = returns_5_arr[-min_len:] - btc_ret_5
                    new_features['cross_relative_strength'] = np.pad(rs, (n_len - min_len, 0), constant_values=0)
        except:
            pass

        # ==================== INDICADORES (TA-Lib con NumPy) ====================
        new_features['rsi_3'] = talib.RSI(close, 3)
        new_features['rsi_5'] = talib.RSI(close, 5)
        new_features['rsi_7'] = talib.RSI(close, 7)
        new_features['rsi_14'] = talib.RSI(close, 14)
        new_features['rsi_21'] = talib.RSI(close, 21)
        
        atr = talib.ATR(high, low, close, 14)
        new_features['atr'] = atr
        new_features['atr_pct'] = np.where(close != 0, (atr / close) * 100, 0.0)
        new_features['natr'] = talib.NATR(high, low, close, 14)
        
        new_features['adx'] = talib.ADX(high, low, close, 14)
        new_features['plus_di'] = talib.PLUS_DI(high, low, close, 14)
        new_features['minus_di'] = talib.MINUS_DI(high, low, close, 14)
        
        macd, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
        new_features['macd'] = macd
        new_features['macd_signal'] = macd_signal
        new_features['macd_hist'] = macd_hist
        
        upper, middle, lower_band = talib.BBANDS(close, 20, 2, 2)
        new_features['bb_upper'] = upper
        new_features['bb_middle'] = middle
        new_features['bb_lower'] = lower_band
        new_features['bb_position'] = safe_div(close - lower_band, upper - lower_band, 0.5)
        new_features['bb_width'] = safe_div(upper - lower_band, middle)
        
        slowk, slowd = talib.STOCH(high, low, close, 14, 3, 3)
        new_features['stoch_k'] = slowk
        new_features['stoch_d'] = slowd
        new_features['stoch_cross'] = np.where(slowk > slowd, 1, -1)
        
        new_features['mfi'] = talib.MFI(high, low, close, volume, 14)
        new_features['cci'] = talib.CCI(high, low, close, 20)
        
        for p in [5, 10, 20, 50, 100, 200]:
            if n_len >= p:
                ema = talib.EMA(close, p)
                new_features[f'ema_{p}'] = ema
                new_features[f'dist_ema_{p}'] = safe_div(close - ema, ema)
            else:
                new_features[f'ema_{p}'] = np.zeros(n_len)
                new_features[f'dist_ema_{p}'] = np.zeros(n_len)
                
        for p in [20, 50]:
            new_features[f'sma_{p}'] = talib.SMA(close, p) if n_len >= p else np.zeros(n_len)
            
        v_sma_20 = talib.SMA(volume, 20)
        new_features['volume_sma_20'] = v_sma_20
        new_features['volume_ratio'] = np.where(v_sma_20 != 0, volume / v_sma_20, 0.0)
        
        obv = talib.OBV(close, volume)
        obv_sma = talib.SMA(obv, 20)
        new_features['obv'] = obv
        new_features['obv_sma'] = obv_sma
        new_features['obv_ratio'] = np.where(obv_sma != 0, obv / obv_sma, 1.0)
        
        vol_ret = df['returns_1'].to_numpy()
        new_features['volatility_10'] = talib.STDDEV(vol_ret, 10, 1) * 100
        
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
        new_features['gk_vol'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

        # Post-Indicator Logic
        bbw_ma20 = talib.SMA(new_features['bb_width'], 20)
        new_features['bb_squeeze'] = np.where(new_features['bb_width'] < bbw_ma20 * 0.5, 1, 0)
        
        new_features['ema_5_20_cross'] = np.where(new_features.get('ema_5', np.zeros(n_len)) > new_features.get('ema_20', np.zeros(n_len)), 1, -1)
        new_features['ema_20_50_cross'] = np.where(new_features.get('ema_20', np.zeros(n_len)) > new_features.get('ema_50', np.zeros(n_len)), 1, -1)
        new_features['ema_50_200_cross'] = np.where(new_features.get('ema_50', np.zeros(n_len)) > new_features.get('ema_200', np.zeros(n_len)), 1, -1)
        
        new_features['up_bar'] = np.where(close > fast_shift(close, 1), 1, 0)
        new_features['down_bar'] = np.where(close < fast_shift(close, 1), 1, 0)
        new_features['higher_high'] = np.where(high > fast_shift(high, 1), 1, 0)
        new_features['lower_low'] = np.where(low < fast_shift(low, 1), 1, 0)

        # ==================== REGIME AWARE ====================
        new_features['trend_power'] = np.zeros(n_len)
        new_features['trend_alignment'] = np.zeros(n_len)
        new_features['range_extreme'] = np.zeros(n_len)
        new_features['mean_reversion_potential'] = np.zeros(n_len)
        new_features['volatility_regime'] = np.ones(n_len)
        new_features['panic_index'] = np.zeros(n_len)

        if market_regime == "TRENDING":
            new_features['trend_power'] = new_features['adx'] * new_features['volume_ratio']
            new_features['trend_alignment'] = (np.where(new_features['ema_5_20_cross'] > 0, 1, -1) + np.where(new_features['ema_20_50_cross'] > 0, 1, -1) + np.where(new_features['ema_50_200_cross'] > 0, 1, -1)) / 3
        elif market_regime == "RANGING":
            new_features['range_extreme'] = np.where(new_features['rsi_14'] < 30, 1, 0) - np.where(new_features['rsi_14'] > 70, 1, 0)
            new_features['mean_reversion_potential'] = abs(new_features['bb_position'] - 0.5) * 2
        elif market_regime == "VOLATILE":
            atr_pct_ma10 = talib.SMA(new_features['atr_pct'], 10)
            atr_pct_ma50 = talib.SMA(new_features['atr_pct'], 50)
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
                sent_arr = np.full(n_len, s_val)
                new_features['sentiment'] = sent_arr
                sent_shifted = fast_shift(sent_arr, 1)
                new_features['sentiment_change'] = np.where(np.isnan(sent_shifted), 0.0, sent_arr - sent_shifted)
                new_features['sentiment_momentum'] = talib.SMA(new_features['sentiment_change'], 5)
            except:
                pass

        # ==================== OMEGA MIND / DERIVATIVES ====================
        for k in ['vbi', 'vbi_avg', 'liq_intensity', 'funding_rate', 'oi', 'oi_delta', 'funding_distortion']:
            new_features[k] = np.zeros(n_len)
            
        if data_provider:
            try:
                hft = data_provider.get_hft_indicators(symbol)
                new_features['vbi'] = np.full(n_len, hft.get('vbi', 0.0))
                new_features['vbi_avg'] = np.full(n_len, hft.get('vbi_avg', 0.0))
                new_features['liq_intensity'] = np.full(n_len, hft.get('liq_intensity', 0.0) / 100000.0)
            except:
                pass
            if hasattr(data_provider, 'get_derivatives_metrics'):
                try:
                    deriv = data_provider.get_derivatives_metrics(symbol)
                    new_features['funding_rate'] = np.full(n_len, deriv.get('funding_rate', 0.0))
                    new_features['oi'] = np.full(n_len, deriv.get('oi', 0.0))
                    new_features['oi_delta'] = np.full(n_len, deriv.get('oi_delta', 0.0))
                    fr_ma20 = talib.SMA(new_features['funding_rate'], 20)
                    new_features['funding_distortion'] = fr_ma20 / 0.0001
                except:
                    pass

        # ==================== PHASE 3: SOPHIA KMEANS CLUSTER ====================
        try:
            symbol_key = symbol if symbol else 'default'
            cluster_cols = ['rsi_14', 'atr_pct', 'volume_ratio', 'adx']
            if all(c in new_features for c in cluster_cols) and n_len >= 50:
                feat_data = {c: new_features[c] for c in cluster_cols}
                features_array = pd.DataFrame(feat_data).ffill().fillna(0).values
                # We need current_time properly
                if 'datetime' in df.columns:
                    current_time = df['datetime'][-1]
                elif 'timestamp' in df.columns:
                    current_time = df['timestamp'][-1]
                else:
                    current_time = pd.Timestamp.utcnow()
                
                last_fit_time = self._kmeans_last_fit.get(symbol_key)
                fit_count = getattr(self, '_kmeans_fit_counter', {})
                current_count = fit_count.get(symbol_key, 0) + 1
                fit_count[symbol_key] = current_count
                self._kmeans_fit_counter = fit_count
                
                if last_fit_time is None or current_count % 50 == 0:
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
                    self._kmeans_cache[symbol_key] = kmeans
                    self._scaler_cache[symbol_key] = scaler
                    self._kmeans_last_fit[symbol_key] = current_time
                    
                scaler = self._scaler_cache[symbol_key]
                kmeans = self._kmeans_cache[symbol_key]
                scaled_features = scaler.transform(features_array)
                raw_clusters = kmeans.predict(scaled_features)
                cluster_map = getattr(scaler, 'cluster_map', {0:0,1:1,2:2,3:3})
                anchored_clusters = np.vectorize(cluster_map.get)(raw_clusters, 0)
                new_features['market_cluster'] = anchored_clusters
                
                for i in range(4):
                    new_features[f'cluster_{i}'] = (anchored_clusters == i).astype(int)
            else:
                new_features['market_cluster'] = np.full(n_len, -1)
                for i in range(4): new_features[f'cluster_{i}'] = np.zeros(n_len)
        except:
            new_features['market_cluster'] = np.full(n_len, -1)
            for i in range(4): new_features[f'cluster_{i}'] = np.zeros(n_len)

        # ==================== SCALPING MICROSTRUCTURE ====================
        new_features['micro_velocity_3'] = df['returns_3'].to_numpy() / 3.0
        
        # 🌊 PHASE 10: REAL L2 ORDERBOOK METRICS
        new_features['l2_ofi'] = np.zeros(n_len)
        new_features['l2_spread'] = np.zeros(n_len)
        new_features['l2_microprice_dist'] = np.zeros(n_len)
        
        if data_provider and hasattr(data_provider, 'get_orderbook'):
            try:
                ob = data_provider.get_orderbook(symbol)
                if ob:
                    # Current values at the moment the feature vector is created
                    ofi = ob.calculate_ofi()
                    spread = ob.calculate_spread()
                    microprice = ob.calculate_microprice()
                    
                    # Fill the last value (real-time snapshot). 
                    new_features['l2_ofi'][-1] = ofi
                    new_features['l2_spread'][-1] = spread
                    if close[-1] > 0 and microprice > 0:
                        new_features['l2_microprice_dist'][-1] = (microprice - close[-1]) / close[-1]
            except Exception as e:
                logger.debug(f"Silent exception caught: {e}")
        vol_ma_5 = talib.SMA(volume, 5)
        vol_ma_15 = talib.SMA(volume, 15)
        new_features['volume_accel'] = np.where(vol_ma_15 > 0, vol_ma_5 / vol_ma_15, 1.0)
        
        if 'bb_width' in new_features:
            bbw_ma20 = talib.SMA(new_features['bb_width'], 20)
            new_features['spread_squeeze'] = np.where(bbw_ma20 > 0, new_features['bb_width'] / bbw_ma20, 1.0)
        else:
            new_features['spread_squeeze'] = np.ones(n_len)

        vol_mean_20 = talib.SMA(volume, 20)
        high_vol = volume > (vol_mean_20 * 1.5)
        body_size = abs(close - open_)
        body_avg_20 = talib.SMA(body_size, 20)
        small_body = body_size < (body_avg_20 * 0.5)
        
        new_features['micro_absorption'] = np.where(high_vol & small_body, 1, 0)
        
        # total_wick calculation on numpy natively for SMA
        total_wick_np = (high - np.maximum(open_, close)) + (np.minimum(open_, close) - low)
        wick_avg = talib.SMA(total_wick_np, 20)
        close_shifted_1 = fast_shift(close, 1)
        upper_wick_arr = (high - np.maximum(open_, close))
        lower_wick_arr = (np.minimum(open_, close) - low)
        ex_bear = (upper_wick_arr > wick_avg) & (close < close_shifted_1)
        ex_bull = (lower_wick_arr > wick_avg) & (close > close_shifted_1)
        new_features['micro_exhaustion'] = np.where(ex_bull, 1, np.where(ex_bear, -1, 0))
        
        big_body = body_size > (body_avg_20 * 1.5)
        sw_bull = high_vol & big_body & (close >= (high - (high - low) * 0.1))
        sw_bear = high_vol & big_body & (close <= (low + (high - low) * 0.1))
        new_features['micro_sweep'] = np.where(sw_bull, 1, np.where(sw_bear, -1, 0))
        
        conds = [new_features['micro_sweep'] == 1, new_features['micro_sweep'] == -1, new_features['micro_exhaustion'] == 1, new_features['micro_exhaustion'] == -1, new_features['micro_absorption'] == 1]
        new_features['micro_label'] = np.select(conds, [1, -1, 2, -2, 3], default=0)

        # ==================== HORIZON-SPECIFIC FEATURES ====================
        is_swing = 1 if horizon.upper() == 'SWING' else 0
        new_features['is_swing_horizon'] = np.full(n_len, is_swing)
        
        if is_swing:
            new_features['swing_momentum_ratio'] = np.where(new_features.get('momentum_5', np.zeros(n_len)) != 0, new_features.get('momentum_34', np.zeros(n_len)) / (new_features.get('momentum_5', np.ones(n_len)) + 1e-10), 0.0)
            if 'ema_50' in new_features:
                ema50_shifted = fast_shift(new_features['ema_50'], 10)
                new_features['swing_ema50_slope'] = np.where(ema50_shifted != 0, (new_features['ema_50'] - ema50_shifted) / (ema50_shifted + 1e-10), 0.0)
            else:
                new_features['swing_ema50_slope'] = np.zeros(n_len)
        else:
            new_features['scalp_velocity_1'] = df['returns_1'].to_numpy()
            new_features['scalp_rsi_divergence'] = new_features.get('rsi_3', np.full(n_len, 50)) - new_features.get('rsi_14', np.full(n_len, 50))
        
        for feat_name in ['swing_momentum_ratio', 'swing_ema50_slope', 'scalp_velocity_1', 'scalp_rsi_divergence']:
            if feat_name not in new_features:
                new_features[feat_name] = np.zeros(n_len)

        # ==================== PHASE 3 (AITS): HYPERGRAPH CENTRALITY ====================
        graph_feats = swarm_correlator.get_hypergraph_features(symbol)
        new_features['graph_centrality'] = np.full(n_len, graph_feats.get('graph_centrality', 0.0))
        new_features['graph_pagerank'] = np.full(n_len, graph_feats.get('graph_pagerank', 0.0))
        new_features['graph_connectivity'] = np.full(n_len, graph_feats.get('graph_connectivity', 0.0))

        # ==================== PHASE 2 (AITS): MACRO & ON-CHAIN NERVOUS SYSTEM ====================
        macro_feats = macro_loader.get_macro_features()
        onchain_feats = onchain_loader.get_onchain_features()
        
        new_features['macro_dxy_returns'] = np.full(n_len, macro_feats.get('macro_dxy_returns', 0.0))
        new_features['macro_dxy_trend'] = np.full(n_len, macro_feats.get('macro_dxy_trend', 0.0))
        new_features['macro_nq_returns'] = np.full(n_len, macro_feats.get('macro_nq_returns', 0.0))
        new_features['macro_nq_trend'] = np.full(n_len, macro_feats.get('macro_nq_trend', 0.0))
        new_features['onchain_whale_flow'] = np.full(n_len, onchain_feats.get('onchain_whale_flow', 0.0))

        # Merge dict back into Polars df efficiently
        pl_features = pl.DataFrame(new_features)
        df = pl.concat([df, pl_features], how="horizontal")

        # --------------------------------------------------------------------------------
        # [MEMORY OPTIMIZATION] Downcast to Float32 before turning to Pandas
        # Reduces memory overhead by 50% for XGBoost Inference without precision loss.
        # --------------------------------------------------------------------------------
        float_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Float64, pl.Float32]]
        if float_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])

        # Convert to Pandas for downstream components
        df_out = df.to_pandas()
        
        # Immediate memory release
        del df
        import gc
        gc.collect()
        
        df_out = self.validate_features(df_out)
        
        # [PHASE 12] SAVE TO STORE
        if feature_store and len(df_out) > 1:
            try:
                feature_store.store_features(symbol, df_out)
            except Exception as e:
                logger.debug(f"FeatureStore storage skipped: {e}")

        return df_out

    def validate_features(self, df):
        """Limpieza robusta de features sin bleeding de O.Os"""
        if len(df) == 0: return df
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True) 
        df.fillna(0.0, inplace=True)
        return df
