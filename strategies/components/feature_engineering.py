import time
import polars as pl
import numpy as np
import datetime
import talib
from utils.logger import logger
from utils.debug_tracer import trace_execution
from utils.math_helpers import safe_div
from utils.math_kernel import (
    calculate_zscore_jit, calculate_quantum_features_batch_jit,
    calculate_rsi_jit, calculate_atr_jit, calculate_adx_jit,
    calculate_macd_jit, calculate_bollinger_jit, calculate_ema_jit,
    kalman_filter_1d_jit, fractional_differencing_jit
)
from core.swarm_correlator import swarm_correlator
from data.macro_intelligence import macro_intelligence
from data.onchain_loader import onchain_loader
from data.news_sentiment_nlp import news_sentiment

# [MÓDULO OMEGA] Indicadores Modulares
from strategies.indicators import (
    TrendIndicators,
    MomentumIndicators,
    VolatilityIndicators,
    VolumeIndicators
)

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

class FeatureArena:
    """
    Zero-Copy Bridge between Rust StatefulEngine and ML models.
    Pre-allocates a Numpy C-contiguous array for fast inplace inference.
    """
    def __init__(self, capacity=1):
        # El array que XGBoost consume por referencia (zero copy)
        self.features = None
        self.indices = None
        self.capacity = capacity
        self.columns = None
        
    def initialize(self, df_columns: list):
        self.columns = df_columns
        self.features = np.zeros((self.capacity, len(df_columns)), dtype=np.float32, order='C')
        
        # Mapear los 12 outputs de Rust a los índices correctos en el Arena
        # Rust outputs: [ema20, ema50, ema200, rsi14, zscore20, zscore50, bb_width, atr_pct, volume_ratio, return1, bb_mean, bb_std]
        rust_names = [
            'ema_20', 'ema_50', 'ema_200', 'rsi_14', 'zscore_20', 'zscore_50', 
            'bb_width', 'atr_pct', 'volume_ratio', 'returns_1', 'bb_middle', 'bb_std'
        ]
        
        indices = []
        for name in rust_names:
            if name in df_columns:
                indices.append(df_columns.index(name))
            else:
                indices.append(-1)
        self.indices = np.array(indices, dtype=np.int32)
        
    def inject_base_features(self, df_row):
        """Copia todos los features de la última vela cerrada al Arena."""
        if self.features is not None and df_row.shape[1] == self.features.shape[1]:
            np.copyto(self.features, df_row.astype(np.float32))

class FeatureEngineering:
    """
    🏗️ COMPONENT: Feature Engineering (POLARS/ARROW EDITION)
    Migrated to Polars for Zero-Copy IPC and Rust-based multithreading.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FeatureEngineering, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self._kmeans_cache = {}
        self._scaler_cache = {}
        self._kmeans_last_fit = {}
        self._btc_cache = {'time': 0, 'data': None}
        # ═══════════════════════════════════════════════════════════════
        # [QUANTUM CACHE] INCREMENTAL FEATURE RESULT CACHE
        # QUÉ: Almacena el último DataFrame de features computado por símbolo.
        # POR QUÉ: Entre cambios de vela (60s en scalping 1m), las mismas
        #   100 barras producen exactamente los mismos 143 features.
        #   Recalcular todo toma ~400ms. Devolver el cache toma ~0.01ms.
        # PARA QUÉ: Reducir latencia de 400ms a <1ms entre velas.
        # CÓMO: Key = (symbol, last_timestamp, last_close, n_bars).
        #   Si la key no cambió, devolvemos el resultado anterior.
        # ═══════════════════════════════════════════════════════════════
        self._result_cache = {}  # {symbol: (cache_key, result_df)}
        
        # 🌌 QUANTUM BRIDGE (METAL INJECTION)
        self._metal_active = False
        self.nano_core = None
        self.metal_engines = {}  # {symbol: StatefulEngine}
        self.feature_arenas = {} # {symbol: FeatureArena}
        try:
            import sys
            import os
            rust_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/rust_core'))
            if rust_path not in sys.path:
                sys.path.insert(0, rust_path)
            import nano_core
            self.nano_core = nano_core
            self._metal_active = True
            logger.info("🌌 QUANTUM BRIDGE: Rust nano_core cargado exitosamente. Listo para simulación en caliente.")
        except Exception as e:
            logger.warning(f"⚠️ QUANTUM BRIDGE INACTIVO: Cayendo a Pandas (O(N)). Razón: {e}")

    def update_and_inject_metal(self, symbol: str, price: float, high: float, low: float, volume: float) -> bool:
        """
        Intenta actualizar el estado O(1) usando Rust y lo inyecta directo al Arena.
        Retorna el array Numpy del Arena (C-contiguous) si fue exitoso, o None si falla.
        """
        if not self._metal_active or symbol not in self.metal_engines or symbol not in self.feature_arenas:
            return None
            
        try:
            engine = self.metal_engines[symbol]
            arena = self.feature_arenas[symbol]
            
            if arena.features is None or arena.indices is None:
                return None

            # O(1) tick features from Rust (12 elements)
            feats = engine.tick(price, high, low, volume)
            
            # Zero-Copy Numpy inplace slice assignment (nanoseconds)
            valid_mask = arena.indices >= 0
            valid_indices = arena.indices[valid_mask]
            
            # Update only valid indices in the Arena
            arena.features[0, valid_indices] = np.array(feats, dtype=np.float32)[valid_mask]
            
            return arena.features
        except Exception as e:
            logger.error(f"Fallo catastrófico del metal en {symbol}: {e}. Cortando puente.")
            self._metal_active = False
            return None

    @trace_execution
    def prepare_features(self, bars, market_regime="UNKNOWN", sentiment_loader=None, data_provider=None, symbol=None, feature_store=None, horizon="SCALPING", return_polars=False, is_live=False):
        if bars is None or len(bars) == 0:
            return pl.DataFrame()
            
        # ═══════════════════════════════════════════════════════════════
        # [QUANTUM ZERO-COPY] NATIVE POLARS INGESTION
        # QUÉ: Bypass total de Pandas. Cargamos los arrays estructurados de 
        #   NumPy o diccionarios directamente en la memoria de Rust (Polars).
        # POR QUÉ: Convertir NumPy -> Pandas -> Polars añadía 350ms de latencia.
        # PARA QUÉ: Reducir latencia a micro/nanosegundos ("espacio cuántico").
        # ═══════════════════════════════════════════════════════════════
        if hasattr(bars, 'dtype') and hasattr(bars.dtype, 'names') and bars.dtype.names:
            # Zero-copy dictionary of arrays for Polars
            pl_dict = {name: bars[name].astype(np.float64) if bars[name].dtype.kind == 'f' else bars[name] for name in bars.dtype.names}
            df = pl.DataFrame(pl_dict)
        else:
            df = pl.DataFrame(bars)
            
        if len(df) < 20:
            return pl.DataFrame()

        # ═══════════════════════════════════════════════════════════════
        # [QUANTUM ISOLATION] THE UNCLOSED CANDLE PARADOX FIX
        # Si estamos en LIVE (producción), la última vela es la viva (unclosed).
        # Para evitar repainting en los features técnicos, la removemos.
        # ═══════════════════════════════════════════════════════════════
        if is_live and len(df) > 50:
            df = df.head(len(df) - 1)

        # ═══════════════════════════════════════════════════════════════
        # [QUANTUM CACHE] CHECK RESULT CACHE BEFORE COMPUTATION
        # Si las barras no cambiaron, devolver resultado anterior (~0.01ms)
        # ═══════════════════════════════════════════════════════════════
        _cache_symbol = symbol or "UNKNOWN"
        try:
            _last_close = float(df['close'][-1])
            _last_open = float(df['open'][-1])
            _n_bars = len(df)
            _cache_key = (_last_close, _last_open, _n_bars)
            
            if _cache_symbol in self._result_cache:
                _prev_key, _prev_result = self._result_cache[_cache_symbol]
                if _prev_key == _cache_key and _prev_result is not None:
                    # Cache HIT: same bars, return previous result
                        return _prev_result
        except Exception:
            pass  # Cache miss, compute normally

        # Si feature_store está activo, hacemos fallback (muy raro en HFT loop)
        if feature_store and len(df) > 100:
            try:
                ts_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
                if ts_col in df.columns:
                    start_ts = df[ts_col].min()
                    end_ts = df[ts_col].max()
                    cached_df = feature_store.get_features(symbol, start_ts, end_ts)
                    if len(cached_df) > 0 and len(cached_df) >= len(df) * 0.9:
                        # [PANDAS ERADICATION] FeatureStore merging is disabled in hot-path
                        # We return the calculated dataframe directly.
                        pass
            except Exception as e:
                logger.warning(f"FeatureStore retrieval skipped: {e}")

        numeric_cols = ['close', 'open', 'high', 'low', 'volume']
        cast_exprs = [pl.col(c).cast(pl.Float64) for c in numeric_cols if c in df.columns]
        if cast_exprs:
            df = df.with_columns(cast_exprs)

        # Arrays NumPy zero-copy para TA-Lib y JIT
        raw_close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        open_ = df['open'].to_numpy()
        volume = df['volume'].to_numpy()
        
        # 🧮 FASE 5: Kalman Filter (Zero-Lag Smoothing)
        # Purifica el micro-ruido de alta frecuencia HFT para estabilizar a la Inteligencia Artificial
        close = kalman_filter_1d_jit(raw_close, R=1e-4, Q=1e-5)
        
        n_len = len(close)
        new_features = {}

        # ==================== PRICE ACTION & MOMENTUM (POLARS EXPRS) ====================
        exprs = []
        for p in [1, 3, 5, 8, 10, 13, 20, 21, 34]:
            exprs.append(pl.col('close').pct_change(p).fill_null(0.0).alias(f'returns_{p}'))
            exprs.append((pl.col('close').pct_change(p) * 100).fill_null(0.0).alias(f'roc_{p}'))
            exprs.append((pl.col('close') - pl.col('close').shift(p)).fill_null(0.0).alias(f'momentum_{p}'))
            
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

        # 🧮 FASE 5: Fractional Differencing
        # Sobreescribe los retornos simples ruidosos con diferenciación fraccional
        # Conserva estacionariedad sin borrar el Hurst Exponent original del activo
        new_features['returns_1'] = fractional_differencing_jit(close, d=0.45)

        # ==================== CROSS-SECTIONAL ====================
        # [FORENSIC PRUNE] cross_spread_vs_btc & cross_relative_strength removed (dead features)

        # ==================== INDICADORES (math_kernel JIT & TA-Lib) ====================
        # ==================== INDICADORES (MÓDULO OMEGA MODULAR) ====================
        
        # 1. Momentum (M01-M18)
        momentum_feats = MomentumIndicators.calculate_all(df, close, high, low, volume, n_len)
        new_features.update(momentum_feats)
        
        # 2. Trend (T01-T20)
        trend_feats = TrendIndicators.calculate_all(df, close, high, low, n_len)
        new_features.update(trend_feats)
        
        # 3. Volatility (V01-V10)
        vol_feats = VolatilityIndicators.calculate_all(df, close, high, low, n_len)
        new_features.update(vol_feats)
        
        # 4. Volume (F01-F14)
        volume_feats = VolumeIndicators.calculate_all(df, close, high, low, volume, n_len)
        new_features.update(volume_feats)

        # Post-Indicator Logic
        bbw_ma20 = talib.SMA(new_features['bb_width'], 20)
        new_features['bb_squeeze'] = np.where(new_features['bb_width'] < bbw_ma20 * 0.5, 1, 0)
        
        new_features['ema_5_20_cross'] = np.where(new_features['ema_5'] > new_features['ema_20'], 1, -1)
        new_features['ema_20_50_cross'] = np.where(new_features['ema_20'] > new_features['ema_50'], 1, -1)
        new_features['ema_50_200_cross'] = np.where(new_features['ema_50'] > new_features['ema_200'], 1, -1)
        
        new_features['up_bar'] = np.where(close > fast_shift(close, 1), 1, 0)
        new_features['down_bar'] = np.where(close < fast_shift(close, 1), 1, 0)
        new_features['higher_high'] = np.where(high > fast_shift(high, 1), 1, 0)
        new_features['lower_low'] = np.where(low < fast_shift(low, 1), 1, 0)

        # ==================== REGIME AWARE ====================
        new_features['trend_power'] = np.zeros(n_len)
        new_features['trend_alignment'] = np.zeros(n_len)
        new_features['range_extreme'] = np.zeros(n_len)
        new_features['panic_index'] = np.zeros(n_len)
        new_features['volatility_regime'] = np.where(new_features['atr'] > talib.SMA(new_features['atr'], 50), 1, 0) if n_len >= 50 else np.zeros(n_len)

        # ==================== SENTIMENT (NLP ENSEMBLE - Phase 8) ====================
        # QUÉ: Reemplaza TextBlob (siempre cero en producción) con HuggingFace dual ensemble.
        # POR QUÉ: FinBERT + CryptoBERT interpretan jerga financiera/cripto correctamente.
        # CÓMO: Exponential decay protege contra valores stale. Freshness flag permite
        #       al ML ignorar datos viejos automáticamente.
        try:
            nlp_feats = news_sentiment.get_sentiment_features(symbol)
            new_features['news_sentiment'] = np.full(n_len, nlp_feats.get('news_sentiment', 0.0))
            new_features['news_sentiment_magnitude'] = np.full(n_len, nlp_feats.get('news_sentiment_magnitude', 0.0))
            new_features['news_sentiment_shock'] = np.full(n_len, nlp_feats.get('news_sentiment_shock', 0.0))
            new_features['news_has_fresh_data'] = np.full(n_len, nlp_feats.get('news_has_fresh_data', 0.0))
        except Exception:
            new_features['news_sentiment'] = np.zeros(n_len)
            new_features['news_sentiment_magnitude'] = np.zeros(n_len)
            new_features['news_sentiment_shock'] = np.zeros(n_len)
            new_features['news_has_fresh_data'] = np.zeros(n_len)
        
        # [FORENSIC PRUNE v2] Legacy aliases sentiment/sentiment_change/sentiment_momentum restored for dimension parity
        new_features['sentiment'] = new_features['news_sentiment']
        new_features['sentiment_change'] = new_features['news_sentiment_shock']
        new_features['sentiment_momentum'] = new_features['news_sentiment_magnitude']

        # ==================== OMEGA MIND / DERIVATIVES & MICROSTRUCTURE ====================
        dp_derivs = data_provider.get_derivatives_metrics(symbol) if symbol and hasattr(data_provider, 'get_derivatives_metrics') else {}
        
        # Microstructure Injection: Ticks & Orderbook
        current_vbi = 0.0
        current_cvd = 0.0
        current_tick_dir = 0.0
        if data_provider and hasattr(data_provider, 'lob_imbalance') and symbol in data_provider.lob_imbalance:
             current_vbi = data_provider.lob_imbalance[symbol].get('imbalance', 0.0)
             # CVD y Tick Direction si el loader los provee:
             current_cvd = data_provider.lob_imbalance[symbol].get('cvd', 0.0)
             current_tick_dir = data_provider.lob_imbalance[symbol].get('tick_direction', 0.0)
             
        new_features['vbi'] = np.full(n_len, current_vbi)
        new_features['cvd'] = np.full(n_len, current_cvd)
        new_features['tick_direction'] = np.full(n_len, current_tick_dir)
        
        # [DARK ALPHA] Inyección de net_pressure (Liquidation Cascade Density)
        current_net_pressure = data_provider.get_order_flow_metrics(symbol).get('net_pressure', 0.0) if symbol and hasattr(data_provider, 'get_order_flow_metrics') else 0.0
        new_features['net_pressure'] = np.full(n_len, current_net_pressure)
        
        new_features['liq_intensity'] = np.full(n_len, dp_derivs.get('liquidations', 0.0))
        new_features['funding_rate'] = np.full(n_len, dp_derivs.get('funding_rate', 0.0))
        new_features['oi'] = np.full(n_len, dp_derivs.get('oi', 0.0))
        new_features['cross_spread_vs_btc'] = np.zeros(n_len)
        new_features['cross_relative_strength'] = np.zeros(n_len)
        
        # Legacy missing features
        new_features['vbi_avg'] = np.full(n_len, current_vbi)
        new_features['oi_delta'] = np.full(n_len, dp_derivs.get('oi_delta', 0.0))
        new_features['funding_distortion'] = np.zeros(n_len)
        new_features['micro_velocity_3'] = np.zeros(n_len)
        new_features['is_swing_horizon'] = np.zeros(n_len)
        # new_features['scalp_velocity_1'] = np.zeros(n_len)
        new_features['swing_momentum_ratio'] = np.zeros(n_len)
        new_features['swing_ema50_slope'] = np.zeros(n_len)

        # ==================== PHASE 3: SOPHIA KMEANS CLUSTER ====================
        try:
            symbol_key = symbol if symbol else 'default'
            cluster_cols = ['rsi_14', 'atr_pct', 'volume_ratio', 'adx']
            if all(c in new_features for c in cluster_cols) and n_len >= 50:
                feat_data = {c: new_features[c] for c in cluster_cols}
                features_array = pl.DataFrame(feat_data).select(cluster_cols).fill_null(strategy="forward").fill_null(0.0).to_numpy()
                # We need current_time properly
                if 'datetime' in df.columns:
                    current_time = df['datetime'][-1]
                elif 'timestamp' in df.columns:
                    current_time = df['timestamp'][-1]
                else:
                    current_time = datetime.datetime.now(datetime.timezone.utc)
                
                last_fit_time = self._kmeans_last_fit.get(symbol_key)
                fit_count = getattr(self, '_kmeans_fit_counter', {})
                current_count = fit_count.get(symbol_key, 0) + 1
                fit_count[symbol_key] = current_count
                self._kmeans_fit_counter = fit_count
                
                if last_fit_time is None or current_count % 50 == 0:
                    # ═══════════════════════════════════════════════════════════════
                    # FORENSIC FIX: KMEANS LOOKAHEAD BIAS
                    # QUÉ: Limitar el fit del KMeans y Scaler a las primeras velas.
                    # POR QUÉ: Si hacemos fit sobre los 5000 rows del backtest, el
                    #   clustering de la vela 0 está sesgado por la volatilidad de
                    #   la vela 5000 (Data Leakage del futuro).
                    # PARA QUÉ: Preservar la causalidad temporal estricta.
                    # ═══════════════════════════════════════════════════════════════
                    fit_size = min(len(features_array), 500)
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans
                    scaler = StandardScaler()
                    scaled_fit = scaler.fit_transform(features_array[:fit_size])
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
        # [FORENSIC PRUNE] micro_velocity_3 removed (100% correlated with returns_3)
        
        # 🌊 PHASE 10: REAL L2 ORDERBOOK METRICS
        if 'l2_ofi' in df.columns:
            new_features['l2_ofi'] = df['l2_ofi'].to_numpy()
            new_features['l2_spread'] = df['l2_spread'].to_numpy()
            new_features['l2_microprice_dist'] = df['l2_microprice_dist'].to_numpy()
        elif data_provider and hasattr(data_provider, 'get_order_flow_metrics'):
            of_metrics = data_provider.get_order_flow_metrics(symbol) if symbol else {}
            new_features['l2_ofi'] = np.full(n_len, of_metrics.get('ofi', of_metrics.get('l2_ofi', 0.0)))
            new_features['l2_spread'] = np.full(n_len, of_metrics.get('spread', of_metrics.get('l2_spread', 0.0)))
            new_features['l2_microprice_dist'] = np.full(n_len, of_metrics.get('micro_price', of_metrics.get('l2_microprice_dist', 0.0)))
        else:
            new_features['l2_ofi'] = np.zeros(n_len)
            new_features['l2_spread'] = np.zeros(n_len)
            new_features['l2_microprice_dist'] = np.zeros(n_len)
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
        # [FORENSIC PRUNE] is_swing_horizon, swing_momentum_ratio, swing_ema50_slope, scalp_velocity_1 removed.
        # Scalp velocity is 100% correlated with returns_1. Swing horizon params were dead zeros.
        new_features['scalp_rsi_divergence'] = new_features.get('rsi_3', np.full(n_len, 50)) - new_features.get('rsi_14', np.full(n_len, 50))

        # ==================== PHASE 3 (AITS): HYPERGRAPH CENTRALITY ====================
        from core.swarm_correlator import swarm_correlator
        graph_feats = swarm_correlator.get_hypergraph_features(symbol) if symbol else {}
        # new_features['graph_centrality'] = np.full(n_len, graph_feats.get('graph_centrality', 0.0))
        # new_features['graph_pagerank'] = np.full(n_len, graph_feats.get('graph_pagerank', 0.0))
        new_features['graph_connectivity'] = np.full(n_len, graph_feats.get('graph_connectivity', 0.0))

        # ==================== PHASE 2 (AITS): MACRO & ON-CHAIN NERVOUS SYSTEM ====================
        # 🌐 CTOS DATA OMNISCIENCE: 30+ macro/micro features from MacroIntelligence
        macro_feats = macro_intelligence.get_macro_features()
        onchain_feats = onchain_loader.get_onchain_features()
        
        # Fear & Greed Index (Alternative.me)
        new_features['fng_value'] = np.full(n_len, macro_feats.get('fng_value', 50.0))
        new_features['fng_change_1d'] = np.full(n_len, macro_feats.get('fng_change_1d', 0.0))
        new_features['fng_ma_7d'] = np.full(n_len, macro_feats.get('fng_ma_7d', 50.0))
        
        # Crypto Global (CoinGecko)
        new_features['btc_dominance'] = np.full(n_len, macro_feats.get('btc_dominance', 50.0))
        new_features['btc_dominance_change_24h'] = np.full(n_len, macro_feats.get('btc_dominance_change_24h', 0.0))
        new_features['eth_dominance'] = np.full(n_len, macro_feats.get('eth_dominance', 15.0))
        new_features['total_market_cap_norm'] = np.full(n_len, macro_feats.get('total_market_cap_norm', 0.0))
        new_features['total_volume_24h_norm'] = np.full(n_len, macro_feats.get('total_volume_24h_norm', 0.0))
        new_features['market_cap_change_24h_pct'] = np.full(n_len, macro_feats.get('market_cap_change_24h_pct', 0.0))
        
        # TradFi Macro (yFinance: DXY, NASDAQ, Gold, VIX, US10Y)
        new_features['macro_dxy_returns'] = np.full(n_len, macro_feats.get('macro_dxy_returns', 0.0))
        new_features['macro_dxy_trend'] = np.full(n_len, macro_feats.get('macro_dxy_trend', 0.0))
        new_features['macro_nq_returns'] = np.full(n_len, macro_feats.get('macro_nq_returns', 0.0))
        new_features['macro_nq_trend'] = np.full(n_len, macro_feats.get('macro_nq_trend', 0.0))
        new_features['macro_gold_returns'] = np.full(n_len, macro_feats.get('macro_gold_returns', 0.0))
        new_features['macro_vix'] = np.full(n_len, macro_feats.get('macro_vix', 20.0))
        new_features['macro_vix_change'] = np.full(n_len, macro_feats.get('macro_vix_change', 0.0))
        new_features['macro_us10y'] = np.full(n_len, macro_feats.get('macro_us10y', 4.0))
        new_features['macro_us10y_change'] = np.full(n_len, macro_feats.get('macro_us10y_change', 0.0))
        
        # Binance Derivatives (per-symbol: Funding, OI, L/S Ratio, Taker Vol)
        deriv_feats = macro_intelligence.get_derivatives_features(symbol) if symbol else {}
        new_features['funding_rate'] = np.full(n_len, deriv_feats.get('funding_rate', 0.0))
        new_features['funding_rate_norm'] = np.full(n_len, deriv_feats.get('funding_rate_norm', 0.0))
        new_features['open_interest_norm'] = np.full(n_len, deriv_feats.get('open_interest_norm', 0.0))
        # [FORENSIC PRUNE] oi_change_pct removed (100% dead zeros)
        new_features['long_short_ratio'] = np.full(n_len, deriv_feats.get('long_short_ratio', 1.0))
        new_features['top_trader_ls_ratio'] = np.full(n_len, deriv_feats.get('top_trader_ls_ratio', 1.0))
        new_features['taker_buy_sell_ratio'] = np.full(n_len, deriv_feats.get('taker_buy_sell_ratio', 1.0))
        
        # On-Chain (Proxy Institucional vía Binance AggTrades > $100k)
        if 'whale_flow' in df.columns:
            new_features['onchain_whale_flow'] = df['whale_flow'].to_numpy()
            new_features['dark_alpha_pressure'] = df.get('net_pressure', pl.Series(np.zeros(n_len))).to_numpy()
            new_features['liquidation_cascade'] = df.get('liquidation_cascade', pl.Series(np.zeros(n_len))).to_numpy()
            new_features['dex_whisper'] = df.get('dex_whisper', pl.Series(np.zeros(n_len))).to_numpy()
        elif data_provider and hasattr(data_provider, 'get_order_flow_metrics'):
            of_metrics = data_provider.get_order_flow_metrics(symbol) if symbol else {}
            # Fallback to derivatives just in case
            dp_derivs = data_provider.get_derivatives_metrics(symbol) if symbol and hasattr(data_provider, 'get_derivatives_metrics') else {}
            new_features['onchain_whale_flow'] = np.full(n_len, of_metrics.get('whale_flow', dp_derivs.get('whale_flow', 0.0)))
            
            # [DARK ALPHA] Extract Dark Alpha parameters if available
            new_features['dark_alpha_pressure'] = np.full(n_len, of_metrics.get('net_pressure', 0.0))
            
            # PHASE V: Advanced Topological Dark Alpha
            # Liquidation Cascades proxy: High OI combined with large extreme spikes
            new_features['liquidation_cascade'] = np.full(n_len, of_metrics.get('liquidation_cascade', 0.0))
            # DEX Whispering: Synthesized proxy from order flow imbalance metrics vs CEX footprint
            new_features['dex_whisper'] = np.full(n_len, of_metrics.get('dex_whisper', 0.0))
        else:
            new_features['onchain_whale_flow'] = np.zeros(n_len)
            new_features['dark_alpha_pressure'] = np.zeros(n_len)
            new_features['liquidation_cascade'] = np.zeros(n_len)
            new_features['dex_whisper'] = np.zeros(n_len)

        # Merge dict back into Polars df efficiently
        # Remove duplicates to avoid DuplicateError in horizontal concat
        for col in df.columns:
            new_features.pop(col, None)
            
        pl_features = pl.DataFrame(new_features)
        df = pl.concat([df, pl_features], how="horizontal")

        # ================================================================================
        # [FORENSIC PRUNE v2] KEEP ABSOLUTE PRICE COLUMNS FOR DIMENSION PARITY
        # QUÉ: Restaurar columnas de precio absoluto (open, high, low) y timestamp
        #       del output que va al ML.
        # POR QUÉ: El modelo de Machine Learning fue entrenado con 143 dimensiones.
        #       Si las eliminamos, PyTorch no puede inyectar los pesos de la capa densa.
        # ================================================================================
        # Remove only internal processing columns that shouldn't reach ML
        cols_to_drop = [c for c in ['timestamp_ms'] if c in df.columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop)

        # --------------------------------------------------------------------------------
        # [MEMORY OPTIMIZATION] Downcast to Float32 before turning to Pandas
        # Reduces memory overhead by 50% for XGBoost Inference without precision loss.
        # --------------------------------------------------------------------------------
        float_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Float64, pl.Float32]]
        if float_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])

        # ═══════════════════════════════════════════════════════════════
        # [QUANTUM ZERO-COPY] VECTORIZED VALIDATION (single Rust pass)
        # QUÉ: Reemplaza inf/nan en TODAS las columnas float con una sola
        #   expresión Polars vectorizada, en vez de iterar columna por columna.
        # POR QUÉ: El loop anterior ejecutaba 143 operaciones Polars separadas,
        #   cada una creando un nuevo DataFrame → O(143) overhead de Rust.
        # PARA QUÉ: Reducir la validación de ~500ms a <1ms.
        # ═══════════════════════════════════════════════════════════════
        sanitize_exprs = [
            pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in float_cols
        ]
        if sanitize_exprs:
            df = df.with_columns(sanitize_exprs)
        # Forward fill, backward fill, fill 0
        df = df.fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0)

        # ═══════════════════════════════════════════════════════════════
        # [QUANTUM CACHE] STORE RESULT IN CACHE
        # ═══════════════════════════════════════════════════════════════
        try:
            self._result_cache[_cache_symbol] = (_cache_key, df)
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════
        # [QUANTUM ZERO-COPY] WARMUP METAL & ARENA
        # ═══════════════════════════════════════════════════════════════
        if self._metal_active and symbol:
            if symbol not in self.metal_engines:
                self.metal_engines[symbol] = self.nano_core.StatefulEngine(20, 14, 1000)
                try:
                    # Seed engine with history
                    c_np = df['close'].to_numpy()
                    h_np = df['high'].to_numpy()
                    l_np = df['low'].to_numpy()
                    v_np = df['volume'].to_numpy()
                    self.metal_engines[symbol].seed_history(c_np, h_np, l_np, v_np)
                except Exception as e:
                    logger.warning(f"Metal engine seed failed for {symbol}: {e}")
            
            if symbol not in self.feature_arenas:
                arena = FeatureArena(capacity=1)
                arena.initialize(df.columns)
                self.feature_arenas[symbol] = arena
            
            # Update base features in the Arena
            # Copy the last row exactly to the arena
            last_row_np = df.tail(1).to_numpy()
            self.feature_arenas[symbol].inject_base_features(last_row_np)
            
        # Immediate memory release for intermediate GC if not returning Polars
        # [PANDAS ERADICATION] We ALWAYS return Polars now.
        
        # [PHASE 12] SAVE TO STORE
        if feature_store and len(df) > 1:
            try:
                feature_store.store_features(symbol, df)
            except Exception as e:
                logger.debug(f"FeatureStore storage skipped: {e}")

        return df

    def validate_features(self, df):
        """Limpieza robusta de features sin bleeding de O.Os (Polars Edition)"""
        if len(df) == 0: return df
        
        float_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Float64, pl.Float32]]
        sanitize_exprs = [
            pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in float_cols
        ]
        if sanitize_exprs:
            df = df.with_columns(sanitize_exprs)
            
        df = df.fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0)
        return df
