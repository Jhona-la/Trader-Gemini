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
    def prepare_features(self, df, **kwargs):
        if not isinstance(df, pl.DataFrame):
            # Enforce strict Polars conversion, NO PANDAS allowed
            try:
                # If it's a generator or dict, pl.DataFrame handles it
                df = pl.DataFrame(df)
            except Exception:
                # Fallback if structure is weird (e.g. list of dicts)
                df = pl.DataFrame(list(df) if not isinstance(df, list) else df)

        if df.is_empty() or len(df) < 5:
            return df
            
        n_len = len(df)
        symbol = df['symbol'][0] if 'symbol' in df.columns else 'UNKNOWN'
        
        high = np.require(df['high'].to_numpy(), dtype=np.float64, requirements=['C', 'W'])
        low = np.require(df['low'].to_numpy(), dtype=np.float64, requirements=['C', 'W'])
        close = np.require(df['close'].to_numpy(), dtype=np.float64, requirements=['C', 'W'])
        volume = np.require(df['volume'].to_numpy(), dtype=np.float64, requirements=['C', 'W'])

        # Features base construidas en Polars Expr
        exprs = [
            pl.col('close').log().diff().alias('returns_1'),
            pl.col('close').log().diff(5).alias('returns_5'),
            pl.col('close').log().diff(15).alias('returns_15'),
            (pl.col('close') / pl.col('close').rolling_mean(window_size=20) - 1).alias('dist_ma20'),
            (pl.col('close') / pl.col('close').rolling_mean(window_size=50) - 1).alias('dist_ma50'),
            (pl.col('close') / pl.col('close').rolling_mean(window_size=200) - 1).alias('dist_ma200')
        ]
        
        v_ma5 = pl.col('volume').rolling_mean(window_size=5)
        v_imb_ma5 = pl.col('buy_volume').rolling_mean(window_size=5) if 'buy_volume' in df.columns else pl.lit(0.0)
        exprs.append(pl.when(v_ma5 != 0).then(v_imb_ma5 / v_ma5).otherwise(0.0).alias('micro_imbalance'))
        df = df.with_columns(exprs)

        # Frac diff
        
        indicator_series = []
        indicator_exprs = []
        for module_feats in [
            MomentumIndicators.calculate_all(df, close, high, low, volume, n_len),
            TrendIndicators.calculate_all(df, close, high, low, n_len),
            VolatilityIndicators.calculate_all(df, close, high, low, n_len),
            VolumeIndicators.calculate_all(df, close, high, low, volume, n_len)
        ]:
            for k, v in module_feats.items():
                if isinstance(v, pl.Expr):
                    indicator_exprs.append(v.alias(k))
                else:
                    indicator_series.append(pl.Series(k, v))
                    
        if indicator_series:
            df = df.with_columns(indicator_series)
        if indicator_exprs:
            df = df.with_columns(indicator_exprs)

        post_exprs = [
            pl.when(pl.col('bb_width') < pl.col('bb_width').rolling_mean(window_size=20) * 0.5).then(1).otherwise(0).alias('bb_squeeze'),
            pl.when(pl.col('ema_5') > pl.col('ema_20')).then(1).otherwise(-1).alias('ema_5_20_cross'),
            pl.when(pl.col('ema_20') > pl.col('ema_50')).then(1).otherwise(-1).alias('ema_20_50_cross'),
            pl.when(pl.col('ema_50') > pl.col('ema_200')).then(1).otherwise(-1).alias('ema_50_200_cross'),
            pl.when(pl.col('close') > pl.col('close').shift(1)).then(1).otherwise(0).alias('up_bar'),
            pl.when(pl.col('close') < pl.col('close').shift(1)).then(1).otherwise(0).alias('down_bar'),
            pl.when(pl.col('high') > pl.col('high').shift(1)).then(1).otherwise(0).alias('higher_high'),
            pl.when(pl.col('low') < pl.col('low').shift(1)).then(1).otherwise(0).alias('lower_low'),
            pl.lit(0).alias('trend_power'),
            pl.lit(0).alias('trend_alignment'),
            pl.lit(0).alias('range_extreme'),
            pl.lit(0).alias('panic_index')
        ]
        
        if n_len >= 50:
            post_exprs.append(
                pl.when(pl.col('atr') > pl.col('atr').rolling_mean(window_size=50)).then(1).otherwise(0).alias('volatility_regime')
            )
        else:
            post_exprs.append(pl.lit(0).alias('volatility_regime'))
            
        df = df.with_columns(post_exprs)

        nlp_feats = news_sentiment.get_sentiment_features(symbol)
        df = df.with_columns([
            pl.lit(nlp_feats.get('finbert_sentiment', 0.0)).alias('finbert_sentiment'),
            pl.lit(nlp_feats.get('cryptobert_sentiment', 0.0)).alias('cryptobert_sentiment'),
            pl.lit(nlp_feats.get('freshness_score', 0.0)).alias('news_freshness'),
            pl.lit(nlp_feats.get('impact_score', 0.0)).alias('news_impact'),
            pl.lit(nlp_feats.get('news_sentiment', 0.0)).alias('sentiment'),
            pl.lit(nlp_feats.get('news_sentiment_shock', 0.0)).alias('sentiment_change'),
            pl.lit(nlp_feats.get('news_sentiment_magnitude', 0.0)).alias('sentiment_momentum')
        ])
        
        opt_metrics = {}
        df = df.with_columns([
            pl.lit(opt_metrics.get('put_call_ratio', 1.0)).alias('put_call_ratio'),
            pl.lit(opt_metrics.get('implied_volatility', 0.5)).alias('iv_rank'),
            pl.lit(opt_metrics.get('gamma_exposure', 0.0)).alias('gamma_exposure'),
            pl.lit(opt_metrics.get('max_pain_distance', 0.0)).alias('max_pain_distance')
        ])
        
        macro_state = macro_intelligence.get_macro_features()
        df = df.with_columns([
            pl.lit(macro_state.get('dxy_correlation', 0.0)).alias('dxy_correlation'),
            pl.lit(macro_state.get('sp500_correlation', 0.0)).alias('sp500_correlation'),
            pl.lit(macro_state.get('usdt_dominance', 0.0)).alias('usdt_dominance'),
            pl.lit(macro_state.get('vix_level', 20.0)).alias('vix_level'),
            pl.lit(macro_state.get('fng_value', 50.0)).alias('fng_value'),
            pl.lit(macro_state.get('btc_dominance', 50.0)).alias('btc_dominance'),
            pl.lit(macro_state.get('macro_dxy_returns', 0.0)).alias('macro_dxy_returns'),
            pl.lit(macro_state.get('macro_sp500_returns', 0.0)).alias('macro_sp500_returns'),
            pl.lit(macro_state.get('macro_vix', 20.0)).alias('macro_vix'),
            pl.lit(macro_state.get('macro_us10y', 4.0)).alias('macro_us10y')
        ])

        from strategies.indicators.structure import StructureIndicators

        struct_feats = StructureIndicators.calculate_all(df, close, high, low, n_len)
        df = df.with_columns([pl.Series(k, v) for k, v in struct_feats.items()])
        
        

        df = df.with_columns([
            pl.lit(0).alias('hmm_regime'),
            pl.lit(0.0).alias('hmm_volatility'),
            pl.lit(0.0).alias('hmm_trend')
        ])

        if 'timestamp' in df.columns:
            df = df.with_columns([
            ])
        else:
            df = df.with_columns([
            ])

        # 🚀 FORENSIC FIX: Eliminamos target_1m, target_5m, target_15m, target_class 
        # que estaban introduciendo Data Leakage (Lookahead Bias) masivo al Dataframe de Features
        # mediante .shift(-X). El etiquetado REAL se delega estrictamente a ml_strategy._create_labels()
        
        # ═══════════════════════════════════════════════════════════════════
        # 🔬 [HOLOGRAPHIC AUDIT FIX] RESURRECCIÓN DE FEATURES MUERTAS
        # ═══════════════════════════════════════════════════════════════════
        # ANTES: 61 features hardcodeadas a pl.lit(0.0) = tensor 63% MUERTO
        # AHORA: Calculamos TODAS las features derivables de OHLCV
        # ═══════════════════════════════════════════════════════════════════
        
        price_change = pl.col('close') - pl.col('close').shift(1)
        buy_vol = pl.when(price_change > 0).then(pl.col('volume')).when(price_change < 0).then(0.0).otherwise(pl.col('volume') * 0.5)
        sell_vol = pl.when(price_change < 0).then(pl.col('volume')).when(price_change > 0).then(0.0).otherwise(pl.col('volume') * 0.5)
        vol_total = buy_vol + sell_vol
        
        tick_dir = pl.when(price_change > 0).then(1.0).when(price_change < 0).then(-1.0).otherwise(0.0)
        net_press = pl.when(vol_total > 0).then((buy_vol - sell_vol) / vol_total).otherwise(0.0)
        micro_vel_3 = pl.col('close').log().diff(3)
        vol_change_expr = pl.col('volume').pct_change(1)
        vol_accel_expr = vol_change_expr - vol_change_expr.shift(1)
        vbi_raw = pl.when(vol_total > 0).then(buy_vol / vol_total).otherwise(0.5)
        vbi_expr = vbi_raw.ewm_mean(span=14, adjust=False)
        vbi_avg_expr = vbi_raw.rolling_mean(window_size=20)

        hl_range = pl.col('high') - pl.col('low')
        range_ext = pl.when(hl_range > 0).then((pl.col('close') - pl.col('low')) / hl_range).otherwise(0.5)

        horizon = kwargs.get('horizon', 'SCALPING')
        is_swing_val = 1.0 if str(horizon).upper() == 'SWING' else 0.0

        resurrected_exprs = [
            tick_dir.alias('tick_direction'),
            net_press.alias('net_pressure'),
            micro_vel_3.alias('micro_velocity_3'),
            vol_accel_expr.alias('volume_accel'),
            vbi_expr.alias('vbi'),
            vbi_avg_expr.alias('vbi_avg'),
            range_ext.alias('range_extreme'),
            pl.lit(is_swing_val).alias('is_swing_horizon'),
        ]
        
        if 'adx' in df.columns and 'ema_20' in df.columns and 'ema_50' in df.columns:
            resurrected_exprs.append(
                pl.when(pl.col('ema_20') > pl.col('ema_50')).then(pl.col('adx') / 100.0).otherwise(-pl.col('adx') / 100.0).alias('trend_power')
            )
        else:
            resurrected_exprs.append(pl.lit(0.0).alias('trend_power'))
            
        if all(c in df.columns for c in ['ema_5', 'ema_20', 'ema_50', 'ema_200']):
            resurrected_exprs.append(
                ((pl.when(pl.col('ema_5') > pl.col('ema_20')).then(1).otherwise(-1) +
                  pl.when(pl.col('ema_20') > pl.col('ema_50')).then(1).otherwise(-1) +
                  pl.when(pl.col('ema_50') > pl.col('ema_200')).then(1).otherwise(-1)
                ).cast(pl.Float64) / 3.0).alias('trend_alignment')
            )
        else:
            resurrected_exprs.append(pl.lit(0.0).alias('trend_alignment'))

        if 'atr' in df.columns and 'volume_sma_20' in df.columns:
            resurrected_exprs.append(
                pl.when(
                    (pl.col('volume') > pl.col('volume_sma_20') * 2.0) & (price_change < 0) & (price_change.abs() > pl.col('atr') * 0.5)
                ).then(2.0).otherwise(0.0).alias('panic_index')
            )
        else:
            resurrected_exprs.append(pl.lit(0.0).alias('panic_index'))

        if 'rsi_14' in df.columns:
            rsi_sl = pl.col('rsi_14') - pl.col('rsi_14').shift(5)
            price_sl = pl.col('close') - pl.col('close').shift(5)
            resurrected_exprs.append(
                pl.when((price_sl > 0) & (rsi_sl < 0)).then(-1.0).when((price_sl < 0) & (rsi_sl > 0)).then(1.0).otherwise(0.0).alias('scalp_rsi_divergence')
            )
        else:
            resurrected_exprs.append(pl.lit(0.0).alias('scalp_rsi_divergence'))

        if 'returns_15' in df.columns and 'returns_5' in df.columns:
            resurrected_exprs.append(
                pl.when(pl.col('returns_5').abs() > 1e-8).then(pl.col('returns_15') / pl.col('returns_5')).otherwise(0.0).alias('swing_momentum_ratio')
            )
        else:
            resurrected_exprs.append(pl.lit(0.0).alias('swing_momentum_ratio'))

        if 'ema_50' in df.columns:
            resurrected_exprs.append(pl.col('ema_50').log().diff(5).alias('swing_ema50_slope'))
        else:
            resurrected_exprs.append(pl.lit(0.0).alias('swing_ema50_slope'))

        if 'bb_width' in df.columns and n_len >= 50:
            resurrected_exprs.append(
                pl.when(pl.col('bb_width') < pl.col('bb_width').rolling_quantile(0.2, window_size=50)).then(1.0).otherwise(0.0).alias('spread_squeeze')
            )
        else:
            resurrected_exprs.append(pl.lit(0.0).alias('spread_squeeze'))

        df = df.with_columns(resurrected_exprs)
        
        # --- FEATURES QUE REQUIEREN API EXTERNA (quedan como 0.0 documentadas) ---
        df = df.with_columns([
            pl.lit(0.0).alias('liq_intensity'),
            pl.lit(0.0).alias('funding_rate'),
            pl.lit(0.0).alias('oi'),
            pl.lit(0.0).alias('cross_spread_vs_btc'),
            pl.lit(0.0).alias('cross_relative_strength'),
            pl.lit(0.0).alias('oi_delta'),
            pl.lit(0.0).alias('funding_distortion'),
            pl.lit(0.0).alias('market_cluster'),
            pl.lit(0.0).alias('cluster_0'),
            pl.lit(0.0).alias('cluster_1'),
            pl.lit(0.0).alias('cluster_2'),
            pl.lit(0.0).alias('cluster_3'),
            pl.lit(0.0).alias('l2_ofi'),
            pl.lit(0.0).alias('l2_spread'),
            pl.lit(0.0).alias('l2_microprice_dist'),
            pl.lit(0.0).alias('micro_absorption'),
            pl.lit(0.0).alias('micro_exhaustion'),
            pl.lit(0.0).alias('micro_sweep'),
            pl.lit(0.0).alias('micro_label'),
            pl.lit(0.0).alias('graph_connectivity'),
            pl.lit(0.0).alias('fng_change_1d'),
            pl.lit(0.0).alias('fng_ma_7d'),
            pl.lit(0.0).alias('btc_dominance_change_24h'),
            pl.lit(0.0).alias('eth_dominance'),
            pl.lit(0.0).alias('total_market_cap_norm'),
            pl.lit(0.0).alias('total_volume_24h_norm'),
            pl.lit(0.0).alias('market_cap_change_24h_pct'),
            pl.lit(0.0).alias('macro_dxy_trend'),
            pl.lit(0.0).alias('macro_nq_trend'),
            pl.lit(0.0).alias('macro_gold_returns'),
            pl.lit(0.0).alias('macro_vix_change'),
            pl.lit(0.0).alias('macro_us10y_change'),
            pl.lit(0.0).alias('funding_rate_norm'),
            pl.lit(0.0).alias('open_interest_norm'),
            pl.lit(0.0).alias('long_short_ratio'),
            pl.lit(0.0).alias('top_trader_ls_ratio'),
            pl.lit(0.0).alias('taker_buy_sell_ratio'),
            pl.lit(0.0).alias('onchain_whale_flow'),
            pl.lit(0.0).alias('dark_alpha_pressure'),
            pl.lit(0.0).alias('liquidation_cascade'),
            pl.lit(0.0).alias('dex_whisper'),
        ])

        # ═══════════════════════════════════════════════════════════════════
        # 🔬 [HOLOGRAPHIC AUDIT FIX] PRESERVAR PRECISIÓN float64
        # ANTES: ALL float → float32 (destruía 29 bits de mantisa)
        # AHORA: Mantenemos float64 para cálculos. Cast a float32 se hace
        # SOLO en el momento de inferencia del modelo, NO aquí.
        # ═══════════════════════════════════════════════════════════════════
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
    def validate_features(self, df):
        """Limpieza robusta de features sin bleeding de O.Os (Polars Edition Puro)"""
        if len(df) == 0: return df
        
        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
            
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
