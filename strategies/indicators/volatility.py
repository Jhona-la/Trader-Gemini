import numpy as np
import talib
from utils.math_kernel import calculate_atr_jit, calculate_bollinger_jit, calculate_ema_jit
from utils.math_helpers import safe_div

class VolatilityIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, n_len):
        """Calcula todos los indicadores de Volatilidad V01-V10."""
        import polars as pl
        import numpy as np
        features = {}
        
        # V01, V02: ATR & NATR (Keep JIT/TA-Lib for exact Wilders smoothing match)
        atr = calculate_atr_jit(high, low, close, 14)
        features['atr'] = pl.Series('atr', atr)
        features['natr'] = pl.Series('natr', talib.NATR(high, low, close, 14) if n_len > 14 else np.zeros(n_len))
        features['atr_pct'] = pl.when(pl.col('close') != 0).then((pl.col('atr') / pl.col('close')) * 100).otherwise(0.0)
        
        # V03: TRANGE
        features['trange'] = pl.Series('trange', talib.TRANGE(high, low, close))
        
        # V04, V05: Bollinger Bands (Pure Polars Expressions)
        sma_20 = pl.col('close').rolling_mean(window_size=20)
        std_20 = pl.col('close').rolling_std(window_size=20)
        upper = sma_20 + 2.0 * std_20
        lower_band = sma_20 - 2.0 * std_20
        middle = sma_20
        
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower_band
        features['bb_position'] = pl.when((upper - lower_band) != 0).then((pl.col('close') - lower_band) / (upper - lower_band)).otherwise(0.5)
        features['bb_width'] = pl.when(middle != 0).then((upper - lower_band) / middle).otherwise(0.0)
        features['mean_reversion_potential'] = pl.when((upper - lower_band) != 0).then((pl.col('close') - middle) / (upper - lower_band)).otherwise(0.0)
        
        # V05: Keltner Channels (Pure Polars Expressions)
        if n_len > 20:
            ema20 = pl.col('close').ewm_mean(span=20, adjust=False)
            kc_upper = ema20 + (1.5 * pl.col('atr'))
            kc_lower = ema20 - (1.5 * pl.col('atr'))
            features['kc_upper'] = kc_upper
            features['kc_lower'] = kc_lower
            features['kc_width'] = pl.when(ema20 != 0).then((kc_upper - kc_lower) / ema20).otherwise(0.0)
            
        # V06: Donchian Channels (Pure Polars Expressions)
        if n_len > 20:
            dc_upper = pl.col('high').rolling_max(window_size=20)
            dc_lower = pl.col('low').rolling_min(window_size=20)
            features['dc_upper'] = dc_upper
            features['dc_lower'] = dc_lower
            
            mid = (dc_upper + dc_lower) / 2
            features['dc_width'] = pl.when(mid != 0).then((dc_upper - dc_lower) / mid).otherwise(0.0)
        
        # gk_vol (Garman-Klass Volatility) (Pure Polars Expressions)
        if 'open' in df.columns:
            log_hl = (pl.when(pl.col('low') != 0).then(pl.col('high') / pl.col('low')).otherwise(1.0)).log().pow(2)
            log_co = (pl.when(pl.col('open') != 0).then(pl.col('close') / pl.col('open')).otherwise(1.0)).log().pow(2)
            features['gk_vol'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # V07: Historical Volatility (Pure Polars Expressions)
        if 'returns_1' in df.columns:
            features['volatility_10'] = pl.col('returns_1').rolling_std(window_size=10) * 100
            if n_len > 30:
                features['volatility_30'] = pl.col('returns_1').rolling_std(window_size=30) * 100
        else:
            features['volatility_10'] = pl.lit(0.0)
            if n_len > 30:
                features['volatility_30'] = pl.lit(0.0)
        
        return features
