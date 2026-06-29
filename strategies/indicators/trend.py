import numpy as np
import talib
from utils.math_kernel import calculate_ema_jit
from utils.math_helpers import safe_div

from numba import njit

@njit
def _supertrend_core(close, basic_ub, basic_lb):
    n = len(close)
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    supertrend = np.zeros(n)
    
    # Initialize first valid index
    for i in range(1, n):
        if np.isnan(basic_ub[i]):
            continue
            
        final_ub[i] = basic_ub[i] if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1] else final_ub[i-1]
        final_lb[i] = basic_lb[i] if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1] else final_lb[i-1]
        
        if supertrend[i-1] == final_ub[i-1] and close[i] <= final_ub[i]:
            supertrend[i] = final_ub[i]
        elif supertrend[i-1] == final_ub[i-1] and close[i] > final_ub[i]:
            supertrend[i] = final_lb[i]
        elif supertrend[i-1] == final_lb[i-1] and close[i] >= final_lb[i]:
            supertrend[i] = final_lb[i]
        elif supertrend[i-1] == final_lb[i-1] and close[i] < final_lb[i]:
            supertrend[i] = final_ub[i]
            
    return supertrend

def _get_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calcula SuperTrend (Numpy vectorizado aproximado)"""
    n = len(close)
    atr = talib.ATR(high, low, close, timeperiod=period)
    hl2 = (high + low) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)
    
    return _supertrend_core(close, basic_ub, basic_lb)

class TrendIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, n_len):
        """Calcula todos los indicadores de Tendencia T01-T20."""
        import polars as pl
        features = {}
        
        # T01-T04: EMAs y SMAs (Pure Polars Expressions)
        for p in [5, 10, 20, 50, 100, 200]:
            ema_expr = pl.col('close').ewm_mean(span=p, adjust=False)
            features[f'ema_{p}'] = ema_expr
            # (close - ema) / ema
            features[f'dist_ema_{p}'] = pl.when(ema_expr != 0).then((pl.col('close') - ema_expr) / ema_expr).otherwise(0.0)
        
        features['sma_20'] = pl.col('close').rolling_mean(window_size=20)
        features['sma_50'] = pl.col('close').rolling_mean(window_size=50)
        
        # T03, T04: DEMA, TEMA, WMA, KAMA (TA-Lib C-Level)
        if n_len >= 20:
            features['dema_20'] = pl.Series('dema_20', talib.DEMA(close, 20))
            features['tema_20'] = pl.Series('tema_20', talib.TEMA(close, 20))
            features['wma_20'] = pl.Series('wma_20', talib.WMA(close, 20))
            features['kama_20'] = pl.Series('kama_20', talib.KAMA(close, 20))
        
        # T11: Supertrend (Custom Numba JIT)
        if n_len > 10:
            st = _get_supertrend(high, low, close, 10, 3.0)
            features['supertrend'] = pl.Series('supertrend', st)
            features['supertrend_dist'] = pl.when(pl.col('supertrend') != 0).then((pl.col('close') - pl.col('supertrend')) / pl.col('supertrend')).otherwise(0.0)
            features['supertrend_dir'] = pl.when(pl.col('close') > pl.col('supertrend')).then(1.0).otherwise(-1.0)
            
        # T13: Parabolic SAR (TA-Lib C-Level)
        if n_len > 2:
            sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            features['sar'] = pl.Series('sar', sar)
            features['sar_dist'] = pl.when(pl.col('sar') != 0).then((pl.col('close') - pl.col('sar')) / pl.col('sar')).otherwise(0.0)
            
        # T14, T15: ADX & DI (JIT & TA-Lib)
        from utils.math_kernel import calculate_adx_jit
        features['adx'] = pl.Series('adx', calculate_adx_jit(high, low, close, 14))
        features['plus_di'] = pl.Series('plus_di', talib.PLUS_DI(high, low, close, 14))
        features['minus_di'] = pl.Series('minus_di', talib.MINUS_DI(high, low, close, 14))
        
        # T16: Aroon
        if n_len > 14:
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            features['aroon_up'] = pl.Series('aroon_up', aroon_up)
            features['aroon_down'] = pl.Series('aroon_down', aroon_down)
            features['aroon_osc'] = pl.Series('aroon_osc', talib.AROONOSC(high, low, timeperiod=14))
            
        # T20: TRIX
        if n_len > 15:
            features['trix'] = pl.Series('trix', talib.TRIX(close, timeperiod=15))
            
        # Ichimoku Cloud (Pure Polars Expressions)
        tenkan = (pl.col('high').rolling_max(window_size=9) + pl.col('low').rolling_min(window_size=9)) / 2
        kijun = (pl.col('high').rolling_max(window_size=26) + pl.col('low').rolling_min(window_size=26)) / 2
        
        features['ichimoku_tenkan'] = tenkan
        features['ichimoku_kijun'] = kijun
        features['ichimoku_cross'] = pl.when(tenkan > kijun).then(1.0).otherwise(-1.0)
        
        senkou_a = (tenkan + kijun) / 2
        features['ichimoku_senkou_a'] = senkou_a

        senkou_b = (pl.col('high').rolling_max(window_size=52) + pl.col('low').rolling_min(window_size=52)) / 2
        features['ichimoku_senkou_b'] = senkou_b
        
        features['ichimoku_cloud_width'] = pl.when(pl.col('close') != 0).then((senkou_a - senkou_b) / pl.col('close')).otherwise(0.0)
        
        cloud_top = pl.max_horizontal(senkou_a, senkou_b)
        cloud_bot = pl.min_horizontal(senkou_a, senkou_b)
        
        features['ichimoku_price_vs_cloud'] = pl.when(pl.col('close') > cloud_top).then(1.0).when(pl.col('close') < cloud_bot).then(-1.0).otherwise(0.0)

        # T07: ALMA (Custom Vectorized Python -> Keep as pl.Series)
        if n_len > 21:
            alma = TrendIndicators._alma(close, period=21, offset=0.85, sigma=6.0)
            features['alma_21'] = pl.Series('alma_21', alma)
            features['alma_dist'] = pl.when(pl.col('alma_21') != 0).then((pl.col('close') - pl.col('alma_21')) / pl.col('alma_21')).otherwise(0.0)

        # T08: Hull MA
        if n_len > 16:
            hma = TrendIndicators._hull_ma(close, period=16)
            features['hull_ma_16'] = pl.Series('hull_ma_16', hma)
            features['hull_dist'] = pl.when(pl.col('hull_ma_16') != 0).then((pl.col('close') - pl.col('hull_ma_16')) / pl.col('hull_ma_16')).otherwise(0.0)
            
            # hull_direction needs a shift (diff > 0)
            features['hull_direction'] = pl.when((pl.col('hull_ma_16') - pl.col('hull_ma_16').shift(1)) > 0).then(1.0).otherwise(-1.0)

        # T10: VWMA (Pure Polars Expressions)
        if 'volume' in df.columns:
            vp = pl.col('close') * pl.col('volume')
            vwma = vp.rolling_mean(window_size=20) / pl.col('volume').rolling_mean(window_size=20)
            features['vwma_20'] = vwma
            features['vwma_dist'] = pl.when(vwma != 0).then((pl.col('close') - vwma) / vwma).otherwise(0.0)

        # T19: Elder Ray (Bull Power / Bear Power)
        ema13 = pl.col('close').ewm_mean(span=13, adjust=False)
        features['elder_bull_power'] = pl.col('high') - ema13
        features['elder_bear_power'] = pl.col('low') - ema13

        return features

    @staticmethod
    def _alma(close, period=21, offset=0.85, sigma=6.0):
        """Arnaud Legoux Moving Average — zero lag MA with Gaussian distribution."""
        n = len(close)
        result = np.full(n, np.nan)
        m = offset * (period - 1)
        s = period / sigma

        weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
        weights /= np.sum(weights)

        # Vectorized convolution instead of python loop
        conv = np.convolve(close, weights[::-1], mode='valid')
        result[period - 1:] = conv

        # Fill NaN with close
        result = np.where(np.isnan(result), close, result)
        return result

    @staticmethod
    def _hull_ma(close, period=16):
        """Hull Moving Average — minimum lag with smooth output."""
        half_period = max(1, period // 2)
        sqrt_period = max(1, int(np.sqrt(period)))

        wma_half = talib.WMA(close, half_period)
        wma_full = talib.WMA(close, period)

        # Handle NaN from early bars
        wma_half = np.where(np.isnan(wma_half), close, wma_half)
        wma_full = np.where(np.isnan(wma_full), close, wma_full)

        diff = 2 * wma_half - wma_full
        hma = talib.WMA(diff, sqrt_period)
        hma = np.where(np.isnan(hma), close, hma)

        return hma
