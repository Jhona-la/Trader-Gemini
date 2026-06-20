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
        features = {}
        
        # T01-T04: EMAs y SMAs
        for p in [5, 10, 20, 50, 100, 200]:
            if n_len >= p:
                ema = calculate_ema_jit(close, p)
                features[f'ema_{p}'] = ema
                features[f'dist_ema_{p}'] = safe_div(close - ema, ema)
            else:
                features[f'ema_{p}'] = np.copy(close)
                features[f'dist_ema_{p}'] = np.zeros(n_len)
        
        features['sma_20'] = talib.SMA(close, 20) if n_len >= 20 else np.zeros(n_len)
        features['sma_50'] = talib.SMA(close, 50) if n_len >= 50 else np.zeros(n_len)
        
        # T03, T04: DEMA, TEMA, WMA
        if n_len >= 20:
            features['dema_20'] = talib.DEMA(close, 20)
            features['tema_20'] = talib.TEMA(close, 20)
            features['wma_20'] = talib.WMA(close, 20)
            features['kama_20'] = talib.KAMA(close, 20)
        
        # T11: Supertrend
        if n_len > 10:
            st = _get_supertrend(high, low, close, 10, 3.0)
            features['supertrend'] = st
            features['supertrend_dist'] = safe_div(close - st, st)
            features['supertrend_dir'] = np.where(close > st, 1, -1)
            
        # T13: Parabolic SAR
        if n_len > 2:
            sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            features['sar'] = sar
            features['sar_dist'] = safe_div(close - sar, sar)
            
        # T14, T15: ADX & DI
        from utils.math_kernel import calculate_adx_jit
        features['adx'] = calculate_adx_jit(high, low, close, 14)
        features['plus_di'] = talib.PLUS_DI(high, low, close, 14)
        features['minus_di'] = talib.MINUS_DI(high, low, close, 14)
        
        # T16: Aroon
        if n_len > 14:
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down
            features['aroon_osc'] = talib.AROONOSC(high, low, timeperiod=14)
            
        # T20: TRIX
        if n_len > 15:
            features['trix'] = talib.TRIX(close, timeperiod=15)
            
        # Ichimoku Cloud (simplificado Tenkan/Kijun)
        if n_len > 26:
            high_9 = talib.MAX(high, 9)
            low_9 = talib.MIN(low, 9)
            features['ichimoku_tenkan'] = (high_9 + low_9) / 2
            
            high_26 = talib.MAX(high, 26)
            low_26 = talib.MIN(low, 26)
            features['ichimoku_kijun'] = (high_26 + low_26) / 2
            
            features['ichimoku_cross'] = np.where(features['ichimoku_tenkan'] > features['ichimoku_kijun'], 1, -1)

            # Senkou Span A (cloud leading edge)
            features['ichimoku_senkou_a'] = (features['ichimoku_tenkan'] + features['ichimoku_kijun']) / 2

            # Senkou Span B (cloud lagging edge)
            if n_len > 52:
                high_52 = talib.MAX(high, 52)
                low_52 = talib.MIN(low, 52)
                features['ichimoku_senkou_b'] = (high_52 + low_52) / 2
                
                # Cloud thickness (positive = bullish, negative = bearish)
                features['ichimoku_cloud_width'] = safe_div(
                    features['ichimoku_senkou_a'] - features['ichimoku_senkou_b'],
                    close
                )
                
                # Price relative to cloud
                cloud_top = np.maximum(features['ichimoku_senkou_a'], features['ichimoku_senkou_b'])
                cloud_bot = np.minimum(features['ichimoku_senkou_a'], features['ichimoku_senkou_b'])
                features['ichimoku_price_vs_cloud'] = np.where(
                    close > cloud_top, 1.0,
                    np.where(close < cloud_bot, -1.0, 0.0)
                )

        # ═══════════════════════════════════════════════════════════
        # T07: ALMA (Arnaud Legoux MA, período 21, offset 0.85, sigma 6)
        # Suaviza sin el lag de la EMA. Ideal para cruces de señal.
        # ═══════════════════════════════════════════════════════════
        if n_len > 21:
            alma = TrendIndicators._alma(close, period=21, offset=0.85, sigma=6.0)
            features['alma_21'] = alma
            features['alma_dist'] = safe_div(close - alma, alma)

        # ═══════════════════════════════════════════════════════════
        # T08: Hull MA (período 16)
        # Moving average con el menor lag manteniendo suavidad.
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        # ═══════════════════════════════════════════════════════════
        if n_len > 16:
            hma = TrendIndicators._hull_ma(close, period=16)
            features['hull_ma_16'] = hma
            features['hull_dist'] = safe_div(close - hma, hma)
            features['hull_direction'] = np.where(
                np.diff(hma, prepend=hma[0]) > 0, 1.0, -1.0
            )

        # ═══════════════════════════════════════════════════════════
        # T10: VWMA (Volume Weighted MA, período 20)
        # Media ponderada por volumen. Más "honesta" que SMA.
        # ═══════════════════════════════════════════════════════════
        if n_len > 20 and 'volume' in df.columns:
            volume = df['volume'].to_numpy().astype(np.float64)
            vp = close * volume
            vwma = safe_div(talib.SMA(vp, 20), talib.SMA(volume, 20), close)
            features['vwma_20'] = vwma
            features['vwma_dist'] = safe_div(close - vwma, vwma)

        # ═══════════════════════════════════════════════════════════
        # T19: Elder Ray (Bull Power / Bear Power)
        # Bull Power = High - EMA-13 | Bear Power = Low - EMA-13
        # ═══════════════════════════════════════════════════════════
        if n_len > 13:
            ema13 = calculate_ema_jit(close, 13)
            features['elder_bull_power'] = high - ema13
            features['elder_bear_power'] = low - ema13

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
