import numpy as np
import talib
from numba import njit

@njit
def _hurst_exponent_core(close, window):
    n = len(close)
    result = np.full(n, 0.50)
    if n < window:
        return result
    for i in range(window, n):
        segment = close[i - window:i]
        seg_std = np.std(segment)
        if seg_std == 0:
            continue
        
        diff_len = len(segment) - 1
        returns = np.zeros(diff_len)
        for j in range(diff_len):
            returns[j] = segment[j+1] - segment[j]
            
        mean_ret = np.mean(returns)
        
        dev_sum = 0.0
        max_dev = -1e9
        min_dev = 1e9
        for j in range(diff_len):
            dev_sum += (returns[j] - mean_ret)
            if dev_sum > max_dev: max_dev = dev_sum
            if dev_sum < min_dev: min_dev = dev_sum
            
        R = max_dev - min_dev
        S = np.std(returns)
        
        if S > 0 and R > 0:
            rs = R / S
            result[i] = np.log(rs) / np.log(len(returns))
    return result

@njit
def _fractal_dimension_core(close, window):
    n = len(close)
    result = np.full(n, 1.5)
    if n < window:
        return result
    for i in range(window, n):
        segment = close[i - window:i]
        max_val = np.max(segment)
        min_val = np.min(segment)
        rng = max_val - min_val
        if rng == 0:
            continue
        path_length = 0.0
        for j in range(len(segment)-1):
            path_length += np.abs(segment[j+1] - segment[j])
        normalized = path_length / rng
        if normalized > 0:
            result[i] = 1.0 + np.log(normalized) / np.log(2.0 * (window - 1.0))
    return result

@njit
def _detect_swings_core(high, low, lookback):
    n = len(high)
    swing_highs = np.zeros(n)
    swing_lows = np.zeros(n)
    for i in range(lookback, n - lookback):
        is_high = True
        is_low = True
        for j in range(i - lookback, i + lookback + 1):
            if high[j] > high[i]:
                is_high = False
            if low[j] < low[i]:
                is_low = False
        if is_high:
            swing_highs[i] = high[i]
        if is_low:
            swing_lows[i] = low[i]
    return swing_highs, swing_lows

@njit
def _classify_structure_core(swing_highs, swing_lows):
    n = len(swing_highs)
    result = np.zeros(n)
    sh1 = 0.0; sh2 = 0.0
    sl1 = 0.0; sl2 = 0.0
    for i in range(n):
        if swing_highs[i] > 0:
            sh1 = sh2
            sh2 = swing_highs[i]
        if swing_lows[i] > 0:
            sl1 = sl2
            sl2 = swing_lows[i]
            
        if sh1 > 0 and sh2 > 0 and sl1 > 0 and sl2 > 0:
            hh = sh2 > sh1
            hl = sl2 > sl1
            lh = sh2 < sh1
            ll = sl2 < sl1
            if hh and hl:
                result[i] = 1.0
            elif lh and ll:
                result[i] = -1.0
    return result

@njit
def _detect_choch_core(close, swing_highs, swing_lows):
    n = len(close)
    result = np.zeros(n)
    last_structure = 0
    sh1 = 0.0; sh2 = 0.0
    sl1 = 0.0; sl2 = 0.0
    
    for i in range(1, n):
        if swing_highs[i-1] > 0:
            sh1 = sh2
            sh2 = swing_highs[i-1]
        if swing_lows[i-1] > 0:
            sl1 = sl2
            sl2 = swing_lows[i-1]
            
        if sh1 > 0 and sh2 > 0 and sl1 > 0 and sl2 > 0:
            if sh2 > sh1 and sl2 > sl1:
                last_structure = 1
            elif sh2 < sh1 and sl2 < sl1:
                last_structure = -1
                
        if last_structure == -1 and sh2 > 0:
            if close[i] > sh2 and close[i-1] <= sh2:
                result[i] = 1.0
        elif last_structure == 1 and sl2 > 0:
            if close[i] < sl2 and close[i-1] >= sl2:
                result[i] = -1.0
    return result

@njit
def _fibonacci_distance_core(close, high, low, lookback):
    n = len(close)
    fib_382 = np.zeros(n)
    fib_618 = np.zeros(n)
    fib_zone = np.zeros(n)
    if n < lookback:
        return fib_382, fib_618, fib_zone
    
    for i in range(lookback, n):
        seg_high = np.max(high[i - lookback:i])
        seg_low = np.min(low[i - lookback:i])
        rng = seg_high - seg_low
        if rng == 0:
            continue
            
        level_382 = seg_high - 0.382 * rng
        level_618 = seg_high - 0.618 * rng
        fib_382[i] = (close[i] - level_382) / rng
        fib_618[i] = (close[i] - level_618) / rng
        
        price_pos = (seg_high - close[i]) / rng
        if price_pos < 0.236:
            fib_zone[i] = 1.0
        elif price_pos > 0.618:
            fib_zone[i] = -1.0
            
    return fib_382, fib_618, fib_zone

class StructureIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, n_len):
        features = {}
        features["hurst"] = StructureIndicators._hurst_exponent(close)
        features["fdi"] = StructureIndicators._fractal_dimension(close)
        
        swing_highs, swing_lows = StructureIndicators._detect_swings(high, low)
        features["market_structure"] = StructureIndicators._classify_structure(swing_highs, swing_lows, n_len)
        features["choch"] = StructureIndicators._detect_choch(close, swing_highs, swing_lows, n_len)
        
        fib_data = StructureIndicators._fibonacci_distance(close, high, low, n_len, lookback=50)
        features.update(fib_data)
        return features

    @staticmethod
    def _hurst_exponent(close, window=100):
        res = _hurst_exponent_core(close, window)
        return np.clip(res, 0.0, 1.0)

    @staticmethod
    def _fractal_dimension(close, window=30):
        res = _fractal_dimension_core(close, window)
        return np.clip(res, 1.0, 2.0)

    @staticmethod
    def _detect_swings(high, low, lookback=5):
        return _detect_swings_core(high, low, lookback)

    @staticmethod
    def _classify_structure(swing_highs, swing_lows, n_len):
        return _classify_structure_core(swing_highs, swing_lows)

    @staticmethod
    def _detect_choch(close, swing_highs, swing_lows, n_len):
        return _detect_choch_core(close, swing_highs, swing_lows)

    @staticmethod
    def _fibonacci_distance(close, high, low, n_len, lookback=50):
        fib_382, fib_618, fib_zone = _fibonacci_distance_core(close, high, low, lookback)
        return {"fib_382_dist": fib_382, "fib_618_dist": fib_618, "fib_zone": fib_zone}
