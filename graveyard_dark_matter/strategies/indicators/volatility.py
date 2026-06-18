import numpy as np
import talib
from utils.math_kernel import calculate_atr_jit, calculate_bollinger_jit, calculate_ema_jit
from utils.math_helpers import safe_div

class VolatilityIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, n_len):
        """Calcula todos los indicadores de Volatilidad V01-V10."""
        features = {}
        
        # V01, V02: ATR & NATR
        atr = calculate_atr_jit(high, low, close, 14)
        features['atr'] = atr
        features['natr'] = talib.NATR(high, low, close, 14) if n_len > 14 else np.zeros(n_len)
        features['atr_pct'] = np.where(close != 0, (atr / close) * 100, 0.0)
        
        # V03: TRANGE
        features['trange'] = talib.TRANGE(high, low, close)
        
        # V04, V05: Bollinger Bands
        upper, middle, lower_band = calculate_bollinger_jit(close, 20, 2.0)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower_band
        features['bb_position'] = safe_div(close - lower_band, upper - lower_band, 0.5)
        features['bb_width'] = safe_div(upper - lower_band, middle)
        features['mean_reversion_potential'] = safe_div(close - middle, upper - lower_band, 0.0)
        
        # V05: Keltner Channels
        if n_len > 20:
            ema20 = calculate_ema_jit(close, 20)
            features['kc_upper'] = ema20 + (1.5 * atr)
            features['kc_lower'] = ema20 - (1.5 * atr)
            features['kc_width'] = safe_div(features['kc_upper'] - features['kc_lower'], ema20)
            
        # V06: Donchian Channels
        if n_len > 20:
            features['dc_upper'] = talib.MAX(high, 20)
            features['dc_lower'] = talib.MIN(low, 20)
            features['dc_width'] = safe_div(features['dc_upper'] - features['dc_lower'], (features['dc_upper'] + features['dc_lower']) / 2)
        
        # gk_vol (Garman-Klass Volatility)
        if 'open' in df.columns:
            open_ = df['open'].to_numpy()
            log_hl = np.log(safe_div(high, low, 1.0)) ** 2
            log_co = np.log(safe_div(close, open_, 1.0)) ** 2
            features['gk_vol'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # V07: Historical Volatility
        vol_ret = df['returns_1'].to_numpy() if 'returns_1' in df.columns else np.zeros(n_len)
        features['volatility_10'] = talib.STDDEV(vol_ret, 10, 1) * 100
        if n_len > 30:
            features['volatility_30'] = talib.STDDEV(vol_ret, 30, 1) * 100
        
        return features
