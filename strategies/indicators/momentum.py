import numpy as np
import talib
from utils.math_kernel import calculate_rsi_jit, calculate_macd_jit

class MomentumIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, volume, n_len):
        """Calcula todos los indicadores de Momentum M01-M18 y los retorna en un diccionario."""
        features = {}
        
        # M01, M02, M03: RSIs
        features['rsi_3'] = calculate_rsi_jit(close, 3)
        features['rsi_5'] = calculate_rsi_jit(close, 5)
        features['rsi_7'] = calculate_rsi_jit(close, 7)
        features['rsi_14'] = calculate_rsi_jit(close, 14)
        features['rsi_21'] = calculate_rsi_jit(close, 21)
        
        # M03: StochRSI
        if n_len > 14:
            rsi_arr = features['rsi_14']
            rsi_min = talib.MIN(rsi_arr, 14)
            rsi_max = talib.MAX(rsi_arr, 14)
            features['stoch_rsi'] = np.where(rsi_max != rsi_min, (rsi_arr - rsi_min) / (rsi_max - rsi_min), 0.0)
        
        # M05, M06: MACD & PPO & APO
        macd, macd_signal, macd_hist = calculate_macd_jit(close, 12, 26, 9)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist
        if n_len > 26:
            features['ppo'] = talib.PPO(close, 12, 26, 0)
            features['apo'] = talib.APO(close, 12, 26, 0)
        
        # M04: Stochastic
        slowk, slowd = talib.STOCH(high, low, close, 14, 3, 3)
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        features['stoch_cross'] = np.where(slowk > slowd, 1, -1)
        
        # M10: MFI
        features['mfi'] = talib.MFI(high, low, close, volume, 14)
        
        # M08: CCI
        features['cci'] = talib.CCI(high, low, close, 20)
        
        # M09: ROC
        features['roc_1'] = talib.ROC(close, 1)
        features['roc_5'] = talib.ROC(close, 5)
        features['roc_14'] = talib.ROC(close, 14)
        
        # M14: MOM
        features['mom_10'] = talib.MOM(close, 10)
        
        # M16: CMO
        if n_len > 14:
            features['cmo_14'] = talib.CMO(close, 14)
            
        # M17: Williams %R
        if n_len > 14:
            features['willr_14'] = talib.WILLR(high, low, close, 14)
            
        # M12: Ultimate Oscillator
        if n_len > 28:
            features['ultosc'] = talib.ULTOSC(high, low, close, 7, 14, 28)

        # ═══════════════════════════════════════════════════════════
        # M15: Fisher Transform (período 10)
        # Convierte precios a distribución Gaussiana para identificar
        # extremos estadísticos. Reversiones desde extremos son potentes.
        # ═══════════════════════════════════════════════════════════
        if n_len > 10:
            fisher = MomentumIndicators._fisher_transform(high, low, period=10)
            features['fisher_transform'] = fisher
            features['fisher_signal'] = np.where(fisher > 1.5, -1.0,
                                        np.where(fisher < -1.5, 1.0, 0.0))

        # ═══════════════════════════════════════════════════════════
        # M11: TSI (True Strength Index, largo=25, corto=13, señal=7)
        # Momentum doblemente suavizado. Menos whipsaws que MACD.
        # ═══════════════════════════════════════════════════════════
        if n_len > 25:
            tsi = MomentumIndicators._true_strength_index(close, long=25, short=13)
            features['tsi'] = tsi
            if n_len > 32:
                from utils.math_kernel import calculate_ema_jit
                features['tsi_signal'] = calculate_ema_jit(tsi, 7)

        # ═══════════════════════════════════════════════════════════
        # M13: Awesome Oscillator (AO)
        # SMA(5) - SMA(34) de los midpoints. Twin peaks y saucers.
        # ═══════════════════════════════════════════════════════════
        if n_len > 34:
            midpoints = (high + low) / 2
            ao = talib.SMA(midpoints, 5) - talib.SMA(midpoints, 34)
            features['awesome_osc'] = ao
            # AO histogram change (acceleration)
            features['ao_acceleration'] = np.diff(ao, prepend=ao[0])

        # ═══════════════════════════════════════════════════════════
        # M18: DPO (Detrended Price Oscillator, período 20)
        # Elimina la tendencia para ver ciclos. Identifica duración.
        # ═══════════════════════════════════════════════════════════
        if n_len > 20:
            shift = 20 // 2 + 1
            sma_20 = talib.SMA(close, 20)
            dpo = np.zeros(n_len)
            dpo[shift:] = close[:-shift] - sma_20[shift:]
            features['dpo'] = np.where(np.isnan(sma_20), 0.0, dpo)

        # ═══════════════════════════════════════════════════════════
        # RSI Divergence Detection (precio vs RSI-14)
        # Divergencia bearish: precio HH + RSI LH
        # Divergencia bullish: precio LL + RSI HL
        # ═══════════════════════════════════════════════════════════
        if n_len > 20:
            rsi_arr = features.get('rsi_14', np.full(n_len, 50.0))
            rsi_slope = talib.LINEARREG_SLOPE(rsi_arr, 10)
            price_slope = talib.LINEARREG_SLOPE(close, 10)
            features['rsi_divergence'] = np.where(
                (price_slope > 0) & (rsi_slope < 0), -1.0,  # Bearish divergence
                np.where((price_slope < 0) & (rsi_slope > 0), 1.0, 0.0)  # Bullish divergence
            )

        return features

    @staticmethod
    def _fisher_transform(high, low, period=10):
        """Fisher Transform: convierte precios a distribución Gaussiana."""
        n = len(high)
        if n < period + 1:
            return np.zeros(n)
            
        midpoints = (high + low) / 2
        
        # Use talib for O(1) rolling max/min
        max_val = talib.MAX(midpoints, timeperiod=period + 1)
        min_val = talib.MIN(midpoints, timeperiod=period + 1)
        
        rng = max_val - min_val
        # Avoid division by zero
        rng = np.where(rng == 0, 1e-10, rng)
        
        x = 2 * ((midpoints - min_val) / rng) - 1
        x = np.clip(x, -0.999, 0.999)
        
        result = 0.5 * np.log((1 + x) / (1 - x))
        
        # Fill the first `period` elements with 0.0
        result[:period] = 0.0
        result = np.nan_to_num(result, nan=0.0)
        
        return result

    @staticmethod
    def _true_strength_index(close, long=25, short=13):
        """True Strength Index: momentum doblemente suavizado."""
        from utils.math_kernel import calculate_ema_jit
        n = len(close)
        mom = np.diff(close, prepend=close[0])

        # Double smoothing of momentum
        ema1 = calculate_ema_jit(mom, long)
        tsi_num = calculate_ema_jit(ema1, short)

        # Double smoothing of |momentum|
        abs_mom = np.abs(mom)
        ema1_abs = calculate_ema_jit(abs_mom, long)
        tsi_den = calculate_ema_jit(ema1_abs, short)

        # TSI = 100 * smoothed_mom / smoothed_abs_mom
        tsi = np.where(tsi_den != 0, 100 * tsi_num / tsi_den, 0.0)
        return tsi
