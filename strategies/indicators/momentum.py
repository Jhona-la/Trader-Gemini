import numpy as np
import talib
from utils.math_kernel import calculate_rsi_jit, calculate_macd_jit

class MomentumIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, volume, n_len):
        """Calcula todos los indicadores de Momentum M01-M18 y los retorna en un diccionario."""
        import polars as pl
        features = {}
        
        # M01, M02, M03: RSIs (Wrap JIT in Series for exact TA-Lib match)
        features['rsi_3'] = pl.Series('rsi_3', calculate_rsi_jit(close, 3))
        features['rsi_5'] = pl.Series('rsi_5', calculate_rsi_jit(close, 5))
        features['rsi_7'] = pl.Series('rsi_7', calculate_rsi_jit(close, 7))
        features['rsi_14'] = pl.Series('rsi_14', calculate_rsi_jit(close, 14))
        features['rsi_21'] = pl.Series('rsi_21', calculate_rsi_jit(close, 21))
        
        # M03: StochRSI (Pure Polars Expressions)
        rsi_14 = pl.col('rsi_14')
        rsi_min = rsi_14.rolling_min(window_size=14)
        rsi_max = rsi_14.rolling_max(window_size=14)
        features['stoch_rsi'] = pl.when(rsi_max != rsi_min).then((rsi_14 - rsi_min) / (rsi_max - rsi_min)).otherwise(0.0)
        
        # M05, M06: MACD & PPO & APO (Pure Polars Expressions)
        ema_12 = pl.col('close').ewm_mean(span=12, adjust=False)
        ema_26 = pl.col('close').ewm_mean(span=26, adjust=False)
        macd = ema_12 - ema_26
        macd_signal = macd.ewm_mean(span=9, adjust=False)
        macd_hist = macd - macd_signal
        
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist
        
        if n_len > 26:
            features['ppo'] = pl.Series('ppo', talib.PPO(close, 12, 26, 0))
            features['apo'] = pl.Series('apo', talib.APO(close, 12, 26, 0))
        
        # M04: Stochastic (TA-Lib)
        slowk, slowd = talib.STOCH(high, low, close, 14, 3, 3)
        features['stoch_k'] = pl.Series('stoch_k', slowk)
        features['stoch_d'] = pl.Series('stoch_d', slowd)
        features['stoch_cross'] = pl.when(pl.col('stoch_k') > pl.col('stoch_d')).then(1.0).otherwise(-1.0)
        
        # M10: MFI (TA-Lib)
        features['mfi'] = pl.Series('mfi', talib.MFI(high, low, close, volume, 14))
        
        # M08: CCI (TA-Lib)
        features['cci'] = pl.Series('cci', talib.CCI(high, low, close, 20))
        
        # M09: ROC (Pure Polars)
        features['roc_1'] = pl.col('close').pct_change(1) * 100
        features['roc_5'] = pl.col('close').pct_change(5) * 100
        features['roc_14'] = pl.col('close').pct_change(14) * 100
        
        # M14: MOM (Pure Polars)
        features['mom_10'] = pl.col('close') - pl.col('close').shift(10)
        
        # M16: CMO (TA-Lib)
        if n_len > 14:
            features['cmo_14'] = pl.Series('cmo_14', talib.CMO(close, 14))
            
        # M17: Williams %R (TA-Lib)
        if n_len > 14:
            features['willr_14'] = pl.Series('willr_14', talib.WILLR(high, low, close, 14))
            
        # M12: Ultimate Oscillator (TA-Lib)
        if n_len > 28:
            features['ultosc'] = pl.Series('ultosc', talib.ULTOSC(high, low, close, 7, 14, 28))

        # M15: Fisher Transform (Pure Polars Expression)
        midpoints = (pl.col('high') + pl.col('low')) / 2
        fisher_max = midpoints.rolling_max(window_size=11)
        fisher_min = midpoints.rolling_min(window_size=11)
        fisher_rng = fisher_max - fisher_min
        fisher_rng = pl.when(fisher_rng == 0).then(1e-10).otherwise(fisher_rng)
        
        fisher_x = 2 * ((midpoints - fisher_min) / fisher_rng) - 1
        fisher_x = pl.when(fisher_x > 0.999).then(0.999).when(fisher_x < -0.999).then(-0.999).otherwise(fisher_x)
        fisher = 0.5 * ((1 + fisher_x) / (1 - fisher_x)).log()
        
        features['fisher_transform'] = fisher.fill_null(0.0)
        features['fisher_signal'] = pl.when(features['fisher_transform'] > 1.5).then(-1.0).when(features['fisher_transform'] < -1.5).then(1.0).otherwise(0.0)

        # M11: TSI (Pure Polars Expression)
        tsi_mom = pl.col('close').diff()
        tsi_abs_mom = tsi_mom.abs()
        
        tsi_ema1 = tsi_mom.ewm_mean(span=25, adjust=False)
        tsi_num = tsi_ema1.ewm_mean(span=13, adjust=False)
        
        tsi_ema1_abs = tsi_abs_mom.ewm_mean(span=25, adjust=False)
        tsi_den = tsi_ema1_abs.ewm_mean(span=13, adjust=False)
        
        tsi_expr = pl.when(tsi_den != 0).then(100 * tsi_num / tsi_den).otherwise(0.0)
        features['tsi'] = tsi_expr
        
        features['tsi_signal'] = tsi_expr.ewm_mean(span=7, adjust=False)

        # M13: Awesome Oscillator (Pure Polars)
        midpoints = (pl.col('high') + pl.col('low')) / 2
        ao = midpoints.rolling_mean(window_size=5) - midpoints.rolling_mean(window_size=34)
        features['awesome_osc'] = ao
        features['ao_acceleration'] = ao - ao.shift(1).fill_null(ao)

        # M18: DPO (Pure Polars)
        shift = 20 // 2 + 1
        sma_20_shifted = pl.col('close').rolling_mean(window_size=20).shift(-shift)
        features['dpo'] = pl.col('close') - sma_20_shifted

        # RSI Divergence Detection
        if n_len > 20:
            # Requires talib.LINEARREG_SLOPE so keep as Series
            rsi_slope = talib.LINEARREG_SLOPE(calculate_rsi_jit(close, 14), 10)
            price_slope = talib.LINEARREG_SLOPE(close, 10)
            features['rsi_divergence'] = pl.Series('rsi_divergence', np.where(
                (price_slope > 0) & (rsi_slope < 0), -1.0,
                np.where((price_slope < 0) & (rsi_slope > 0), 1.0, 0.0)
            ))

        return features


