"""
MÓDULO OMEGA — Categoría 8: Indicadores Estadísticos y de Régimen (R01-R06)
===========================================================================

QUÉ: Indicadores que usan propiedades estadísticas de la serie temporal
  para clasificar regímenes, detectar anomalías, y cuantificar la
  predictibilidad del mercado.

POR QUÉ: Los indicadores técnicos clásicos asumen que el mercado
  siempre tiene estructura explotable. Los estadísticos cuantifican
  CUÁNTA estructura hay, y en qué dirección.

PARA QUÉ: Reducir el sizing cuando la predictibilidad es baja (R01),
  detectar cuando la combinación de indicadores es anómala (R04),
  y explotar patrones temporales recurrentes (R05).

CÓMO: Autocorrelación, Z-scores, seasonal patterns.
CUÁNDO: En cada ciclo de feature engineering.
DÓNDE: strategies/indicators/statistical.py
QUIÉN: Quant Developer
"""

import numpy as np
import talib
from utils.math_helpers import safe_div


class StatisticalIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, volume, n_len):
        """Calcula todos los indicadores Estadísticos R01-R06."""
        features = {}

        # ═══════════════════════════════════════════════════════════
        # R01: Autocorrelación de Retornos (lag 1, 5, 20)
        # Autocorrelación positiva: momentum persiste (usar TFTF)
        # Autocorrelación negativa: movimientos se revierten (mean reversion)
        # Autocorrelación ≈ 0: no hay estructura explotable
        # ═══════════════════════════════════════════════════════════
        returns = np.diff(close, prepend=close[0]) / np.where(close != 0, close, 1.0)

        for lag in [1, 5, 20]:
            ac = StatisticalIndicators._rolling_autocorrelation(returns, lag=lag, window=50)
            features[f'autocorr_lag{lag}'] = ac

        # ═══════════════════════════════════════════════════════════
        # R03: Z-Score de Precio vs Media Histórica
        # Z > 2.5: precio alejado al alza (extensión o continuación extrema)
        # Z < -2.5: precio alejado a la baja
        # ═══════════════════════════════════════════════════════════
        for window in [20, 50]:
            if n_len > window:
                sma = talib.SMA(close, window)
                std = talib.STDDEV(close, window, 1)
                features[f'zscore_{window}'] = np.where(std != 0, (close - sma) / std, 0.0)

        # ═══════════════════════════════════════════════════════════
        # R05: Seasonal Pattern Score
        # Patrones por hora del día, día de la semana.
        # BTC históricamente tiene mayor volatilidad en ciertos horarios.
        # ═══════════════════════════════════════════════════════════
        if hasattr(df, 'index') and hasattr(df.index, 'hour'):
            try:
                hour = df.index.hour.to_numpy().astype(np.float64)
                features['hour_of_day'] = hour / 24.0  # Normalizado [0, 1]

                # High activity hours (UTC): 13-16 (US open), 8-10 (EU open)
                features['high_activity_session'] = np.where(
                    ((hour >= 13) & (hour <= 16)) | ((hour >= 8) & (hour <= 10)),
                    1.0, 0.0
                )
            except Exception:
                features['hour_of_day'] = np.zeros(n_len)
                features['high_activity_session'] = np.zeros(n_len)

            try:
                dow = df.index.dayofweek.to_numpy().astype(np.float64)
                features['day_of_week'] = dow / 6.0  # Normalizado [0, 1]

                # Weekend effect: crypto has different dynamics on weekends
                features['is_weekend'] = np.where((dow >= 5), 1.0, 0.0)
            except Exception:
                features['day_of_week'] = np.zeros(n_len)
                features['is_weekend'] = np.zeros(n_len)
        else:
            features['hour_of_day'] = np.zeros(n_len)
            features['high_activity_session'] = np.zeros(n_len)
            features['day_of_week'] = np.zeros(n_len)
            features['is_weekend'] = np.zeros(n_len)

        # ═══════════════════════════════════════════════════════════
        # R: Return Distribution Statistics (Rolling)
        # Skewness y Kurtosis de retornos — capturan la "forma"
        # de la distribución de retornos (colas gordas, asimetría).
        # ═══════════════════════════════════════════════════════════
        if n_len > 30:
            features['returns_skew_30'] = StatisticalIndicators._rolling_skewness(returns, 30)
            features['returns_kurtosis_30'] = StatisticalIndicators._rolling_kurtosis(returns, 30)

        # ═══════════════════════════════════════════════════════════
        # Volatility Regime (ATR ratio short/long)
        # ATR-7 / ATR-50 > 1.5 = alta volatilidad relativa
        # ATR-7 / ATR-50 < 0.7 = compresión (squeeze inminente)
        # ═══════════════════════════════════════════════════════════
        if n_len > 50:
            atr_7 = talib.ATR(high, low, close, 7)
            atr_50 = talib.ATR(high, low, close, 50)
            features['vol_regime_ratio'] = safe_div(atr_7, atr_50, 1.0)
            features['vol_compression'] = np.where(
                safe_div(atr_7, atr_50, 1.0) < 0.7, 1.0, 0.0
            )

        # ═══════════════════════════════════════════════════════════
        # Mean Reversion Probability
        # Combinación de Z-score + autocorrelación para estimar
        # la probabilidad de reversión vs continuación.
        # ═══════════════════════════════════════════════════════════
        if n_len > 50:
            zscore_50 = features['zscore_50']
            autocorr_1 = features['autocorr_lag1']
            # High z-score + negative autocorrelation = high reversion probability
            features['mean_reversion_prob'] = np.clip(
                (np.abs(zscore_50) / 3.0) * np.where(autocorr_1 < 0, 1.5, 0.5),
                0.0, 1.0
            )

        return features

from numba import njit

@njit
def _rolling_autocorr_core(returns, lag, window):
    n = len(returns)
    result = np.zeros(n)
    if n < window + lag:
        return result
        
    for i in range(window + lag, n):
        x = returns[i - window:i - lag]
        y = returns[i - window + lag:i]
        
        if len(x) == 0:
            continue
            
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)
        
        if x_std > 0 and y_std > 0:
            result[i] = np.mean((x - x_mean) * (y - y_mean)) / (x_std * y_std)
            
    return result

@njit
def _rolling_skewness_core(data, window):
    n = len(data)
    result = np.zeros(n)
    for i in range(window, n):
        segment = data[i - window:i]
        mean = np.mean(segment)
        std = np.std(segment)
        if std > 0:
            # Numba requires explicit element-wise or explicit loop for power sometimes, but this should work:
            val = 0.0
            for j in range(len(segment)):
                val += ((segment[j] - mean) / std) ** 3
            result[i] = val / len(segment)
    return result

@njit
def _rolling_kurtosis_core(data, window):
    n = len(data)
    result = np.full(n, 3.0)
    for i in range(window, n):
        segment = data[i - window:i]
        mean = np.mean(segment)
        std = np.std(segment)
        if std > 0:
            val = 0.0
            for j in range(len(segment)):
                val += ((segment[j] - mean) / std) ** 4
            result[i] = val / len(segment)
    return result

    @staticmethod
    def _rolling_autocorrelation(returns, lag=1, window=50):
        """Autocorrelación rolling de retornos con lag específico."""
        res = _rolling_autocorr_core(returns, lag, window)
        return np.clip(res, -1.0, 1.0)

    @staticmethod
    def _rolling_skewness(data, window):
        """Skewness rolling — asimetría de la distribución."""
        return _rolling_skewness_core(data, window)

    @staticmethod
    def _rolling_kurtosis(data, window):
        """Kurtosis rolling — colas gordas de la distribución. >3 = leptokurtic."""
        return _rolling_kurtosis_core(data, window)
