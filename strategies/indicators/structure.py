"""
MÓDULO OMEGA — Categoría 5: Indicadores de Estructura de Mercado (E01-E10)
==========================================================================

QUÉ: Indicadores que analizan la estructura interna del precio:
  persistencia de tendencia (Hurst), puntos de giro (Swing), cambios
  de carácter (CHoCH), rupturas de estructura (BOS), y zonas de valor.

POR QUÉ: Sin estructura, el sistema opera a ciegas sobre si el mercado
  es trending o ranging. Hurst solo ya puede multiplicar la selectividad
  de señales: H > 0.55 = tendencia (TFTF), H < 0.45 = mean reversion.

PARA QUÉ: Alimentar al ML con features que capturan la geometría
  del precio, no solo su dirección o velocidad.

CÓMO: Cálculos vectorizados con numpy/talib para nanosegundos.
CUÁNDO: En cada ciclo de feature engineering.
DÓNDE: strategies/indicators/structure.py
QUIÉN: Quant Developer + Arquitecto
"""

import numpy as np
from utils.math_helpers import safe_div


class StructureIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, n_len):
        """Calcula todos los indicadores Estructurales E01-E10."""
        features = {}

        # ═══════════════════════════════════════════════════════════
        # E01: Hurst Exponent (Rescaled Range, ventana deslizante)
        # H > 0.55 → serie persistente (tendencias continúan)
        # H ≈ 0.50 → random walk (no predecible)
        # H < 0.45 → antipersistente (reversión a la media)
        # ═══════════════════════════════════════════════════════════
        hurst = StructureIndicators._hurst_exponent(close, window=100)
        features['hurst_exponent'] = hurst
        features['hurst_trending'] = np.where(hurst > 0.55, 1.0, np.where(hurst < 0.45, -1.0, 0.0))

        # ═══════════════════════════════════════════════════════════
        # E02: Fractal Dimension Index (FDI)
        # FDI bajo = tendencia fuerte. FDI alto = lateral complejo.
        # Complemento del Hurst para confirmar régimen.
        # ═══════════════════════════════════════════════════════════
        fdi = StructureIndicators._fractal_dimension(close, window=30)
        features['fractal_dimension'] = fdi

        # ═══════════════════════════════════════════════════════════
        # E07: Swing High/Low Detector
        # Identifica HH-HL (alcista) vs LH-LL (bajista)
        # ═══════════════════════════════════════════════════════════
        swing_highs, swing_lows = StructureIndicators._detect_swings(high, low, lookback=5)
        features['swing_high'] = swing_highs
        features['swing_low'] = swing_lows

        # Structure classification: HH-HL = 1 (bullish), LH-LL = -1 (bearish), mixed = 0
        structure_dir = StructureIndicators._classify_structure(swing_highs, swing_lows, n_len)
        features['structure_direction'] = structure_dir

        # ═══════════════════════════════════════════════════════════
        # E08: CHoCH Detector (Change of Character)
        # Primera quiebra de estructura opuesta.
        # ═══════════════════════════════════════════════════════════
        choch = StructureIndicators._detect_choch(close, swing_highs, swing_lows, n_len)
        features['choch_signal'] = choch

        # ═══════════════════════════════════════════════════════════
        # E10: Fibonacci Retracement Zones
        # Distancia del precio a niveles Fib del último impulso.
        # ═══════════════════════════════════════════════════════════
        fib_dist = StructureIndicators._fibonacci_distance(close, high, low, n_len, lookback=50)
        features['fib_382_dist'] = fib_dist.get('fib_382_dist', np.zeros(n_len))
        features['fib_618_dist'] = fib_dist.get('fib_618_dist', np.zeros(n_len))
        features['fib_zone'] = fib_dist.get('fib_zone', np.zeros(n_len))

        return features

    # ────────────────────────────────────────────────────────────
    # IMPLEMENTACIONES INTERNAS
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _hurst_exponent(close, window=100):
        """Calcula Hurst Exponent con Rescaled Range (R/S) análisis."""
        n = len(close)
        result = np.full(n, 0.50)  # Default: random walk

        if n < window:
            return result

        for i in range(window, n):
            segment = close[i - window:i]
            if np.std(segment) == 0:
                continue

            returns = np.diff(segment)
            if len(returns) == 0:
                continue

            mean_ret = np.mean(returns)
            deviations = np.cumsum(returns - mean_ret)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(returns, ddof=1)

            if S > 0 and R > 0:
                # H = log(R/S) / log(n)
                rs = R / S
                result[i] = np.log(rs) / np.log(len(returns))

        # Clamp to valid range [0, 1]
        result = np.clip(result, 0.0, 1.0)
        return result

    @staticmethod
    def _fractal_dimension(close, window=30):
        """Fractal Dimension Index usando el método de Higuchi simplificado."""
        n = len(close)
        result = np.full(n, 1.5)  # Default: entre 1 (línea) y 2 (ruido)

        if n < window:
            return result

        for i in range(window, n):
            segment = close[i - window:i]
            max_val = np.max(segment)
            min_val = np.min(segment)
            rng = max_val - min_val

            if rng == 0:
                result[i] = 1.5
                continue

            # Normalized length of the price path
            path_length = np.sum(np.abs(np.diff(segment)))
            normalized = path_length / rng

            # FDI approximation: 1 + log(normalized) / log(2 * (window - 1))
            if normalized > 0:
                result[i] = 1.0 + np.log(normalized) / np.log(2 * (window - 1))

        result = np.clip(result, 1.0, 2.0)
        return result

    @staticmethod
    def _detect_swings(high, low, lookback=5):
        """Detecta Swing Highs y Swing Lows con confirmación de lookback."""
        n = len(high)
        swing_highs = np.zeros(n)  # Precio del swing high, 0 si no hay
        swing_lows = np.zeros(n)

        for i in range(lookback, n - lookback):
            # Swing High: high[i] es mayor que los lookback anteriores y posteriores
            if high[i] == np.max(high[i - lookback:i + lookback + 1]):
                swing_highs[i] = high[i]

            # Swing Low: low[i] es menor que los lookback anteriores y posteriores
            if low[i] == np.min(low[i - lookback:i + lookback + 1]):
                swing_lows[i] = low[i]

        return swing_highs, swing_lows

    @staticmethod
    def _classify_structure(swing_highs, swing_lows, n_len):
        """Clasifica la estructura de mercado: HH-HL=1, LH-LL=-1, mixed=0."""
        result = np.zeros(n_len)

        # Collect the last 4 swing points
        high_indices = np.where(swing_highs > 0)[0]
        low_indices = np.where(swing_lows > 0)[0]

        if len(high_indices) < 2 or len(low_indices) < 2:
            return result

        for i in range(max(high_indices[1], low_indices[1]), n_len):
            # Find last 2 swing highs and lows before index i
            prev_highs = high_indices[high_indices <= i]
            prev_lows = low_indices[low_indices <= i]

            if len(prev_highs) < 2 or len(prev_lows) < 2:
                continue

            sh1, sh2 = swing_highs[prev_highs[-2]], swing_highs[prev_highs[-1]]
            sl1, sl2 = swing_lows[prev_lows[-2]], swing_lows[prev_lows[-1]]

            hh = sh2 > sh1  # Higher High
            hl = sl2 > sl1  # Higher Low
            lh = sh2 < sh1  # Lower High
            ll = sl2 < sl1  # Lower Low

            if hh and hl:
                result[i] = 1.0   # Bullish structure
            elif lh and ll:
                result[i] = -1.0  # Bearish structure
            # else: mixed = 0

        return result

    @staticmethod
    def _detect_choch(close, swing_highs, swing_lows, n_len):
        """
        Change of Character: primera quiebra de un swing en dirección opuesta.
        +1 = CHoCH alcista (precio rompe un swing high en tendencia bajista)
        -1 = CHoCH bajista (precio rompe un swing low en tendencia alcista)
        """
        result = np.zeros(n_len)
        last_structure = 0  # 0=undefined, 1=bullish, -1=bearish

        high_indices = np.where(swing_highs > 0)[0]
        low_indices = np.where(swing_lows > 0)[0]

        if len(high_indices) < 2 or len(low_indices) < 2:
            return result

        for i in range(1, n_len):
            # Update structure from swing classification
            prev_highs = high_indices[high_indices < i]
            prev_lows = low_indices[low_indices < i]

            if len(prev_highs) >= 2 and len(prev_lows) >= 2:
                sh1, sh2 = swing_highs[prev_highs[-2]], swing_highs[prev_highs[-1]]
                sl1, sl2 = swing_lows[prev_lows[-2]], swing_lows[prev_lows[-1]]

                if sh2 > sh1 and sl2 > sl1:
                    last_structure = 1
                elif sh2 < sh1 and sl2 < sl1:
                    last_structure = -1

            # CHoCH detection
            if last_structure == -1 and len(prev_highs) >= 1:
                # In bearish: if price breaks above last swing high → CHoCH bullish
                last_sh = swing_highs[prev_highs[-1]]
                if last_sh > 0 and close[i] > last_sh and (i == 0 or close[i - 1] <= last_sh):
                    result[i] = 1.0

            elif last_structure == 1 and len(prev_lows) >= 1:
                # In bullish: if price breaks below last swing low → CHoCH bearish
                last_sl = swing_lows[prev_lows[-1]]
                if last_sl > 0 and close[i] < last_sl and (i == 0 or close[i - 1] >= last_sl):
                    result[i] = -1.0

        return result

    @staticmethod
    def _fibonacci_distance(close, high, low, n_len, lookback=50):
        """
        Calcula la distancia del precio a niveles Fibonacci del último impulso.
        Retorna distancias normalizadas a 0.382 y 0.618.
        """
        fib_382 = np.zeros(n_len)
        fib_618 = np.zeros(n_len)
        fib_zone = np.zeros(n_len)

        if n_len < lookback:
            return {'fib_382_dist': fib_382, 'fib_618_dist': fib_618, 'fib_zone': fib_zone}

        for i in range(lookback, n_len):
            seg_high = np.max(high[i - lookback:i])
            seg_low = np.min(low[i - lookback:i])
            rng = seg_high - seg_low

            if rng == 0:
                continue

            # Fibonacci levels
            level_382 = seg_high - 0.382 * rng
            level_618 = seg_high - 0.618 * rng

            # Normalized distance from current price to each level
            fib_382[i] = (close[i] - level_382) / rng
            fib_618[i] = (close[i] - level_618) / rng

            # Zone classification: 0.0-0.236=premium, 0.236-0.5=neutral, 0.5-1.0=discount
            price_pos = (seg_high - close[i]) / rng  # 0=at top, 1=at bottom
            if price_pos < 0.236:
                fib_zone[i] = 1.0   # Premium zone (overextended)
            elif price_pos > 0.618:
                fib_zone[i] = -1.0  # Discount zone (undervalued)
            # else 0 = equilibrium zone

        return {'fib_382_dist': fib_382, 'fib_618_dist': fib_618, 'fib_zone': fib_zone}
