"""
MÓDULO OMEGA — Categoría 4: Indicadores de Volumen y Flujo (F01-F14)
====================================================================

QUÉ: Indicadores basados en volumen que revelan la presión real
  de compradores y vendedores detrás de cada movimiento de precio.

POR QUÉ: El precio puede mentir (manipulación de mecha), pero el volumen
  combinado con la dirección del aggressor (CVD) es el indicador más
  honesto de intención del mercado en corto plazo.

PARA QUÉ: Detectar acumulación/distribución, divergencias precio-flujo,
  y presión institucional agresiva.

CÓMO: Cálculos vectorizados con numpy/talib.
CUÁNDO: En cada ciclo de feature engineering.
DÓNDE: strategies/indicators/volume.py
QUIÉN: Quant Developer
"""

import numpy as np
import talib
from utils.math_helpers import safe_div


class VolumeIndicators:
    @staticmethod
    def calculate_all(df, close, high, low, volume, n_len):
        """Calcula todos los indicadores de Volumen F01-F14."""
        features = {}

        v_sma_20 = talib.SMA(volume, 20)
        features['volume_sma_20'] = v_sma_20
        features['volume_ratio'] = np.where(v_sma_20 != 0, volume / v_sma_20, 0.0)

        # ═══════════════════════════════════════════════════════════
        # F01: OBV (On-Balance Volume)
        # Acumulación/distribución. OBV subiendo con precio lateral = acumulación
        # ═══════════════════════════════════════════════════════════
        obv = talib.OBV(close, volume)
        obv_sma = talib.SMA(obv, 20)
        features['obv'] = obv
        features['obv_sma'] = obv_sma
        features['obv_ratio'] = np.where(obv_sma != 0, obv / obv_sma, 1.0)

        # OBV divergence: precio sube + OBV baja = distribución
        if n_len > 10:
            price_trend = talib.LINEARREG_SLOPE(close, 10)
            obv_trend = talib.LINEARREG_SLOPE(obv.astype(np.float64), 10)
            features['obv_divergence'] = np.where(
                (price_trend > 0) & (obv_trend < 0), -1.0,
                np.where((price_trend < 0) & (obv_trend > 0), 1.0, 0.0)
            )

        # ═══════════════════════════════════════════════════════════
        # F02: CVD (Cumulative Volume Delta) — CALCULADO
        # Diferencia entre volumen de compras y ventas agresoras.
        # Si viene inyectado del footprint, usamos ese. Si no, lo
        # aproximamos con el método tick-rule (close vs prev close).
        # ═══════════════════════════════════════════════════════════
        if 'cvd' in df.columns:
            features['cvd'] = df['cvd'].to_numpy()
        else:
            # Aproximación tick-rule: si close > prev_close, el volumen
            # se atribuye a compradores agresores, y viceversa.
            price_change = np.diff(close, prepend=close[0])
            buy_volume = np.where(price_change > 0, volume,
                         np.where(price_change < 0, 0, volume * 0.5))
            sell_volume = np.where(price_change < 0, volume,
                          np.where(price_change > 0, 0, volume * 0.5))
            delta = buy_volume - sell_volume
            features['cvd'] = np.cumsum(delta)

        # F03: CVD Rate of Change — velocidad del cambio del CVD
        if n_len > 5:
            cvd_arr = features['cvd'].astype(np.float64)
            features['cvd_roc_5'] = talib.ROC(cvd_arr, 5)
        if n_len > 14:
            features['cvd_roc_14'] = talib.ROC(cvd_arr, 14)

        # CVD divergence with price
        if n_len > 10:
            cvd_trend = talib.LINEARREG_SLOPE(features['cvd'].astype(np.float64), 10)
            features['cvd_divergence'] = np.where(
                (price_trend > 0) & (cvd_trend < 0), -1.0,  # Price up, CVD down = bearish
                np.where((price_trend < 0) & (cvd_trend > 0), 1.0, 0.0)  # Price down, CVD up = bullish
            )

        # ═══════════════════════════════════════════════════════════
        # F02b: Chaikin A/D Line & Oscillator
        # ═══════════════════════════════════════════════════════════
        ad = talib.AD(high, low, close, volume)
        features['ad_line'] = ad
        if n_len > 10:
            features['adosc'] = talib.ADOSC(high, low, close, volume, 3, 10)

        # ═══════════════════════════════════════════════════════════
        # F04/F05: VWAP (Rolling 20 y 50)
        # ═══════════════════════════════════════════════════════════
        typ_price = (high + low + close) / 3
        vp = typ_price * volume
        if n_len > 20:
            features['vwap_20'] = safe_div(talib.SUM(vp, 20), talib.SUM(volume, 20), typ_price)
            features['vwap_dist'] = safe_div(close - features['vwap_20'], features['vwap_20'])
        if n_len > 50:
            features['vwap_50'] = safe_div(talib.SUM(vp, 50), talib.SUM(volume, 50), typ_price)
            features['vwap_50_dist'] = safe_div(close - features['vwap_50'], features['vwap_50'])

        # ═══════════════════════════════════════════════════════════
        # F07: Ease of Movement (EMV)
        # ═══════════════════════════════════════════════════════════
        if n_len > 14:
            hl2 = (high + low) / 2
            hl2_prev = np.roll(hl2, 1)
            hl2_prev[0] = hl2[0]
            box_ratio = safe_div(volume, (high - low), 1.0)
            emv_raw = safe_div(hl2 - hl2_prev, box_ratio, 0.0)
            features['emv_14'] = talib.SMA(emv_raw, 14)

        # ═══════════════════════════════════════════════════════════
        # F08: Chaikin Money Flow (CMF, período 21)
        # CMF > 0: acumulación neta. CMF < 0: distribución neta.
        # ═══════════════════════════════════════════════════════════
        if n_len > 21:
            mfm = safe_div((close - low) - (high - close), high - low, 0.0)  # Money Flow Multiplier
            mfv = mfm * volume  # Money Flow Volume
            features['cmf_21'] = safe_div(talib.SUM(mfv, 21), talib.SUM(volume, 21), 0.0)

        # ═══════════════════════════════════════════════════════════
        # F10: Force Index (período 2 y 13)
        # Fuerza = cambio_precio × volumen. Spike = agresión institucional.
        # ═══════════════════════════════════════════════════════════
        price_change_arr = np.diff(close, prepend=close[0])
        force_raw = price_change_arr * volume
        if n_len > 2:
            from utils.math_kernel import calculate_ema_jit
            features['force_index_2'] = calculate_ema_jit(force_raw.astype(np.float64), 2)
        if n_len > 13:
            features['force_index_13'] = calculate_ema_jit(force_raw.astype(np.float64), 13)

        # ═══════════════════════════════════════════════════════════
        # F11: Volume Oscillator (ratio de medias de volumen)
        # ═══════════════════════════════════════════════════════════
        if n_len > 10:
            vol_sma_5 = talib.SMA(volume, 5)
            vol_sma_10 = talib.SMA(volume, 10)
            features['vol_oscillator'] = safe_div(vol_sma_5 - vol_sma_10, vol_sma_10)

        # ═══════════════════════════════════════════════════════════
        # F12: Micro Imbalance (si viene del footprint)
        # ═══════════════════════════════════════════════════════════
        if 'micro_imbalance' in df.columns:
            features['micro_imbalance'] = df['micro_imbalance'].to_numpy()

        # ═══════════════════════════════════════════════════════════
        # F14: Large Trade Detection (Whale Prints)
        # Trades con volumen > 2σ del promedio = actividad institucional
        # ═══════════════════════════════════════════════════════════
        if n_len > 20:
            vol_mean = talib.SMA(volume, 20)
            vol_std = talib.STDDEV(volume, 20, 1)
            whale_threshold = vol_mean + 2 * vol_std
            features['whale_print'] = np.where(volume > whale_threshold, 1.0, 0.0)
            # Direction of whale: if close > open, buyer whale
            if 'open' in df.columns:
                open_arr = df['open'].to_numpy().astype(np.float64)
                features['whale_direction'] = np.where(
                    (volume > whale_threshold) & (close > open_arr), 1.0,
                    np.where((volume > whale_threshold) & (close < open_arr), -1.0, 0.0)
                )

        return features
