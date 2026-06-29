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
        import polars as pl
        features = {}

        v_sma_20 = pl.col('volume').rolling_mean(window_size=20)
        features['volume_sma_20'] = v_sma_20
        features['volume_ratio'] = pl.when(v_sma_20 != 0).then(pl.col('volume') / v_sma_20).otherwise(0.0)

        # ═══════════════════════════════════════════════════════════
        # F01: OBV (On-Balance Volume)
        # Acumulación/distribución. OBV subiendo con precio lateral = acumulación
        # ═══════════════════════════════════════════════════════════
        obv = talib.OBV(close, volume)
        features['obv'] = pl.Series('obv', obv)
        obv_sma = pl.col('obv').rolling_mean(window_size=20)
        features['obv_sma'] = obv_sma
        features['obv_ratio'] = pl.when(obv_sma != 0).then(pl.col('obv') / obv_sma).otherwise(1.0)

        # OBV divergence: precio sube + OBV baja = distribución
        if n_len > 10:
            price_trend = talib.LINEARREG_SLOPE(close, 10)
            obv_trend = talib.LINEARREG_SLOPE(obv.astype(np.float64), 10)
            features['obv_divergence'] = pl.Series('obv_divergence', np.where(
                (price_trend > 0) & (obv_trend < 0), -1.0,
                np.where((price_trend < 0) & (obv_trend > 0), 1.0, 0.0)
            ))

        # ═══════════════════════════════════════════════════════════
        # F02: CVD (Cumulative Volume Delta) — CALCULADO
        # Diferencia entre volumen de compras y ventas agresoras.
        # Si viene inyectado del footprint, usamos ese. Si no, lo
        # aproximamos con el método tick-rule (close vs prev close).
        # ═══════════════════════════════════════════════════════════
        if 'cvd' in df.columns:
            features['cvd'] = pl.col('cvd')
        else:
            # Aproximación tick-rule: si close > prev_close, el volumen
            # se atribuye a compradores agresores, y viceversa.
            price_change = pl.col('close') - pl.col('close').shift(1)
            buy_volume = pl.when(price_change > 0).then(pl.col('volume')).when(price_change < 0).then(0.0).otherwise(pl.col('volume') * 0.5)
            sell_volume = pl.when(price_change < 0).then(pl.col('volume')).when(price_change > 0).then(0.0).otherwise(pl.col('volume') * 0.5)
            delta = buy_volume - sell_volume
            features['cvd'] = delta.cum_sum().fill_null(0.0)

        # F03: CVD Rate of Change — velocidad del cambio del CVD
        if n_len > 14:
            features['cvd_roc_1'] = features['cvd'].pct_change(1) * 100
            features['cvd_roc_5'] = features['cvd'].pct_change(5) * 100
            features['cvd_roc_14'] = features['cvd'].pct_change(14) * 100

        # CVD divergence with price

        if n_len > 10:
            if 'cvd' in df.columns:
                cvd_arr = df['cvd'].to_numpy()
            else:
                price_change_np = np.diff(close, prepend=close[0])
                buy_np = np.where(price_change_np > 0, volume, np.where(price_change_np < 0, 0, volume * 0.5))
                sell_np = np.where(price_change_np < 0, volume, np.where(price_change_np > 0, 0, volume * 0.5))
                cvd_arr = np.cumsum(buy_np - sell_np)
                
            cvd_trend = talib.LINEARREG_SLOPE(cvd_arr.astype(np.float64), 10)
            features['cvd_divergence'] = pl.Series('cvd_divergence', np.where(
                (price_trend > 0) & (cvd_trend < 0), -1.0,  # Price up, CVD down = bearish
                np.where((price_trend < 0) & (cvd_trend > 0), 1.0, 0.0)  # Price down, CVD up = bullish
            ))

        # ═══════════════════════════════════════════════════════════
        # F02b: Chaikin A/D Line & Oscillator
        # ═══════════════════════════════════════════════════════════
        ad = talib.AD(high, low, close, volume)
        features['ad_line'] = pl.Series('ad_line', ad)
        if n_len > 10:
            features['adosc'] = pl.Series('adosc', talib.ADOSC(high, low, close, volume, 3, 10))

        # ═══════════════════════════════════════════════════════════
        # F04/F05: VWAP (Rolling 20 y 50)
        # ═══════════════════════════════════════════════════════════
        typ_price = (pl.col('high') + pl.col('low') + pl.col('close')) / 3
        vp = typ_price * pl.col('volume')
        if n_len > 20:
            vwap_20 = vp.rolling_sum(window_size=20) / pl.col('volume').rolling_sum(window_size=20)
            features['vwap_20'] = vwap_20
            features['vwap_dist'] = pl.when(vwap_20 != 0).then((pl.col('close') - vwap_20) / vwap_20).otherwise(0.0)
        if n_len > 50:
            vwap_50 = vp.rolling_sum(window_size=50) / pl.col('volume').rolling_sum(window_size=50)
            features['vwap_50'] = vwap_50
            features['vwap_50_dist'] = pl.when(vwap_50 != 0).then((pl.col('close') - vwap_50) / vwap_50).otherwise(0.0)

        # ═══════════════════════════════════════════════════════════
        # F07: Ease of Movement (EMV)
        # ═══════════════════════════════════════════════════════════
        if n_len > 14:
            hl2 = (pl.col('high') + pl.col('low')) / 2
            hl2_prev = hl2.shift(1).fill_null(hl2)
            box_ratio = pl.when((pl.col('high') - pl.col('low')) != 0).then(pl.col('volume') / (pl.col('high') - pl.col('low'))).otherwise(1.0)
            emv_raw = pl.when(box_ratio != 0).then((hl2 - hl2_prev) / box_ratio).otherwise(0.0)
            features['emv_14'] = emv_raw.rolling_mean(window_size=14)

        # ═══════════════════════════════════════════════════════════
        # F08: Chaikin Money Flow (CMF, período 21)
        # CMF > 0: acumulación neta. CMF < 0: distribución neta.
        # ═══════════════════════════════════════════════════════════
        if n_len > 21:
            mfm = pl.when((pl.col('high') - pl.col('low')) != 0).then(
                ((pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))) / (pl.col('high') - pl.col('low'))
            ).otherwise(0.0)
            mfv = mfm * pl.col('volume')
            cmf = mfv.rolling_sum(window_size=21) / pl.col('volume').rolling_sum(window_size=21)
            features['cmf_21'] = cmf.fill_null(0.0)

        # ═══════════════════════════════════════════════════════════
        # F10: Force Index (período 2 y 13)
        # Fuerza = cambio_precio × volumen. Spike = agresión institucional.
        # ═══════════════════════════════════════════════════════════
        price_change_expr = pl.col('close') - pl.col('close').shift(1).fill_null(pl.col('close'))
        force_raw = price_change_expr * pl.col('volume')
        if n_len > 2:
            features['force_index_2'] = force_raw.ewm_mean(span=2, adjust=False)
        if n_len > 13:
            features['force_index_13'] = force_raw.ewm_mean(span=13, adjust=False)

        # ═══════════════════════════════════════════════════════════
        # F11: Volume Oscillator (ratio de medias de volumen)
        # ═══════════════════════════════════════════════════════════
        if n_len > 10:
            vol_sma_5 = pl.col('volume').rolling_mean(window_size=5)
            vol_sma_10 = pl.col('volume').rolling_mean(window_size=10)
            features['vol_oscillator'] = pl.when(vol_sma_10 != 0).then((vol_sma_5 - vol_sma_10) / vol_sma_10).otherwise(0.0)

        # ═══════════════════════════════════════════════════════════
        # F12: Micro Imbalance (si viene del footprint)
        # ═══════════════════════════════════════════════════════════
        if 'micro_imbalance' in df.columns:
            features['micro_imbalance'] = pl.col('micro_imbalance')

        # ═══════════════════════════════════════════════════════════
        # F14: Large Trade Detection (Whale Prints)
        # Trades con volumen > 2σ del promedio = actividad institucional
        # ═══════════════════════════════════════════════════════════
        if n_len > 20:
            vol_mean = pl.col('volume').rolling_mean(window_size=20)
            vol_std = pl.col('volume').rolling_std(window_size=20)
            whale_threshold = vol_mean + 2 * vol_std
            features['whale_print'] = pl.when(pl.col('volume') > whale_threshold).then(1.0).otherwise(0.0)
            # Direction of whale: if close > open, buyer whale
            if 'open' in df.columns:
                features['whale_direction'] = pl.when(
                    (pl.col('volume') > whale_threshold) & (pl.col('close') > pl.col('open'))
                ).then(1.0).when(
                    (pl.col('volume') > whale_threshold) & (pl.col('close') < pl.col('open'))
                ).then(-1.0).otherwise(0.0)

        return features
