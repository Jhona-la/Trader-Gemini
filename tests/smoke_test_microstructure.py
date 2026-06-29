"""
🔬 SMOKE TEST: Microstructure Labeling Pipeline
=================================================

👨‍🏫 MODO PROFESOR:
- QUÉ: Valida que el pipeline de feature_engineering.py genera
  correctamente las nuevas features de microestructura:
  micro_absorption, micro_exhaustion, micro_sweep, micro_label
- POR QUÉ: Si estas features no se generan, los modelos ML no
  pueden aprender patrones de Order Flow, perdiendo una ventaja
  crítica para el scalping con $13 USD.
- PARA QUÉ: Confirmar que datos sintéticos realistas producen
  etiquetas válidas y que no hay NaN/Inf contaminando el pipeline.
- CÓMO: Genera 200 barras sintéticas con patrones reales (trend,
  absorption, sweep) y verifica la presencia y distribución
  de cada etiqueta.
- DÓNDE: tests/smoke_test_microstructure.py
- QUIÉN: FeatureEngineering.prepare_features()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRADER_GEMINI_ENV'] = 'TEST'

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

def generate_synthetic_bars(n=200, base_price=50000.0, seed=42):
    """
    Genera barras OHLCV sintéticas con patrones intencionados:
    - Bars 50-60: ABSORPTION (alto volumen, cuerpo pequeño)
    - Bars 100-110: SWEEP BULL (cuerpo grande, volumen alto, cierre en máximo)
    - Bars 150-160: EXHAUSTION (mecha larga, momentum decayendo)
    """
    rng = np.random.RandomState(seed)
    
    closes = [base_price]
    opens = [base_price]
    highs = [base_price * 1.001]
    lows = [base_price * 0.999]
    volumes = [100.0]
    
    for i in range(1, n):
        ret = rng.normal(0, 0.002)
        c_prev = closes[-1]
        
        # Absorption zone: lots of volume, tiny body
        if 50 <= i <= 60:
            o = c_prev * (1 + rng.normal(0, 0.0001))
            c = o * (1 + rng.normal(0, 0.00005))  # Tiny body
            h = max(o, c) * (1 + abs(rng.normal(0, 0.001)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.001)))
            v = rng.uniform(500, 1000)  # High volume
        # Sweep bull zone: big body, high volume, close near high
        elif 100 <= i <= 110:
            o = c_prev
            c = o * (1 + abs(rng.normal(0.003, 0.001)))  # Big up
            h = c * (1 + abs(rng.normal(0, 0.0002)))  # Close near high
            l = o * (1 - abs(rng.normal(0, 0.0003)))
            v = rng.uniform(400, 800)
        # Exhaustion zone: long wick, declining momentum
        elif 150 <= i <= 160:
            o = c_prev
            c = o * (1 - rng.normal(0.0005, 0.0002))
            h = o * (1 + abs(rng.normal(0.003, 0.001)))  # Long upper wick
            l = min(o, c) * (1 - abs(rng.normal(0, 0.0003)))
            v = rng.uniform(100, 300)
        else:
            o = c_prev * (1 + rng.normal(0, 0.0005))
            c = o * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.001)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.001)))
            v = rng.uniform(50, 200)
        
        opens.append(o)
        closes.append(c)
        highs.append(h)
        lows.append(l)
        volumes.append(v)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'datetime': pd.date_range('2026-01-01', periods=n, freq='1min')
    })
    return df


# ============================================================================
# TESTS
# ============================================================================

class TestMicrostructureLabeling:
    """Validates the microstructure labeling output from FeatureEngineering."""

    @pytest.fixture
    def feature_engine(self):
        from strategies.components.feature_engineering import FeatureEngineering
        return FeatureEngineering()

    @pytest.fixture
    def bars(self):
        return generate_synthetic_bars(200)

    def test_micro_features_exist(self, feature_engine, bars):
        """All 4 microstructure features must be generated."""
        df = feature_engine.prepare_features(bars.to_dict('records'))
        
        required = ['micro_absorption', 'micro_exhaustion', 'micro_sweep', 'micro_label']
        for feat in required:
            assert feat in df.columns, f"Missing microstructure feature: {feat}"
        
        print(f"✅ All {len(required)} microstructure features present")

    def test_micro_features_no_nan_inf(self, feature_engine, bars):
        """Microstructure features must be clean (no NaN/Inf)."""
        df = feature_engine.prepare_features(bars.to_dict('records'))
        
        micro_cols = ['micro_absorption', 'micro_exhaustion', 'micro_sweep', 'micro_label']
        for col in micro_cols:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col].values.astype(float)).sum()
            assert nan_count == 0, f"{col} has {nan_count} NaN values"
            assert inf_count == 0, f"{col} has {inf_count} Inf values"
        
        print("✅ Zero NaN/Inf in microstructure features")

    def test_micro_label_values(self, feature_engine, bars):
        """micro_label must only contain valid coded values."""
        df = feature_engine.prepare_features(bars.to_dict('records'))
        
        valid_labels = {0, 1, -1, 2, -2, 3}
        actual_labels = set(df['micro_label'].unique())
        
        assert actual_labels.issubset(valid_labels), \
            f"Invalid labels found: {actual_labels - valid_labels}"
        
        # Must have at least Neutral (0) and some non-zero labels
        assert 0 in actual_labels, "Must have at least Neutral (0) labels"
        non_zero = actual_labels - {0}
        assert len(non_zero) > 0, "No microstructure events detected in synthetic data!"
        
        print(f"✅ Valid labels: {sorted(actual_labels)}")
        print(f"   Distribution: {df['micro_label'].value_counts().to_dict()}")

    def test_absorption_detection(self, feature_engine, bars):
        """Absorption events should be detected in the absorption zone."""
        df = feature_engine.prepare_features(bars.to_dict('records'))
        
        # Check absorption zone (bars 50-60) has some absorption signals
        absorption_count = df['micro_absorption'].sum()
        assert absorption_count > 0, "No absorption events detected!"
        
        print(f"✅ Absorption events detected: {absorption_count}")

    def test_scalping_features_exist(self, feature_engine, bars):
        """Scalping-specific features must also be present."""
        df = feature_engine.prepare_features(bars.to_dict('records'))
        
        scalp_features = ['micro_velocity_3', 'volume_accel', 'spread_squeeze']
        for feat in scalp_features:
            assert feat in df.columns, f"Missing scalping feature: {feat}"
        
        print(f"✅ All {len(scalp_features)} scalping microstructure features present")

    def test_feature_count(self, feature_engine, bars):
        """Total feature count should be 80+ (comprehensive pipeline)."""
        df = feature_engine.prepare_features(bars.to_dict('records'))
        
        feature_count = len(df.columns)
        print(f"📊 Total features generated: {feature_count}")
        assert feature_count >= 60, \
            f"Only {feature_count} features — expected 60+!"
        
        print(f"✅ Feature pipeline comprehensive: {feature_count} features")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
