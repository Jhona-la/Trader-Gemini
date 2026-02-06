import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.ml_strategy import MLStrategyHybridUltimate
from data.feature_store import FeatureStore
from core.ml_governance import MLGovernance
from queue import Queue
import os
import sqlite3

def verify_phase_12():
    print("🧪 [TEST] Verificando Fase 12: ML Governance & Feature Store...")
    
    # 1. Setup mock components
    events_queue = Queue()
    symbol = "BTC/USDT"
    
    class MockDataProvider:
        def __init__(self):
            self.events_queue = events_queue
    
    data_provider = MockDataProvider()
    strategy = MLStrategyHybridUltimate(data_provider, events_queue, symbol=symbol)
    
    # 2. Test Feature Store Save
    print("📝 Generando datos sintéticos para Feature Store...")
    bars = []
    base_ts = datetime.now()
    for i in range(200):
        bars.append({
            'datetime': base_ts + timedelta(minutes=i),
            'open': 70000 + i,
            'high': 70100 + i,
            'low': 69900 + i,
            'close': 70050 + i,
            'volume': 100 + i
        })
    
    # Trigger feature calculation and storage
    df = strategy._prepare_features(bars)
    
    # Check if DB has data
    conn = sqlite3.connect("data/feature_store.db")
    count = conn.execute("SELECT COUNT(*) FROM features WHERE symbol = ?", (symbol,)).fetchone()[0]
    conn.close()
    
    if count >= 200:
        print(f"✅ Feature Store: {count} registros encontrados para {symbol}.")
    else:
        print(f"❌ Feature Store: Solo se encontraron {count} registros.")

    # 3. Test Feature Store Lookup
    print("🔍 Verificando recuperación de cache...")
    cached_df = strategy._prepare_features(bars)
    if 'rsi_14' in cached_df.columns:
        print("✅ Cache Lookup: Features recuperadas exitosamente.")
    else:
        print("❌ Cache Lookup: No se encontraron indicadores en el DF retornado.")

    # 4. Test ML Governance Registration
    print("⚖️ Verificando Registro de Gobernanza...")
    strategy.last_training_score = 1.8  # Simulamos un buen Sharpe para Quality Gate
    strategy.performance_history.append(1)
    strategy.performance_history.append(1)
    strategy.performance_history.append(1)
    
    from sklearn.ensemble import RandomForestClassifier
    strategy.rf_model = RandomForestClassifier()
    strategy.xgb_model = RandomForestClassifier()
    strategy.gb_model = RandomForestClassifier()
    
    strategy._save_models()
    
    conn = sqlite3.connect("data/feature_store.db")
    prod_model = conn.execute("SELECT model_id, is_production FROM model_registry WHERE symbol = ? ORDER BY created_at DESC", (symbol,)).fetchone()
    conn.close()
    
    if prod_model:
        print(f"✅ Governance Registry: Modelo {prod_model[0]} registrado.")
        if prod_model[1] == 1:
            print("✅ Quality Gate: Modelo promovido a PRODUCCIÓN exitosamente.")
        else:
            print("❌ Quality Gate: El modelo no fue promovido a pesar del score alto.")
    else:
        print("❌ Governance Registry: No se encontró registro del modelo.")

if __name__ == "__main__":
    try:
        verify_phase_12()
    except Exception as e:
        print(f"💥 Error durante la verificación: {e}")
        import traceback
        traceback.print_exc()
