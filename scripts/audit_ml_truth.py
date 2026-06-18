import os
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb_lib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum.mmap_storage import QuantumMMAP
from strategies.components.feature_engineering import FeatureEngineering

def audit_ml_truth(symbol="BTCUSDT", horizon="SCALPING"):
    print(f"🔍 [AUDIT ML TRUTH] OOS Verification for {symbol} ({horizon})")
    
    # 1. Load Real Data from Quantum Data Lake
    mmap_path = f"data/quantum_lake/{symbol}.qbin"
    if not os.path.exists(mmap_path):
        print(f"❌ No DataLake for {symbol} en {mmap_path}")
        return
        
    storage = QuantumMMAP(symbol)
    df_raw = storage.to_dataframe()
    if len(df_raw) < 110000:
        print(f"⚠️ Not enough candles in DataLake ({len(df_raw)})")
        return
        
    df_raw = df_raw.sort_index()
    
    # 2. Extract Validation Set Indices
    # In sandbox we do:
    # train_size = int(len(df) * 0.7)
    train_size = int(len(df_raw) * 0.7)
    print(f"📊 Dataset Total: {len(df_raw)} velas")
    print(f"📈 Train Set Size (0-{train_size}): {train_size} velas")
    print(f"📉 Validation Set Size ({train_size}-end): {len(df_raw) - train_size} velas")
    
    # 3. Load Model and Features Metadata
    clean_sym = symbol.replace("/", "")
    
    import glob
    meta_paths = glob.glob(f".models/{clean_sym}_*calping_meta.joblib") + glob.glob(f".models/{clean_sym}_*CALPING_meta.joblib")
    ubj_paths = glob.glob(f".models/{clean_sym}_*calping_xgb.ubj") + glob.glob(f".models/{clean_sym}_*CALPING_xgb.ubj")
    
    if not meta_paths or not ubj_paths:
        print(f"❌ Model missing for {symbol} in .models/")
        return
        
    meta_path = meta_paths[0]
    ubj_path = ubj_paths[0]
    
    model_data = joblib.load(meta_path)
    feature_cols = model_data.get('feature_cols', [])
    if not feature_cols:
        print(f"❌ No feature_cols found in meta.joblib for {symbol}")
        return
        
    xgb = xgb_lib.XGBClassifier()
    xgb.load_model(ubj_path)
    
    # 4. Feature Engineering on FULL dataset
    # Computamos features completas. Esto NO genera leakage OOS porque 
    # FeatureEngineering calcula medias moviles hacia atrás unicamente.
    print(f"🧠 Computing Features...")
    fe = FeatureEngineering()
    df_features = fe.prepare_features(df_raw, symbol=symbol, horizon=horizon)
    df_features = df_features.to_pandas()
    
    # Cortar el OOS
    df_val = df_features.iloc[train_size:].copy()
    
    # Crear Targets OOS (Idéntico a train_supreme.py)
    closes = df_val['close'].values.astype(np.float32)
    lookahead = 15 if horizon == 'SCALPING' else 60
    
    N = len(closes)
    y_true = np.zeros(N, dtype=int)
    
    for i in range(N - lookahead):
        p0 = closes[i]
        pf = closes[i+lookahead]
        y_true[i] = 1 if pf > p0 else 0
        
    # Descartar las ultimas velas que no tienen target resoluble
    df_val = df_val.iloc[:-lookahead]
    y_true = y_true[:-lookahead]
    
    # Fill missing columns por si acaso
    for c in feature_cols:
        if c not in df_val.columns:
            df_val[c] = 0.0
            
    X_val = df_val[feature_cols].values.astype(np.float32)
    
    # 5. Predict
    print(f"🔮 Predicting OOS ({len(X_val)} samples)...")
    y_pred = xgb.predict(X_val)
    
    # 6. Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*60)
    print(f"🛡️ AUDITORÍA DE LA VERDAD PREDICTIVA: {symbol} ({horizon})")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   Precision: {prec*100:.2f}%")
    print(f"   Recall   : {rec*100:.2f}%")
    print(f"   Base Rate (Ratio de 1s): {(np.sum(y_true)/len(y_true))*100:.2f}%")
    
    if acc < 0.55:
        print(f"\n❌ VEREDICTO: RUIDO DETECTADO (< 55% Acc).")
        print("   El modelo NO superó la prueba de fuego OOS.")
    else:
        print(f"\n✅ VEREDICTO: EDGE PROBABLE (> 55% Acc).")
        print("   El modelo posee poder predictivo real fuera de muestra.")
    print("="*60 + "\n")
    
    return acc

if __name__ == "__main__":
    import logging
    logging.getLogger('core.transparent_logger').setLevel(logging.CRITICAL) # Ocultar logs del core
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "COMPUSDT", "WIFUSDT"]
    for sym in symbols:
        audit_ml_truth(sym, "SCALPING")
