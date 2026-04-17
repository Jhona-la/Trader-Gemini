import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score

# Ensure root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from strategies.components.feature_engineering import FeatureEngineering

def train_micro_scalping():
    print("🧠 [TRAINING] Protocol Micro-Scalping (Option A) - Re-entrenamiento Inteligente")
    print("⚠️ Target predictivo: +0.25% en marco 1m-5m. Ignorando volatilidad de swing account.")
    
    cache_dir = "data/cache_parquet"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    files = [f for f in os.listdir(cache_dir) if f.endswith('1m.parquet')]
    
    global_metrics = {'precisions': [], 'accuracies': []}
    
    for f in files:
        symbol = f.replace('_1m.parquet', '')
        path = os.path.join(cache_dir, f)
        
        try:
            print(f"🏋️ Entrenando Célula Micro-Scalping: {symbol}...")
            df = pd.read_parquet(path)
            
            # --- Feature Engineering adaptativo ---
            fe = FeatureEngineering()
            df_features = fe.prepare_features(df, symbol=symbol, horizon="SCALPING")
            if df_features.empty:
                print(f"⚠️ {symbol}: Not enough features data.")
                continue
                
            df_features = df_features.dropna()
            if len(df_features) < 100:
                print(f"⚠️ {symbol}: Not enough valid features post-dropna (len: {len(df_features)}).")
                continue
                
            feature_cols = [c for c in df_features.columns if c not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            closes = df_features['close'].values.astype(np.float32)
            
            # 🎯 [MICRO-ACCOUNT LABELING]: Buscar ganancias pequeñas aseguradas (>0.25%) considerando fee (0.1% x2).
            # Lookahead = 5 minutos/velas
            lookahead = 5
            target_pct = 0.0025 # 0.25% gain
            
            target = np.zeros(len(closes), dtype=int)
            for i in range(len(closes) - lookahead):
                future_max = np.max(closes[i+1 : i+1+lookahead])
                # Evaluar rentabilidad real
                if (future_max - closes[i]) / closes[i] > target_pct:
                    target[i] = 1
                    
            X = df_features[feature_cols].values[:-lookahead]
            y = target[:-lookahead]
            
            # Time Series Validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Hiperparámetros optimizados para alta reacción / Overfit tolerado para Micro-scalping (Fast Decay)
            model = XGBClassifier(
                n_estimators=150, 
                learning_rate=0.08, # Más alto para priorizar features recientes
                max_depth=4, 
                eval_metric='logloss',
                n_jobs=-1
            )
            
            fold_precisions = []
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                if len(np.unique(y_test)) < 2: continue # Evitar colapsos
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                p = precision_score(y_test, preds, zero_division=0)
                fold_precisions.append(p)
                
            avg_precision = np.mean(fold_precisions) if fold_precisions else 0.5
            global_metrics['precisions'].append(avg_precision)
            
            # Fit final robusto
            model.fit(X, y)
            
            # [OPTION A FIX] Salvar el modelo compatible con Micro-Scalp
            safe_sym = symbol.replace('/', '')
            
            # Compatible with both versions of prediction engines inside Trade Gemini
            model_path_ubj = os.path.join(models_dir, f"{safe_sym}_scalping_xgb.ubj")
            model_path_json = os.path.join(models_dir, f"{safe_sym}_xgb.json")
            
            model.save_model(model_path_ubj)
            model.save_model(model_path_json) # Doble respaldo de formato
            
            meta_path = os.path.join(models_dir, f"{safe_sym}_scalping_meta.joblib")
            joblib.dump({'feature_cols': feature_cols}, meta_path)
            
            print(f"✅ {symbol}: Precision={avg_precision:.3f} | Pesos purgados y actualizados.")
            
        except Exception as e:
            print(f"❌ {symbol}: Falla Célula XGB - {e}")
            
    print("\n" + "="*50)
    print("🧠 [REENTRENAMIENTO COMPLETADO] Precision prom:", round(np.mean(global_metrics['precisions']), 3) if global_metrics['precisions'] else 'N/A')

if __name__ == "__main__":
    train_micro_scalping()
