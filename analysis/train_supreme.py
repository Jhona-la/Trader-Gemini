
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

def train_supreme():
    print("🧠 [TRAINING] Protocol Supreme: Phase 26-30 - Model Retraining...")
    
    cache_dir = "data/cache_parquet"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    files = [f for f in os.listdir(cache_dir) if f.endswith('1m.parquet')]
    
    global_metrics = {'precisions': [], 'accuracies': []}
    
    for f in files:
        symbol = f.replace('_1m.parquet', '')
        path = os.path.join(cache_dir, f)
        
        try:
            print(f"🏋️ Training {symbol}...")
            df = pd.read_parquet(path)
            
            # --- Feature Engineering (Phase 27) ---
            fe = FeatureEngineering()
            df_features = fe.prepare_features(df, symbol=symbol)
            if df_features.empty:
                print(f"⚠️ {symbol}: Not enough features data.")
                continue
                
            df_features = df_features.dropna()
            if len(df_features) < 50:
                print(f"⚠️ {symbol}: Not enough valid features post-dropna (len: {len(df_features)}).")
                continue
                
            feature_cols = [c for c in df_features.columns if c not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            closes = df_features['close'].values.astype(np.float32)
            
            # Target: Next Candle Close > Current Close (Binary Classification)
            target = (np.roll(closes, -1) > closes).astype(int)
            target[-1] = 0 # Invalid last
            
            X = df_features[feature_cols].values[:-1]
            y = target[:-1]
            
            # --- Time Series Split (Phase 28) ---
            tscv = TimeSeriesSplit(n_splits=3)
            
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3, # Prevent overfitting
                eval_metric='logloss',
                n_jobs=1
            )
            
            fold_precisions = []
            
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                p = precision_score(y_test, preds, zero_division=0)
                fold_precisions.append(p)
                
            avg_precision = np.mean(fold_precisions)
            global_metrics['precisions'].append(avg_precision)
            
            # Fit on full data
            model.fit(X, y)
            
            # Save Model (Phase 30)
            safe_sym = symbol.replace('/', '')
            model_path = os.path.join(models_dir, f"{safe_sym}_xgb.ubj")
            model.save_model(model_path)
            
            meta_path = os.path.join(models_dir, f"{safe_sym}_meta.joblib")
            joblib.dump({'feature_cols': feature_cols}, meta_path)
            
            print(f"✅ {symbol}: Precision={avg_precision:.3f} | Saved to {model_path} & {meta_path}")
            
        except Exception as e:
            print(f"❌ {symbol}: Training Failed - {e}")
            
    print("🧠 [TRAINING] Complete.")
    print(f"📊 Global Avg Precision: {np.mean(global_metrics['precisions']):.3f}")

if __name__ == "__main__":
    train_supreme()
