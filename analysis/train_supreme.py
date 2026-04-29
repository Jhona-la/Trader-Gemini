
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

def train_model_for_horizon(df_features, feature_cols, symbol, horizon, models_dir):
    closes = df_features['close'].values.astype(np.float32)
    
    # Horizon-specific lookahead
    # For SCALPING (1m data): 3 bars ahead (3 minutes)
    # For SWING (1m data): 60 bars ahead (1 hour)
    lookahead = 3 if horizon == 'SCALPING' else 60
    
    # Target: Future Close > Current Close (Binary Classification)
    target = (np.roll(closes, -lookahead) > closes).astype(int)
    
    # Invalidate the last `lookahead` rows
    for i in range(1, lookahead + 1):
        target[-i] = 0
        
    X = df_features[feature_cols].values[:-lookahead]
    y = target[:-lookahead]
    
    if len(X) < 50:
        print(f"⚠️ {symbol} [{horizon}]: Not enough valid samples.")
        return 0.0
        
    # --- Time Series Split (Phase 28) ---
    tscv = TimeSeriesSplit(n_splits=3)
    
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4, # Slightly deeper to capture complex micro-structures
        eval_metric='logloss',
        n_jobs=-1 # Use all cores
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
    
    # Fit on full data
    model.fit(X, y)
    
    # Save Model (Phase 30)
    safe_sym = symbol.replace('/', '')
    suffix = "_scalping" if horizon == 'SCALPING' else "_swing"
    
    model_path = os.path.join(models_dir, f"{safe_sym}{suffix}_xgb.ubj")
    model.save_model(model_path)
    
    meta_path = os.path.join(models_dir, f"{safe_sym}{suffix}_meta.joblib")
    joblib.dump({'feature_cols': feature_cols}, meta_path)
    
    print(f"✅ {symbol} [{horizon}]: Precision={avg_precision:.3f} | Saved to {model_path} & {meta_path}")
    return avg_precision

def train_supreme():
    print("🧠 [TRAINING] Protocol Supreme: Multi-Horizon Model Retraining...")
    
    cache_dir = "data/historical"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    files = [f for f in os.listdir(cache_dir) if f.endswith('_1m.csv')]
    
    global_metrics = {'SCALPING': [], 'SWING': []}
    
    for f in files:
        symbol = f.replace('_1m.csv', '').replace('_', '/')
        path = os.path.join(cache_dir, f)
        
        try:
            print(f"🏋️ Feature Engineering {symbol}...")
            df = pd.read_csv(path)
            
            fe = FeatureEngineering()
            df_features = fe.prepare_features(df, symbol=symbol)
            
            if df_features.empty:
                print(f"⚠️ {symbol}: Features data empty.")
                continue
                
            df_features = df_features.dropna()
            if len(df_features) < 100:
                print(f"⚠️ {symbol}: Not enough valid features post-dropna (len: {len(df_features)}).")
                continue
                
            feature_cols = [c for c in df_features.columns if c not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Train SCALPING
            p_scalp = train_model_for_horizon(df_features, feature_cols, symbol, 'SCALPING', models_dir)
            if p_scalp > 0: global_metrics['SCALPING'].append(p_scalp)
            
            # Train SWING
            p_swing = train_model_for_horizon(df_features, feature_cols, symbol, 'SWING', models_dir)
            if p_swing > 0: global_metrics['SWING'].append(p_swing)
            
        except Exception as e:
            import traceback
            print(f"❌ {symbol}: Training Failed - {e}")
            traceback.print_exc()
            
    print("🧠 [TRAINING] Complete.")
    if global_metrics['SCALPING']:
        print(f"📊 Global Avg Precision (SCALPING): {np.mean(global_metrics['SCALPING']):.3f}")
    if global_metrics['SWING']:
        print(f"📊 Global Avg Precision (SWING): {np.mean(global_metrics['SWING']):.3f}")

if __name__ == "__main__":
    train_supreme()
