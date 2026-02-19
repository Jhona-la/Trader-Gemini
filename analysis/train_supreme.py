
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
from utils.math_kernel import calculate_rsi_jit, calculate_zscore_jit

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
            closes = df['close'].values.astype(np.float32)
            
            # 1. RSI
            rsi = calculate_rsi_jit(closes, period=14)
            
            # 2. Z-Score (Volatility)
            zscore = calculate_zscore_jit(closes, period=20)
            
            # 3. Log Return
            returns = np.diff(np.log(closes), prepend=np.log(closes[0]))
            
            # Target: Next Candle Close > Current Close (Binary Classification)
            # 1 = UP, 0 = DOWN
            # Lag target by -1
            target = (np.roll(closes, -1) > closes).astype(int)
            target[-1] = 0 # Invalid last
            
            # Prepare X, y
            # Drop NaN from indicators (first 20)
            start_idx = 20
            end_idx = len(closes) - 1 # Drop last target
            
            X = np.column_stack([rsi, zscore, returns])[start_idx:end_idx]
            y = target[start_idx:end_idx]
            
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
            model_path = os.path.join(models_dir, f"{safe_sym}_xgb.json")
            model.save_model(model_path)
            
            print(f"✅ {symbol}: Precision={avg_precision:.3f} | Saved to {model_path}")
            
        except Exception as e:
            print(f"❌ {symbol}: Training Failed - {e}")
            
    print("🧠 [TRAINING] Complete.")
    print(f"📊 Global Avg Precision: {np.mean(global_metrics['precisions']):.3f}")

if __name__ == "__main__":
    train_supreme()
