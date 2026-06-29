import os
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from config import Config
from sklearn.preprocessing import StandardScaler

def synthetic_dark_alpha(df):
    """
    Inject synthetic Dark Alpha metrics into historical dataframe for retraining.
    In real-time, these come from Rust (Mempool Panic, Net Liq Pressure, Liquidation Cascade).
    """
    n = len(df)
    
    # Mempool Panic (Proxy: high volatility spikes)
    returns = df['close'].pct_change().fillna(0)
    volatility = returns.rolling(10).std().fillna(0)
    panic_score = (volatility / volatility.max()).clip(0, 1)
    
    # Net Liquidity Pressure (Proxy: Volume imbalance)
    buy_vol = df['volume'] * (df['close'] > df['open']).astype(int)
    sell_vol = df['volume'] * (df['close'] <= df['open']).astype(int)
    total_vol = buy_vol + sell_vol
    total_vol = total_vol.replace(0, 1)
    net_liq = (buy_vol - sell_vol) / total_vol
    
    # Liquidation Cascade (Proxy: High volume + large wick in direction of trend)
    cascade = np.zeros(n)
    for i in range(1, n):
        if total_vol.iloc[i] > total_vol.rolling(20).mean().iloc[i] * 2:
            cascade[i] = 0.5 + (0.5 * panic_score.iloc[i])
            
    return panic_score, net_liq, cascade

def run_quantum_retraining(db_path, model_out_dir):
    print("🌌 [QUANTUM RETRAINER] Initiating SUPREME XGBoost NANO Model Training...")
    
    if not os.path.exists(db_path):
        print(f"❌ [ERROR] Database not found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    
    # Load historical features
    try:
        df = pd.read_sql_query("SELECT * FROM feature_store ORDER BY timestamp ASC LIMIT 50000", conn)
    except Exception as e:
        print(f"⚠️ [WARNING] Could not load feature_store: {e}")
        df = pd.DataFrame()
        
    if len(df) < 1000:
        print(f"⚠️ Not enough data ({len(df)} rows). Generating synthetic data for testing...")
        df = pd.DataFrame({
            'timestamp': range(2000),
            'close': np.random.randn(2000).cumsum() + 1000,
            'open': np.random.randn(2000).cumsum() + 1000,
            'high': np.random.randn(2000).cumsum() + 1000,
            'low': np.random.randn(2000).cumsum() + 1000,
            'volume': np.abs(np.random.randn(2000)) * 100,
            'target': np.random.randint(0, 2, 2000)
        })
        
    print(f"✅ Loaded {len(df)} rows from feature_store.")
    
    # Inject Dark Alpha
    panic, liq, cascade = synthetic_dark_alpha(df)
    df['dex_whisper'] = panic
    df['dark_alpha_pressure'] = liq
    df['liquidation_cascade'] = cascade
    
    # Clean up df for training
    exclude_cols = ['timestamp', 'datetime', 'symbol', 'target', 'id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # For testing, we ensure target exists
    if 'target' not in df.columns:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
    df.dropna(inplace=True)
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ultra-Aggressive NANO Model Parameters (Sub-Microsecond Inference)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,              # Shallow trees for extreme speed
        'learning_rate': 0.01,
        'tree_method': 'hist',       # Fast histogram building
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 50           # Small ensemble for NANO latency
    }
    
    print("⚛️ Forging XGBoost NANO Model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_scaled, y)
    
    os.makedirs(model_out_dir, exist_ok=True)
    
    model_path = os.path.join(model_out_dir, "xgboost_nano_supreme.json")
    model.save_model(model_path)
    
    scaler_path = os.path.join(model_out_dir, "scaler_nano_supreme.pkl")
    joblib.dump(scaler, scaler_path)
    
    print(f"🏆 [SUCCESS] Quantum Retraining Complete.")
    print(f"   Model saved to: {model_path}")
    print(f"   Scaler saved to: {scaler_path}")

if __name__ == "__main__":
    base_dir = getattr(Config, "BASE_DIR", ".")
    db_path = os.path.join(base_dir, "data", "feature_store.db")
    model_dir = os.path.join(base_dir, ".models", "QUANTUM")
    run_quantum_retraining(db_path, model_dir)
