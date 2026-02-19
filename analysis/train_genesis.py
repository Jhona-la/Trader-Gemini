import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Fix path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config

class GenesisTrainer:
    """
    OMEGA GENESIS: PHASE 26-40
    Train the Universal Ensemble (RF + GB) on fresh Z-Score data.
    """
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.models_dir = 'models_genesis'
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_dataset(self):
        print("📂 Loading Processed Datasets...")
        files = [f for f in os.listdir(self.data_dir) if '_processed.parquet' in f]
        dfs = []
        for f in files:
            df = pd.read_parquet(os.path.join(self.data_dir, f))
            dfs.append(df)
        
        if not dfs:
            print("❌ No processed data found!")
            return None
            
        full_df = pd.concat(dfs)
        print(f"✅ Total Rows: {len(full_df)}")
        
        # LABELING (Simple Strategy for Genesis)
        # Target: Next candle close > Current close (1) or < (0)
        # In real life we want more complex targets, but for Genesis Audit we prove the pipeline.
        full_df['target'] = (full_df['close'].shift(-1) > full_df['close']).astype(int)
        full_df.dropna(inplace=True)
        return full_df

    def train(self):
        df = self.load_dataset()
        if df is None: return
        
        features = ['z_score', 'log_ret', 'volatility']
        X = df[features]
        y = df['target']
        
        # TimeSeries Split (Phase 27)
        print("⏳ Splitting TimeSeries (No Shuffle)...")
        # Use last 20% for testing
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"🧠 Training Ensemble on {len(X_train)} samples...")
        
        # Model 1: Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        
        # Model 2: Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        
        # Evaluation
        preds_rf = rf.predict(X_test)
        preds_gb = gb.predict(X_test)
        
        # Hybrid Voting (Soft)
        # For simplicity in Phase 26, we check individual performance
        acc_rf = accuracy_score(y_test, preds_rf)
        acc_gb = accuracy_score(y_test, preds_gb)
        
        print("-" * 40)
        print(f"🏆 RF Accuracy: {acc_rf:.4f}")
        print(f"🏆 GB Accuracy: {acc_gb:.4f}")
        print("-" * 40)
        
        # Save Models
        joblib.dump(rf, f"{self.models_dir}/genesis_rf.joblib")
        joblib.dump(gb, f"{self.models_dir}/genesis_gb.joblib")
        print(f"💾 Models saved to {self.models_dir}")
        
        if acc_rf > 0.5 or acc_gb > 0.5:
            print("✅ GENESIS TRAINING SUCCESSFUL (Better than random on raw data)")
        else:
            print("⚠️ TRAINING NEEDS OPTIMIZATION (Random Walk detected)")

if __name__ == "__main__":
    GenesisTrainer().train()
