"""
AITS Phase 9: Real Data Training Pipeline
Trains PyTorch models on REAL Binance historical data.

Uses the existing Parquet feature files in data/cache_parquet/
to construct supervised learning datasets and train DeepLOB / LSTM
with Walk-Forward temporal validation.
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import pandas as pd
except ImportError:
    pd = None

from pytorch_models import RecurrentMemoryNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "cache_parquet")
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")


def load_features(symbol: str = "BTC_USDT") -> pd.DataFrame:
    """Loads real feature data from the project's Parquet cache."""
    path = os.path.join(PARQUET_DIR, f"features_{symbol}.parquet")
    if not os.path.exists(path):
        logging.error(f"Parquet file not found: {path}")
        return None
    df = pd.read_parquet(path)
    logging.info(f"Loaded {symbol}: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def prepare_dataset(df: pd.DataFrame, seq_length: int = 30, target_col: str = "returns_1"):
    """
    Converts a DataFrame into supervised learning sequences.
    
    Target: Binary classification based on returns_1:
        0 = DOWN (returns_1 < 0)
        1 = FLAT (|returns_1| < threshold)
        2 = UP   (returns_1 > 0)
    """
    # Select numeric features only (exclude timestamp)
    feature_cols = [c for c in df.columns if c != "timestamp" and df[c].dtype in [np.float64, np.float32]]
    
    # Remove target from features
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    # Drop NaN rows
    df_clean = df[feature_cols + [target_col]].dropna()
    
    if len(df_clean) < seq_length + 10:
        logging.error(f"Not enough data: {len(df_clean)} rows (need {seq_length + 10})")
        return None, None

    # Create target labels
    threshold = 0.0001  # 0.01% threshold for FLAT
    targets = np.where(
        df_clean[target_col].values > threshold, 2,  # UP
        np.where(df_clean[target_col].values < -threshold, 0, 1)  # DOWN / FLAT
    )
    
    features = df_clean[feature_cols].values.astype(np.float32)
    
    # Replace infinities and remaining NaNs
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize features (z-score per column)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std
    
    # Build sequences
    X_seqs = []
    y_seqs = []
    for i in range(seq_length, len(features)):
        X_seqs.append(features[i - seq_length:i])
        y_seqs.append(targets[i])
    
    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_seqs, dtype=np.int64)
    
    logging.info(f"Dataset: X={X.shape}, y={y.shape}")
    logging.info(f"Class distribution: DOWN={np.sum(y==0)}, FLAT={np.sum(y==1)}, UP={np.sum(y==2)}")
    
    return torch.tensor(X), torch.tensor(y)


def train_walk_forward(X: torch.Tensor, y: torch.Tensor, feature_dim: int):
    """
    Walk-Forward Training: Train on 70% → Validate on 30%.
    This prevents look-ahead bias (the cardinal sin of quant finance).
    """
    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logging.info(f"Train set: {X_train.shape[0]} samples | Val set: {X_val.shape[0]} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = RecurrentMemoryNetwork(
        input_dim=feature_dim,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training
    epochs = 20
    batch_size = 64
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Mini-batch training
        indices = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            batch_idx = indices[start:end]

            xb = X_train[batch_idx].to(device)
            yb = y_train[batch_idx].to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == yb).sum().item()
            total += len(batch_idx)

        train_loss = total_loss / total
        train_acc = correct / total * 100

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = criterion(val_out, y_val.to(device)).item()
            _, val_pred = torch.max(val_out, 1)
            val_acc = (val_pred == y_val.to(device)).sum().item() / len(y_val) * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        logging.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.1f}% | "
            f"Best={best_val_acc:.1f}%"
        )

    # Save model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_btcusdt_real.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"✅ Model saved to {model_path}")

    return best_val_acc


def main():
    logging.info("╔══════════════════════════════════════════════════════╗")
    logging.info("║  AITS REAL DATA TRAINING PIPELINE                   ║")
    logging.info("╚══════════════════════════════════════════════════════╝")

    if not pd:
        logging.error("pandas is required. Aborting.")
        return

    df = load_features("BTC_USDT")
    if df is None:
        return

    X, y = prepare_dataset(df, seq_length=30, target_col="returns_1")
    if X is None:
        return

    feature_dim = X.shape[2]
    logging.info(f"Feature dimensions: {feature_dim}")

    t0 = time.time()
    best_acc = train_walk_forward(X, y, feature_dim)
    elapsed = time.time() - t0

    logging.info(f"\n{'═'*60}")
    logging.info(f"  TRAINING COMPLETE")
    logging.info(f"  Best Validation Accuracy: {best_acc:.2f}%")
    logging.info(f"  Training Time: {elapsed:.1f}s")
    logging.info(f"  Data: REAL Binance BTC/USDT ({X.shape[0]} sequences)")
    logging.info(f"{'═'*60}")


if __name__ == "__main__":
    main()
