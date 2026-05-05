"""
AITS Post-Training: Feature Leakage Audit
Detects if any feature in the Parquet dataset contains future information
(target leakage), which would invalidate the 98.86% accuracy result.

Tests:
1. Correlation Scan: Find features with suspiciously high correlation to returns_1.
2. Temporal Integrity: Verify features are computed from past data only.
3. Ablation Test: Remove top-correlated features and retrain to measure real accuracy.
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_models import RecurrentMemoryNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "cache_parquet", "features_BTC_USDT.parquet")


def audit_correlations(df: pd.DataFrame, target: str = "returns_1"):
    """Finds features with dangerously high correlation to the target."""
    logging.info("═" * 60)
    logging.info("  AUDIT 1: CORRELATION SCAN")
    logging.info("═" * 60)

    numeric = df.select_dtypes(include=[np.number]).drop(columns=["timestamp"], errors="ignore")
    corr = numeric.corr()[target].drop(target).abs().sort_values(ascending=False)

    # Flag anything above 0.5 as suspicious
    suspicious = corr[corr > 0.5]
    moderate = corr[(corr > 0.3) & (corr <= 0.5)]

    if len(suspicious) > 0:
        logging.warning(f"  🚨 {len(suspicious)} FEATURES WITH |corr| > 0.5 (HIGH LEAKAGE RISK):")
        for feat, val in suspicious.items():
            logging.warning(f"     {feat:40s} corr={val:.4f}")
    else:
        logging.info("  ✅ No features with |corr| > 0.5 found.")

    if len(moderate) > 0:
        logging.info(f"  ⚠️ {len(moderate)} features with 0.3 < |corr| <= 0.5 (monitor):")
        for feat, val in moderate.head(10).items():
            logging.info(f"     {feat:40s} corr={val:.4f}")

    logging.info(f"  Top 15 correlated features:")
    for feat, val in corr.head(15).items():
        logging.info(f"     {feat:40s} corr={val:.4f}")

    return list(suspicious.index), corr


def audit_temporal_integrity(df: pd.DataFrame):
    """Checks if returns columns might leak future info."""
    logging.info("\n" + "═" * 60)
    logging.info("  AUDIT 2: TEMPORAL INTEGRITY CHECK")
    logging.info("═" * 60)

    # Check for forward-looking column names
    forward_suspects = [c for c in df.columns if any(kw in c.lower() for kw in
        ["future", "forward", "next", "target", "label", "y_"])]

    if forward_suspects:
        logging.warning(f"  🚨 Columns with forward-looking names: {forward_suspects}")
    else:
        logging.info("  ✅ No forward-looking column names detected.")

    # Check returns columns for autocorrelation patterns
    returns_cols = [c for c in df.columns if c.startswith("returns_")]
    logging.info(f"  Returns columns found: {returns_cols}")

    for col in returns_cols:
        # If returns_N is perfectly correlated with a shifted version, it's leaking
        if col in df.columns:
            ac1 = df[col].autocorr(lag=1)
            logging.info(f"     {col}: autocorr(lag=1) = {ac1:.4f}")


def ablation_retrain(df: pd.DataFrame, features_to_remove: list, target: str = "returns_1"):
    """Retrains the model WITHOUT the suspicious features to measure true accuracy."""
    logging.info("\n" + "═" * 60)
    logging.info("  AUDIT 3: ABLATION RETRAIN (without leaked features)")
    logging.info("═" * 60)

    feature_cols = [c for c in df.columns
                    if c != "timestamp" and c != target and df[c].dtype in [np.float64, np.float32]]

    # Remove suspicious features
    clean_features = [c for c in feature_cols if c not in features_to_remove]
    logging.info(f"  Original features: {len(feature_cols)}")
    logging.info(f"  Removed (leaked):  {len(features_to_remove)}")
    logging.info(f"  Clean features:    {len(clean_features)}")

    # Prepare dataset
    df_clean = df[clean_features + [target]].dropna()
    features = df_clean[clean_features].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    threshold = 0.0001
    targets = np.where(df_clean[target].values > threshold, 2,
                       np.where(df_clean[target].values < -threshold, 0, 1))

    seq_length = 30
    X_seqs, y_seqs = [], []
    for i in range(seq_length, len(features)):
        X_seqs.append(features[i - seq_length:i])
        y_seqs.append(targets[i])

    X = torch.tensor(np.array(X_seqs, dtype=np.float32))
    y = torch.tensor(np.array(y_seqs, dtype=np.int64))

    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Train
    model = RecurrentMemoryNetwork(input_dim=len(clean_features), hidden_dim=64,
                                    num_layers=2, num_classes=3, dropout=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    for epoch in range(20):
        model.train()
        indices = torch.randperm(len(X_train))
        for start in range(0, len(X_train), 64):
            end = min(start + 64, len(X_train))
            batch_idx = indices[start:end]
            optimizer.zero_grad()
            out = model(X_train[batch_idx])
            loss = criterion(out, y_train[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            _, val_pred = torch.max(val_out, 1)
            val_acc = (val_pred == y_val).sum().item() / len(y_val) * 100
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        if (epoch + 1) % 5 == 0:
            logging.info(f"  Epoch {epoch+1:3d}/20 | Val Acc={val_acc:.1f}% | Best={best_val_acc:.1f}%")

    logging.info(f"\n  ┌─────────────────────────────────────────────┐")
    logging.info(f"  │ ABLATION RESULT (clean features only)        │")
    logging.info(f"  │ Best Validation Accuracy: {best_val_acc:.2f}%          │")
    logging.info(f"  └─────────────────────────────────────────────┘")

    return best_val_acc


def main():
    logging.info("╔══════════════════════════════════════════════════════╗")
    logging.info("║  AITS FEATURE LEAKAGE AUDIT                        ║")
    logging.info("╚══════════════════════════════════════════════════════╝")

    df = pd.read_parquet(PARQUET_PATH)
    logging.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Audit 1: Correlation scan
    suspicious_feats, corr_series = audit_correlations(df)

    # Audit 2: Temporal integrity
    audit_temporal_integrity(df)

    # Audit 3: Ablation (remove ALL returns_* and momentum_* which could leak)
    all_returns = [c for c in df.columns if c.startswith("returns_")]
    to_remove = list(set(suspicious_feats + all_returns))
    logging.info(f"\n  Features marked for removal: {to_remove}")

    ablation_acc = ablation_retrain(df, to_remove)

    # Final verdict
    logging.info("\n" + "═" * 60)
    logging.info("  LEAKAGE AUDIT — FINAL VERDICT")
    logging.info("═" * 60)
    if ablation_acc > 65:
        logging.info(f"  ✅ Clean accuracy ({ablation_acc:.1f}%) remains strong after ablation.")
        logging.info(f"  The model has genuine predictive power.")
    elif ablation_acc > 55:
        logging.warning(f"  ⚠️ Clean accuracy ({ablation_acc:.1f}%) dropped significantly.")
        logging.warning(f"  Some predictive power exists but original 98.86% was inflated.")
    else:
        logging.error(f"  🚨 Clean accuracy ({ablation_acc:.1f}%) near random (33%).")
        logging.error(f"  The original 98.86% was almost entirely due to feature leakage!")


if __name__ == "__main__":
    main()
