"""
AITS: Multi-Symbol Training Pipeline
Trains an LSTM model for each of the 25 assets in the Parquet cache.
Saves each model to trained_models/{SYMBOL}_lstm.pt
"""

import logging, os, sys, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_models import RecurrentMemoryNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "cache_parquet")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
SEQ_LEN = 30
EPOCHS = 15
BATCH = 64

# Features to exclude (identified by leakage audit)
LEAKAGE_FEATURES = {
    "scalp_velocity_1", "up_bar", "down_bar", "dist_ema_5",
    "scalp_rsi_divergence", "rsi_3", "higher_high", "lower_low",
    "momentum_3", "micro_velocity_3", "dist_ema_10",
    "returns_1", "returns_3", "returns_5", "returns_10"
}


def prepare(df, target="returns_1"):
    cols = [c for c in df.columns if c not in {"timestamp", target} | LEAKAGE_FEATURES
            and df[c].dtype in [np.float64, np.float32]]
    clean = df[cols + [target]].dropna()
    if len(clean) < SEQ_LEN + 50:
        return None, None, 0

    feats = np.nan_to_num(clean[cols].values.astype(np.float32), nan=0., posinf=0., neginf=0.)
    m, s = feats.mean(0), feats.std(0) + 1e-8
    feats = (feats - m) / s

    thr = 0.0001
    tgt = np.where(clean[target].values > thr, 2, np.where(clean[target].values < -thr, 0, 1))

    X, y = [], []
    for i in range(SEQ_LEN, len(feats)):
        X.append(feats[i-SEQ_LEN:i])
        y.append(tgt[i])
    return torch.tensor(np.array(X, dtype=np.float32)), torch.tensor(np.array(y, dtype=np.int64)), len(cols)


def train_one(symbol, X, y, fdim):
    split = int(len(X) * 0.7)
    Xt, Xv, yt, yv = X[:split], X[split:], y[:split], y[split:]

    model = RecurrentMemoryNetwork(input_dim=fdim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xt))
        for s in range(0, len(Xt), BATCH):
            e = min(s+BATCH, len(Xt))
            bi = idx[s:e]
            opt.zero_grad()
            loss = crit(model(Xt[bi]), yt[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            _, p = torch.max(model(Xv), 1)
            acc = (p == yv).sum().item() / len(yv) * 100
            if acc > best:
                best = acc

    path = os.path.join(MODEL_DIR, f"{symbol}_lstm.pt")
    torch.save(model.state_dict(), path)
    return best


def main():
    logging.info("╔═════════════════════════════════════════════╗")
    logging.info("║  AITS MULTI-SYMBOL TRAINING (25 Assets)    ║")
    logging.info("╚═════════════════════════════════════════════╝")
    os.makedirs(MODEL_DIR, exist_ok=True)

    parquets = sorted([f for f in os.listdir(PARQUET_DIR) if f.startswith("features_") and f.endswith(".parquet")])
    results = []
    t0 = time.time()

    for pf in parquets:
        symbol = pf.replace("features_", "").replace(".parquet", "")
        df = pd.read_parquet(os.path.join(PARQUET_DIR, pf))
        X, y, fdim = prepare(df)
        if X is None:
            logging.warning(f"  ⚠️ {symbol}: Not enough data, skipped.")
            results.append((symbol, 0, "SKIP"))
            continue

        acc = train_one(symbol, X, y, fdim)
        status = "✅" if acc > 60 else "⚠️"
        results.append((symbol, acc, status))
        logging.info(f"  {status} {symbol:15s} | Samples={len(X):6d} | Val Acc={acc:.1f}%")

    elapsed = time.time() - t0
    logging.info(f"\n{'═'*55}")
    logging.info(f"  MULTI-SYMBOL TRAINING REPORT ({elapsed:.0f}s)")
    logging.info(f"{'═'*55}")
    for sym, acc, st in sorted(results, key=lambda x: -x[1]):
        logging.info(f"  {st} {sym:15s} → {acc:.1f}%")
    avg = np.mean([a for _, a, s in results if s != "SKIP"])
    logging.info(f"{'═'*55}")
    logging.info(f"  Average Accuracy: {avg:.1f}%")
    logging.info(f"  Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
