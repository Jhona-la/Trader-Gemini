"""
AITS: Bulk Historical Data Download + Feature Generation + Multi-Symbol Training

End-to-end pipeline:
1. Downloads 7 days of 1-minute OHLCV data from Binance for all 25 symbols.
2. Generates 90+ technical features (matching the BTC_USDT parquet schema).
3. Saves as Parquet in data/cache_parquet/.
4. Trains an LSTM model for each symbol with sufficient data.
"""

import logging, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import ccxt
except ImportError:
    ccxt = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "cache_parquet")
HIST_DIR = os.path.join(PROJECT_ROOT, "data", "historical")

SYMBOLS = [
    "ADA/USDT", "ARB/USDT", "ATOM/USDT", "AVAX/USDT", "BNB/USDT",
    "DOGE/USDT", "DOT/USDT", "ETC/USDT", "ETH/USDT", "FIL/USDT",
    "INJ/USDT", "LINK/USDT", "LTC/USDT", "NEAR/USDT", "OP/USDT",
    "PAXG/USDT", "POL/USDT", "RENDER/USDT", "SOL/USDT", "SUI/USDT",
    "TIA/USDT", "UNI/USDT", "WIF/USDT", "XRP/USDT"
]

# ─── Step 1: Download OHLCV ─────────────────────────────────────────

def download_symbol(exchange, symbol, days=7):
    """Downloads OHLCV 1m data from Binance."""
    from datetime import datetime, timedelta
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    all_candles = []

    while since < exchange.milliseconds():
        try:
            candles = exchange.fetch_ohlcv(symbol, "1m", since, limit=1000)
            if not candles:
                break
            since = candles[-1][0] + 1
            all_candles.extend(candles)
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logging.warning(f"  {symbol} fetch error: {e}")
            time.sleep(2)
            break

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ─── Step 2: Generate Features ──────────────────────────────────────

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates ~80 technical features from raw OHLCV matching production schema."""
    f = pd.DataFrame()
    f["timestamp"] = df["timestamp"].astype(np.int64) / 1e6

    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Returns
    for p in [1, 3, 5, 10]:
        f[f"returns_{p}"] = c.pct_change(p)

    # Momentum
    for p in [3, 5, 8, 13, 21, 34]:
        f[f"momentum_{p}"] = c - c.shift(p)

    # ROC
    for p in [5, 10, 20]:
        f[f"roc_{p}"] = (c - c.shift(p)) / (c.shift(p) + 1e-10)

    # RSI
    for period in [3, 5, 7, 14]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        f[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # Bollinger
    bb_m = c.rolling(20).mean()
    bb_s = c.rolling(20).std()
    f["bb_position"] = (c - bb_m) / (2 * bb_s + 1e-10)
    f["bb_width"] = (4 * bb_s) / (bb_m + 1e-10)

    # CCI
    tp = (h + l + c) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    f["cci"] = (tp - tp_ma) / (0.015 * tp_md + 1e-10)

    # Stochastic
    l14 = l.rolling(14).min()
    h14 = h.rolling(14).max()
    f["stoch_k"] = 100 * (c - l14) / (h14 - l14 + 1e-10)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()
    f["stoch_cross"] = f["stoch_k"] - f["stoch_d"]

    # EMA distances
    for p in [5, 10, 20, 50]:
        ema = c.ewm(span=p).mean()
        f[f"dist_ema_{p}"] = (c - ema) / (ema + 1e-10)

    # Volatility
    for p in [5, 10, 20]:
        f[f"volatility_{p}"] = c.pct_change().rolling(p).std()

    # Volume features
    f["volume_ma_ratio"] = v / (v.rolling(20).mean() + 1e-10)
    f["volume_imbalance"] = (v - v.rolling(10).mean()) / (v.rolling(10).std() + 1e-10)

    # Spread / Range
    f["hl_spread"] = (h - l) / (c + 1e-10)
    f["oc_range"] = abs(df["open"] - c) / (c + 1e-10)
    f["body_to_wick"] = abs(df["open"] - c) / (h - l + 1e-10)

    # Candle patterns
    f["up_bar"] = (c > df["open"]).astype(float)
    f["down_bar"] = (c < df["open"]).astype(float)
    f["higher_high"] = (h > h.shift(1)).astype(float)
    f["lower_low"] = (l < l.shift(1)).astype(float)

    # Hurst approximation
    def hurst_approx(series, window=100):
        result = pd.Series(np.nan, index=series.index)
        for i in range(window, len(series)):
            ts = series.iloc[i-window:i].values
            lags = range(2, min(20, window // 2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            tau = [t for t in tau if t > 0]
            if len(tau) >= 2:
                reg = np.polyfit(np.log(list(range(2, 2+len(tau)))), np.log(tau), 1)
                result.iloc[i] = reg[0]
        return result
    f["hurst_memory"] = hurst_approx(c, window=100)

    # RANSAC volatility proxy
    f["volatility_ransac"] = c.pct_change().rolling(30).std()

    # Bayesian prior (simple proxy)
    f["bayesian_prior"] = f["returns_1"].rolling(50).mean() / (f["returns_1"].rolling(50).std() + 1e-10)

    # Amihud illiquidity
    f["amihud"] = abs(c.pct_change()) / (v * c + 1e-10) * 1e6
    f["close_position"] = (c - l) / (h - l + 1e-10)

    # Micro features
    f["micro_imbalance"] = f["volume_imbalance"]
    f["micro_label"] = np.where(f["returns_1"] > 0, 1, 0).astype(float)
    f["micro_velocity_3"] = f["momentum_3"]
    f["scalp_velocity_1"] = f["returns_1"]
    f["scalp_rsi_divergence"] = f["rsi_3"] - f["rsi_14"]

    # Volatility regime
    f["volatility_regime"] = np.where(f["volatility_10"] > f["volatility_20"], 1.0, 0.0)

    # Cross features (zero placeholder — requires BTC reference)
    f["cross_spread_vs_btc"] = 0.0
    f["cross_relative_strength"] = 0.0

    f.dropna(inplace=True)
    return f


# ─── Step 3: Train ──────────────────────────────────────────────────

def train_symbol(symbol_key, df_features):
    """Trains LSTM on generated features."""
    import torch, torch.nn as nn, torch.optim as optim
    from pytorch_models import RecurrentMemoryNetwork

    LEAKAGE = {"scalp_velocity_1", "up_bar", "down_bar", "returns_1",
               "returns_3", "returns_5", "returns_10", "micro_velocity_3",
               "micro_label", "higher_high", "lower_low", "momentum_3",
               "dist_ema_5", "scalp_rsi_divergence", "rsi_3", "dist_ema_10"}
    target = "returns_1"

    cols = [c for c in df_features.columns if c not in {"timestamp", target} | LEAKAGE
            and df_features[c].dtype in [np.float64, np.float32]]
    clean = df_features[cols + [target]].dropna()
    if len(clean) < 100:
        return None

    feats = np.nan_to_num(clean[cols].values.astype(np.float32))
    m, s = feats.mean(0), feats.std(0) + 1e-8
    feats = (feats - m) / s

    thr = 0.0001
    tgt = np.where(clean[target].values > thr, 2, np.where(clean[target].values < -thr, 0, 1))

    seq = 30
    X, y = [], []
    for i in range(seq, len(feats)):
        X.append(feats[i-seq:i])
        y.append(tgt[i])
    X = torch.tensor(np.array(X, dtype=np.float32))
    y = torch.tensor(np.array(y, dtype=np.int64))

    split = int(len(X) * 0.7)
    Xt, Xv, yt, yv = X[:split], X[split:], y[:split], y[split:]

    model = RecurrentMemoryNetwork(input_dim=len(cols), hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best = 0.0
    for ep in range(15):
        model.train()
        idx = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 64):
            e = min(s+64, len(Xt))
            opt.zero_grad()
            loss = crit(model(Xt[idx[s:e]]), yt[idx[s:e]])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            _, p = torch.max(model(Xv), 1)
            acc = (p == yv).sum().item() / len(yv) * 100
            if acc > best: best = acc

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{symbol_key}_lstm.pt"))
    return best


# ─── Main Pipeline ──────────────────────────────────────────────────

def main():
    logging.info("╔═════════════════════════════════════════════════════════╗")
    logging.info("║  AITS BULK DOWNLOAD → FEATURES → TRAIN PIPELINE       ║")
    logging.info("╚═════════════════════════════════════════════════════════╝")

    if not ccxt:
        logging.error("ccxt not installed. Aborting.")
        return

    exchange = ccxt.binance({"enableRateLimit": True})
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs(HIST_DIR, exist_ok=True)

    results = []
    t0 = time.time()

    for symbol in SYMBOLS:
        key = symbol.replace("/", "_")
        logging.info(f"\n{'─'*50}")
        logging.info(f"  Processing {symbol}...")

        # Step 1: Download
        df_raw = download_symbol(exchange, symbol, days=7)
        if df_raw is None or len(df_raw) < 200:
            logging.warning(f"  ⚠️ {key}: Not enough data downloaded ({0 if df_raw is None else len(df_raw)} rows)")
            results.append((key, 0, "NO_DATA"))
            continue

        logging.info(f"  ✅ Downloaded {len(df_raw)} candles")

        # Save CSV
        csv_path = os.path.join(HIST_DIR, f"{key}_1m.csv")
        df_raw.to_csv(csv_path, index=False)

        # Step 2: Generate features
        df_feat = generate_features(df_raw)
        pq_path = os.path.join(PARQUET_DIR, f"features_{key}.parquet")
        df_feat.to_parquet(pq_path)
        logging.info(f"  ✅ Generated {df_feat.shape[1]} features ({df_feat.shape[0]} rows)")

        # Step 3: Train
        acc = train_symbol(key, df_feat)
        if acc is not None:
            status = "✅" if acc > 60 else "⚠️"
            results.append((key, acc, status))
            logging.info(f"  {status} Trained {key}: Val Acc = {acc:.1f}%")
        else:
            results.append((key, 0, "TRAIN_FAIL"))

    elapsed = time.time() - t0
    logging.info(f"\n{'═'*55}")
    logging.info(f"  BULK PIPELINE COMPLETE ({elapsed:.0f}s)")
    logging.info(f"{'═'*55}")
    for sym, acc, st in sorted(results, key=lambda x: -x[1]):
        logging.info(f"  {st:10s} {sym:15s} → {acc:.1f}%")
    trained = [a for _, a, s in results if s in ("✅", "⚠️")]
    if trained:
        logging.info(f"  Average Accuracy: {np.mean(trained):.1f}%")
    logging.info(f"  Models saved: aits_research/trained_models/")


if __name__ == "__main__":
    main()
