import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import torch
from typing import Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from strategies.components.feature_engineering import FeatureEngineering
from data.binance_loader import BinanceData
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OmniDatasetBuilder")

def process_chunk(df_chunk: pd.DataFrame, seq_len: int = 60, horizon: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes a chunk of the dataframe to extract X and Y pairs.
    """
    feature_cols = [c for c in df_chunk.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target_1']]
    
    # We need to drop NaNs before extracting sequences
    df_clean = df_chunk.dropna(subset=feature_cols).copy()
    
    if len(df_clean) < seq_len + horizon:
        return np.array([]), np.array([])
        
    features = df_clean[feature_cols].values
    close_prices = df_clean['close'].values
    opens = df_clean['open'].values
    highs = df_clean['high'].values
    lows = df_clean['low'].values
    
    X_list = []
    Y_list = []
    
    # We iterate up to len(df_clean) - horizon
    for i in range(seq_len, len(df_clean) - horizon):
        # X: seq_len features
        x = features[i - seq_len: i]
        
        # Y: horizon [open_pct, high_pct, low_pct, close_pct] relative to current close
        current_close = close_prices[i - 1]
        
        # Future prices
        f_o = opens[i: i + horizon]
        f_h = highs[i: i + horizon]
        f_l = lows[i: i + horizon]
        f_c = close_prices[i: i + horizon]
        
        # Percentage changes relative to current_close
        y_o = (f_o - current_close) / current_close
        y_h = (f_h - current_close) / current_close
        y_l = (f_l - current_close) / current_close
        y_c = (f_c - current_close) / current_close
        
        # Stack into [horizon, 4]
        y = np.column_stack((y_o, y_h, y_l, y_c))
        
        X_list.append(x)
        Y_list.append(y)
        
    return np.array(X_list), np.array(Y_list)


def build_dataset(symbol: str, timeframe: str = '1m', days: int = 30, seq_len: int = 60, horizon: int = 1000):
    """
    Builds the dataset for the Omniscience model.
    """
    logger.info(f"Building Omniscience dataset for {symbol} ({timeframe}) - {days} days")
    
    q = queue.Queue()
    loader = BinanceData(events_queue=q, symbol_list=[symbol])
    
    # Calculate how many limit rows we need (approx)
    bars_per_day = 1440 if timeframe == '1m' else 24 if timeframe == '1h' else 288
    limit = int(days * bars_per_day) + 2000  # Extra for feature calculation
    
    raw_data = loader.get_latest_bars(symbol, n=limit, timeframe=timeframe)
    if raw_data is None:
        logger.error(f"Failed to fetch data for {symbol}.")
        return
        
    df_raw = pd.DataFrame(raw_data)
    
    if df_raw.empty or len(df_raw) < seq_len + horizon:
        logger.error(f"Insufficient data for {symbol} {timeframe}. Got {len(df_raw)} bars.")
        return
        
    logger.info(f"Applying Feature Engineering...")
    # Add target_1 simply to satisfy any internal checks if needed, but we don't use it
    df_raw['target_1'] = 0 
    
    fe = FeatureEngineering()
    df_features = fe.prepare_features(df_raw, symbol=symbol, horizon=timeframe)
    
    # We must replace infinity with NaNs, then drop them
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    logger.info(f"Extracting X and Y sequences (seq_len={seq_len}, horizon={horizon})...")
    X, Y = process_chunk(df_features, seq_len=seq_len, horizon=horizon)
    
    if len(X) == 0:
        logger.error("Generated 0 sequences. Data might contain too many NaNs.")
        return
        
    logger.info(f"Generated {len(X)} sequences.")
    logger.info(f"X shape: {X.shape} (N, seq_len, features)")
    logger.info(f"Y shape: {Y.shape} (N, horizon, 4)")
    
    # Save to disk as PyTorch tensors
    save_dir = os.path.join(Config.BASE_DIR, "data", "omniscience")
    os.makedirs(save_dir, exist_ok=True)
    
    symbol_safe = symbol.replace("/", "_")
    x_path = os.path.join(save_dir, f"{symbol_safe}_{timeframe}_X.pt")
    y_path = os.path.join(save_dir, f"{symbol_safe}_{timeframe}_Y.pt")
    
    torch.save(torch.tensor(X, dtype=torch.float32), x_path)
    torch.save(torch.tensor(Y, dtype=torch.float32), y_path)
    
    logger.info(f"✅ Saved tensors to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    
    build_dataset(args.symbol, args.timeframe, args.days)
