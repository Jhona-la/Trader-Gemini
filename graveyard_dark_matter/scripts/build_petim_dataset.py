import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime

# Setup Paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from strategies.components.feature_engineering import FeatureEngineering
from utils.logger import logger

logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def extract_trajectory_labels(df: pd.DataFrame, lookahead: int = 30) -> pd.DataFrame:
    """
    Extracts MFE, MAE, and Survival Time (Time-to-Exhaustion) for every row in the dataframe.
    This assumes a LONG position entry at the `close` of the current bar.
    For a SHORT position, the MFE and MAE are simply inverted.
    
    Args:
        df: DataFrame with OHLCV data.
        lookahead: Number of future bars to look ahead to determine maximum excursion.
    """
    logger.info(f"Extracting trajectory labels with lookahead={lookahead} bars...")
    
    # We will use numpy arrays for speed
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    n = len(df)
    
    mfe_arr = np.zeros(n)
    mae_arr = np.zeros(n)
    surv_time_arr = np.zeros(n)
    
    # Fast rolling window calculation using numpy stride tricks or simple iteration
    # Since n can be large (e.g. 500k), we use a sliding window approach
    for i in tqdm(range(n - lookahead)):
        entry_price = close_arr[i]
        if entry_price == 0:
            continue
            
        future_highs = high_arr[i+1 : i+1+lookahead]
        future_lows = low_arr[i+1 : i+1+lookahead]
        
        # MFE for Long: Max high in the window
        max_high = np.max(future_highs)
        # MAE for Long: Min low in the window
        min_low = np.min(future_lows)
        
        mfe_arr[i] = (max_high - entry_price) / entry_price
        mae_arr[i] = (entry_price - min_low) / entry_price
        
        # Time-to-exhaustion (survival time): number of bars until the peak (MFE) is reached
        peak_idx = np.argmax(future_highs) + 1  # +1 because it's relative to entry (i+1)
        surv_time_arr[i] = peak_idx
        
    df['label_mfe'] = mfe_arr
    df['label_mae'] = mae_arr
    df['label_survival_time'] = surv_time_arr
    
    # Also add binary continuation targets for pure classification
    # e.g., did we reach 0.5% profit before hitting 0.2% loss?
    profit_target = 0.005
    stop_loss = 0.002
    
    cont_arr = np.zeros(n)
    for i in tqdm(range(n - lookahead)):
        entry = close_arr[i]
        if entry == 0: continue
        
        reached_tp = False
        reached_sl = False
        
        for j in range(1, lookahead + 1):
            if high_arr[i+j] >= entry * (1 + profit_target):
                reached_tp = True
            if low_arr[i+j] <= entry * (1 - stop_loss):
                reached_sl = True
                
            if reached_tp and not reached_sl:
                cont_arr[i] = 1
                break
            elif reached_sl:
                cont_arr[i] = 0
                break
                
    df['label_continuation'] = cont_arr
    
    return df

def process_symbol(symbol: str, timeframe: str = '1m', lookahead: int = 60):
    logger.info(f"Processing PETIM dataset for {symbol} ({timeframe})")
    symbol_safe = symbol.replace("/", "_")
    csv_path = os.path.join(Config.BASE_DIR, "data", "historical", f"{symbol_safe}_{timeframe}.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"Data file not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns and 'datetime' not in df.columns:
        # standard historical format: timestamp, open, high, low, close, volume
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    
    # 1. Feature Engineering
    logger.info("Running Feature Engineering...")
    fe = FeatureEngineering()
    df_features = fe.prepare_features(df, market_regime="UNKNOWN", horizon="SCALPING")
    
    if df_features.empty:
        logger.error("Feature engineering returned an empty DataFrame.")
        return
        
    # Ensure OHLCV are kept
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df_features.columns and col in df.columns:
            df_features[col] = df[col].values[:len(df_features)]
            
    # 2. Label Extraction
    df_labeled = extract_trajectory_labels(df_features, lookahead=lookahead)
    
    # 3. Drop rows with NaNs (which will be at the end due to lookahead)
    df_labeled = df_labeled.iloc[:-lookahead]
    
    # 4. Save to PETIM directory
    out_dir = os.path.join(Config.BASE_DIR, "data", "petim")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol_safe}_labeled.parquet")
    
    df_labeled.to_parquet(out_path)
    logger.info(f"✅ PETIM dataset saved to {out_path} ({len(df_labeled)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PETIM Dataset")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol to process")
    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe (e.g., 1m, 5m)")
    parser.add_argument("--lookahead", type=int, default=30, help="Forward bars for MFE/MAE")
    
    args = parser.parse_args()
    process_symbol(args.symbol, args.timeframe, args.lookahead)
