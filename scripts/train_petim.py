import os
import sys
import argparse
import pandas as pd
import logging

# Setup Paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from ml.petim_model import GeometryPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainPETIM")

def train_petim(symbol: str, timeframe: str = '1m'):
    symbol_safe = symbol.replace("/", "_")
    dataset_path = os.path.join(Config.BASE_DIR, "data", "petim", f"{symbol_safe}_labeled.parquet")
    
    if not os.path.exists(dataset_path):
        logger.error(f"PETIM dataset not found at {dataset_path}")
        return
        
    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded labeled dataset: {len(df)} rows")
    
    # Exclude non-features and targets
    exclude_cols = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 
                    'label_mfe', 'label_mae', 'label_survival_time', 'label_continuation']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    predictor = GeometryPredictor(symbol, timeframe)
    predictor.train(df, feature_cols)
    
    model_dir = os.path.join(Config.BASE_DIR, "models", "petim")
    predictor.save(model_dir)
    logger.info(f"PETIM Models saved to {model_dir}")
    
    # Quick sanity check
    sample_feat = df[feature_cols].iloc[-1].values
    prediction = predictor.predict(sample_feat)
    logger.info(f"Sample Prediction on last row: {prediction}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PETIM Multi-Task Engine")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol to train")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe")
    
    args = parser.parse_args()
    train_petim(args.symbol, args.timeframe)
