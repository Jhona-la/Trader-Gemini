import os
import sys
import logging
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.omniscient_predictor import OmniscientTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OmniscienceTrainer")

def train_model(symbol: str, timeframe: str = '1m', epochs: int = 10, batch_size: int = 64):
    """
    Trains the Omniscient Seq2Seq Transformer model using real historical data.
    """
    symbol_safe = symbol.replace("/", "_")
    data_dir = os.path.join(Config.BASE_DIR, "data", "omniscience")
    x_path = os.path.join(data_dir, f"{symbol_safe}_{timeframe}_X.pt")
    y_path = os.path.join(data_dir, f"{symbol_safe}_{timeframe}_Y.pt")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        logger.error(f"Dataset tensors not found for {symbol} {timeframe}. Run build_omni_dataset.py first.")
        return
        
    logger.info(f"Loading datasets for {symbol} ({timeframe})...")
    X = torch.load(x_path)
    Y = torch.load(y_path)
    
    logger.info(f"X shape: {X.shape}")
    logger.info(f"Y shape: {Y.shape}")
    
    # Validation split (last 10% for validation)
    split_idx = int(len(X) * 0.9)
    train_dataset = TensorDataset(X[:split_idx], Y[:split_idx])
    val_dataset = TensorDataset(X[split_idx:], Y[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        
    logger.info(f"Using device: {device}")
    
    # Model configuration
    input_dim = X.shape[2]
    seq_len = X.shape[1]
    horizon = Y.shape[1]
    
    model = OmniscientTransformer(
        input_dim=input_dim,
        seq_len=seq_len,
        horizon=horizon,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    criterion = nn.MSELoss()
    
    logger.info(f"Starting Training for {epochs} epochs...")
    
    best_val_loss = float('inf')
    model_dir = os.path.join(Config.BASE_DIR, "models", "omniscience")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"omni_weights_{timeframe}.pt")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        t0 = time.time()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            
            # Loss computation
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"✨ New best model saved to {save_path}")

    logger.info("🏆 Training complete.")
    logger.info(f"Final Best Validation Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    train_model(args.symbol, args.timeframe, args.epochs, args.batch_size)
