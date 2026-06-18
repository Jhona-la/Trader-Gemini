"""
AITS Phase 4: Machine Learning Predictive Layer
PyTorch Institutional Architectures

Defines the core Neural Networks for the AITS Layer 3:
1. DeepLOB (Convolutional): For analyzing the L2 Orderbook spatial structure (HFT).
2. TemporalTransformer: For cross-asset and macro multi-variate attention (Swing).
3. RecurrentMemoryNetwork (LSTM): For sequential momentum memory.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class DeepLOB(nn.Module):
    """
    Deep Limit Order Book Network (DeepLOB).
    Uses 1D Convolutions to extract spatial features from the Bid/Ask spread,
    followed by an LSTM layer to capture short-term temporal dynamics.
    
    Expected Input Shape: (Batch_Size, Sequence_Length, 40)
    40 Features = 10 Levels of (Bid Price, Bid Vol, Ask Price, Ask Vol)
    """
    def __init__(self, num_classes=3):
        super(DeepLOB, self).__init__()
        
        # Spatial Feature Extraction (Order Book Depth)
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, padding=1)
        
        # Inception Module (Simulated for simplicity)
        self.inception_conv = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        
        # Temporal Feature Extraction
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        
        # Output Classifier (Down, Flat, Up)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (B, Seq, Features). Conv1D expects (B, Features, Seq)
        x = x.permute(0, 2, 1)
        
        # Convolutions
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.inception_conv(x))
        
        # Back to LSTM shape: (B, Seq, Features)
        x = x.permute(0, 2, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class TemporalTransformer(nn.Module):
    """
    Financial Transformer using Self-Attention.
    Ideal for processing multi-variate features (e.g., BTC, ETH, and Macro indicators)
    simultaneously to understand complex non-linear correlations.
    """
    def __init__(self, feature_dim=15, num_heads=3, num_layers=2, num_classes=3, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        
        # Linear projection to embedding size
        self.d_model = 60 # Must be divisible by num_heads (3)
        self.input_linear = nn.Linear(feature_dim, self.d_model)
        
        # Positional Encoding is required for Transformers to understand sequence order
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, self.d_model))
        
        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=num_heads, 
            dim_feedforward=128, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # x shape: (B, Seq, Feature_Dim)
        seq_len = x.size(1)
        
        x = self.input_linear(x)
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Pass through Transformer
        transformer_out = self.transformer(x)
        
        # Global Average Pooling across the sequence
        pooled = torch.mean(transformer_out, dim=1)
        
        return self.fc_out(pooled)


class RecurrentMemoryNetwork(nn.Module):
    """
    Standard Deep LSTM architecture.
    Robust baseline for sequential momentum and classical trend following.
    """
    def __init__(self, input_dim=15, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.2):
        super(RecurrentMemoryNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Block
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    logging.info("Initializing AITS PyTorch Architectures...")
    
    # Verify DeepLOB Initialization
    batch_size = 32
    seq_length = 50
    lob_features = 40
    dummy_input = torch.randn(batch_size, seq_length, lob_features)
    
    model = DeepLOB(num_classes=3)
    output = model(dummy_input)
    logging.info(f"✅ DeepLOB Forward Pass Successful. Output Shape: {output.shape}")
    
    # Verify Transformer Initialization
    transformer = TemporalTransformer()
    dummy_transformer_input = torch.randn(32, 50, 15)
    trans_out = transformer(dummy_transformer_input)
    logging.info(f"✅ TemporalTransformer Forward Pass Successful. Output Shape: {trans_out.shape}")
