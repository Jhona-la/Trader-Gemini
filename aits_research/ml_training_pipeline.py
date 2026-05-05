"""
AITS Phase 4: Machine Learning Predictive Layer
PyTorch Training Pipeline

Demonstrates the institutional deep learning loop:
1. Instantiation of the Neural Network (DeepLOB).
2. Data loading (Mock Batches).
3. Forward Pass (Inference).
4. Loss Calculation (CrossEntropyLoss).
5. Backpropagation and Weight Optimization (AdamW).
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_models import DeepLOB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_mock_dataloader(num_batches=10, batch_size=32, seq_length=50, features=40):
    """Simulates a PyTorch DataLoader yielding (X, y) tensors."""
    for _ in range(num_batches):
        X = torch.randn(batch_size, seq_length, features)
        
        # Labels: 0 (Down), 1 (Flat), 2 (Up)
        y = torch.randint(0, 3, (batch_size,))
        yield X, y

def run_training_loop():
    logging.info("--- Starting AITS PyTorch Training Pipeline ---")
    
    # 1. Check hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Compute Device Selected: {device}")
    
    # 2. Instantiate Model
    model = DeepLOB(num_classes=3).to(device)
    
    # 3. Define Institutional Loss Function and Optimizer
    # CrossEntropy is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # AdamW (Adam with Weight Decay) is the gold standard for stabilizing deep nets
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 4. Training Epochs
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Iterate over batches
        for batch_idx, (inputs, targets) in enumerate(generate_mock_dataloader()):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(inputs)
            
            # Calculate Loss
            loss = criterion(outputs, targets)
            
            # Backward Pass (Calculate Gradients)
            loss.backward()
            
            # Gradient Clipping (Institutional safeguard against exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimize Weights
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
        epoch_loss = running_loss / 10 # 10 batches
        epoch_acc = (correct_predictions / total_predictions) * 100
        logging.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    logging.info("✅ Training Pipeline Validation Complete.")

if __name__ == "__main__":
    run_training_loop()
