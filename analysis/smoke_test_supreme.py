
import os
import sys
import pandas as pd
import warnings
from unittest.mock import MagicMock

# Ensure root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Config
from config import Config
Config.TRADING_PAIRS = ['BTC/USDT']

from strategies.ml_strategy import UniversalEnsembleStrategy

def smoke_test():
    print("💨 [SMOKE TEST] Protocol Supreme: Validation...")
    
    # Mock mocks
    data_provider = MagicMock()
    events_queue = MagicMock()
    
    try:
        strategy = UniversalEnsembleStrategy(
            data_provider=data_provider,
            events_queue=events_queue,
            symbol='BTC/USDT'
        )
        
        if strategy.is_trained:
            print("✅ Strategy Initialized & Trained (Model Loaded).")
            print("DEBUG: Checking xgb_model...")
            if getattr(strategy, 'xgb_model', None):
                print("✅ XGBoost Model Present.")
            else:
                print("❌ XGBoost Model Missing.")
                
            print("DEBUG: Checking rf_model...")
            if getattr(strategy, 'rf_model', None) is None:
                print("✅ RF Model None (Correct for Supreme Mode).")
            else:
                print("⚠️ RF Model Loaded.")
                # Try to access it to see if it crashes
                print(f"DEBUG: RF Type: {type(strategy.rf_model)}")
                
        else:
            print("❌ Strategy Not Trained.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Crash during Init: {e}")
        sys.exit(1)
        
    print("✅ Smoke Test Passed.")

if __name__ == "__main__":
    smoke_test()
