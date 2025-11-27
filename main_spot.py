"""
Trader Gemini - SPOT Trading Instance
This is a separate instance configured for Spot trading on Binance Testnet.
Run simultaneously with main.py (Futures) for dual-market operation.
"""

import sys
import os

# CRITICAL: Replace config import to use Spot configuration
import config_spot as Config
sys.modules['config'] = Config

# Now import everything else (they will use Config_spot)
from main import main

if __name__ == "__main__":
    print("=" * 60)
    print("🟢 SPOT TRADING INSTANCE")
    print("=" * 60)
    print(f"Market: SPOT (Testnet)")
    print(f"Pairs: {len(Config.Config.TRADING_PAIRS)} symbols")
    print(f"Data Directory: {Config.Config.DATA_DIR}")
    print("=" * 60)
    main()
