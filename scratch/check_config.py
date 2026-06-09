import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
print(f"TRADING_PAIRS: {Config.TRADING_PAIRS}")
print(f"CORE_SYMBOLS: {getattr(Config, 'CORE_SYMBOLS', 'N/A')}")
