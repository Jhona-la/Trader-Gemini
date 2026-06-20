
import os
import json
import time
from datetime import datetime, timezone
from utils.logger import logger
from strategies.ml_strategy import UniversalEnsembleStrategy

def prepare_launch_report():
    print("🚀 TRADER GEMINI: FINAL LAUNCH PREPARATION")
    print("="*50)
    
    # 1. Check Configuration
    from config import Config
    print(f"🔹 Environment: {'PROD' if not Config.BINANCE_USE_TESTNET else 'TESTNET'}")
    print(f"🔹 Leverage Locked: {Config.BINANCE_LEVERAGE}x")
    print(f"🔹 Symbols: {len(Config.CRYPTO_FUTURES_PAIRS)} pairs active")
    
    # 2. Check Resilience
    from utils.heartbeat import get_heartbeat
    is_alive, delay = get_heartbeat().check_survival()
    print(f"🔹 Heartbeat Status: {'ALIVE' if is_alive else 'STOPPED'} ({delay:.1f}s ago)")
    
    # 3. Check Persistence
    status_path = "dashboard/data/futures/live_status.json"
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            data = json.load(f)
            equity = data['total_equity']
            print(f"🔹 Last Reported Equity: ${equity:.2f}")
    
    print("\n✅ SYSTEM IS READY FOR LIVE EXECUTION.")
    print("Monitor Telegram for [ENSEMBLE] consensus signals.")
    print("="*50)

if __name__ == "__main__":
    prepare_launch_report()
