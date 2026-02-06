
import os
import sys
from risk.kill_switch import KillSwitch
from utils.logger import logger

def test_kill_switch_resilience():
    print("🚀 Testing Operational Resilience: Kill Switch + Telegram")
    
    # Initialize KillSwitch with $12.50 floor
    ks = KillSwitch(min_equity=12.50)
    
    # 1. Simulate safe equity
    print("\n[TEST 1] Testing safe equity ($13.50)...")
    ks.update_equity(13.50)
    print(f"Status active: {ks.is_active} (Expected: False)")
    
    # 2. Simulate Hard Floor breach ($12.00)
    print("\n[TEST 2] Testing hard floor breach ($12.00)...")
    ks.update_equity(12.00)
    print(f"Status active: {ks.is_active} (Expected: True)")
    print(f"Reason: {ks.activation_reason}")
    
    # Check logs for Telegram attempt
    print("\n🔍 Verification complete. Check logs/bot_*.json for 'Notifier.send_telegram' calls.")

if __name__ == "__main__":
    test_kill_switch_resilience()
