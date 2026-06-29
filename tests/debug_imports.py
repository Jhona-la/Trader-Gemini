
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("1. Importing utils.safe_leverage...")
try:
    from utils import safe_leverage
    print("   ✅ Success")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("2. Importing risk.risk_manager...")
try:
    from risk import risk_manager
    print("   ✅ Success")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("3. Importing RiskManager class...")
try:
    from risk.risk_manager import RiskManager
    print("   ✅ Success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
