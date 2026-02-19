import os
import sys
import psutil
import requests
import shutil
from datetime import datetime

class OmegaHealthCheck:
    """
    OMEGA PROTOCOL: FINAL PRE-FLIGHT CHECK
    Verifies system readiness before GO LIVE.
    """
    def __init__(self):
        self.status = True
        print(f"🚀 OMEGA HEALTH CHECK - {datetime.now()}")
        print("="*60)

    def check_network(self):
        try:
            requests.get("https://www.google.com", timeout=3)
            print("✅ [NETWORK] Internet Connectivity: OK")
        except:
            print("❌ [NETWORK] Internet Connectivity: FAILED")
            self.status = False

    def check_binance(self):
        try:
            r = requests.get("https://api.binance.com/api/v3/ping", timeout=3)
            if r.status_code == 200:
                print("✅ [API] Binance Public Endpoint: OK")
            else:
                print(f"❌ [API] Binance returned {r.status_code}")
                self.status = False
        except Exception as e:
            print(f"❌ [API] Binance Connectivity Error: {e}")
            self.status = False

    def check_resources(self):
        # Disk Space
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        if free_gb > 1:
            print(f"✅ [DISK] Free Space: {free_gb}GB (Passed)")
        else:
            print(f"❌ [DISK] Free Space Low: {free_gb}GB")
            self.status = False

        # RAM
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        if avail_gb > 0.5:
             print(f"✅ [RAM] Available Memory: {avail_gb:.2f}GB (Passed)")
        else:
             print(f"❌ [RAM] Low Memory: {avail_gb:.2f}GB")
             self.status = False

    def run(self):
        self.check_network()
        self.check_binance()
        self.check_resources()
        
        print("="*60)
        if self.status:
            print("🟢 SYSTEM READY FOR LAUNCH")
            return True
        else:
            print("🔴 SYSTEM CHECKS FAILED - ABORT LAUNCH")
            return False

if __name__ == "__main__":
    check = OmegaHealthCheck()
    if not check.run():
        sys.exit(1)
