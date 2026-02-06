from utils.system_monitor import system_monitor
import os
import json
import time

def verify_monitor():
    print("🧪 Testing SystemMonitor...")
    
    # 1. Run check
    metrics = system_monitor.check_health()
    
    if not metrics:
        print("⏳ Monitor requires cooldown or init... waiting 11s")
        time.sleep(11)
        metrics = system_monitor.check_health()
        
    print(f"📊 Metrics: {metrics}")
    
    if 'cpu_pct' in metrics and 'ram_pct' in metrics:
        print("✅ Metrics Collected")
    else:
        print("❌ Metrics Failed")
        
    # 2. Check File
    path = "dashboard/data/system_health.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            saved = json.load(f)
        if saved.get('cpu_pct') == metrics.get('cpu_pct'):
             print(f"✅ File Persistence Verified: {path}")
        else:
             print("❌ File Content Mismatch")
    else:
        print(f"❌ File Not Found: {path}")

if __name__ == "__main__":
    verify_monitor()
