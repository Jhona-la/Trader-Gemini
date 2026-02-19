import psutil
import time
import os
import sys
import threading
import statistics
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.getcwd())

from core.engine import Engine
from config import Config

def run_idle_test():
    print("="*60)
    print("🌱 PHASE 48: ENERGY EFFICIENCY / IDLE RESOURCE AUDIT")
    print("="*60)
    
    # Mock Config to prevent actual connections
    Config.BINANCE_USE_TESTNET = True
    
    # Initialize Engine
    print("Initializing Engine components...")
    engine = Engine()
    
    # Mock Data Handler
    mock_dh = MagicMock()
    mock_dh.get_latest_bars.return_value = [] # Return empty list so it doesn't crash
    engine.register_data_handler(mock_dh)
    
    # Mock Execution Handler
    mock_exec = MagicMock()
    engine.register_execution_handler(mock_exec)
    
    # Run Engine Loop in background thread
    print("🚀 Starting Engine (Dry Run) in background thread...")
    
    # Engine.run() is blocking, so we run it in a thread
    t = threading.Thread(target=engine.run, daemon=True)
    t.start()
    
    # Allow startup
    time.sleep(2)
    
    print("💤 Measuring IDLE State (30 seconds)...")
    
    current_pid = os.getpid()
    p = psutil.Process(current_pid)
    
    # Warmup CPU counter
    p.cpu_percent()
    
    cpu_stats = []
    mem_stats = []
    
    start_measure = time.time()
    
    while (time.time() - start_measure) < 30:
        time.sleep(1.0)
        c = p.cpu_percent(interval=None)
        m = p.memory_info().rss / 1024 / 1024
        cpu_stats.append(c)
        mem_stats.append(m)
        print(f"   CPU: {c:5.1f}% | RAM: {m:6.1f} MB")
        
    # Stop Engine
    engine.stop()
    t.join(timeout=5)
    
    # Analyze
    avg_cpu = statistics.mean(cpu_stats)
    max_cpu = max(cpu_stats)
    avg_mem = statistics.mean(mem_stats)
    mem_drift = mem_stats[-1] - mem_stats[0]
    
    print("\n" + "-"*60)
    print("📊 EFFICIENCY REPORT")
    print("-"*60)
    print(f"Avg CPU Usage: {avg_cpu:.2f}% (Target < 2.0%)")
    print(f"Max CPU Usage: {max_cpu:.2f}%")
    print(f"Avg RAM Usage: {avg_mem:.2f} MB")
    print(f"RAM Drift:     {mem_drift:+.2f} MB (Target ~0)")
    print("-"*60)
    
    if avg_cpu < 2.0: # 1-2% overhead is acceptable for Python
        print("✅ PASS: System is energy efficient (No busy loops detected).")
    else:
        print("❌ FAIL: High idle CPU usage. Check for 'while True: pass' loops.")
        
    print("="*60)

if __name__ == "__main__":
    run_idle_test()
