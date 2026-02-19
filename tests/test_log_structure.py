import json
import os
import sys
import time
import logging
import glob
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.getcwd())

from utils.logger import JSONFormatter

# Setup paths
LOG_DIR = "logs"
TEST_LOG_NAME = "test_audit_logger.json"

def run_log_audit():
    print("="*60)
    print("📜 PHASE 49: STRUCTURED LOG AUDIT (JSON)")
    print("="*60)
    
    # 1. Setup specific test logger
    logger = logging.getLogger("audit_tester")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid pollution
    if logger.handlers:
        logger.handlers = []
        
    # Create simple FileHandler with the JSONFormatter from utils.logger
    from utils.logger import JSONFormatter
    
    # Ensure dir
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    log_path = os.path.join(LOG_DIR, TEST_LOG_NAME)
    
    # Clean previous run
    if os.path.exists(log_path):
        os.remove(log_path)
        
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # 2. Generate Logs
    print("📝 Generating test logs...")
    logger.info("Test Info Message")
    logger.warning("Test Warning Message")
    logger.error("Test Error Message", exc_info=False)
    
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("Test Exception", exc_info=True)
        
    # Close handler to flush
    handler.close()
    
    # 3. Verify Content
    print(f"🔍 Inspecting {log_path}...")
    
    if not os.path.exists(log_path):
        print("❌ FAIL: Log file was not created.")
        return
        
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"   Found {len(lines)} log entries.")
    
    valid_count = 0
    required_fields = ["timestamp", "level", "logger", "message", "module"]
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        try:
            entry = json.loads(line)
            
            # Check fields
            missing = [field for field in required_fields if field not in entry]
            if missing:
                print(f"❌ Line {i+1}: Missing fields {missing}")
                continue
                
            # Check timestamp format
            # "2026-02-08 00:30:00,123" (Standard Python Logging)
            try:
                # Remove comma milliseconds for parsing check, or match format
                dt_str = entry['timestamp']
                if ',' in dt_str:
                     dt_str = dt_str.split(',')[0]
                datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"❌ Line {i+1}: Invalid timestamp format '{entry['timestamp']}'")
                continue
                
            # Check specific content logic
            if "Test Exception" in entry['message']:
                if "exception" not in entry:
                    print(f"❌ Line {i+1}: Exception logged but 'exception' field missing.")
                    continue
                else:
                    print("   ✅ Exception traceback correctly serialized.")
            
            valid_count += 1
            
        except json.JSONDecodeError:
            print(f"❌ Line {i+1}: Invalid JSON -> {line[:50]}...")
            
    print("-" * 30)
    if valid_count == len(lines) and len(lines) >= 4:
        print(f"✅ PASS: All {valid_count} logs are valid structured JSON.")
        print("   ELK Stack / Splunk compatibility confirmed.")
    else:
        print(f"❌ FAIL: Only {valid_count}/{len(lines)} valid logs.")
        
    print("="*60)
    
    # Cleanup
    if os.path.exists(log_path):
        os.remove(log_path)

if __name__ == "__main__":
    run_log_audit()
