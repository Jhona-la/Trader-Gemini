import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import main
    print("MAIN LOADED OK")
except Exception as e:
    print(f"MAIN LOAD FAILED: {e}")
